from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os, requests, io
from bs4 import BeautifulSoup
from pypdf import PdfReader

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

vectorstore = None
current_source = ""


class QueryRequest(BaseModel):
    question: str


class URLRequest(BaseModel):
    url: str


def ingest_text(text: str, source: str):
    global vectorstore, current_source

    # Clear previous content completely before loading new
    if vectorstore:
        try:
            vectorstore.delete_collection()
        except:
            pass
        vectorstore = None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text], metadatas=[{"source": source}])
    vectorstore = Chroma.from_documents(docs, embeddings)
    current_source = source
    return len(docs)


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    reader = PdfReader(io.BytesIO(contents))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    if not text.strip():
        return {"error": "Could not extract text from PDF"}
    chunks = ingest_text(text, f"PDF: {file.filename}")
    return {"status": "ingested", "source": file.filename, "chunks": chunks}


@app.post("/ingest/url")
async def ingest_url(req: URLRequest):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        response = requests.get(req.url, headers=headers, timeout=15)
        response.raise_for_status()
        response.encoding = response.apparent_encoding

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "iframe", "noscript"]):
            tag.decompose()

        main_content = (
            soup.find("article") or
            soup.find("main") or
            soup.find(id="content") or
            soup.find(id="main-content") or
            soup.find(class_="post-content") or
            soup.find(class_="article-body") or
            soup.find(class_="entry-content") or
            soup.find("body")
        )

        text = main_content.get_text(separator="\n", strip=True) if main_content else ""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        text = "\n".join(lines)

        if not text.strip():
            return {"error": "Could not extract text from URL"}

        # Filter out non-English lines
        cleaned = "\n".join(
            line for line in text.splitlines()
            if sum(1 for c in line if ord(c) < 128) > len(line) * 0.5
        )

        if len(cleaned) < 100:
            return {"error": "Page appears to be in a foreign language or is blocked. Try pasting the text directly."}

        chunks = ingest_text(cleaned, f"URL: {req.url}")
        return {"status": "ingested", "source": req.url, "chunks": chunks}

    except Exception as e:
        return {"error": str(e)}


@app.post("/ingest/text")
async def ingest_raw_text(text: str = Form(...)):
    chunks = ingest_text(text, "Pasted text")
    return {"status": "ingested", "chunks": chunks}


@app.post("/ask")
async def ask(req: QueryRequest):
    global vectorstore, current_source

    rag_context = ""
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.invoke(req.question)
        rag_context = "\n\n".join([d.page_content for d in relevant_docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful reading assistant. A user has uploaded 
or linked some content and wants to understand it.

Source: {source}

Relevant content retrieved from the document:
\"\"\"{rag_context}\"\"\"

Answer clearly and concisely based on the content above. 
For summaries, keep it under 4 sentences so it sounds natural when read aloud.
If the answer is not in the content, say so honestly."""),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "source": current_source,
        "rag_context": rag_context[:2000],
        "question": req.question
    })

    return {"answer": answer, "source": current_source}


@app.get("/status")
async def status():
    return {
        "status": "running",
        "has_content": vectorstore is not None,
        "source": current_source
    }

# Serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")