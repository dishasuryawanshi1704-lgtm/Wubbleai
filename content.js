let dwellTimer = null;

document.addEventListener("mouseover", (e) => {
  const el = e.target;
  const text = el.innerText?.trim();
  if (!text || text.length < 20) return;
  clearTimeout(dwellTimer);
  dwellTimer = setTimeout(() => {
    chrome.storage.local.set({ focusedText: text.slice(0, 1000) });
  }, 800);
});

document.addEventListener("mouseup", () => {
  const selected = window.getSelection()?.toString()?.trim();
  if (selected && selected.length > 10) {
    chrome.storage.local.set({ focusedText: selected });
  }
});

window.addEventListener("load", async () => {
  const pageText = document.body.innerText.slice(0, 15000);
  try {
    await fetch("http://127.0.0.1:8000/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: pageText, url: location.href }),
    });
  } catch (e) {
    console.log("Backend not running:", e.message);
  }
});