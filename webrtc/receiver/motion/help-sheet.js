// /hungryface/webrtc/receiver/shared/help-sheet.js
(async () => {
  // Avoid double-insert if this script runs twice
  if (document.getElementById('helpSheet') || document.getElementById('helpBackdrop')) return;

  try {
    const res = await fetch('/hungryface/webrtc/receiver/motion/help-sheet.html', { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const html = await res.text();

    const wrap = document.createElement('div');
    wrap.innerHTML = html.trim();

    // Move all elements out of the wrapper into the DOM (keeps two top-level nodes)
    const frag = document.createDocumentFragment();
    while (wrap.firstChild) frag.appendChild(wrap.firstChild);
    document.body.appendChild(frag);

    // Let other scripts know the sheet is in the DOM
    document.dispatchEvent(new CustomEvent('help:ready'));
  } catch (err) {
    console.error('[HelpSheet] failed to load:', err);
  }
})();
