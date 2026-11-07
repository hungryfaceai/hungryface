// /hungryface/webrtc/receiver/shared/help-sheet.js
(async () => {
  // Avoid double-insert if this script runs twice
  if (document.getElementById('helpSheet') || document.getElementById('helpBackdrop')) return;

  try {
    const res = await fetch('/hungryface/webrtc/receiver/motion/help-sheet.html', { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const html = await res.text();
    // Shared, readable help styles (both Movement and Prone help use them)
    if (!document.querySelector('link[data-help-css]')) {
      const l = document.createElement('link');
      l.rel = 'stylesheet';
      l.href = '/hungryface/webrtc/receiver/shared/help-sheet.css';
      l.setAttribute('data-help-css', '');
      document.head.appendChild(l);
    }

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
