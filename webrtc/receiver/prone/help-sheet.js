// /hungryface/webrtc/receiver/prone/help-sheet.js
// Loads prone/help-sheet.html and injects the Help side-sheet (#helpSheet, #helpBackdrop) into the page,
// plus the shared help-sheet.css. Fires a 'help:ready' event on document once the markup is in the DOM, so the
// Prone page can wire up open/close handlers. Does nothing if the sheet was already injected.
(async () => {
  if (document.getElementById('helpSheet') || document.getElementById('helpBackdrop')) return;
  try {
    const res = await fetch('/hungryface/webrtc/receiver/prone/help-sheet.html', { cache: 'no-cache' });
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

    const frag = document.createDocumentFragment();
    while (wrap.firstChild) frag.appendChild(wrap.firstChild);
    document.body.appendChild(frag);

    document.dispatchEvent(new CustomEvent('help:ready'));
  } catch (err) {
    console.error('[HelpSheet] failed to load:', err);
  }
})();
