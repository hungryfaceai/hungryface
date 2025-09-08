// Injects help-sheet.html and announces when it's ready (mirrors motion page)
(async () => {
  if (document.getElementById('helpSheet') || document.getElementById('helpBackdrop')) return;
  try {
    const res = await fetch('./help-sheet.html', { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const html = await res.text();
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
