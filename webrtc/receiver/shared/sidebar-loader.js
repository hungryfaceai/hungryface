// sidebar-loader.js — inject the shared receiver sidebar (markup + behaviour) into a page.
//
// Usage (one line per page):
//   <script type="module" src="/hungryface/webrtc/receiver/shared/sidebar-loader.js"></script>
//
// Same behaviour as the inline loader it replaces: fetch sidebar.html, prepend it to <body>
// so the hamburger sits above page content, then load sidebar.js once the markup exists.

const BASE = '/hungryface/webrtc/receiver/shared/';

async function loadSidebar() {
  try {
    const res = await fetch(BASE + 'sidebar.html', { cache: 'no-cache' });
    if (!res.ok) throw new Error('sidebar.html ' + res.status);   // never inject an error page into the menu
    const html = await res.text();
    const wrap = document.createElement('div');
    wrap.innerHTML = html.trim();
    document.body.prepend(...wrap.childNodes);

    await new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = BASE + 'sidebar.js';
      s.onload = resolve;
      s.onerror = reject;
      document.body.appendChild(s);
    });
  } catch (err) {
    console.error('[Sidebar] failed to load shared assets:', err);
  }
}

loadSidebar();
