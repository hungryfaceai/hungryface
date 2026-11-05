(() => {
  // Initialize after DOM is ready (works whether markup was injected or inline)
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }

  function init() {
    const sidebar   = document.getElementById('sidebar');
    const hamburger = document.getElementById('hamburger');
    const inviteModal = document.getElementById('invite-modal'); //made optional
    // If any core element is missing, do nothing.
    //if (!sidebar || !hamburger || !inviteModal) return;
    if (!sidebar || !hamburger) return;

    const navItems  = Array.from(sidebar.querySelectorAll('.nav-item'));
    const isTouch = matchMedia('(pointer: coarse)').matches || 'ontouchstart' in window;

    /* =========================
       Sidebar / Hamburger logic
       ========================= */
    const SIDEBAR_AUTOHIDE_MS = 10000;
    let sidebarTimer = null;

    function hideSidebar() {
      sidebar.classList.remove('visible');
      hamburger.classList.remove('hidden');         // show button when drawer closes
      hamburger.setAttribute('aria-expanded', 'false');
      clearTimeout(sidebarTimer); sidebarTimer = null;
      resetHamburgerFade();
    }
    function showSidebar() {
      sidebar.classList.add('visible');
      hamburger.classList.add('hidden');            // hide button entirely while open
      hamburger.setAttribute('aria-expanded', 'true');
      clearTimeout(sidebarTimer);
      sidebarTimer = setTimeout(hideSidebar, SIDEBAR_AUTOHIDE_MS);
    }
    function toggleSidebar() {
      if (sidebar.classList.contains('visible')) hideSidebar();
      else showSidebar();
    }

    // Click/tap outside the sidebar closes it (ignore clicks on hamburger)
    function clickOutsideHandler(e) {
      if (!sidebar.classList.contains('visible')) return;
      const insideSidebar   = e.target.closest('#sidebar');
      const insideHamburger = e.target.closest('#hamburger');
      if (!insideSidebar && !insideHamburger) hideSidebar();
    }
    document.addEventListener('pointerdown', clickOutsideHandler, { passive: true });
    document.addEventListener('click', clickOutsideHandler, { passive: true });

    // Hamburger auto-fade (disabled on touch so it's always visible there)
    const HAMBURGER_FADE_MS = 4000;
    let hamburgerFadeTimer = null;
    function showHamburger() {
      hamburger.classList.remove('faded');
      hamburger.setAttribute('aria-hidden', 'false');
    }
    function hideHamburgerBtn() {
      hamburger.classList.add('faded');
      hamburger.setAttribute('aria-hidden', 'true');
    }
    function resetHamburgerFade() {
      if (hamburger.classList.contains('hidden')) return; // don't touch while drawer open
      showHamburger();
      clearTimeout(hamburgerFadeTimer);
      if (isTouch) return; // keep visible on iPhone/iPad
      if (document.fullscreenElement) {
        hamburgerFadeTimer = setTimeout(() => {
          if (!sidebar.classList.contains('visible') && !hamburger.classList.contains('hidden')) {
            hideHamburgerBtn();
          }
        }, HAMBURGER_FADE_MS);
      } else {
        showHamburger(); // keep visible when not in fullscreen
      }
    }

    // Discoverability + initial state
    if (isTouch) { hideSidebar(); showHamburger(); } else { showSidebar(); }

    // Toggle via hamburger
    hamburger.addEventListener('click', (e) => {
      e.stopPropagation();
      toggleSidebar();
      resetHamburgerFade();
    });

    // Keep drawer open while interacting
    ['mousemove','pointermove','wheel','touchstart','click','keydown'].forEach(ev =>
      sidebar.addEventListener(ev, () => {
        clearTimeout(sidebarTimer);
        sidebarTimer = setTimeout(hideSidebar, SIDEBAR_AUTOHIDE_MS);
      }, { passive: true })
    );

    // Global activity revives hamburger (mainly in fullscreen)
    ['mousemove','pointermove','touchstart','keydown','click'].forEach(ev =>
      window.addEventListener(ev, () => resetHamburgerFade(), { passive: true })
    );

    // Close drawer with Escape
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') hideSidebar();
    });

    // Keep behavior in sync with fullscreen transitions
    document.addEventListener('fullscreenchange', resetHamburgerFade);

    // ---------- Menu interactions (highlight only; navigation via anchors) ----------
    /*function setActive(itemName) {
      navItems.forEach(btn => btn.classList.toggle('active', btn.dataset.item === itemName));
    }*/

    // ---------- Menu interactions (highlight only; navigation via anchors) ----------
    function setActive(itemName) {
      navItems.forEach(btn => {
        const isActive = btn.dataset.item === itemName;
        btn.classList.toggle('active', isActive);
        if (isActive) btn.setAttribute('aria-current', 'page');
        else btn.removeAttribute('aria-current');
      });
    }
    
    // Auto-highlight based on current URL (longest matching path prefix)
    function norm(p) {
      return new URL(p, location.href).pathname
        .replace(/index\.html$/,'')
        .replace(/\/+$/,'/') || '/';
    }
    function setActiveFromUrl() {
      const here = norm(location.pathname);
      const anchors = Array.from(sidebar.querySelectorAll('.nav-item[href]'));
      let best = null, bestLen = -1;
    
      for (const a of anchors) {
        const path = norm(a.getAttribute('href'));
        if (here.startsWith(path) && path.length > bestLen) {
          best = a; bestLen = path.length;
        }
      }
      if (best) {
        navItems.forEach(btn => {
          const isActive = btn === best;
          btn.classList.toggle('active', isActive);
          if (isActive) btn.setAttribute('aria-current', 'page');
          else btn.removeAttribute('aria-current');
        });
        return true;
      }
      return false;
    }
    
    // Initialize highlight: URL first, fallback to <body data-menu-active="...">
    if (!setActiveFromUrl()) {
      const initialActive = document.body?.dataset?.menuActive;
      if (initialActive) setActive(initialActive);
    }
    
    sidebar.addEventListener('click', (e) => {
      const btn = e.target.closest('.nav-item');
      if (!btn) return;
      const item = btn.dataset.item;
      if (item) setActive(item);
      // Let anchors navigate normally; buttons (invite/contact) are handled below.
      clearTimeout(sidebarTimer);
      sidebarTimer = setTimeout(hideSidebar, SIDEBAR_AUTOHIDE_MS);
    });

    // Optional: initialize active item from <body data-menu-active="video">
    /*const initialActive = document.body?.dataset?.menuActive;
    if (initialActive) setActive(initialActive);*/

    // Cleanup timers on unload
    window.addEventListener('beforeunload', () => {
      clearTimeout(sidebarTimer);
      clearTimeout(hamburgerFadeTimer);
    });

    /* =======================
       QR Invite modal handling
       ======================= */
    (function initInvite() {
      const trigger  = document.getElementById('menu-invite');
      const modal    = document.getElementById('invite-modal');
      if (!trigger || !modal) return;

      const canvas   = document.getElementById('invite-qr');
      const urlText  = document.getElementById('invite-url');
      const btnCopy  = document.getElementById('invite-copy');
      const btnShare = document.getElementById('invite-share');
      const btnClose = modal.querySelector('.invite-close');
      const statusEl = document.getElementById('invite-status');

      // Per-page override: window.HF_INVITE_URL or <body data-invite-url="…">; default: Naptio home.
      const INVITE_URL = window.HF_INVITE_URL || document.body?.dataset?.inviteUrl || 'https://hungryfaceai.github.io/hungryface/home/';
      const SHARE = { title: 'Naptio', text: 'Naptio turns two phones into a private baby monitor. Take a look:' };

      // Share where the device supports it (phones); otherwise Copy is the main action.
      const canShare = 'share' in navigator && (!navigator.canShare || navigator.canShare({ url: INVITE_URL }));
      if (!canShare) { btnShare.hidden = true; btnCopy.classList.add('invite-btn-primary'); }

      // QR code drawn on this device (QRious, as on the pairing page): the link isn't sent to a QR service.
      let qrReady = null;
      function loadQRious() {
        if (window.QRious) return Promise.resolve();
        return qrReady || (qrReady = new Promise((resolve, reject) => {
          const s = document.createElement('script');
          s.src = 'https://cdn.jsdelivr.net/npm/qrious@4.0.2/dist/qrious.min.js';
          s.onload = resolve; s.onerror = () => { qrReady = null; reject(new Error('QR library failed to load')); };
          document.head.appendChild(s);
        }));
      }
      async function drawQR() {
        try {
          await loadQRious();
          // Dark modules on white, with a quiet zone: the most reliable for phone cameras.
          new window.QRious({ element: canvas, value: INVITE_URL, size: 440, padding: 24, level: 'M', background: '#ffffff', foreground: '#000000' });
          canvas.classList.remove('invite-qr-failed');
        } catch {
          canvas.classList.add('invite-qr-failed');
          canvas.setAttribute('aria-label', 'QR code unavailable — use Share or Copy link');
        }
      }

      function say(msg) { statusEl.textContent = msg; clearTimeout(say.t); say.t = setTimeout(() => { statusEl.textContent = ''; }, 2500); }

      let lastFocus = null;
      function focusables() { return [...modal.querySelectorAll('button:not([hidden])')].filter(b => b.offsetParent !== null); }
      function onKey(e) {
        if (e.key === 'Escape') { e.preventDefault(); closeInvite(); return; }
        if (e.key !== 'Tab') return;
        const f = focusables(); if (!f.length) return;
        const first = f[0], last = f[f.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
      function openInvite() {
        lastFocus = document.activeElement;
        urlText.textContent = INVITE_URL.replace(/^https?:\/\//, '').replace(/\/$/, '');
        drawQR();
        statusEl.textContent = '';
        modal.classList.add('visible');
        modal.setAttribute('aria-hidden', 'false');
        document.addEventListener('keydown', onKey);
        const primary = focusables().find(b => b.classList.contains('invite-btn-primary'));
        setTimeout(() => (primary || btnClose).focus(), 0);
      }
      function closeInvite() {
        modal.classList.remove('visible');
        modal.setAttribute('aria-hidden', 'true');
        document.removeEventListener('keydown', onKey);
        if (lastFocus && lastFocus.focus && document.contains(lastFocus)) lastFocus.focus();
      }

      async function copyLink() {
        try { await navigator.clipboard.writeText(INVITE_URL); say('Link copied'); btnCopy.textContent = 'Copied'; setTimeout(() => { btnCopy.textContent = 'Copy link'; }, 1500); }
        catch { const r = document.createRange(); r.selectNodeContents(urlText); const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r); say('Press and hold the link to copy it'); }
      }

      trigger.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); openInvite(); });
      modal.addEventListener('click', (e) => { if (e.target === modal) closeInvite(); });
      btnClose && btnClose.addEventListener('click', closeInvite);
      btnCopy && btnCopy.addEventListener('click', copyLink);
      btnShare && btnShare.addEventListener('click', async () => {
        try { await navigator.share({ ...SHARE, url: INVITE_URL }); }
        catch (err) { if (!err || err.name !== 'AbortError') copyLink(); }   // cancelled: do nothing; failed: copy instead
      });
    })();
  }
})();
