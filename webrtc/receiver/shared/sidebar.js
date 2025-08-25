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
    const inviteModal = document.getElementById('invite-modal');
    // If any core element is missing, do nothing.
    if (!sidebar || !hamburger || !inviteModal) return;

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
    //https://chatgpt.com/c/68ac564d-b3e4-8328-ae07-598697bbd50c
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
      if (!trigger) return;

      const modal    = document.getElementById('invite-modal');
      const qrImg    = document.getElementById('invite-qr');
      const urlInput = document.getElementById('invite-url');
      const btnCopy  = document.getElementById('invite-copy');
      const btnOpen  = document.getElementById('invite-open');
      const btnShare = document.getElementById('invite-share');
      const btnClose = modal.querySelector('.invite-close');

      // Allow per-page override via:
      // 1) window.HF_INVITE_URL
      // 2) <body data-invite-url="...">
      // 3) default to HungryFace home
      const INVITE_URL = window.HF_INVITE_URL
        || document.body?.dataset?.inviteUrl
        || 'https://hungryfaceai.github.io/hungryface/home/';

      const QR_ENDPOINT = 'https://api.qrserver.com/v1/create-qr-code/?size=220x220&format=svg&data=';

      function toast(msg) {
        const el = document.createElement('div');
        el.className = 'invite-toast';
        el.textContent = msg;
        document.body.appendChild(el);
        requestAnimationFrame(() => el.classList.add('show'));
        setTimeout(() => {
          el.classList.remove('show');
          el.addEventListener('transitionend', () => el.remove(), { once: true });
        }, 1400);
      }

      function openInvite() {
        const url = INVITE_URL;
        if (qrImg) {
          qrImg.src = QR_ENDPOINT + encodeURIComponent(url);
          qrImg.alt = 'QR code for ' + url;
        }
        if (urlInput) {
          urlInput.value = url;
          setTimeout(() => {
            try { urlInput.focus({ preventScroll: true }); urlInput.setSelectionRange(url.length, url.length); } catch {}
            urlInput.scrollLeft = urlInput.scrollWidth;
          }, 0);
        }
        if (btnOpen) btnOpen.href = url;

        modal.classList.add('visible');
        modal.setAttribute('aria-hidden', 'false');

        document.addEventListener('keydown', onEsc, { once: true });
      }

      function closeInvite() {
        modal.classList.remove('visible');
        modal.setAttribute('aria-hidden', 'true');
      }

      function onEsc(e) { if (e.key === 'Escape') closeInvite(); }

      trigger.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        openInvite();
      });

      modal.addEventListener('click', (e) => { if (e.target === modal) closeInvite(); });
      if (btnClose) btnClose.addEventListener('click', closeInvite);

      if (btnCopy) {
        btnCopy.addEventListener('click', async () => {
          try {
            await navigator.clipboard.writeText(urlInput.value);
            toast('Link copied');
          } catch {
            urlInput?.select?.();
            toast('Copy failed — select and copy');
          }
        });
      }
      
      if (btnShare) {
      const canWebShare = typeof navigator !== 'undefined' && 'share' in navigator;
      // Hide the Share button if unsupported (most desktops)
      if (!canWebShare) btnShare.style.display = 'none';
    
      btnShare.addEventListener('click', async () => {
        const url = (urlInput && urlInput.value) || INVITE_URL;
        try {
          // Extra guard for some iOS versions
          if (navigator.canShare && !navigator.canShare({ url })) {
            throw new Error('URL sharing not supported');
          }
          await navigator.share({
            title: 'HungryFace',
            text: 'Join me on HungryFace:',
            url
          });
        } catch (err) {
          // Ignore user-cancel; for other errors, fall back to copying the link
          if (err && err.name === 'AbortError') return;
          try {
            await navigator.clipboard.writeText(url);
            toast('Link copied');
          } catch {
            urlInput?.select?.();
            toast('Share failed — link selected to copy');
          }
        }
      });
    }
          
    })();
  }
})();
