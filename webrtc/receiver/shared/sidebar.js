(() => {
  // Initialize after DOM is ready (works whether markup was injected or inline)
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }

  function init() {
    const sidebar = document.getElementById('sidebar');
    const hamburger = document.getElementById('hamburger');
    // If any core element is missing, do nothing.
    if (!sidebar || !hamburger) return;

    const navItems = Array.from(sidebar.querySelectorAll('.nav-item'));
    const isTouch = matchMedia('(pointer: coarse)').matches || 'ontouchstart' in window;

    /* =========================
       Sidebar / Hamburger logic
       ========================= */
    const SIDEBAR_AUTOHIDE_MS = 10000;
    let sidebarTimer = null;

    function hideSidebar() {
      sidebar.classList.remove('visible');
      hamburger.classList.remove('hidden'); // show button when drawer closes
      hamburger.setAttribute('aria-expanded', 'false');
      clearTimeout(sidebarTimer);
      sidebarTimer = null;
      resetHamburgerFade();
    }
    function showSidebar() {
      sidebar.classList.add('visible');
      hamburger.classList.add('hidden'); // hide button entirely while open
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
      const insideSidebar = e.target.closest('#sidebar');
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
    if (isTouch) {
      hideSidebar();
      showHamburger();
    } else {
      showSidebar();
    }

    // Toggle via hamburger
    hamburger.addEventListener('click', (e) => {
      e.stopPropagation();
      toggleSidebar();
      resetHamburgerFade();
    });

    // Keep drawer open while interacting
    ['mousemove', 'pointermove', 'wheel', 'touchstart', 'click', 'keydown'].forEach((ev) =>
      sidebar.addEventListener(
        ev,
        () => {
          clearTimeout(sidebarTimer);
          sidebarTimer = setTimeout(hideSidebar, SIDEBAR_AUTOHIDE_MS);
        },
        { passive: true }
      )
    );

    // Global activity revives hamburger (mainly in fullscreen)
    ['mousemove', 'pointermove', 'touchstart', 'keydown', 'click'].forEach((ev) =>
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
      navItems.forEach((btn) => {
        const isActive = btn.dataset.item === itemName;
        btn.classList.toggle('active', isActive);
        if (isActive) btn.setAttribute('aria-current', 'page');
        else btn.removeAttribute('aria-current');
      });
    }

    // Auto-highlight based on current URL (longest matching path prefix)
    function norm(p) {
      return new URL(p, location.href).pathname.replace(/index\.html$/, '').replace(/\/+$/, '/') || '/';
    }
    function setActiveFromUrl() {
      const here = norm(location.pathname);
      const anchors = Array.from(sidebar.querySelectorAll('.nav-item[href]'));
      let best = null,
        bestLen = -1;

      for (const a of anchors) {
        const path = norm(a.getAttribute('href'));
        if (here.startsWith(path) && path.length > bestLen) {
          best = a;
          bestLen = path.length;
        }
      }
      if (best) {
        navItems.forEach((btn) => {
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
       "Invite a friend" window: shared with the home page (/hungryface/shared/invite.js)
       ======================= */
    (function initInvite() {
      const trigger = document.getElementById('menu-invite');
      if (!trigger) return;
      if (window.NaptioInvite) {
        window.NaptioInvite.attach(trigger);
        return;
      }
      const s = document.createElement('script');
      s.src = '/hungryface/shared/invite.js';
      s.onload = () => window.NaptioInvite && window.NaptioInvite.attach(trigger);
      document.head.appendChild(s);
    })();
  }
})();
