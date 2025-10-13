//https://chatgpt.com/c/68ec3f8a-24cc-832e-a723-6bab67bf563b
// /webrtc/receiver/shared/mini-dash.js
// Reusable "Mini dashboard" overlay with drag, resize, zoom, persistence.
// Mounts only if localStorage['naptio:miniDash:enabled'] === 'on'.
// Works on desktop & phones (Pointer Events). Safe-area aware.
//
// Public API (attached to window.MiniDash):
//   installMiniDashboard(opts)
//   MiniDash.show(), MiniDash.hide(), MiniDash.toggle()
//   MiniDash.setSrc(url), MiniDash.refresh()          // re-checks enabled flag
//   MiniDash.resetState()                              // forget pos/size/zoom
//
// Storage keys:
//   enabled: localStorage['naptio:miniDash:enabled']   -> "on"/"off"
//   pos:     localStorage['naptio:miniDash:pos']       -> {mode:'absolute',left,top} or {mode:'corner',corner:'TR'}
//   size:    localStorage['naptio:miniDash:size']      -> {w,h}
//   zoom:    localStorage['naptio:miniDash:zoom']      -> number (0.25–1.25)

export function installMiniDashboard(opts = {}) {
  const SRC_DEFAULT = 'https://hungryfaceai.github.io/hungryface/webrtc/receiver/shared/alerts/dashboard/';
  const src = opts.src || SRC_DEFAULT;
  const featureKey = opts.featureKey || 'naptio:miniDash:enabled';

  const KEYS = {
    pos:  'naptio:miniDash:pos',
    size: 'naptio:miniDash:size',
    zoom: 'naptio:miniDash:zoom',
  };

  const CSS_ID = 'naptio-mini-dash-style';
  const ROOT_ID = 'naptio-mini-dash';

  // ---- Utilities ----
  const px = (n) => `${Math.round(n)}px`;
  const save = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} };
  const load = (k, fallback=null) => { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : fallback; } catch { return fallback; } };
  const el = (sel) => document.querySelector(sel);
  const make = (tag, cls) => { const x = document.createElement(tag); if (cls) x.className = cls; return x; };

  function ensureStyle() {
    if (document.getElementById(CSS_ID)) return;
    const s = document.createElement('style');
    s.id = CSS_ID;
    s.textContent = `
:root {
  --miniDash-safe-top:    env(safe-area-inset-top, 0px);
  --miniDash-safe-right:  env(safe-area-inset-right, 0px);
  --miniDash-safe-bottom: env(safe-area-inset-bottom, 0px);
  --miniDash-safe-left:   env(safe-area-inset-left, 0px);
  --miniDash-zoom: 1;
}
#${ROOT_ID} {
  position: fixed;
  top: calc(12px + var(--miniDash-safe-top));
  right: calc(12px + var(--miniDash-safe-right));
  width: min(40vw, 420px);
  height: min(40vh, 280px);
  background: #0b0b0b;
  border: 1px solid #2a2a2a;
  border-radius: 14px;
  box-shadow: 0 10px 30px rgba(0,0,0,.55);
  overflow: hidden;
  z-index: 9999;
  touch-action: none; /* avoid scroll hijacking during drag on touch */
}
#${ROOT_ID}[hidden] { display: none !important; }

#${ROOT_ID} .md-titlebar {
  position: relative;
  height: 38px;
  display: grid;
  grid-template-columns: 1fr auto auto auto;
  align-items: center;
  gap: 8px;
  padding: 0 8px 0 12px;
  background: #121212; border-bottom: 1px solid #222;
  font-size: 13px; letter-spacing: .2px; color:#bbb;
  user-select: none;
  cursor: move;
}
#${ROOT_ID}.dragging .md-titlebar { cursor: grabbing; }

#${ROOT_ID} .md-title {
  font-weight: 600; color:#ddd;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

#${ROOT_ID} .md-btn {
  appearance:none; border:0; margin:0;
  height: 26px; border-radius: 8px; background:#1a1a1a; color:#eee; cursor:pointer;
  display:flex; align-items:center; justify-content:center;
  padding:0 8px; font-weight:600;
}
#${ROOT_ID} .md-btn:hover { background:#222; }
#${ROOT_ID} .md-icon {
  width: 26px; height: 26px; border-radius: 50%; background:#1a1a1a; color:#eee; cursor:pointer;
  display:flex; align-items:center; justify-content:center; border:0;
}
#${ROOT_ID} .md-icon:hover { background:#252525; }

#${ROOT_ID} .md-zoom {
  display:flex; align-items:center; gap:6px; cursor: default; user-select: none;
}
#${ROOT_ID} .md-zoom input[type="range"] {
  width: 120px; height: 6px; appearance: none; background:#1a1a1a; border-radius: 999px; outline:none;
}
#${ROOT_ID} .md-zoom input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none; appearance: none;
  width: 16px; height: 16px; border-radius: 50%; background:#ddd; border:1px solid #444;
}
#${ROOT_ID} .md-zoom input[type="range"]::-moz-range-thumb {
  width: 16px; height: 16px; border-radius: 50%; background:#ddd; border:1px solid #444;
}
#${ROOT_ID} .md-zoom .pct { width:48px; text-align:right; }

#${ROOT_ID} .md-viewport { position:absolute; top:38px; left:0; right:0; bottom:0; overflow:hidden; }
#${ROOT_ID} .md-scale {
  width: calc(100% / var(--miniDash-zoom));
  height: calc(100% / var(--miniDash-zoom));
  transform: scale(var(--miniDash-zoom));
  transform-origin: 0 0;
}
#${ROOT_ID} iframe {
  width:100%; height:100%; border:0; display:block; background:#0f0f14;
}

#${ROOT_ID} .md-resize {
  position: absolute;
  right: 6px; bottom: 6px;
  width: 16px; height: 16px;
  cursor: nwse-resize; opacity: .75;
  background: linear-gradient(135deg, transparent 0 45%, #404040 45% 55%, transparent 55% 100%);
  z-index: 3;
}
#${ROOT_ID}.resizing .md-resize { opacity: 1; }

@media (max-width: 640px) {
  #${ROOT_ID} { width: min(96vw, 520px); height: min(48vh, 380px); right: calc(8px + var(--miniDash-safe-right)); }
  #${ROOT_ID} .md-zoom input[type="range"] { width: 90px; }
}
`;
    document.head.appendChild(s);
  }

  function clampToViewport(box) {
    const rect = box.getBoundingClientRect();
    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;
    const styles = getComputedStyle(document.documentElement);
    const safeTop = parseFloat(styles.getPropertyValue('--miniDash-safe-top')) || 0;
    const safeRight = parseFloat(styles.getPropertyValue('--miniDash-safe-right')) || 0;
    const safeBottom = parseFloat(styles.getPropertyValue('--miniDash-safe-bottom')) || 0;
    const safeLeft = parseFloat(styles.getPropertyValue('--miniDash-safe-left')) || 0;
    const margin = 8;

    const minLeft = margin + safeLeft;
    const minTop = margin + safeTop;
    const maxLeft = vw - rect.width - margin - safeRight;
    const maxTop = vh - rect.height - margin - safeBottom;

    return {
      left: Math.max(minLeft, Math.min(maxLeft, rect.left)),
      top: Math.max(minTop, Math.min(maxTop, rect.top)),
    };
  }

  function clampSize(box, w, h) {
    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;
    const styles = getComputedStyle(document.documentElement);
    const safeRight = parseFloat(styles.getPropertyValue('--miniDash-safe-right')) || 0;
    const safeBottom = parseFloat(styles.getPropertyValue('--miniDash-safe-bottom')) || 0;
    const margin = 8;
    const maxW = vw - margin*2 - safeRight;
    const maxH = vh - margin*2 - safeBottom;
    const minW = 220, minH = 140;
    return {
      w: Math.max(minW, Math.min(maxW, w)),
      h: Math.max(minH, Math.min(maxH, h)),
    };
  }

  function snapTopRight(box) {
    box.style.left = 'auto';
    box.style.bottom = 'auto';
    box.style.right = 'calc(12px + var(--miniDash-safe-right))';
    box.style.top = 'calc(12px + var(--miniDash-safe-top))';
    save(KEYS.pos, { mode: 'corner', corner: 'TR' });
  }

  function setZoom(root, z, slider, label) {
    z = Math.max(0.25, Math.min(1.25, z));
    root.style.setProperty('--miniDash-zoom', z);
    if (slider) slider.value = String(z);
    if (label)  label.textContent = `${Math.round(z * 100)}%`;
    save(KEYS.zoom, z);
  }

  function build() {
    ensureStyle();
    if (document.getElementById(ROOT_ID)) return document.getElementById(ROOT_ID);

    const root = make('div'); root.id = ROOT_ID; root.hidden = true;

    const titlebar = make('div', 'md-titlebar');
    const title = make('div', 'md-title'); title.textContent = 'Dashboard';

    // Zoom controls
    const zoomWrap = make('div', 'md-zoom');
    const zMinus = make('button', 'md-btn'); zMinus.textContent = '−';
    const zSlider = document.createElement('input');
    zSlider.type = 'range'; zSlider.min = '0.25'; zSlider.max = '1.25'; zSlider.step = '0.05'; zSlider.value = '1';
    const zPlus  = make('button', 'md-btn'); zPlus.textContent = '+';
    const zPct   = make('span', 'pct'); zPct.textContent = '100%';
    zoomWrap.append(zMinus, zSlider, zPlus, zPct);

    // Snap-to-top-right
    const snapBtn = make('button', 'md-btn'); snapBtn.title = 'Snap top-right'; snapBtn.textContent = '↗︎';

    // Close
    const closeBtn = make('button', 'md-icon'); closeBtn.setAttribute('aria-label', 'Close'); closeBtn.textContent = '×';

    titlebar.append(title, zoomWrap, snapBtn, closeBtn);

    // Viewport + scale wrapper + iframe
    const viewport = make('div', 'md-viewport');
    const scaleWrap = make('div', 'md-scale');
    const iframe = document.createElement('iframe');
    iframe.loading = 'lazy';
    iframe.referrerPolicy = 'no-referrer';
    iframe.allow = 'autoplay; fullscreen; picture-in-picture; clipboard-read; clipboard-write; camera; microphone';
    iframe.allowFullscreen = true;
    scaleWrap.appendChild(iframe);
    viewport.appendChild(scaleWrap);

    // Resize grip
    const resizeGrip = make('div', 'md-resize');

    root.append(titlebar, viewport, resizeGrip);
    document.body.appendChild(root);

    // ----- Wire up behavior -----

    // Set src lazily once we show the UI
    function ensureSrc() {
      if (!iframe.src) iframe.src = src;
    }

    // Zoom
    const onSetZoom = (z) => setZoom(document.documentElement, z, zSlider, zPct);
    zSlider.addEventListener('input', () => onSetZoom(parseFloat(zSlider.value)));
    zMinus.addEventListener('click', () => onSetZoom((parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--miniDash-zoom')) || 1) - 0.05));
    zPlus.addEventListener('click',  () => onSetZoom((parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--miniDash-zoom')) || 1) + 0.05));

    // Ctrl/⌘ + wheel to zoom
    root.addEventListener('wheel', (e) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      const cur = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--miniDash-zoom')) || 1;
      onSetZoom(cur + (e.deltaY > 0 ? -0.05 : 0.05));
    }, { passive: false });

    // Double-click/tap title to reset
    let lastTap = 0;
    titlebar.addEventListener('click', () => {
      const now = Date.now();
      if (now - lastTap < 350) onSetZoom(1);
      lastTap = now;
    });

    // Drag
    let dragging = false, dragDX = 0, dragDY = 0;
    titlebar.addEventListener('pointerdown', (e) => {
      // ignore drags that start on controls
      if (e.target.closest('.md-btn') || e.target === closeBtn) return;
      if (root.hidden) return;
      if (e.pointerType === 'mouse' && e.button !== 0) return;

      dragging = true;
      root.classList.add('dragging');
      const rect = root.getBoundingClientRect();
      dragDX = e.clientX - rect.left;
      dragDY = e.clientY - rect.top;

      root.style.left = px(rect.left);
      root.style.top  = px(rect.top);
      root.style.right = 'auto';
      root.style.bottom = 'auto';

      iframe.style.pointerEvents = 'none';
      titlebar.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    titlebar.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const left = e.clientX - dragDX;
      const top  = e.clientY - dragDY;
      root.style.left = px(left);
      root.style.top  = px(top);
    });
    function endDrag(e) {
      if (!dragging) return;
      dragging = false;
      root.classList.remove('dragging');
      iframe.style.pointerEvents = '';
      titlebar.releasePointerCapture?.(e.pointerId);

      // Clamp into viewport and persist
      const clamped = clampToViewport(root);
      root.style.left = px(clamped.left);
      root.style.top  = px(clamped.top);
      save(KEYS.pos, { mode: 'absolute', left: clamped.left, top: clamped.top });
    }
    titlebar.addEventListener('pointerup', endDrag);
    titlebar.addEventListener('pointercancel', endDrag);

    // Resize
    let resizing = false, startX = 0, startY = 0, startW = 0, startH = 0;
    resizeGrip.addEventListener('pointerdown', (e) => {
      if (root.hidden) return;
      resizing = true;
      root.classList.add('resizing');
      const r = root.getBoundingClientRect();
      startX = e.clientX; startY = e.clientY;
      startW = r.width;   startH = r.height;
      iframe.style.pointerEvents = 'none';
      resizeGrip.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    resizeGrip.addEventListener('pointermove', (e) => {
      if (!resizing) return;
      const { w, h } = clampSize(root, startW + (e.clientX - startX), startH + (e.clientY - startY));
      root.style.width = px(w);
      root.style.height = px(h);
      // keep on-screen while resizing
      const clamped = clampToViewport(root);
      root.style.left = px(clamped.left);
      root.style.top  = px(clamped.top);
    });
    function endResize(e) {
      if (!resizing) return;
      resizing = false;
      root.classList.remove('resizing');
      iframe.style.pointerEvents = '';
      resizeGrip.releasePointerCapture?.(e.pointerId);
      save(KEYS.size, { w: root.offsetWidth, h: root.offsetHeight });
      // persist new position as well (after clamp)
      const r = root.getBoundingClientRect();
      save(KEYS.pos, { mode: 'absolute', left: r.left, top: r.top });
    }
    resizeGrip.addEventListener('pointerup', endResize);
    resizeGrip.addEventListener('pointercancel', endResize);

    // Snap & Close
    snapBtn.addEventListener('click', () => {
      snapTopRight(root);
      const r = root.getBoundingClientRect();
      save(KEYS.pos, { mode: 'absolute', left: r.left, top: r.top });
    });
    closeBtn.addEventListener('click', () => { root.hidden = true; });

    // Keep inside viewport on resize/rotate
    window.addEventListener('resize', () => {
      if (root.hidden) return;
      const clamp = clampToViewport(root);
      root.style.left = px(clamp.left);
      root.style.top  = px(clamp.top);
    });

    // Restore zoom/size/pos
    (function restore() {
      const z = load(KEYS.zoom, 1); setZoom(document.documentElement, z, zSlider, zPct);
      const sz = load(KEYS.size, null);
      if (sz?.w && sz?.h) { root.style.width = px(sz.w); root.style.height = px(sz.h); }
      const pos = load(KEYS.pos, null);
      if (pos?.mode === 'absolute') {
        root.style.left = px(pos.left); root.style.top = px(pos.top);
        root.style.right = 'auto'; root.style.bottom = 'auto';
      } else {
        snapTopRight(root);
      }
      // final clamp
      const c = clampToViewport(root);
      root.style.left = px(c.left);
      root.style.top  = px(c.top);
    })();

    // Public helpers bound to window for convenience
    function show() { ensureSrc(); root.hidden = false; }
    function hide() { root.hidden = true; }
    function toggle() { if (root.hidden) show(); else hide(); }
    function setSrc(url) { iframe.removeAttribute('srcdoc'); iframe.src = url || SRC_DEFAULT; }
    function resetState() {
      ['naptio:miniDash:pos','naptio:miniDash:size','naptio:miniDash:zoom'].forEach(k => localStorage.removeItem(k));
      // Reset zoom/pos/size immediately
      setZoom(document.documentElement, 1, zSlider, zPct);
      root.style.width = ''; root.style.height = '';
      snapTopRight(root);
    }

    window.MiniDash = Object.assign(window.MiniDash || {}, {
      show, hide, toggle, setSrc, resetState,
      refresh() { // re-check enabled flag and show/hide
        const enabled = (localStorage.getItem(featureKey) || 'off') === 'on';
        if (enabled) { setSrc(src); show(); } else { hide(); }
      }
    });

    // Initial visibility (only if enabled)
    const enabled = (localStorage.getItem(featureKey) || 'off') === 'on';
    if (enabled) { setSrc(src); root.hidden = false; } else { root.hidden = true; }

    // Esc to hide
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hide(); });

    return root;
  }

  const root = build();
  return root;
}

// Auto-install when imported as a module (with default options)
if (typeof window !== 'undefined' && !window.__MiniDashAuto) {
  window.__MiniDashAuto = true;
  installMiniDashboard();
}
