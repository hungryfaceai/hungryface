// /hungryface/webrtc/receiver/shared/switcher.js
// Lightweight “Trusted device switcher” (receiver-side)
// - NO fetch('wss://…'). You provide `getTrusted()` or we fall back to SecureSignal.
// - Top-right floating button
// - Full fingerprint shown (no truncation)
// - Shows last known room, highlights current peer

/**
 * @typedef {Object} TrustedDevice
 * @property {string} fp              - Device fingerprint (id)
 * @property {string=} name           - Optional friendly name
 * @property {{room?: string}=} lastMeta - Optional metadata previously saved
 */

/**
 * Initialize the Trusted Device Switcher UI.
 *
 * @param {Object} opts
 * @param {() => Promise<TrustedDevice[]>=} opts.getTrusted - async function that returns trusted devices
 * @param {string=} opts.wsEndpoint      - If getTrusted not provided, we’ll use SecureSignal with this endpoint
 * @param {string=} opts.currentPeer     - The fingerprint currently selected in the URL (?peer=…)
 * @param {string=} opts.desiredRoom     - Desired room filter (optional, only used for hinting)
 * @param {(fp: string) => void=} opts.onPick - Called when a device is selected
 */
export function initTrustedSwitcher(opts = {}) {
  const {
    getTrusted = null,
    wsEndpoint = '',
    currentPeer = '',
    desiredRoom = '',
    onPick = () => {},
  } = opts;

  // ---- DOM scaffold --------------------------------------------------------
  const root = document.createElement('div');
  root.id = 'switcher-root';

  const btn = document.createElement('button');
  btn.id = 'switcher-btn';
  btn.type = 'button';
  btn.textContent = 'Switch device';

  // Position top-right if CSS not present
  Object.assign(btn.style, {
    position: 'fixed',
    top: 'max(env(safe-area-inset-top, 0px) + 8px, 8px)',
    right: 'max(env(safe-area-inset-right, 0px) + 8px, 8px)',
    zIndex: 1000,
  });

  const panel = document.createElement('div');
  panel.id = 'switcher-panel';
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-modal', 'false');
  panel.style.position = 'fixed';
  panel.style.top = 'calc(max(env(safe-area-inset-top, 0px) + 8px, 8px) + 40px)';
  panel.style.right = 'max(env(safe-area-inset-right, 0px) + 8px, 8px)';
  panel.style.zIndex = 1001;
  panel.style.display = 'none'; // toggled by JS

  panel.innerHTML = `
    <div class="switcher-card">
      <div class="switcher-head">
        <strong>Trusted devices</strong>
        <div class="spacer"></div>
        <button class="switcher-refresh" type="button" title="Refresh">↻</button>
        <button class="switcher-close" type="button" title="Close">✕</button>
      </div>
      <div class="switcher-subtle">
        ${desiredRoom ? `Desired room: <b>${escapeHtml(desiredRoom)}</b>` : ''}
      </div>
      <ul class="switcher-list"></ul>
      <div class="switcher-foot">
        <small>Tap a device to switch.</small>
      </div>
    </div>
  `;

  // Minimal inline styles to work without external CSS
  const style = document.createElement('style');
  style.textContent = `
    #switcher-panel .switcher-card{
      background: rgba(20,20,20,0.98);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 12px;
      min-width: 300px; max-width: 92vw;
      color: #eee;
      box-shadow: 0 10px 24px rgba(0,0,0,0.35);
      backdrop-filter: blur(10px);
    }
    #switcher-panel .switcher-head{
      display:flex; align-items:center; gap:8px;
      padding:10px 12px; border-bottom:1px solid rgba(255,255,255,0.08);
    }
    #switcher-panel .switcher-head .spacer{flex:1}
    #switcher-panel .switcher-head button{
      background:#222; color:#ddd; border:1px solid #333;
      border-radius:8px; padding:4px 8px; cursor:pointer;
    }
    #switcher-panel .switcher-subtle{
      padding:6px 12px; color:#bbb; font-size:12px; border-bottom:1px solid rgba(255,255,255,0.06);
    }
    #switcher-panel .switcher-list{
      list-style:none; margin:0; padding:4px 0;
      max-height: 45vh; overflow:auto;
    }
    #switcher-panel .switcher-list li{
      padding:10px 12px; border-bottom:1px solid rgba(255,255,255,0.06);
      cursor:pointer;
    }
    #switcher-panel .switcher-list li:last-child{ border-bottom: 0; }
    #switcher-panel .switcher-list li:hover{ background:#121212; }
    #switcher-panel .dev .row1{
      display:flex; align-items:center; gap:8px; margin-bottom:4px;
    }
    #switcher-panel .dev .dev-name{
      font-weight:600; color:#fff; font-size:14px;
    }
    #switcher-panel .dev .badge{
      font-size:11px; color:#eee; background:#2a2a2a;
      border:1px solid #3a3a3a; padding:1px 6px; border-radius:999px;
    }
    #switcher-panel .dev .row2{
      display:flex; align-items:baseline; gap:8px;
    }
    /* Full fingerprint — no truncation */
    #switcher-panel .dev .dev-fp{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size:12px; white-space: pre-wrap; word-break: break-all; overflow: visible; max-width: none;
      color:#cfe;
    }
    #switcher-panel .dev .dev-room{
      margin-left:auto; font-size:12px; color:#bbb;
    }
    #switcher-panel .empty{
      padding:12px; color:#bbb; font-style: italic;
    }
    #switcher-panel .switcher-foot{
      padding:8px 12px; color:#aaa; font-size:12px; border-top:1px solid rgba(255,255,255,0.06);
    }
  `;

  document.head.appendChild(style);
  root.appendChild(btn);
  root.appendChild(panel);
  document.body.appendChild(root);

  const listEl = panel.querySelector('.switcher-list');
  const closeBtn = panel.querySelector('.switcher-close');
  const refreshBtn = panel.querySelector('.switcher-refresh');

  // ---- data source ---------------------------------------------------------
  const fetchTrusted = getTrusted || (async () => {
    if (!wsEndpoint) throw new Error('No getTrusted() or wsEndpoint provided.');
    // Lazy import SecureSignal only if needed
    const { SecureSignal } = await import('/hungryface/webrtc/shared/secure-signal.js');
    const ss = new SecureSignal({ url: wsEndpoint });
    await ss.init();
    return ss.trusted || [];
  });

  // ---- populate list -------------------------------------------------------
  async function populate() {
    listEl.innerHTML = '';
    try {
      const items = await fetchTrusted();
      if (!Array.isArray(items) || !items.length) {
        listEl.innerHTML = `<li class="empty">No trusted devices found.</li>`;
        return;
      }

      // sort by: current first, then name, then fp
      items.sort((a, b) => {
        const ac = a?.fp === currentPeer ? -1 : 0;
        const bc = b?.fp === currentPeer ? -1 : 0;
        if (ac !== bc) return ac - bc;
        const an = (a?.name || '').toLowerCase();
        const bn = (b?.name || '').toLowerCase();
        if (an !== bn) return an < bn ? -1 : 1;
        const af = (a?.fp || '');
        const bf = (b?.fp || '');
        return af < bf ? -1 : 1;
      });

      for (const dev of items) {
        if (!dev || !dev.fp) continue;
        const { fp, name, lastMeta } = dev;
        const room = lastMeta?.room || '(no room)';
        const li = document.createElement('li');
        li.className = 'dev';

        li.innerHTML = `
          <div class="row1">
            <span class="dev-name">${escapeHtml(name || 'Device')}</span>
            ${fp === currentPeer ? '<span class="badge">current</span>' : ''}
          </div>
          <div class="row2">
            <span class="dev-fp" title="${escapeHtml(fp)}">${escapeHtml(fp)}</span>
            <span class="dev-room" title="Last seen room">${escapeHtml(room)}</span>
          </div>
        `;

        // ensure full fp shows even if page CSS overrides
        const fpEl = li.querySelector('.dev-fp');
        if (fpEl) {
          fpEl.style.whiteSpace = 'pre-wrap';
          fpEl.style.wordBreak = 'break-all';
          fpEl.style.overflow = 'visible';
          fpEl.style.maxWidth = 'none';
        }

        li.addEventListener('click', () => {
          try { onPick(fp); } catch (e) { console.error('[Switcher] onPick failed:', e); }
          hidePanel();
        });

        listEl.appendChild(li);
      }
    } catch (e) {
      console.error('[Switcher] failed to load trusted devices:', e);
      listEl.innerHTML = `<li class="empty">Failed to load trusted devices.</li>`;
    }
  }

  // ---- interactions --------------------------------------------------------
  function showPanel() {
    panel.style.display = 'block';
    populate();
    // click-outside to close
    setTimeout(() => {
      document.addEventListener('mousedown', onDocDown, true);
      document.addEventListener('touchstart', onDocDown, { passive: true, capture: true });
      document.addEventListener('keydown', onEsc, true);
    }, 0);
  }
  function hidePanel() {
    panel.style.display = 'none';
    document.removeEventListener('mousedown', onDocDown, true);
    document.removeEventListener('touchstart', onDocDown, true);
    document.removeEventListener('keydown', onEsc, true);
  }
  function togglePanel() {
    (panel.style.display === 'none' || !panel.style.display) ? showPanel() : hidePanel();
  }
  function onDocDown(e) {
    if (!panel.contains(e.target) && e.target !== btn) hidePanel();
  }
  function onEsc(e) {
    if (e.key === 'Escape') hidePanel();
  }

  btn.addEventListener('click', togglePanel);
  closeBtn?.addEventListener('click', hidePanel);
  refreshBtn?.addEventListener('click', populate);
}

// -------- utils --------
function escapeHtml(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
