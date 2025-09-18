// webrtc/receiver/shared/alerts/pill.js
// Creates (or wires) the Alerts button + red badge (iOS style)

function injectStyleOnce(id, css) {
  if (document.getElementById(id)) return;
  const s = document.createElement('style'); s.id = id; s.textContent = css; document.head.appendChild(s);
}

injectStyleOnce('alerts-pill-css', `
  .alerts-btn{ padding:10px 12px; border-radius:10px; border:1px solid rgba(255,255,255,.4);
               background:transparent; color:#fff; font-weight:600; cursor:pointer; }
  .alerts-btn:active{ transform:scale(.98); }
  .alerts-badge{
    display:inline-flex; align-items:center; justify-content:center;
    min-width:18px; height:18px; padding:0 5px; border-radius:999px;
    background:#ef4444; color:#fff; font-size:12px; line-height:1; font-weight:800; margin-left:8px;
  }
`);

export function mountAlertsPill({ drawerOpenBtnId='btnOpenAlerts', drawerBadgeId='alertsBadge', injectIfMissing=true } = {}) {
  let openBtn = document.getElementById(drawerOpenBtnId);
  let badge   = document.getElementById(drawerBadgeId);

  if (!openBtn && injectIfMissing) {
    // Try to append in a sensible spot (topbar if present, else controls, else body)
    const host = document.querySelector('.topbar .row') || document.getElementById('controls') || document.body;
    const wrap = document.createElement('div');
    wrap.innerHTML = `
      <button id="${drawerOpenBtnId}" class="alerts-btn" type="button">
        Alerts <span id="${drawerBadgeId}" class="alerts-badge">0</span>
      </button>
    `.trim();
    host.appendChild(wrap.firstChild);
    openBtn = document.getElementById(drawerOpenBtnId);
    badge   = document.getElementById(drawerBadgeId);
  } else {
    // Add classes for consistent styling if you supplied your own elements
    if (openBtn) openBtn.classList.add('alerts-btn');
    if (badge)   badge.classList.add('alerts-badge');
  }

  return { openBtn, badge };
}
