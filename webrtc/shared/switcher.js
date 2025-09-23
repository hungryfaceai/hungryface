// /webrtc/receiver/shared/switcher.js
import { SecureSignal } from '/hungryface/webrtc/shared/secure-signal.js';
import { annotateTrusted, formatFp } from '/hungryface/webrtc/shared/device-registry.js';

export async function initTrustedSwitcher({
  wsEndpoint,
  mount = document.body,
  currentPeer = '',
  desiredRoom = '',
  onPick,                 // fn(fp) => void   (usually navigates with ?peer=fp)
} = {}) {
  // ----- UI -----
  const wrap = document.createElement('div');
  wrap.innerHTML = `
    <style>
      .tsw-toggle {
        position: fixed; left: 14px; top: 14px; z-index: 50;
        padding: 9px 12px; border-radius: 10px; border:1px solid #2a2a2a;
        background:#111; color:#eee; font-weight: 600; cursor:pointer;
      }
      .tsw-panel {
        position: fixed; left: 14px; top: 56px; z-index: 49;
        width: min(94vw, 360px); max-height: 70vh; overflow:auto;
        background: rgba(10,10,10,.92); border:1px solid #2a2a2a; border-radius: 12px;
        padding: 10px; display:none;
        box-shadow: 0 10px 24px rgba(0,0,0,.45);
        backdrop-filter: blur(4px);
      }
      .tsw-row {
        display:flex; align-items:center; justify-content:space-between;
        gap: 10px; padding: 10px; border-radius: 10px; border:1px solid #222;
        background:#0d0d0d; margin: 8px 0;
      }
      .tsw-row.current { border-color:#356ad8; box-shadow:0 0 0 1px #356ad8 inset; }
      .tsw-col { display:flex; flex-direction:column; gap:4px; min-width:0; }
      .tsw-name { font-weight:700; color:#fff; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
      .tsw-sub  { color:#aab; font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
      .tsw-btn  {
        padding: 8px 10px; border-radius: 10px; border:1px solid #2a2a2a;
        background:#171717; color:#eee; font-weight:600; cursor:pointer;
      }
      .tsw-empty { color:#99a; padding: 10px; }
      .tsw-head  { display:flex; align-items:center; justify-content:space-between; gap:10px; padding:4px 2px 8px; }
      .tsw-head input {
        flex:1; min-width: 0; padding: 8px 10px; border-radius: 10px; border:1px solid #222;
        background:#0b0b0b; color:#ddd;
      }
    </style>
    <button class="tsw-toggle" type="button">Switch device</button>
    <div class="tsw-panel" role="dialog" aria-label="Trusted devices">
      <div class="tsw-head">
        <div style="font-weight:700">Trusted devices</div>
        <input type="search" placeholder="Filter by name/room…" />
      </div>
      <div class="tsw-list"></div>
    </div>
  `;
  const btn = wrap.querySelector('.tsw-toggle');
  const panel = wrap.querySelector('.tsw-panel');
  const listEl = wrap.querySelector('.tsw-list');
  const searchEl = wrap.querySelector('input[type="search"]');
  mount.appendChild(wrap);

  btn.addEventListener('click', () => {
    panel.style.display = (panel.style.display === 'block') ? 'none' : 'block';
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') panel.style.display = 'none';
  });

  // ----- Data -----
  const ss = new SecureSignal({
    url: wsEndpoint,
    onTrustedList: (list) => render(annotateTrusted(list)),
  });
  await ss.init();

  function render(items) {
    const q = (searchEl.value || '').toLowerCase().trim();
    const filtered = items
      .filter(d => {
        if (!q) return true;
        const room = (d.meta?.room || '').toLowerCase();
        const name = (d.name || 'Device').toLowerCase();
        return name.includes(q) || room.includes(q) || d.fp.toLowerCase().includes(q);
      })
      .sort((a,b) => Number(b.meta?.ts || 0) - Number(a.meta?.ts || 0));

    listEl.innerHTML = filtered.length
      ? filtered.map(d => {
          const room = d.meta?.room || '';
          const when = d.meta?.ts ? new Date(d.meta.ts).toLocaleString() : '';
          const sub  = [room && `room: ${room}`, when && `seen: ${when}`].filter(Boolean).join(' • ');
          const isCurrent = (d.fp === currentPeer);
          return `
            <div class="tsw-row ${isCurrent ? 'current' : ''}">
              <div class="tsw-col" style="flex:1;min-width:0">
                <div class="tsw-name">${(d.name || 'Device')}</div>
                <div class="tsw-sub">${formatFp(d.fp)}${sub ? ' — ' + sub : ''}</div>
              </div>
              <button class="tsw-btn" data-fp="${d.fp}">${isCurrent ? 'Connected' : 'Switch'}</button>
            </div>
          `;
        }).join('')
      : `<div class="tsw-empty">No trusted devices yet. Pair on the Pair page.</div>`;

    listEl.querySelectorAll('.tsw-btn').forEach(b => {
      b.addEventListener('click', () => {
        const fp = b.getAttribute('data-fp');
        if (fp && typeof onPick === 'function') onPick(fp);
      });
    });
  }

  searchEl.addEventListener('input', () => {
    // re-render with current cached trusted list
    render(annotateTrusted(ss.trusted || []));
  });
}
