// webrtc/receiver/shared/alerts/drawer.js
// Generic Alerts drawer: search, per-type filters, sortable columns, export & clear.

import { getAllAlerts, clearAllAlerts } from './store.js';

function injectStyleOnce(id, css) {
  if (document.getElementById(id)) return;
  const s = document.createElement('style'); s.id = id; s.textContent = css; document.head.appendChild(s);
}
injectStyleOnce('alerts-drawer-css', `
  .modal{position:fixed;inset:0;display:none;align-items:center;justify-content:center;background:rgba(0,0,0,.6);z-index:140;}
  .modal.show{display:flex;}
  .modal-card{width:min(96%,1100px);max-height:90vh;overflow:auto;background:#0b0b0b;border:1px solid #242424;border-radius:14px;}
  .modal-head{position:sticky;top:0;display:flex;justify-content:space-between;align-items:center;gap:10px;padding:10px 12px;border-bottom:1px solid #1a1a1a;background:#0d0d0d;border-top-left-radius:14px;border-top-right-radius:14px;}
  .modal-actions{display:flex;gap:8px;align-items:center;}
  .close{appearance:none;border:none;background:transparent;color:#aaa;font-size:22px;cursor:pointer;padding:2px 6px;}

  .toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
  .table{width:100%;border-collapse:collapse}
  .table th,.table td{padding:8px 10px;border-bottom:1px solid #1f2937;text-align:left;font-size:13px}
  .table th{color:#ddd;font-weight:700;cursor:pointer;user-select:none}
  .table td{color:#bbb;white-space:nowrap}
  .muted{color:#aaa}
  .btn{padding:8px 10px;border-radius:10px;border:1px solid rgba(255,255,255,.4);background:transparent;color:#fff;font-weight:600;cursor:pointer;}
  .btn:active{transform:scale(.98)}
  input[type="text"]{padding:8px 10px;border-radius:10px;border:1px solid #222;background:#0b0b0b;color:#ddd;}
  label{font-size:12px;color:#ddd;display:inline-flex;gap:6px;align-items:center;}
  .chip{display:inline-flex;align-items:center;gap:6px;padding:6px 10px;border:1px solid #333;border-radius:999px;font-size:12px}
`);

let els = {};
const filterState = { q: '', types: new Set(['Audio','Prone','Motion','Fence']) };
const sortState   = { key: 'startAt', dir: 'desc' };

function $(id){ return document.getElementById(id); }
function fmtDT(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (!isFinite(+d)) return '—';
  return d.toLocaleString([], { hour12:false, year:'numeric', month:'2-digit', day:'2-digit',
    hour:'2-digit', minute:'2-digit', second:'2-digit' });
}
function fmtDur(ms) {
  if (ms == null || !isFinite(ms) || ms < 0) return '—';
  const s = Math.round(ms/1000); const mm = Math.floor(s/60), ss = s%60;
  return `${mm}m ${ss}s`;
}

export function setupAlertsDrawer({ openBtnEl, badgeEl, hostId='alertsModalHost' } = {}) {
  // Create host if missing
  let host = document.getElementById(hostId);
  if (!host) { host = document.createElement('div'); host.id = hostId; document.body.appendChild(host); }

  const wrap = document.createElement('div');
  wrap.innerHTML = `
    <div id="alertsModal" class="modal" aria-hidden="true">
      <div class="modal-card">
        <div class="modal-head">
          <strong>Alerts history</strong>
          <div class="modal-actions">
            <!-- <button id="btnAlertsExportJSON" class="btn" type="button">Export JSON</button> -->
            <!-- <button id="btnAlertsExportCSV"  class="btn" type="button">Export CSV</button> -->
            <button id="btnAlertsClear"      class="btn" type="button">Clear all</button>
            <button id="btnAlertsClose"      class="close" type="button" title="Close" aria-label="Close">×</button>
          </div>
        </div>
        <div class="modal-body" style="padding:12px;">
          <div class="toolbar">
            <input id="alertsSearch" type="text" placeholder="Search message or type…" style="min-width:220px;">
            <label class="chip"><input type="checkbox" id="chkAudio" checked> Audio</label>
            <label class="chip"><input type="checkbox" id="chkProne" checked> Prone</label>
            <label class="chip"><input type="checkbox" id="chkMotion" checked> Motion</label>
            <label class="chip"><input type="checkbox" id="chkFence" checked> Fence</label>
          </div>
          <div style="overflow:auto">
            <table class="table">
              <thead>
                <tr>
                  <th data-key="startAt">Start</th>
                  <th data-key="endAt">End</th>
                  <th data-key="duration">Duration</th>
                  <th data-key="avgScore">Avg score</th>
                  <th data-key="type">Type</th>
                  <th data-key="message">Message</th>
                </tr>
              </thead>
              <tbody id="alertsTbody"><tr><td class="muted" colspan="6">No alerts yet.</td></tr></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  `.trim();
  host.appendChild(wrap.firstChild);

  els = {
    modal: $('alertsModal'),
    tbody: $('alertsTbody'),
    openBtn: openBtnEl || document.getElementById('btnOpenAlerts'),
    badge: badgeEl || document.getElementById('alertsBadge'),
    btnClose: $('btnAlertsClose'),
    btnClear: $('btnAlertsClear'),
    // btnJSON: $('btnAlertsExportJSON'),   // ← hidden (commented out)
    // btnCSV: $('btnAlertsExportCSV'),     // ← hidden (commented out)
    search: $('alertsSearch'),
    chk: {
      Audio: $('chkAudio'), Prone: $('chkProne'),
      Motion: $('chkMotion'), Fence: $('chkFence')
    },
    head: host.querySelector('thead')
  };

  // Open/close
  els.openBtn?.addEventListener('click', async () => {
    await renderTable();
    await refreshAlertsBadge();
    els.modal.classList.add('show');
    els.modal.setAttribute('aria-hidden','false');
  });
  els.btnClose?.addEventListener('click', () => closeModal());
  els.modal.addEventListener('click', (e) => { if (e.target === els.modal) closeModal(); });
  function closeModal(){ els.modal.classList.remove('show'); els.modal.setAttribute('aria-hidden','true'); }

  // Filters
  els.search?.addEventListener('input', () => { filterState.q = (els.search.value||'').trim().toLowerCase(); renderTable(); });
  for (const [name, box] of Object.entries(els.chk)) {
    box?.addEventListener('change', () => {
      if (box.checked) filterState.types.add(name); else filterState.types.delete(name);
      renderTable();
    });
  }

  // Sorting (click head)
  els.head?.addEventListener('click', (e) => {
    const th = e.target.closest('th'); if (!th) return;
    const key = th.dataset.key; if (!key) return;
    if (sortState.key === key) sortState.dir = (sortState.dir === 'asc') ? 'desc' : 'asc';
    else { sortState.key = key; sortState.dir = 'desc'; }
    renderTable();
  });

  // Toolbar actions
  els.btnClear?.addEventListener('click', async () => { await clearAllAlerts(); await renderTable(); await refreshAlertsBadge(); });
  // els.btnJSON?.addEventListener('click', async () => downloadBlob(makeName('json'), 'application/json', await dumpJSON())); // ← hidden
  // els.btnCSV?.addEventListener('click', async () => downloadBlob(makeName('csv'),  'text/csv',           await dumpCSV()));  // ← hidden

  // Keep badge (and table if open) in sync
  document.addEventListener('alerts:changed', async () => {
    await refreshAlertsBadge();
    if (els.modal?.classList.contains('show')) await renderTable();
  });
}

function makeName(ext){ return `alerts_${new Date().toISOString().replace(/[:.]/g,'-')}.${ext}`; }
function downloadBlob(name, mime, text) {
  const blob = new Blob([text], { type: mime });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click(); URL.revokeObjectURL(a.href);
}

export async function refreshAlertsBadge() {
  try {
    const rows = await getAllAlerts();
    if (els.badge) els.badge.textContent = String(rows.length);
  } catch { if (els.badge) els.badge.textContent = '0'; }
}

async function dumpJSON() {
  const rows = await getAllAlerts();
  return JSON.stringify({ version:1, exportedAt:new Date().toISOString(), rows }, null, 2);
}
async function dumpCSV() {
  const rows = await getAllAlerts();
  const esc = s => `"${String(s??'').replace(/"/g,'""')}"`;
  const header = ['startAt','endAt','durationSec','avgScore','type','message'];
  const lines = [header.join(',')];
  for (const r of rows) {
    const t0 = r.startAt ? +new Date(r.startAt) : NaN;
    const t1 = r.endAt   ? +new Date(r.endAt)   : NaN;
    const dur = (isFinite(t0) && isFinite(t1)) ? Math.round((t1 - t0)/1000) : '';
    lines.push([esc(r.startAt), esc(r.endAt), dur, Number(r.avgScore||0), esc(r.type||''), esc(r.message||'')].join(','));
  }
  return lines.join('\n');
}

async function renderTable() {
  const rows = await getAllAlerts();
  const tb = els.tbody; tb.innerHTML = '';
  if (!rows.length) { tb.innerHTML = `<tr><td class="muted" colspan="6">No alerts yet.</td></tr>`; return; }

  // Filter
  const allowed = filterState.types;
  const q = filterState.q;
  let out = rows.filter(r => {
    const typeOk = allowed.has((r.type||'').toString());
    if (!typeOk) return false;
    if (!q) return true;
    const hay = `${r.type||''} ${r.message||''}`.toLowerCase();
    return hay.includes(q);
  });

  // Sort
  const dir = sortState.dir === 'asc' ? 1 : -1;
  out.sort((a,b) => {
    const key = sortState.key;
    if (key === 'duration') {
      const da = (+new Date(a.endAt||0)) - (+new Date(a.startAt||0));
      const db = (+new Date(b.endAt||0)) - (+new Date(b.startAt||0));
      return (da - db) * dir;
    }
    let va = a[key], vb = b[key];
    if (key === 'avgScore') { va = Number(va||0); vb = Number(vb||0); return (va - vb) * dir; }
    if (key === 'startAt' || key === 'endAt') { va = +new Date(va||0); vb = +new Date(vb||0); return (va - vb) * dir; }
    return String(va||'').localeCompare(String(vb||'')) * dir;
  });

  for (const r of out) {
    const t0 = r.startAt ? new Date(r.startAt) : null;
    const t1 = r.endAt ? new Date(r.endAt) : null;
    const durMs = (t0 && t1) ? (t1 - t0) : null;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${fmtDT(r.startAt)}</td>
      <td>${fmtDT(r.endAt)}</td>
      <td>${fmtDur(durMs)}</td>
      <td>${(Number(r.avgScore)||0).toFixed(4)}</td>
      <td>${r.type || ''}</td>
      <td>${r.message || ''}</td>
    `;
    tb.appendChild(tr);
  }
}
