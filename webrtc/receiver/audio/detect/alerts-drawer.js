// alerts-drawer.js
// Renders the Alerts modal, table, export/clear actions, and keeps the header badge updated.

import { openAlertDB, getAllAlerts } from './alert-banner.js';

function injectStyleOnce(id, css) {
  if (document.getElementById(id)) return;
  const s = document.createElement('style');
  s.id = id;
  s.textContent = css;
  document.head.appendChild(s);
}

function fmtDT(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (!isFinite(+d)) return '—';
  return d.toLocaleString([], { hour12:false, year:'numeric', month:'2-digit', day:'2-digit',
                                hour:'2-digit', minute:'2-digit', second:'2-digit' });
}
function fmtDuration(ms) {
  if (ms == null || !isFinite(ms) || ms < 0) return '—';
  const s = Math.round(ms/1000);
  const mm = Math.floor(s/60), ss = s%60;
  return `${mm}m ${ss}s`;
}

let els = {};
function $(id){ return document.getElementById(id); }

export function setupAlertsDrawer() {
  injectStyleOnce('alerts-drawer-css', `
    .alerts-toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
    .badge{display:inline-flex;align-items:center;justify-content:center;min-width:18px;height:18px;padding:0 6px;border-radius:999px;background:#374151;color:#fff;font-size:12px;font-weight:700;margin-left:6px}
    .table{width:100%;border-collapse:collapse}
    .table th,.table td{padding:8px 10px;border-bottom:1px solid #1f2937;text-align:left;font-size:13px}
    .table th{color:#ddd;font-weight:700}
    .table td{color:#bbb;white-space:nowrap}
    .table td.nowrap{white-space:nowrap}
    .table .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
    .table .muted{color:#aaa}
    .table .actions{display:flex;gap:8px}
  `);

  const host = document.getElementById('alertsModalHost') || document.body;
  const wrap = document.createElement('div');
  wrap.innerHTML = `
    <div id="alertsModal" class="modal" aria-hidden="true">
      <div class="modal-card">
        <div class="modal-head">
          <strong>Alerts history</strong>
          <div class="modal-actions">
            <button id="btnAlertsExportJSON" class="btn small" type="button">Export JSON</button>
            <button id="btnAlertsExportCSV"  class="btn small" type="button">Export CSV</button>
            <button id="btnAlertsClear"      class="btn small" type="button">Clear all</button>
            <button id="btnAlertsClose"      class="close"     type="button" title="Close" aria-label="Close">×</button>
          </div>
        </div>
        <div class="modal-body">
          <div class="alerts-toolbar muted">
            Saved episodes are stored in your browser (IndexedDB).
          </div>
          <div style="overflow:auto">
            <table class="table" id="alertsTable">
              <thead>
                <tr>
                  <th>Start</th>
                  <th>End</th>
                  <th>Duration</th>
                  <th>Avg score</th>
                  <th>Type</th>
                </tr>
              </thead>
              <tbody id="alertsTbody">
                <tr><td class="muted" colspan="5">No alerts yet.</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  `.trim();
  const node = wrap.firstChild;
  host.appendChild(node);

  els = {
    modal: $('alertsModal'),
    tbody: $('alertsTbody'),
    openBtn: $('btnOpenAlerts'),
    badge: $('alertsBadge'),
    btnClose: $('btnAlertsClose'),
    btnClear: $('btnAlertsClear'),
    btnJSON: $('btnAlertsExportJSON'),
    btnCSV: $('btnAlertsExportCSV')
  };

  // Open/close wiring
  els.openBtn?.addEventListener('click', async () => {
    await renderAlertsTable();
    await refreshAlertsBadge();
    els.modal.classList.add('show');
    els.modal.setAttribute('aria-hidden','false');
  });
  els.btnClose?.addEventListener('click', () => {
    els.modal.classList.remove('show');
    els.modal.setAttribute('aria-hidden','true');
  });
  els.modal.addEventListener('click', (e) => {
    if (e.target === els.modal) {
      els.modal.classList.remove('show');
      els.modal.setAttribute('aria-hidden','true');
    }
  });

  // Toolbar
  els.btnClear?.addEventListener('click', () => clearAllAlerts());
  els.btnJSON?.addEventListener('click', () => exportAlertsJSON());
  els.btnCSV?.addEventListener('click', () => exportAlertsCSV());

  // Keep badge (and table if open) in sync when alerts change
  document.addEventListener('alerts:changed', async () => {
    await refreshAlertsBadge();
    if (els.modal?.classList.contains('show')) await renderAlertsTable();
  });
}

async function renderAlertsTable() {
  const rows = await getAllAlerts();
  const tb = els.tbody;
  tb.innerHTML = '';
  if (!rows.length) {
    tb.innerHTML = `<tr><td class="muted" colspan="5">No alerts yet.</td></tr>`;
    return;
  }
  for (const r of rows) {
    const start = r.startAt ? new Date(r.startAt) : null;
    const end   = r.endAt   ? new Date(r.endAt)   : null;
    const durMs = (start && end) ? (end - start) : null;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="nowrap">${fmtDT(r.startAt)}</td>
      <td class="nowrap">${fmtDT(r.endAt)}</td>
      <td class="nowrap">${fmtDuration(durMs)}</td>
      <td class="nowrap mono">${(Number(r.avgScore)||0).toFixed(4)}</td>
      <td class="nowrap">${r.type || 'audio detection'}</td>
    `;
    tb.appendChild(tr);
  }
}

function downloadBlob(name, mime, text) {
  const blob = new Blob([text], { type: mime });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

export async function refreshAlertsBadge() {
  try {
    const rows = await getAllAlerts();
    if (els.badge) els.badge.textContent = String(rows.length);
  } catch {
    if (els.badge) els.badge.textContent = '0';
  }
}

async function exportAlertsJSON() {
  const rows = await getAllAlerts();
  downloadBlob(`alerts_${new Date().toISOString().replace(/[:.]/g,'-')}.json`,
               'application/json',
               JSON.stringify({ version:1, exportedAt:new Date().toISOString(), rows }, null, 2));
}

function rowsToCSV(rows) {
  const esc = s => `"${String(s??'').replace(/"/g,'""')}"`;
  const header = ['startAt','endAt','durationSec','avgScore','type'];
  const lines = [header.join(',')];
  for (const r of rows) {
    const t0 = r.startAt ? +new Date(r.startAt) : NaN;
    const t1 = r.endAt   ? +new Date(r.endAt)   : NaN;
    const dur = (isFinite(t0) && isFinite(t1)) ? Math.round((t1 - t0)/1000) : '';
    lines.push([esc(r.startAt), esc(r.endAt), dur, Number(r.avgScore||0), esc(r.type||'audio detection')].join(','));
  }
  return lines.join('\n');
}
async function exportAlertsCSV() {
  const rows = await getAllAlerts();
  downloadBlob(`alerts_${new Date().toISOString().replace(/[:.]/g,'-')}.csv`,
               'text/csv',
               rowsToCSV(rows));
}

async function clearAllAlerts() {
  const db = await openAlertDB();
  await new Promise((resolve, reject) => {
    const tx = db.transaction('alerts','readwrite');
    tx.oncomplete = resolve;
    tx.onerror    = () => reject(tx.error);
    tx.objectStore('alerts').clear();
  });
  document.dispatchEvent(new CustomEvent('alerts:changed', { detail: { action: 'clear' } }));
  await renderAlertsTable();
}
