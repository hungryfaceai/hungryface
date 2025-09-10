// alerts-drawer.js
// Renders an "Alerts history" modal, with export (JSON/CSV), clear, and a badge counter.
// Depends on window.alertBanner.{openAlertDB,getAllAlerts} provided by alert-banner.js

(function () {
  const STYLE_ID = 'alertsDrawerStyles';

  function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const css = `
.alerts-toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
.table{width:100%;border-collapse:collapse}
.table th,.table td{padding:8px 10px;border-bottom:1px solid #1f2937;text-align:left;font-size:13px}
.table th{color:#ddd;font-weight:700}
.table td{color:#bbb;white-space:nowrap}
.table td.nowrap{white-space:nowrap}
.table .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.table .muted{color:#aaa}
`.trim();
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = css;
    document.head.appendChild(style);
  }

  function mountAlertsModal(hostId = 'alertsModalHost') {
    injectStyles();
    let host = document.getElementById(hostId);
    if (!host) {
      host = document.createElement('div');
      host.id = hostId;
      document.body.appendChild(host);
    }
    // Avoid remount
    if (document.getElementById('alertsModal')) return;

    host.innerHTML = `
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
              Saved episodes are stored locally in your browser (IndexedDB).
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
      </div>`;
    wireModal();
  }

  function fmtDT(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    if (!isFinite(+d)) return '—';
    return d.toLocaleString([], { hour12: false, year: 'numeric', month: '2-digit', day: '2-digit',
                                  hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }
  function fmtDuration(ms) {
    if (ms == null || !isFinite(ms) || ms < 0) return '—';
    const s = Math.round(ms / 1000);
    const mm = Math.floor(s / 60), ss = s % 60;
    return `${mm}m ${ss}s`;
  }
  function rowsToCSV(rows) {
    const esc = s => `"${String(s ?? '').replace(/"/g, '""')}"`;
    const header = ['startAt','endAt','durationSec','avgScore','type'];
    const lines = [header.join(',')];
    for (const r of rows) {
      const t0 = r.startAt ? +new Date(r.startAt) : NaN;
      const t1 = r.endAt   ? +new Date(r.endAt)   : NaN;
      const dur = (isFinite(t0) && isFinite(t1)) ? Math.round((t1 - t0) / 1000) : '';
      lines.push([esc(r.startAt), esc(r.endAt), dur, Number(r.avgScore || 0), esc(r.type || 'audio detection')].join(','));
    }
    return lines.join('\n');
  }
  function downloadBlob(name, mime, text) {
    const blob = new Blob([text], { type: mime });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  async function getRows() {
    const api = window.alertBanner;
    if (!api?.getAllAlerts) return [];
    const rows = await api.getAllAlerts();
    // newest first
    return rows.sort((a, b) => (+new Date(b.startAt || 0)) - (+new Date(a.startAt || 0)));
  }

  async function renderAlertsTable() {
    const tb = document.getElementById('alertsTbody');
    if (!tb) return;
    const rows = await getRows();
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

  async function refreshAlertsBadge() {
    const el = document.getElementById('alertsBadge');
    if (!el) return;
    try {
      const rows = await getRows();
      el.textContent = String(rows.length);
    } catch {
      el.textContent = '0';
    }
  }

  function openModal() {
    const m = document.getElementById('alertsModal');
    if (!m) return;
    m.classList.add('show');
    m.setAttribute('aria-hidden', 'false');
  }
  function closeModal() {
    const m = document.getElementById('alertsModal');
    if (!m) return;
    m.classList.remove('show');
    m.setAttribute('aria-hidden', 'true');
  }

  function wireModal() {
    const modal = document.getElementById('alertsModal');
    const btnClose = document.getElementById('btnAlertsClose');
    const btnClear = document.getElementById('btnAlertsClear');
    const btnJSON  = document.getElementById('btnAlertsExportJSON');
    const btnCSV   = document.getElementById('btnAlertsExportCSV');

    // open on header button
    const openBtn = document.getElementById('btnOpenAlerts');
    openBtn?.addEventListener('click', async () => {
      await renderAlertsTable();
      await refreshAlertsBadge();
      openModal();
    });

    btnClose?.addEventListener('click', closeModal);
    modal?.addEventListener('click', (e) => { if (e.target === modal) closeModal(); });

    btnClear?.addEventListener('click', async () => {
      const api = window.alertBanner;
      if (!api?.openAlertDB) return;
      const db = await api.openAlertDB();
      await new Promise((resolve, reject) => {
        const tx = db.transaction('alerts', 'readwrite');
        tx.oncomplete = resolve;
        tx.onerror = () => reject(tx.error);
        tx.objectStore('alerts').clear();
      });
      await renderAlertsTable();
      await refreshAlertsBadge();
    });

    btnJSON?.addEventListener('click', async () => {
      const rows = await getRows();
      downloadBlob(`alerts_${new Date().toISOString().replace(/[:.]/g,'-')}.json`,
                   'application/json',
                   JSON.stringify({ version:1, exportedAt:new Date().toISOString(), rows }, null, 2));
    });

    btnCSV?.addEventListener('click', async () => {
      const rows = await getRows();
      downloadBlob(`alerts_${new Date().toISOString().replace(/[:.]/g,'-')}.csv`,
                   'text/csv',
                   rowsToCSV(rows));
    });

    // keep badge in sync
    document.addEventListener('alerts:changed', refreshAlertsBadge);
  }

  // public (optional) API
  window.alertsDrawer = Object.freeze({
    mountAlertsModal,
    refreshAlertsBadge,
    openModal
  });

  // auto-mount on load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      mountAlertsModal();
      refreshAlertsBadge();
    });
  } else {
    mountAlertsModal();
    refreshAlertsBadge();
  }
})();
