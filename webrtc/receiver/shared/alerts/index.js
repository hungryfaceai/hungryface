// Modal with table, filtering, column sorting, export & clear.
async function refreshBadge() {
try {
const rows = await getAllAlerts();
if (els.badge) els.badge.textContent = String(rows.length);
} catch { if (els.badge) els.badge.textContent = '0'; }
}
export { refreshBadge as refreshAlertsBadge };


async function renderTable() {
const rows = await getAllAlerts();
const tb = els.tbody; tb.innerHTML = '';
if (!rows.length) { tb.innerHTML = `<tr><td class="muted" colspan="6">No alerts yet.</td></tr>`; return; }


// Filter
const allowed = filterState.types; // Set of names: Audio/Prone/Motion/Fence
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
// default string compare
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
