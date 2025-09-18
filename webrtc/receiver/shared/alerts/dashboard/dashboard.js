// Alerts Dashboard
// Swimlane timeline + activity band + brush/zoom + search/filter/sort + table

import {
  getAllAlerts,
  clearAllAlerts,
  exportAlertsJSON,
  exportAlertsCSV,
  AlertTypes,
} from '/hungryface/webrtc/receiver/shared/alerts/store.js';

// ------- State -------
const state = {
  rowsAll: [],
  // filter/search
  allowed: new Set(Object.keys(AlertTypes)), // "Audio","Prone","Motion","Fence"
  q: '',
  // sort
  sortKey: 'startAt',
  sortDir: 'desc',
  // time window (rolling, default 12h)
  windowHours: 12,
  windowStartMs: null,
  windowEndMs: null, // usually "now"
  // brush
  brushing: false,
  brushStartX: 0,
  brushEndX: 0,
};

// ------- DOM -------
const $ = (id) => document.getElementById(id);
const els = {
  svg: $('timeline'),
  tbody: $('alertsTbody'),
  count: $('countInWindow'),
  winLabel: $('windowLabel'),
  q: $('q'),
  chips: Array.from(document.querySelectorAll('.chip[data-type]')),
  winBtns: Array.from(document.querySelectorAll('button[data-win]')),
  ths: Array.from(document.querySelectorAll('th[data-key]')),
  btnJSON: document.getElementById('btnExportJSON'),
  btnCSV: document.getElementById('btnExportCSV'),
  btnClear: document.getElementById('btnClearAll'),
};

// ------- Utils -------
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
function fmtDT(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (!isFinite(+d)) return '—';
  return d.toLocaleString([], { hour12:false, year:'numeric', month:'2-digit', day:'2-digit',
                                hour:'2-digit', minute:'2-digit', second:'2-digit' });
}
function fmtDur(ms) {
  if (ms == null || !isFinite(ms) || ms < 0) return '—';
  const s = Math.round(ms/1000);
  const mm = Math.floor(s/60), ss = s % 60;
  return `${mm}m ${ss}s`;
}
function download(name, type, text) {
  const blob = new Blob([text], { type });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ------- Time window helpers -------
function setRollingWindow(hours) {
  state.windowHours = hours;
  const now = Date.now();
  state.windowEndMs = now;
  state.windowStartMs = now - hours * 3600 * 1000;
}
function setExplicitWindow(startMs, endMs) {
  state.windowStartMs = Math.min(startMs, endMs);
  state.windowEndMs   = Math.max(startMs, endMs);
}
function xToTime(x, box, startMs, endMs) {
  const t = x / Math.max(1, box.width);
  return startMs + t * (endMs - startMs);
}
function timeToX(t, box, startMs, endMs) {
  const r = (t - startMs) / Math.max(1, endMs - startMs);
  return box.x + clamp(r, 0, 1) * box.width;
}

// ------- Data filtering/sorting -------
function filterRowsInWindow(rows) {
  const { allowed, q, windowStartMs, windowEndMs } = state;
  const qtok = (q||'').trim().toLowerCase();
  return rows.filter(r => {
    // type filter
    const typeOk = allowed.has(String(r.type || ''));
    if (!typeOk) return false;
    // search
    if (qtok) {
      const hay = `${r.type||''} ${r.message||''}`.toLowerCase();
      if (!hay.includes(qtok)) return false;
    }
    // time window intersection: include if any overlap within [start,end]
    const t0 = +new Date(r.startAt || 0);
    const t1 = +new Date(r.endAt   || r.startAt || 0);
    const a0 = Math.min(t0, t1), a1 = Math.max(t0, t1);
    return a1 >= windowStartMs && a0 <= windowEndMs;
  });
}

function sortRows(rows) {
  const { sortKey, sortDir } = state;
  const dir = sortDir === 'asc' ? 1 : -1;
  const out = [...rows];
  out.sort((a,b) => {
    if (sortKey === 'duration') {
      const da = (+new Date(a.endAt||0)) - (+new Date(a.startAt||0));
      const db = (+new Date(b.endAt||0)) - (+new Date(b.startAt||0));
      return (da - db) * dir;
    }
    let va = a[sortKey], vb = b[sortKey];
    if (sortKey === 'avgScore') { va = Number(va||0); vb = Number(vb||0); return (va - vb) * dir; }
    if (sortKey === 'startAt' || sortKey === 'endAt') { va = +new Date(va||0); vb = +new Date(vb||0); return (va - vb) * dir; }
    return String(va||'').localeCompare(String(vb||'')) * dir;
  });
  return out;
}

// ------- Activity band (alerts/min) -------
function binActivity(rows, binMs) {
  const { windowStartMs, windowEndMs } = state;
  const nBins = Math.max(1, Math.ceil((windowEndMs - windowStartMs) / binMs));
  const bins = new Array(nBins).fill(0);
  for (const r of rows) {
    const t0 = +new Date(r.startAt||0);
    const t1 = +new Date(r.endAt||r.startAt||0);
    const a0 = Math.max(windowStartMs, Math.min(t0, t1));
    const a1 = Math.min(windowEndMs,   Math.max(t0, t1));
    if (a1 < windowStartMs || a0 > windowEndMs) continue;
    // increment all bins touched by [a0,a1]
    let i0 = Math.floor((a0 - windowStartMs) / binMs);
    let i1 = Math.floor((a1 - windowStartMs) / binMs);
    i0 = clamp(i0, 0, nBins-1); i1 = clamp(i1, 0, nBins-1);
    for (let i=i0; i<=i1; i++) bins[i]++;
  }
  return { bins, binMs, start: windowStartMs };
}

// ------- SVG timeline -------
function renderTimeline(rows) {
  const svg = els.svg;
  const W = svg.clientWidth || svg.viewBox.baseVal.width || 1000;
  const H = svg.clientHeight || 300;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const PAD_L = 60, PAD_R = 16, PAD_T = 10, PAD_B = 24;
  const lanes = ['Audio','Prone','Motion','Fence'];
  const laneH = (H - PAD_T - PAD_B) / (lanes.length + 1); // +1 row for activity band
  const bandH = Math.max(14, Math.min(22, laneH * 0.5));
  const lanesTop = PAD_T + bandH + 8;
  const plotX = PAD_L, plotY = lanesTop, plotW = W - PAD_L - PAD_R, plotH = H - lanesTop - PAD_B;

  const startMs = state.windowStartMs, endMs = state.windowEndMs;

  // background
  const bg = rect(plotX, PAD_T, plotW, H - PAD_T - PAD_B, '#0a0a0a'); bg.setAttribute('rx','8'); svg.appendChild(bg);
  // grid (hour ticks)
  const hours = Math.max(1, Math.round((endMs - startMs)/3600000));
  const tickEveryMs = pickNiceTick((endMs - startMs));
  for (let t = Math.ceil(startMs/tickEveryMs)*tickEveryMs; t <= endMs; t += tickEveryMs) {
    const x = plotX + (plotW * (t - startMs) / (endMs - startMs));
    const g = line(x, PAD_T, x, H - PAD_B, 'rgba(255,255,255,0.06)'); svg.appendChild(g);
    const lab = text(new Date(t).toLocaleTimeString([], {hour12:false,hour:'2-digit',minute:'2-digit'}), x+2, H - 6, '#aaa', 'end');
    lab.setAttribute('text-anchor','end'); svg.appendChild(lab);
  }

  // activity band (alerts/min) above lanes
  const bandX = plotX, bandY = PAD_T, bandW = plotW, bandTop = bandY, bandBot = bandY + bandH;
  const { bins, binMs } = binActivity(rows, pickBin(endMs - startMs));
  const maxBin = Math.max(1, ...bins);
  const bw = bandW / bins.length;
  for (let i=0;i<bins.length;i++){
    const v = bins[i] / maxBin;
    const h = Math.round(v * bandH);
    const x = bandX + i*bw;
    const y = bandBot - h;
    const bar = rect(x, y, Math.max(1, bw-1), h, 'rgba(96,165,250,0.45)');
    svg.appendChild(bar);
  }
  const capLine = line(bandX, bandBot+0.5, bandX+bandW, bandBot+0.5, 'rgba(255,255,255,0.15)'); svg.appendChild(capLine);

  // lanes labels
  lanes.forEach((name, idx) => {
    const y = laneY(idx);
    const lbl = text(name, PAD_L - 8, y + laneH/2 + 4, '#ddd', 'end');
    lbl.setAttribute('text-anchor','end'); svg.appendChild(lbl);
    // separator
    const sep = line(plotX, y + laneH, plotX + plotW, y + laneH, 'rgba(255,255,255,0.06)'); svg.appendChild(sep);
  });

  // alert bars
  const colorFor = (type) => ({
    Audio:'#ef4444', Prone:'#f59e0b', Motion:'#60a5fa', Fence:'#22c55e'
  }[type] || '#888');
  for (const r of rows) {
    const t0 = +new Date(r.startAt||0);
    const t1 = +new Date(r.endAt||r.startAt||0);
    const a0 = Math.max(startMs, Math.min(t0,t1));
    const a1 = Math.min(endMs,   Math.max(t0,t1));
    if (a1 < startMs || a0 > endMs) continue;
    const type = String(r.type||'');
    const laneIdx = Math.max(0, lanes.indexOf(type));
    const y = laneY(laneIdx) + 3;
    const x0 = plotX + plotW * (a0 - startMs) / (endMs - startMs);
    const x1 = plotX + plotW * (a1 - startMs) / (endMs - startMs);
    const w  = Math.max(2, x1 - x0);
    const op = String(clamp(Number(r.avgScore||0), 0.15, 1));
    const bar = rect(x0, y, w, Math.max(8, laneH - 6), colorFor(type), op);
    bar.setAttribute('rx','4'); bar.setAttribute('ry','4');
    bar.setAttribute('data-tip', `${type} • ${fmtDT(r.startAt)} → ${fmtDT(r.endAt)} • ${(Number(r.avgScore)||0).toFixed(3)}\n${r.message||''}`);
    svg.appendChild(bar);

    // a small dot at the start
    const dot = circle(x0, y + (laneH/2), 2.2, colorFor(type), op);
    svg.appendChild(dot);
  }

  // "now" marker
  const now = Date.now();
  if (now >= startMs && now <= endMs) {
    const x = plotX + plotW * (now - startMs) / (endMs - startMs);
    const nline = line(x, PAD_T, x, H - PAD_B, 'rgba(239,68,68,0.85)');
    svg.appendChild(nline);
  }

  // brush overlay
  addBrushOverlay(svg, { x: plotX, y: PAD_T, width: plotW, height: H - PAD_T - PAD_B }, startMs, endMs);

  // label
  els.winLabel.textContent = `${new Date(startMs).toLocaleString()} → ${new Date(endMs).toLocaleString()}`;

  function laneY(idx){ return lanesTop + idx * laneH; }
  function rect(x,y,w,h,fill,opacity) { const n = document.createElementNS('http://www.w3.org/2000/svg','rect'); n.setAttribute('x',x); n.setAttribute('y',y); n.setAttribute('width',w); n.setAttribute('height',h); n.setAttribute('fill',fill); if(opacity) n.setAttribute('fill-opacity',opacity); return n; }
  function line(x1,y1,x2,y2,stroke){ const n = document.createElementNS('http://www.w3.org/2000/svg','line'); n.setAttribute('x1',x1); n.setAttribute('y1',y1); n.setAttribute('x2',x2); n.setAttribute('y2',y2); n.setAttribute('stroke',stroke); n.setAttribute('stroke-width','1'); return n; }
  function text(txt,x,y,fill){ const n = document.createElementNS('http://www.w3.org/2000/svg','text'); n.setAttribute('x',x); n.setAttribute('y',y); n.setAttribute('fill',fill); n.setAttribute('font-size','11'); n.textContent = txt; return n; }
  function circle(cx,cy,r,fill,opacity){ const n = document.createElementNS('http://www.w3.org/2000/svg','circle'); n.setAttribute('cx',cx); n.setAttribute('cy',cy); n.setAttribute('r',r); n.setAttribute('fill',fill); if(opacity) n.setAttribute('fill-opacity',opacity); return n; }
}

function pickNiceTick(spanMs){
  // choose roughly 6–10 vertical grid lines
  const targets = [5*60e3, 10*60e3, 15*60e3, 30*60e3, 60*60e3, 2*60*60e3, 3*60*60e3, 6*60*60e3, 12*60*60e3];
  const approx = spanMs / 8;
  let best = targets[0], diff = Math.abs(targets[0]-approx);
  for (const t of targets){ const d = Math.abs(t - approx); if (d < diff) { best = t; diff = d; } }
  return best;
}
function pickBin(spanMs){
  // activity band bin ~ 50–120 bars
  const targetBins = 80;
  const raw = spanMs / targetBins;
  const nice = [60e3, 2*60e3, 5*60e3, 10*60e3, 15*60e3, 30*60e3]; // 1–30 min
  let best = nice[0], diff = Math.abs(nice[0]-raw);
  for (const n of nice){ const d = Math.abs(n-raw); if (d < diff){ best = n; diff = d; } }
  return best;
}

// ------- Brush (drag to zoom) -------
function addBrushOverlay(svg, plotBox, startMs, endMs){
  const overlay = document.createElementNS('http://www.w3.org/2000/svg','rect');
  overlay.setAttribute('x', plotBox.x);
  overlay.setAttribute('y', plotBox.y);
  overlay.setAttribute('width',  plotBox.width);
  overlay.setAttribute('height', plotBox.height);
  overlay.setAttribute('fill','transparent');
  overlay.style.cursor = 'crosshair';
  svg.appendChild(overlay);

  const sel = document.createElementNS('http://www.w3.org/2000/svg','rect');
  sel.setAttribute('fill','rgba(255,255,255,0.12)');
  sel.setAttribute('stroke','rgba(255,255,255,0.35)');
  sel.setAttribute('stroke-width','1');
  sel.style.display = 'none';
  svg.appendChild(sel);

  const toLocalX = (evt) => {
    const p = svg.createSVGPoint();
    p.x = evt.clientX; p.y = evt.clientY;
    const m = svg.getScreenCTM().inverse();
    const s = p.matrixTransform(m);
    return clamp(s.x - plotBox.x, 0, plotBox.width);
  };

  overlay.addEventListener('mousedown', (e) => {
    state.brushing = true;
    state.brushStartX = toLocalX(e);
    state.brushEndX = state.brushStartX;
    sel.style.display = '';
    e.preventDefault();
  });
  window.addEventListener('mousemove', (e) => {
    if (!state.brushing) return;
    state.brushEndX = toLocalX(e);
    const x = Math.min(state.brushStartX, state.brushEndX);
    const w = Math.abs(state.brushEndX - state.brushStartX);
    sel.setAttribute('x', plotBox.x + x);
    sel.setAttribute('y', plotBox.y);
    sel.setAttribute('width', w);
    sel.setAttribute('height', plotBox.height);
  });
  const finish = () => {
    if (!state.brushing) return;
    state.brushing = false;
    sel.style.display = 'none';
    const minSel = 8; // px
    if (Math.abs(state.brushEndX - state.brushStartX) < minSel) return;
    const x0 = Math.min(state.brushStartX, state.brushEndX);
    const x1 = Math.max(state.brushStartX, state.brushEndX);
    const t0 = xToTime(x0, { x:0, width:plotBox.width }, startMs, endMs);
    const t1 = xToTime(x1, { x:0, width:plotBox.width }, startMs, endMs);
    setExplicitWindow(t0, t1);
    renderAll();
  };
  window.addEventListener('mouseup', finish);
  // reset zoom on double click
  svg.addEventListener('dblclick', () => { setRollingWindow(state.windowHours); renderAll(); });
}

// ------- Table -------
function renderTable(rows) {
  const tb = els.tbody;
  tb.innerHTML = '';
  if (!rows.length) {
    tb.innerHTML = `<tr><td class="muted" colspan="6">No alerts in this window.</td></tr>`;
    return;
  }
  for (const r of rows) {
    const t0 = r.startAt ? new Date(r.startAt) : null;
    const t1 = r.endAt   ? new Date(r.endAt)   : null;
    const durMs = (t0 && t1) ? (t1 - t0) : null;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${fmtDT(r.startAt)}</td>
      <td>${fmtDT(r.endAt)}</td>
      <td>${fmtDur(durMs)}</td>
      <td class="mono">${(Number(r.avgScore)||0).toFixed(4)}</td>
      <td>${r.type || ''}</td>
      <td>${r.message || ''}</td>
    `;
    tb.appendChild(tr);
  }
  els.count.textContent = String(rows.length);
}

// ------- Exports -------
els.btnJSON?.addEventListener('click', async () => {
  const rows = await getAllAlerts();
  const name = `alerts_${new Date().toISOString().replace(/[:.]/g,'-')}.json`;
  const payload = exportAlertsJSON ? await exportAlertsJSON(rows) : JSON.stringify({ version:1, exportedAt:new Date().toISOString(), rows }, null, 2);
  download(name, 'application/json', payload);
});
els.btnCSV?.addEventListener('click', async () => {
  const rows = await getAllAlerts();
  const name = `alerts_${new Date().toISOString().replace(/[:.]/g,'-')}.csv`;
  const text = exportAlertsCSV ? await exportAlertsCSV(rows) : rowsToCSV(rows);
  download(name, 'text/csv', text);
});
els.btnClear?.addEventListener('click', async () => {
  if (!confirm('Clear ALL alerts? This cannot be undone.')) return;
  await clearAllAlerts();
  await boot();
});

// Fallback CSV
function rowsToCSV(rows) {
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

// ------- Wiring -------
els.q?.addEventListener('input', () => { state.q = els.q.value; renderAll(); });
els.chips.forEach(ch => ch.addEventListener('click', () => {
  const t = ch.dataset.type;
  if (state.allowed.has(t)) state.allowed.delete(t); else state.allowed.add(t);
  ch.classList.toggle('active');
  renderAll();
}));
els.winBtns.forEach(b => b.addEventListener('click', () => {
  const h = Math.max(1, parseInt(b.dataset.win, 10) || 12);
  setRollingWindow(h);
  renderAll();
}));
els.ths.forEach(th => th.addEventListener('click', () => {
  const key = th.dataset.key;
  if (state.sortKey === key) {
    state.sortDir = (state.sortDir === 'asc') ? 'desc' : 'asc';
  } else {
    state.sortKey = key;
    state.sortDir = (key === 'startAt' || key === 'endAt') ? 'desc' : 'asc';
  }
  renderAll();
}));

// Keep in sync with store updates
document.addEventListener('alerts:changed', async () => {
  state.rowsAll = await getAllAlerts();
  renderAll();
});

// ------- Render all -------
function renderAll() {
  // Filter rows by window + chips + q
  const rowsInWin = filterRowsInWindow(state.rowsAll);
  // Timeline (render with all in window, not yet sorted)
  renderTimeline(rowsInWin);
  // Table (sorted)
  renderTable(sortRows(rowsInWin));
}

// ------- Boot -------
async function boot() {
  setRollingWindow(state.windowHours);
  state.rowsAll = await getAllAlerts();
  renderAll();
}
boot();
