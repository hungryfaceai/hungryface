/* app.js — Live Baby Cry Detector using YAMNet (TFJS)
   - Loads class indices from a local class_map.csv (same folder as index.html)
   - Users specify display_name values in the UI (separate with ; or newlines)
   - No remote CSV fetch (CORS-safe for GitHub Pages/your host)
*/

const els = {
  btnStart: document.getElementById('btnStart'),
  btnStop: document.getElementById('btnStop'),
  status: document.getElementById('status'),
  chartCanvas: document.getElementById('chart'),
  log: document.getElementById('log'),
  inputs: {
    SR_TARGET: document.getElementById('SR_TARGET'),
    FRAME_LEN_S: document.getElementById('FRAME_LEN_S'),
    HOP_S: document.getElementById('HOP_S'),
    SMOOTH_WIN: document.getElementById('SMOOTH_WIN'),
    THRESH: document.getElementById('THRESH'),
    PLOT_WINDOW_S: document.getElementById('PLOT_WINDOW_S'),
    MODEL_URL: document.getElementById('MODEL_URL'),
    INCLUDE_CLASSES: document.getElementById('INCLUDE_CLASSES'),
  }
};

const state = {
  running: false,
  audioCtx: null,
  mediaStream: null,
  processor: null,
  model: null,     // tf.GraphModel
  labelList: [],   // display_name strings
  cryIdxs: [],
  ring16k: new Float32Array(0),
  times: [],
  cryRaw: [],
  crySm: [],
  lastPos: 0,
  chart: null,
  startTime: 0,
};

function getParams() {
  return {
    SR_TARGET: parseInt(els.inputs.SR_TARGET.value, 10) || 16000,
    FRAME_LEN_S: parseFloat(els.inputs.FRAME_LEN_S.value) || 0.96,
    HOP_S: parseFloat(els.inputs.HOP_S.value) || 0.48,
    SMOOTH_WIN: Math.max(1, parseInt(els.inputs.SMOOTH_WIN.value, 10) || 5),
    THRESH: parseFloat(els.inputs.THRESH.value) || 0.25,
    PLOT_WINDOW_S: Math.max(5, parseFloat(els.inputs.PLOT_WINDOW_S.value) || 60),
    MODEL_URL: els.inputs.MODEL_URL.value.trim(),
    INCLUDE_CLASSES: splitClasses(els.inputs.INCLUDE_CLASSES.value),
  };
}

// Split classes by semicolons or new lines (NOT commas)
function splitClasses(text) {
  return text
    .split(/[;\n\r]+/)
    .map(s => s.trim())
    .filter(Boolean);
}

function logln(msg) {
  const t = new Date().toLocaleTimeString();
  els.log.textContent += `[${t}] ${msg}\n`;
  els.log.scrollTop = els.log.scrollHeight;
}

function setStatus(msg) {
  els.status.textContent = msg;
}

/* ===== CSV loading/parsing =====
   Expects a file named class_map.csv next to index.html with header:
   index,mid,display_name
*/
async function loadLocalClassMap() {
  const url = 'class_map.csv';
  const resp = await fetch(url, { cache: 'no-store' });
  if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);
  const text = await resp.text();
  const rows = parseCSV(text);
  // header row expected; find the display_name field
  const header = rows[0] || [];
  const idxDisplay = header.findIndex(h => h.trim().toLowerCase() === 'display_name');
  if (idxDisplay === -1) throw new Error('class_map.csv missing display_name column');
  const labels = rows.slice(1).map(r => (r[idxDisplay] || '').trim());
  return labels;
}

// CSV parser that handles quotes and commas-inside-quotes
function parseCSV(text) {
  const out = [];
  let row = [];
  let cur = '';
  let inQ = false;

  const pushCell = () => { row.push(cur); cur = ''; };
  const pushRow = () => { out.push(row); row = []; };

  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (inQ && text[i + 1] === '"') { cur += '"'; i++; }
      else { inQ = !inQ; }
      continue;
    }
    if (c === ',' && !inQ) { pushCell(); continue; }
    if ((c === '\n' || c === '\r') && !inQ) {
      // handle \r\n or \n
      if (c === '\r' && text[i + 1] === '\n') i++;
      pushCell(); pushRow(); continue;
    }
    cur += c;
  }
  // flush last cell/row if needed
  if (cur.length > 0 || row.length > 0) { pushCell(); pushRow(); }
  return out;
}

function findCryIdxs(labels, includeDisplayNames) {
  // Exact match (case-insensitive) first
  const needles = includeDisplayNames.map(s => s.trim().toLowerCase());
  let idxs = labels
    .map((n, i) => [i, n])
    .filter(([, n]) => needles.includes(n.trim().toLowerCase()))
    .map(([i]) => i);

  // Fallback: substring match if nothing found
  if (idxs.length === 0) {
    idxs = labels
      .map((n, i) => [i, n])
      .filter(([, n]) => {
        const L = n.trim().toLowerCase();
        return needles.some(m => L.includes(m));
      })
      .map(([i]) => i);
  }
  return Array.from(new Set(idxs));
}

function movingAvgTail(arr, k) {
  if (arr.length === 0) return 0;
  const start = Math.max(0, arr.length - k);
  let sum = 0;
  for (let i = start; i < arr.length; i++) sum += arr[i];
  return sum / (arr.length - start);
}

function setupChart() {
  if (state.chart) { state.chart.destroy(); }
  const ctx = els.chartCanvas.getContext('2d');
  state.chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'cry_score (raw)',_
