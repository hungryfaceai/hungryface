/* app.js — Live Baby Cry Detector using YAMNet (TFJS)
   - Loads class indices from a local class_map.csv (same folder as index.html)
   - Users specify display_name values in the UI (separate with ; or newlines)
   - Robust Start button: resumes AudioContext, falls back if AudioWorklet fails, clear logging
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
      if (c === '\r' && text[i + 1] === '\n') i++;
      pushCell(); pushRow(); continue;
    }
    cur += c;
  }
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
        { label: 'cry_score (raw)', data: [], borderWidth: 2, pointRadius: 0, tension: 0.15 },
        { label: 'cry_score (smoothed)', data: [], borderWidth: 2, pointRadius: 0, tension: 0.15 },
        { label: 'threshold', data: [], borderWidth: 1, pointRadius: 0, borderDash: [6,4] }
      ]
    },
    options: {
      animation: false,
      scales: {
        x: { title: { display: true, text: 'Time (s)' } },
        y: {
          title: { display: true, text: 'Cry score' },
          suggestedMin: 0, suggestedMax: 1.0
        }
      },
      plugins: { legend: { display: false } },
      elements: { line: { fill: false } }
    }
  });
}

function updateChartWindow(params) {
  const { PLOT_WINDOW_S, THRESH } = params;
  const t = state.times;
  const r = state.cryRaw;
  const s = state.crySm;
  const chart = state.chart;

  let cut = 0;
  if (t.length && t[t.length - 1] > PLOT_WINDOW_S) {
    const startTime = t[t.length - 1] - PLOT_WINDOW_S;
    cut = Math.max(0, t.findIndex(v => v >= startTime));
    if (cut < 0) cut = 0;
  }

  const labels = t.slice(cut);
  const raw = r.slice(cut);
  const sm = s.slice(cut);
  const thr = labels.map(_ => THRESH);

  chart.data.labels = labels;
  chart.data.datasets[0].data = raw;
  chart.data.datasets[1].data = sm;
  chart.data.datasets[2].data = thr;

  if (raw.length || sm.length) {
    const all = raw.concat(sm);
    const ymin = Math.min(...all);
    const ymax = Math.max(...all);
    const pad = 0.05;
    chart.options.scales.y.min = Math.min(0, ymin - pad);
    chart.options.scales.y.max = Math.min(1.5, ymax + pad);
  }
  chart.update();
}

function setAlert(on) {
  document.body.classList.toggle('alert', !!on);
}

function concatFloat32(a, b) {
  const out = new Float32Array(a.length + b.length);
  out.set(a, 0); out.set(b, a.length);
  return out;
}

// Linear resampler (mono)
function resampleLinear(x, fromSr, toSr) {
  if (fromSr === toSr) return x;
  const ratio = toSr / fromSr;
  const n = Math.max(1, Math.round(x.length * ratio));
  const out = new Float32Array(n);
  const dx = (x.length - 1) / (n - 1);
  for (let i = 0; i < n; i++) {
    const pos = i * dx;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, x.length - 1);
    const frac = pos - i0;
    out[i] = x[i0] * (1 - frac) + x[i1] * frac;
  }
  return out;
}

// Run a single YAMNet inference on a 0.96s window of 16k audio (Float32Array)
async function inferCryScore16k(wave16k, cryIdxs) {
  const { FRAME_LEN_S } = getParams();
  const targetLen = Math.round(FRAME_LEN_S * 16000);
  let x = wave16k;
  if (x.length !== targetLen) {
    if (x.length < targetLen) {
      const pad = new Float32Array(targetLen);
      pad.set(x, targetLen - x.length); // right align like the Python version
      x = pad;
    } else {
      x = x.slice(x.length - targetLen);
    }
  }

  const t = tf.tidy(() => tf.tensor1d(x).expandDims(0)); // [1, num_samples]
  let scores;
  try {
    const out = await state.model.executeAsync(t);
    if (Array.isArray(out)) {
      const cand = out.find(o => o.shape.length === 2 && o.shape[1] >= 500);
      scores = cand || out[0];
      out.forEach(o => { if (o !== scores) o.dispose(); });
    } else if (out && out.shape && out.shape.length === 2) {
      scores = out;
    } else if (out && out['scores']) {
      scores = out['scores'];
      for (const k of Object.keys(out)) if (k !== 'scores') out[k].dispose();
    } else {
      throw new Error('Unexpected YAMNet output structure.');
    }
  } finally {
    t.dispose();
  }

  // scores: [num_patches, 521] -> average over patches, sum selected class columns
  const s = tf.tidy(() => {
    const meanOverFrames = scores.mean(0); // [521]
    const idxs = tf.tensor1d(cryIdxs, 'int32');
    const selected = meanOverFrames.gather(idxs);
    const summed = selected.sum(); // scalar
    return summed.dataSync()[0];
  });
  scores.dispose();
  return s;
}

async function ensureModelAndLabels() {
  if (!state.model) {
    const { MODEL_URL } = getParams();
    setStatus('Loading model…');
    logln(`Loading TFJS graph model from ${MODEL_URL}`);
    state.model = await tf.loadGraphModel(MODEL_URL, { fromTFHub: true });
    logln('Model loaded.');
  }
  if (!state.labelList.length) {
    setStatus('Loading class_map.csv…');
    state.labelList = await loadLocalClassMap(); // display_name array
    logln(`Loaded ${state.labelList.length} labels from class_map.csv`);
  }
  // Compute indices based on current UI selection
  const { INCLUDE_CLASSES } = getParams();
  state.cryIdxs = findCryIdxs(state.labelList, INCLUDE_CLASSES);
  if (state.cryIdxs.length === 0) {
    throw new Error('No matching classes found in class_map.csv. Check spelling/case.');
  }
  logln(`Using ${state.cryIdxs.length} class index(es) for: ${INCLUDE_CLASSES.join(' | ')}`);
  setStatus('Ready');
}

function makeScriptProcessor(ctx, onAudio) {
  const bufSize = 4096;
  const sp = ctx.createScriptProcessor(bufSize, 1, 1);
  sp.onaudioprocess = (e) => {
    const channel = e.inputBuffer.getChannelData(0);
    onAudio(new Float32Array(channel));
  };
  return sp;
}

/* ===================== START/STOP (robust) ===================== */
async function start() {
  if (state.running) return;

  setStatus('Initializing…');
  try {
    await ensureModelAndLabels(); // model + class_map.csv
  } catch (e) {
    logln(`Init error: ${e.message}`);
    setStatus('Init failed');
    return;
  }

  // Reset plot/buffers
  const params = getParams();
  state.ring16k = new Float32Array(0);
  state.times = []; state.cryRaw = []; state.crySm = [];
  state.lastPos = 0;
  setupChart();
  setAlert(false);

  // Audio
  let ctx;
  try {
    ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: "interactive" });
    state.audioCtx = ctx;

    // Some browsers auto-start suspended; make sure we resume on user gesture
    await ctx.resume();

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, noiseSuppression: false, echoCancellation: false, autoGainControl: false },
      video: false
    });
    state.mediaStream = stream;

    const src = ctx.createMediaStreamSource(stream);
    const sink = ctx.createGain(); sink.gain.value = 0; sink.connect(ctx.destination);

    let processor;
    let workletReady = false;

    // Try AudioWorklet, but fall back if anything errors
    if (ctx.audioWorklet && ctx.audioWorklet.addModule) {
      try {
        const workletURL = URL.createObjectURL(new Blob([`
          class MicProcessor extends AudioWorkletProcessor {
            process(inputs){
              const input = inputs[0];
              if (!input || !input[0]) return true;
              const channel = input[0];
              this.port.postMessage(channel.slice(0));
              return true;
            }
          }
          registerProcessor('mic-processor', MicProcessor);
        `], { type: 'application/javascript' }));
        await ctx.audioWorklet.addModule(workletURL);
        processor = new AudioWorkletNode(ctx, 'mic-processor', { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1 });
        processor.port.onmessage = (ev) => onAudioChunk(ev.data, ctx.sampleRate);
        src.connect(processor).connect(sink);
        workletReady = true;
        logln('AudioWorklet active.');
      } catch (e) {
        logln(`AudioWorklet failed, falling back: ${e.message}`);
      }
    }

    if (!workletReady) {
      processor = makeScriptProcessor(ctx, (chunk) => onAudioChunk(chunk, ctx.sampleRate));
      src.connect(processor).connect(sink);
      logln('Using ScriptProcessor fallback.');
    }

    state.processor = processor;
    state.running = true;
    els.btnStart.disabled = true;
    els.btnStop.disabled = false;
    setStatus(`Listening @ ${ctx.sampleRate.toFixed(0)} Hz (device)…`);
    state.startTime = performance.now();
    logln('Listening… allow mic permission if prompted.');
  } catch (e) {
    logln(`Start error: ${e.message}`);
    setStatus('Start failed');
    // Clean up partial init
    try { if (state.processor) state.processor.disconnect(); } catch {}
    try { if (ctx && ctx.state !== 'closed') await ctx.close(); } catch {}
    try { if (state.mediaStream) state.mediaStream.getTracks().forEach(t => t.stop()); } catch {}
    state.audioCtx = null; state.mediaStream = null; state.processor = null; state.running = false;
    els.btnStart.disabled = false; els.btnStop.disabled = true;
  }
}

async function stop() {
  if (!state.running) return;
  try {
    if (state.processor) { try { state.processor.disconnect(); } catch {} }
    if (state.audioCtx)  { try { await state.audioCtx.close(); } catch {} }
    if
