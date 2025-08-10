/* app.js — Debug build
   Purpose: find why Start click isn't firing.
   - Verbose logs for load, DOMContentLoaded, element queries, listener binding
   - Global error & promise rejection hooks
   - Document-level click tracer
   - Forces Start enabled on init
   - Everything else (model/mic/inference) retained
*/

//////////////////// DEBUG WIRING ////////////////////
function dbg(msg) {
  const t = new Date().toLocaleTimeString();
  try {
    const logEl = document.getElementById('log');
    if (logEl) {
      logEl.textContent += `[${t}] ${msg}\n`;
      logEl.scrollTop = logEl.scrollHeight;
    }
  } catch {}
  // Always mirror to console
  // eslint-disable-next-line no-console
  console.log(`[DEBUG] ${msg}`);
}

// Catch script/async errors early
window.addEventListener('error', (e) => {
  // eslint-disable-next-line no-console
  console.error('[ONERROR]', e.message, e.filename, e.lineno, e.colno);
  dbg(`window.onerror: ${e.message} @ ${e.filename}:${e.lineno}`);
});
window.addEventListener('unhandledrejection', (e) => {
  // eslint-disable-next-line no-console
  console.error('[UNHANDLED REJECTION]', e.reason);
  dbg(`unhandledrejection: ${e.reason && e.reason.message ? e.reason.message : String(e.reason)}`);
});

// Click tracer: see what actually gets clicked
document.addEventListener('click', (e) => {
  const id = e.target && e.target.id ? `#${e.target.id}` : '(no id)';
  const tag = e.target && e.target.tagName ? e.target.tagName : '(no tag)';
  dbg(`document click → ${tag} ${id}`);
}, true);

dbg('app.js file evaluated (top-level).');

//////////////////// ELEMENTS ////////////////////
const els = {
  btnStart: null,
  btnStop: null,
  status: null,
  chartCanvas: null,
  log: null,
  inputs: {}
};

function bindElements() {
  els.btnStart = document.getElementById('btnStart');
  els.btnStop  = document.getElementById('btnStop');
  els.status   = document.getElementById('status');
  els.chartCanvas = document.getElementById('chart');
  els.log = document.getElementById('log');

  els.inputs = {
    SR_TARGET: document.getElementById('SR_TARGET'),
    FRAME_LEN_S: document.getElementById('FRAME_LEN_S'),
    HOP_S: document.getElementById('HOP_S'),
    SMOOTH_WIN: document.getElementById('SMOOTH_WIN'),
    THRESH: document.getElementById('THRESH'),
    PLOT_WINDOW_S: document.getElementById('PLOT_WINDOW_S'),
    MODEL_URL: document.getElementById('MODEL_URL'),
    INCLUDE_CLASSES: document.getElementById('INCLUDE_CLASSES'),
  };

  dbg(`bindElements:
    btnStart=${!!els.btnStart}, btnStop=${!!els.btnStop}, status=${!!els.status}, chart=${!!els.chartCanvas}
    inputs: ${Object.keys(els.inputs).map(k => `${k}=${!!els.inputs[k]}`).join(', ')}`);
}

function setStatus(msg) {
  if (els.status) els.status.textContent = msg;
  dbg(`STATUS: ${msg}`);
}
function logln(msg){ dbg(msg); }

//////////////////// STATE ////////////////////
const state = {
  running: false,
  audioCtx: null,
  mediaStream: null,
  processor: null,
  model: null,
  labelList: [],
  cryIdxs: [],
  ring16k: new Float32Array(0),
  times: [],
  cryRaw: [],
  crySm: [],
  lastPos: 0,
  chart: null,
};

//////////////////// PARAMS ////////////////////
function splitClasses(text) {
  return String(text || '')
    .split(/[;\n\r]+/)
    .map(s => s.trim())
    .filter(Boolean);
}

function getParams() {
  return {
    SR_TARGET: parseInt(els.inputs.SR_TARGET && els.inputs.SR_TARGET.value, 10) || 16000,
    FRAME_LEN_S: parseFloat(els.inputs.FRAME_LEN_S && els.inputs.FRAME_LEN_S.value) || 0.96,
    HOP_S: parseFloat(els.inputs.HOP_S && els.inputs.HOP_S.value) || 0.48,
    SMOOTH_WIN: Math.max(1, parseInt(els.inputs.SMOOTH_WIN && els.inputs.SMOOTH_WIN.value, 10) || 5),
    THRESH: parseFloat(els.inputs.THRESH && els.inputs.THRESH.value) || 0.25,
    PLOT_WINDOW_S: Math.max(5, parseFloat(els.inputs.PLOT_WINDOW_S && els.inputs.PLOT_WINDOW_S.value) || 60),
    MODEL_URL: (els.inputs.MODEL_URL && els.inputs.MODEL_URL.value || 'https://tfhub.dev/google/tfjs-model/yamnet/tfjs/1').trim(),
    INCLUDE_CLASSES: splitClasses(els.inputs.INCLUDE_CLASSES && els.inputs.INCLUDE_CLASSES.value),
  };
}

//////////////////// CSV + LABELS ////////////////////
async function loadLocalClassMap() {
  const url = 'class_map.csv';
  dbg(`Fetching ${url} …`);
  const resp = await fetch(url, { cache: 'no-store' });
  if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);
  const text = await resp.text();
  const rows = parseCSV(text);
  const header = rows[0] || [];
  const idxDisplay = header.findIndex(h => String(h).trim().toLowerCase() === 'display_name');
  if (idxDisplay === -1) throw new Error('class_map.csv missing display_name column');
  const labels = rows.slice(1).map(r => (r[idxDisplay] || '').trim());
  dbg(`class_map.csv loaded: ${labels.length} labels`);
  return labels;
}

function parseCSV(text) {
  const out = [];
  let row = [], cur = '', inQ = false;
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
  const needles = includeDisplayNames.map(s => s.trim().toLowerCase());
  let idxs = labels
    .map((n, i) => [i, n])
    .filter(([, n]) => needles.includes(n.trim().toLowerCase()))
    .map(([i]) => i);
  if (idxs.length === 0) {
    idxs = labels
      .map((n, i) => [i, n])
      .filter(([, n]) => {
        const L = n.trim().toLowerCase();
        return needles.some(m => L.includes(m));
      })
      .map(([i]) => i);
  }
  dbg(`findCryIdxs → ${idxs.length} indices`);
  return Array.from(new Set(idxs));
}

//////////////////// CHART ////////////////////
function setupChart() {
  if (state.chart) state.chart.destroy();
  const ctx = els.chartCanvas.getContext('2d');
  state.chart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'cry_score (raw)', data: [], borderWidth: 2, pointRadius: 0, tension: 0.15 },
      { label: 'cry_score (smoothed)', data: [], borderWidth: 2, pointRadius: 0, tension: 0.15 },
      { label: 'threshold', data: [], borderWidth: 1, pointRadius: 0, borderDash: [6,4] }
    ]},
    options: { animation: false, plugins: { legend: { display: false } } }
  });
  dbg('Chart set up.');
}

function updateChartWindow(params) {
  const { PLOT_WINDOW_S, THRESH } = params;
  const t = state.times, r = state.cryRaw, s = state.crySm;
  let cut = 0;
  if (t.length && t[t.length - 1] > PLOT_WINDOW_S) {
    const startTime = t[t.length - 1] - PLOT_WINDOW_S;
    cut = Math.max(0, t.findIndex(v => v >= startTime));
  }
  state.chart.data.labels = t.slice(cut);
  state.chart.data.datasets[0].data = r.slice(cut);
  state.chart.data.datasets[1].data = s.slice(cut);
  state.chart.data.datasets[2].data = new Array(t.length - cut).fill(THRESH);
  state.chart.update();
}

function setAlert(on) { document.body.classList.toggle('alert', !!on); }

//////////////////// AUDIO + TFJS ////////////////////
function concatFloat32(a, b) {
  const out = new Float32Array(a.length + b.length);
  out.set(a, 0); out.set(b, a.length);
  return out;
}
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

function movingAvgTail(arr, k) {
  if (arr.length === 0) return 0;
  const start = Math.max(0, arr.length - k);
  let sum = 0;
  for (let i = start; i < arr.length; i++) sum += arr[i];
  return sum / (arr.length - start);
}

async function inferCryScore16k(wave16k, cryIdxs) {
  const { FRAME_LEN_S } = getParams();
  const targetLen = Math.round(FRAME_LEN_S * 16000);
  if (wave16k.length < targetLen) {
    const pad = new Float32Array(targetLen);
    pad.set(wave16k, targetLen - wave16k.length);
    wave16k = pad;
  } else if (wave16k.length > targetLen) {
    wave16k = wave16k.slice(wave16k.length - targetLen);
  }
  const t = tf.tensor1d(wave16k).expandDims(0);
  const out = await state.model.executeAsync(t);
  let scores;
  if (Array.isArray(out)) scores = out.find(o => o.shape.length === 2 && o.shape[1] >= 500);
  else if (out.shape && out.shape.length === 2) scores = out;
  else scores = out['scores'];

  const s = tf.tidy(() => {
    const meanOverFrames = scores.mean(0);
    const idxs = tf.tensor1d(cryIdxs, 'int32');
    return meanOverFrames.gather(idxs).sum().dataSync()[0];
  });
  tf.dispose([t, scores, out]);
  return s;
}

async function ensureModelAndLabels() {
  if (!state.model) {
    const { MODEL_URL } = getParams();
    dbg(`Loading TFJS model from ${MODEL_URL} …`);
    state.model = await tf.loadGraphModel(MODEL_URL, { fromTFHub: true });
    dbg('Model loaded.');
  }
  if (!state.labelList.length) {
    setStatus('Loading class_map.csv…');
    state.labelList = await loadLocalClassMap();
  }
  const { INCLUDE_CLASSES } = getParams();
  state.cryIdxs = findCryIdxs(state.labelList, INCLUDE_CLASSES);
  if (state.cryIdxs.length === 0) {
    throw new Error('No matching classes found. Check spelling/case and separators (use ; or new lines).');
  }
  dbg(`cryIdxs = [${state.cryIdxs.join(', ')}]`);
  setStatus('Ready');
}

function makeScriptProcessor(ctx, onAudio) {
  const sp = ctx.createScriptProcessor(4096, 1, 1);
  sp.onaudioprocess = (e) => onAudio(new Float32Array(e.inputBuffer.getChannelData(0)));
  return sp;
}

//////////////////// START/STOP ////////////////////
async function start() {
  dbg('=== START CLICKED ===');
  if (state.running) { dbg('Already running.'); return; }

  try {
    setStatus('Initializing…');
    await tf.ready().catch(()=>{});
    try { tf.setBackend('webgl'); } catch {}
    await ensureModelAndLabels();

    // Reset plot/buffers
    const params = getParams();
    state.ring16k = new Float32Array(0);
    state.times = []; state.cryRaw = []; state.crySm = [];
    state.lastPos = 0;
    setupChart();
    setAlert(false);

    // Audio
    dbg('Creating AudioContext…');
    const ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: "interactive" });
    state.audioCtx = ctx;
    await ctx.resume();
    dbg(`AudioContext state: ${ctx.state}`);

    dbg('Requesting microphone (getUserMedia)…');
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, noiseSuppression: false, echoCancellation: false, autoGainControl: false },
      video: false
    });
    state.mediaStream = stream;
    dbg('Microphone access granted.');

    const src = ctx.createMediaStreamSource(stream);
    const sink = ctx.createGain(); sink.gain.value = 0; sink.connect(ctx.destination);

    let processor;
    try {
      dbg('Trying AudioWorklet…');
      const workletURL = URL.createObjectURL(new Blob([`
        class MicProcessor extends AudioWorkletProcessor {
          process(inputs) {
            const input = inputs[0];
            if (input && input[0]) this.port.postMessage(input[0].slice(0));
            return true;
          }
        }
        registerProcessor('mic-processor', MicProcessor);
      `], { type: 'application/javascript' }));
      await ctx.audioWorklet.addModule(workletURL);
      processor = new AudioWorkletNode(ctx, 'mic-processor', { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1 });
      processor.port.onmessage = (ev) => onAudioChunk(ev.data, ctx.sampleRate);
      src.connect(processor).connect(sink);
      dbg('AudioWorklet active.');
    } catch (e) {
      dbg(`AudioWorklet failed (${e.message}); falling back to ScriptProcessor.`);
      processor = makeScriptProcessor(ctx, (chunk) => onAudioChunk(chunk, ctx.sampleRate));
      src.connect(processor).connect(sink);
    }

    state.processor = processor;
    state.running = true;
    if (els.btnStart) els.btnStart.disabled = true;
    if (els.btnStop)  els.btnStop.disabled = false;
    setStatus(`Listening @ ${ctx.sampleRate.toFixed(0)} Hz`);
    dbg('Listening…');
  } catch (err) {
    dbg(`ERROR in start(): ${err && err.message ? err.message : String(err)}`);
    setStatus('Start failed');
    // Cleanup
    try { if (state.processor) state.processor.disconnect(); } catch {}
    try { if (state.audioCtx && state.audioCtx.state !== 'closed') state.audioCtx.close(); } catch {}
    try { if (state.mediaStream) state.mediaStream.getTracks().forEach(t => t.stop()); } catch {}
    state.audioCtx = null; state.mediaStream = null; state.processor = null; state.running = false;
    if (els.btnStart) els.btnStart.disabled = false;
    if (els.btnStop)  els.btnStop.disabled = true;
  }
}

async function stop() {
  dbg('Stop requested.');
  if (state.processor) { try { state.processor.disconnect(); } catch {} }
  if (state.audioCtx)  { try { await state.audioCtx.close(); } catch {} }
  if (state.mediaStream) { state.mediaStream.getTracks().forEach(t => t.stop()); }
  state.running = false;
  if (els.btnStart) els.btnStart.disabled = false;
  if (els.btnStop)  els.btnStop.disabled = true;
  setStatus('Stopped');
  dbg('Stopped.');
}

async function onAudioChunk(chunkFloat32, deviceSr) {
  const params = getParams();
  const x16 = resampleLinear(chunkFloat32, deviceSr, params.SR_TARGET);
  for (let i = 0; i < x16.length; i++) x16[i] = Math.max(-1, Math.min(1, x16[i]));
  state.ring16k = concatFloat32(state.ring16k, x16);

  const frameHop16k = Math.round(params.HOP_S * params.SR_TARGET);
  const frameLen16k = Math.round(params.FRAME_LEN_S * params.SR_TARGET);

  while (state.ring16k.length - state.lastPos >= frameHop16k) {
    const end = state.lastPos + frameHop16k;
    const start = Math.max(0, end - frameLen16k);
    let w = state.ring16k.slice(start, end);
    if (w.length < frameLen16k) {
      const pad = new Float32Array(frameLen16k);
      pad.set(w, frameLen16k - w.length);
      w = pad;
    }
    let score = 0;
    try {
      score = await inferCryScore16k(w, state.cryIdxs);
    } catch (e) {
      dbg(`Inference error: ${e.message}`);
    }
    state.cryRaw.push(score);
    state.crySm.push(movingAvgTail(state.cryRaw, params.SMOOTH_WIN));
    state.times.push(state.times.length * params.HOP_S);
    updateChartWindow(params);
    setAlert(state.crySm[state.crySm.length - 1] >= params.THRESH);
    state.lastPos += frameHop16k;
  }
}

//////////////////// DOM READY: BIND LISTENERS ////////////////////
document.addEventListener('DOMContentLoaded', () => {
  dbg('DOMContentLoaded fired.');
  bindElements();

  // Force Start enabled to avoid stale disabled state
  if (els.btnStart) els.btnStart.disabled = false;
  if (els.btnStop)  els.btnStop.disabled = true;

  if (!els.btnStart) {
    dbg('FATAL: #btnStart not found in DOM.');
  } else {
    els.btnStart.addEventListener('click', start);
    dbg('Bound click listener to #btnStart.');
  }
  if (els.btnStop) {
    els.btnStop.addEventListener('click', stop);
    dbg('Bound click listener to #btnStop.');
  }

  // Param change logging
  Object.keys(els.inputs).forEach(k => {
    if (els.inputs[k]) {
      els.inputs[k].addEventListener('change', () => dbg(`param changed: ${k} → ${els.inputs[k].value}`));
    }
  });

  setStatus('Page ready. Click Start.');
});
