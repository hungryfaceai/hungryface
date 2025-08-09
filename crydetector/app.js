/* app.js — Live Baby Cry Detector using YAMNet (TFJS)
   - Mic capture via Web Audio
   - Resample to SR_TARGET (default 16 kHz)
   - Frame with FRAME_LEN_S / HOP_S
   - Run YAMNet (TFJS) on 0.96 s windows
   - Sum probabilities over selected cry classes
   - Smooth with moving average (SMOOTH_WIN)
   - Plot in real time and flash background when smoothed >= THRESH
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
    LABELS_URL: document.getElementById('LABELS_URL'),
    INCLUDE_CLASSES: document.getElementById('INCLUDE_CLASSES'),
  }
};

const state = {
  running: false,
  audioCtx: null,
  mediaStream: null,
  processor: null, // Worklet or ScriptProcessor
  model: null,     // tf.GraphModel
  labelList: [],
  cryIdxs: [],
  ring16k: new Float32Array(0),
  times: [],
  cryRaw: [],
  crySm: [],
  lastPos: 0, // in 16k samples
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
    LABELS_URL: els.inputs.LABELS_URL.value.trim(),
    INCLUDE_CLASSES: els.inputs.INCLUDE_CLASSES.value.split(',')
      .map(s => s.trim()).filter(Boolean),
  };
}

function logln(msg) {
  const t = new Date().toLocaleTimeString();
  els.log.textContent += `[${t}] ${msg}\n`;
  els.log.scrollTop = els.log.scrollHeight;
}

function setStatus(msg) {
  els.status.textContent = msg;
}

async function loadLabels(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch labels CSV: ${resp.status}`);
  const text = await resp.text();
  // CSV has header: index,mid,display_name
  return text.trim().split(/\r?\n/).slice(1).map(line => {
    const parts = parseCSVLine(line);
    return parts[2] ? parts[2].trim() : '';
  });
}

// Simple CSV row parser (handles commas in quotes)
function parseCSVLine(line) {
  const out = [];
  let cur = '';
  let inQ = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"' && line[i + 1] === '"') { cur += '"'; i++; continue; }
    if (c === '"') { inQ = !inQ; continue; }
    if (c === ',' && !inQ) { out.push(cur); cur = ''; continue; }
    cur += c;
  }
  out.push(cur);
  return out;
}

function findCryIdxs(labels, include) {
  const needles = new Set(include.map(s => s.toLowerCase()));
  let idxs = labels.map((n, i) => [i, n])
    .filter(([, n]) => needles.has(n.trim().toLowerCase()))
    .map(([i]) => i);

  if (idxs.length === 0) {
    // Substring fallback
    idxs = labels.map((n, i) => [i, n])
      .filter(([, n]) => {
        const L = n.trim().toLowerCase();
        return [...needles].some(m => L.includes(m));
      })
      .map(([i]) => i);
  }
  return idxs;
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
    // find first index >= startTime
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

  // y autoscale
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
  const n = Math.round(x.length * ratio);
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
  // wave16k length should be frame_len_16k; pad or crop just in case
  const { FRAME_LEN_S } = getParams();
  const targetLen = Math.round(FRAME_LEN_S * 16000);
  let x = wave16k;
  if (x.length !== targetLen) {
    if (x.length < targetLen) {
      const pad = new Float32Array(targetLen);
      pad.set(x, targetLen - x.length); // right align as in Python
      x = pad;
    } else {
      x = x.slice(x.length - targetLen);
    }
  }

  // Model expects float32 waveform, range [-1,1], 16k
  const t = tf.tidy(() => tf.tensor1d(x).expandDims(0)); // [1, num_samples]
  // TFJS Hub YAMNet returns a dict of tensors from executeAsync; scores is [frames, 521]
  // Try to fetch named output; otherwise use first return.
  let scores;
  try {
    const out = await state.model.executeAsync(t);
    // out can be a Tensor or array/dict depending on graph; try common cases:
    if (Array.isArray(out)) {
      // heuristic: scores have rank 2 and second dim ~521
      const cand = out.find(o => o.shape.length === 2 && o.shape[1] >= 500);
      scores = cand || out[0];
      // dispose the rest
      out.forEach(o => { if (o !== scores) o.dispose(); });
    } else if (out && out.shape && out.shape.length === 2) {
      scores = out;
    } else if (out && out['scores']) {
      scores = out['scores'];
      // dispose non-score tensors
      for (const k of Object.keys(out)) if (k !== 'scores') out[k].dispose();
    } else {
      throw new Error('Unexpected YAMNet output structure.');
    }
  } finally {
    t.dispose();
  }

  // scores: [num_patches, 521] -> average over patches, sum cry class columns
  const s = tf.tidy(() => {
    const meanOverFrames = scores.mean(0); // [521]
    const idxs = tf.tensor1d(cryIdxs, 'int32');
    const selected = meanOverFrames.gather(idxs);
    const summed = selected.sum(); // scalar
    // Convert to JS number
    return summed.dataSync()[0];
  });
  scores.dispose();
  return s;
}

async function ensureModelAndLabels() {
  const params = getParams();
  if (!state.model) {
    setStatus('Loading model…');
    logln(`Loading TFJS graph model from ${params.MODEL_URL}`);
    state.model = await tf.loadGraphModel(params.MODEL_URL, { fromTFHub: true });
    logln('Model loaded.');
  }
  if (!state.labelList.length) {
    setStatus('Loading labels…');
    state.labelList = await loadLabels(params.LABELS_URL);
    state.cryIdxs = findCryIdxs(state.labelList, params.INCLUDE_CLASSES);
    if (state.cryIdxs.length === 0) {
      throw new Error('Could not find any matching cry-related labels.');
    }
    logln(`Using ${state.cryIdxs.length} class index(es) for: ${params.INCLUDE_CLASSES.join(', ')}`);
  }
  setStatus('Ready');
}

function makeScriptProcessor(ctx, onAudio) {
  // Fallback if AudioWorklet isn’t available
  const bufSize = 4096; // 256–16384; latency/battery tradeoff
  const sp = ctx.createScriptProcessor(bufSize, 1, 1);
  sp.onaudioprocess = (e) => {
    const channel = e.inputBuffer.getChannelData(0);
    onAudio(new Float32Array(channel));
  };
  return sp;
}

async function start() {
  if (state.running) return;
  await ensureModelAndLabels();

  const params = getParams();
  state.ring16k = new Float32Array(0);
  state.times = []; state.cryRaw = []; state.crySm = [];
  state.lastPos = 0;
  setupChart();
  setAlert(false);

  // Audio
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: "interactive" });
  state.audioCtx = ctx;
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, noiseSuppression: false, echoCancellation: false, autoGainControl: false }, video: false });
  state.mediaStream = stream;

  const src = ctx.createMediaStreamSource(stream);
  let processor;

  // Prefer AudioWorklet when available for robustness
  if (ctx.audioWorklet && ctx.audioWorklet.addModule) {
    const workletURL = URL.createObjectURL(new Blob([`
      class MicProcessor extends AudioWorkletProcessor {
        process(inputs){
          const input = inputs[0];
          if (!input || !input[0]) return true;
          const channel = input[0];
          // Send raw Float32Array to main thread
          this.port.postMessage(channel.slice(0));
          return true;
        }
      }
      registerProcessor('mic-processor', MicProcessor);
    `], { type: 'application/javascript' }));
    await ctx.audioWorklet.addModule(workletURL);
    processor = new AudioWorkletNode(ctx, 'mic-processor', { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1 });
    processor.port.onmessage = (ev) => onAudioChunk(ev.data, ctx.sampleRate);
    src.connect(processor).connect(ctx.destination);
  } else {
    processor = makeScriptProcessor(ctx, (chunk) => onAudioChunk(chunk, ctx.sampleRate));
    src.connect(processor).connect(ctx.destination);
  }

  state.processor = processor;
  state.running = true;
  els.btnStart.disabled = true;
  els.btnStop.disabled = false;
  setStatus(`Listening @ ${ctx.sampleRate.toFixed(0)} Hz (device)…`);
  state.startTime = performance.now();
  logln('Listening… allow mic permission if prompted.');
}

async function stop() {
  if (!state.running) return;
  try {
    if (state.processor) {
      try { state.processor.disconnect(); } catch {}
    }
    if (state.audioCtx) {
      try { state.audioCtx.close(); } catch {}
    }
    if (state.mediaStream) {
      state.mediaStream.getTracks().forEach(t => t.stop());
    }
  } finally {
    state.running = false;
    els.btnStart.disabled = false;
    els.btnStop.disabled = true;
    setStatus('Stopped');
    logln('Stopped.');
    setAlert(false);
  }
}

async function onAudioChunk(chunkFloat32, deviceSr) {
  if (!state.running) return;
  const params = getParams();

  // Resample to SR_TARGET (default 16k)
  const x16 = resampleLinear(chunkFloat32, deviceSr, params.SR_TARGET);
  // Clip
  for (let i = 0; i < x16.length; i++) {
    if (x16[i] > 1) x16[i] = 1;
    else if (x16[i] < -1) x16[i] = -1;
  }
  state.ring16k = concatFloat32(state.ring16k, x16);

  // Framing
  const frameHop16k = Math.round(params.HOP_S * params.SR_TARGET);
  const frameLen16k = Math.round(params.FRAME_LEN_S * params.SR_TARGET);

  // Process as many hops as available
  while (state.ring16k.length - state.lastPos >= frameHop16k) {
    const end = state.lastPos + frameHop16k;
    const start = Math.max(0, end - frameLen16k);
    let window16k = state.ring16k.slice(start, end);
    if (window16k.length < frameLen16k) {
      const pad = new Float32Array(frameLen16k);
      pad.set(window16k, frameLen16k - window16k.length);
      window16k = pad;
    }

    let score = 0;
    try {
      score = await inferCryScore16k(window16k, state.cryIdxs);
      if (!Number.isFinite(score)) score = 0;
    } catch (e) {
      logln(`Inference error: ${e.message}`);
      score = 0;
    }

    state.cryRaw.push(score);
    const sm = movingAvgTail(state.cryRaw, params.SMOOTH_WIN);
    state.crySm.push(sm);

    const tSec = (state.times.length * params.HOP_S);
    state.times.push(tSec);

    // UI updates
    updateChartWindow(params);
    setAlert(sm >= params.THRESH);

    state.lastPos += frameHop16k;

    // Trim big buffers occasionally (to prevent unbounded growth)
    if (state.times.length % 400 === 0) {
      const keepFrom = Math.max(0, state.times.length - Math.ceil(params.PLOT_WINDOW_S / params.HOP_S) - 10);
      if (keepFrom > 0) {
        state.times = state.times.slice(keepFrom);
        state.cryRaw = state.cryRaw.slice(keepFrom);
        state.crySm = state.crySm.slice(keepFrom);
      }
      // Also drop older audio that’s no longer needed
      const minNeeded = state.lastPos - frameLen16k - frameHop16k;
      if (minNeeded > 0) {
        state.ring16k = state.ring16k.slice(minNeeded);
        state.lastPos -= minNeeded;
      }
    }
  }
}

/* =========== UI wiring =========== */
els.btnStart.addEventListener('click', start);
els.btnStop.addEventListener('click', stop);

// Recompute chart instantly when params change
for (const key of Object.keys(els.inputs)) {
  els.inputs[key].addEventListener('change', () => {
    if (!state.chart) return;
    updateChartWindow(getParams());
  });
}

/* =========== Hints & safeguards =========== */
(async () => {
  logln('Tip: This app needs HTTPS (or http://localhost) for microphone access.');
  logln('If you see a blank plot, start the mic and speak to test levels.');
  try {
    // Warm up TF backend for faster first inference
    await tf.ready();
    tf.setBackend('webgl').catch(() => {});
  } catch (e) {}
})();

/* Notes:
   - YAMNet TFJS URL defaults to: https://tfhub.dev/google/tfjs-model/yamnet/tfjs/1
     If Google updates the model, you can point MODEL_URL to a newer Hub link.
   - Labels CSV defaults to: https://storage.googleapis.com/audioset/yamnet/yamnet_class_map.csv
   - INCLUDE_CLASSES: change to broaden detection, e.g.:
       "Baby cry, infant cry, Crying, sobbing, Whimper"
*/
