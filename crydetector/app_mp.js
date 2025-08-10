// app_mp.js (ES module) — MediaPipe Tasks Audio @0.10.0
// - Uses AudioClassifier.classify(Float32Array, sampleRate)
// - Keeps your UI, spectrum chart, thresholds, hysteresis/hold logic
// - Expects class_map.csv next to index.html
// - Make sure the script tag in index.html is: <script src="./app_mp.js" type="module"></script>

import audio from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio@0.10.0";
const { FilesetResolver, AudioClassifier } = audio;

/* ---------- DOM bindings ---------- */
const els = {
  btnStart: null,
  btnStop: null,
  status: null,
  chartCanvas: null,
  chartSpectrumCanvas: null,
  inputs: {},
  showRawSpec: null,
  showSmSpec: null,
  peakReadout: null,
};

function bindElements() {
  els.btnStart = document.getElementById('btnStart');
  els.btnStop  = document.getElementById('btnStop');
  els.status   = document.getElementById('status');
  els.chartCanvas = document.getElementById('chart');
  els.chartSpectrumCanvas = document.getElementById('chartSpectrum');
  els.showRawSpec = document.getElementById('showRawSpec');
  els.showSmSpec  = document.getElementById('showSmSpec');
  els.peakReadout = document.getElementById('peakReadout');

  els.inputs = {
    SR_TARGET: document.getElementById('SR_TARGET'),
    FRAME_LEN_S: document.getElementById('FRAME_LEN_S'),
    HOP_S: document.getElementById('HOP_S'),
    SMOOTH_WIN: document.getElementById('SMOOTH_WIN'),
    THRESH: document.getElementById('THRESH'),
    PLOT_WINDOW_S: document.getElementById('PLOT_WINDOW_S'),
    MODEL_URL: document.getElementById('MODEL_URL'), // TFLite URL or ./yamnet.tflite
    INCLUDE_CLASSES: document.getElementById('INCLUDE_CLASSES'),
    ALERT_ON_EXTRA: document.getElementById('ALERT_ON_EXTRA'),
    ALERT_OFF_DELTA: document.getElementById('ALERT_OFF_DELTA'),
    ALERT_HOLD_MS: document.getElementById('ALERT_HOLD_MS'),
  };
}

function setStatus(msg) { if (els.status) els.status.textContent = msg; }

/* ---------- App state ---------- */
const state = {
  running: false,
  audioCtx: null,
  mediaStream: null,
  processor: null,
  analyser: null,

  // MediaPipe
  mpFileset: null,
  mpClassifier: null,

  // Labels + mapping
  labelList: [],
  cryIdxs: [],

  // Series / plotting
  ring16k: new Float32Array(0),
  lastPos: 0,
  timesSec: [],
  timesMs: [],
  cryRaw: [],
  crySm: [],
  chart: null,
  spectrumChart: null,

  // Spectrum
  specRawDb: [],
  specSmDb: [],
  specFreqs: [],
  specStartIdx: 0,
  specEndIdx: 0,
  specLastUpdate: 0,

  // Peak
  peakFreq: null,
  peakDb: null,

  // Alert
  alertOn: false,
  alertUntilMs: 0,
};

/* ---------- Params / helpers ---------- */
function splitClasses(text) {
  return String(text || '').split(/[;\n\r]+/).map(s => s.trim()).filter(Boolean);
}
function getParams() {
  return {
    SR_TARGET: parseInt(els.inputs.SR_TARGET?.value, 10) || 16000,
    FRAME_LEN_S: parseFloat(els.inputs.FRAME_LEN_S?.value) || 0.96,
    HOP_S: parseFloat(els.inputs.HOP_S?.value) || 0.48,
    SMOOTH_WIN: Math.max(1, parseInt(els.inputs.SMOOTH_WIN?.value, 10) || 5),
    THRESH: parseFloat(els.inputs.THRESH?.value) || 0.15,
    PLOT_WINDOW_S: Math.max(5, parseFloat(els.inputs.PLOT_WINDOW_S?.value) || 60),
    MODEL_URL: (els.inputs.MODEL_URL?.value || './yamnet.tflite').trim(),
    INCLUDE_CLASSES: splitClasses(els.inputs.INCLUDE_CLASSES?.value),
    ALERT_ON_EXTRA: parseFloat(els.inputs.ALERT_ON_EXTRA?.value ?? '0') || 0,
    ALERT_OFF_DELTA: Math.max(0, parseFloat(els.inputs.ALERT_OFF_DELTA?.value ?? '0.1') || 0.1),
    ALERT_HOLD_MS: Math.max(0, parseInt(els.inputs.ALERT_HOLD_MS?.value, 10) || 3000),
  };
}
function formatClock(ms) {
  return new Date(ms).toLocaleTimeString(undefined, { hour12: false });
}

/* ---------- CSV + labels ---------- */
async function loadLocalClassMap() {
  const resp = await fetch('class_map.csv', { cache: 'no-store' });
  if (!resp.ok) throw new Error(`Failed to fetch class_map.csv: ${resp.status}`);
  const text = await resp.text();
  const rows = parseCSV(text);
  const header = rows[0] || [];
  const idxDisplay = header.findIndex(h => String(h).trim().toLowerCase() === 'display_name');
  if (idxDisplay === -1) throw new Error('class_map.csv missing display_name column');
  return rows.slice(1).map(r => (r[idxDisplay] || '').trim());
}
function parseCSV(text) {
  const out = [];
  let row = [], cur = '', inQ = false;
  const pushCell = () => { row.push(cur); cur = ''; };
  const pushRow = () => { out.push(row); row = []; };
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') { if (inQ && text[i+1] === '"') { cur += '"'; i++; } else { inQ = !inQ; } continue; }
    if (c === ',' && !inQ) { pushCell(); continue; }
    if ((c === '\n' || c === '\r') && !inQ) { if (c === '\r' && text[i+1] === '\n') i++; pushCell(); pushRow(); continue; }
    cur += c;
  }
  if (cur.length > 0 || row.length > 0) { pushCell(); pushRow(); }
  return out;
}
function findCryIdxs(labels, includeDisplayNames) {
  const needles = includeDisplayNames.map(s => s.trim().toLowerCase());
  let idxs = labels.map((n,i)=>[i,n]).filter(([,n])=>needles.includes(n.trim().toLowerCase())).map(([i])=>i);
  if (idxs.length === 0) {
    idxs = labels.map((n,i)=>[i,n]).filter(([,n])=>{
      const L = n.trim().toLowerCase();
      return needles.some(m=>L.includes(m));
    }).map(([i])=>i);
  }
  return Array.from(new Set(idxs));
}

/* ---------- Charts ---------- */
const RAW_COLOR = '#60a5fa';       // blue
const SMOOTH_COLOR = '#ef4444';    // red
const THRESH_ON_COLOR  = '#f97316'; // orange
const THRESH_OFF_COLOR = '#fb923c'; // light orange

function setupChart() {
  if (state.chart) state.chart.destroy();
  const ctx = els.chartCanvas.getContext('2d');
  state.chart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'cry_score (raw)', data: [], borderColor: RAW_COLOR,   borderWidth: 2, pointRadius: 0, tension: 0.15 },
      { label: 'cry_score (smoothed)', data: [], borderColor: SMOOTH_COLOR, borderWidth: 2, pointRadius: 0, tension: 0.15 },
      { label: 'threshold (ON)',  data: [], borderColor: THRESH_ON_COLOR,  borderWidth: 1, pointRadius: 0, borderDash: [6,4] },
      { label: 'threshold (OFF)', data: [], borderColor: THRESH_OFF_COLOR, borderWidth: 1, pointRadius: 0, borderDash: [2,2] }
    ]},
    options: {
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Time' } },
        y: { title: { display: true, text: 'Cry score' }, suggestedMin: 0, suggestedMax: 1.0 }
      },
      elements: { line: { fill: false } }
    }
  });
}

const spectrumOverlay = {
  id: 'spectrumOverlay',
  afterDatasetsDraw(chart) {
    const { ctx, chartArea, scales } = chart;
    if (!chart.$peakFreq) return;
    // Shade 300–3000 Hz band
    const x1 = scales.x.getPixelForValue(300);
    const x2 = scales.x.getPixelForValue(3000);
    ctx.save();
    ctx.fillStyle = 'rgba(239, 68, 68, 0.08)';
    const left = Math.min(x1, x2);
    const width = Math.abs(x2 - x1);
    ctx.fillRect(left, chartArea.top, width, chartArea.bottom - chartArea.top);
    // Peak marker
    const xp = scales.x.getPixelForValue(chart.$peakFreq);
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 2]);
    ctx.beginPath(); ctx.moveTo(xp, chartArea.top); ctx.lineTo(xp, chartArea.bottom); ctx.stroke();
    ctx.restore();
  }
};

function setupSpectrumChart(freqs) {
  if (state.spectrumChart) state.spectrumChart.destroy();
  const ctx = els.chartSpectrumCanvas.getContext('2d');
  state.spectrumChart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        { label: 'spectrum (raw)', data: [], borderColor: RAW_COLOR,   borderWidth: 2, pointRadius: 0, tension: 0 },
        { label: 'spectrum (smoothed)', data: [], borderColor: SMOOTH_COLOR, borderWidth: 2, pointRadius: 0, tension: 0 }
      ]
    },
    options: {
      animation: false,
      plugins: { legend: { display: false } },
      parsing: false,
      scales: {
        x: {
          type: 'logarithmic',
          min: 20,
          max: Math.max(200, Math.round(freqs[freqs.length-1])),
          title: { display: true, text: 'Frequency (Hz, log)' },
          ticks: {
            callback: (val) => {
              const nice = [20,50,100,200,500,1000,2000,5000,8000];
              if (nice.includes(Math.round(val))) return val >= 1000 ? (val/1000)+'k' : val;
              return '';
            }
          }
        },
        y: { min: -100, max: 0, title: { display: true, text: 'Power (dBFS)' } }
      },
      elements: { line: { fill: false } }
    },
    plugins: [spectrumOverlay]
  });
}

function updateChartWindow(params) {
  const { PLOT_WINDOW_S } = params;
  const tMs = state.timesMs, r = state.cryRaw, s = state.crySm;

  let cut = 0;
  if (tMs.length) {
    const startMs = tMs[tMs.length - 1] - (PLOT_WINDOW_S * 1000);
    const idx = tMs.findIndex(v => v >= startMs);
    cut = idx >= 0 ? idx : 0;
  }

  const labels = tMs.slice(cut).map(formatClock);
  const raw = r.slice(cut);
  const sm = s.slice(cut);

  const onVal  = Math.max(0, (params.THRESH || 0) + (params.ALERT_ON_EXTRA || 0));
  const offVal = Math.max(0, onVal - (params.ALERT_OFF_DELTA || 0));
  const thrOn  = new Array(labels.length).fill(onVal);
  const thrOff = new Array(labels.length).fill(offVal);

  state.chart.data.labels = labels;
  state.chart.data.datasets[0].data = raw;
  state.chart.data.datasets[1].data = sm;
  state.chart.data.datasets[2].data = thrOn;
  state.chart.data.datasets[3].data = thrOff;

  if (raw.length || sm.length) {
    const all = raw.concat(sm, [onVal, offVal]);
    const ymin = Math.min(...all), ymax = Math.max(...all), pad = 0.05;
    state.chart.options.scales.y.min = Math.min(0, ymin - pad);
    state.chart.options.scales.y.max = Math.min(1.5, ymax + pad);
  }
  state.chart.update();
}

function updateSpectrumChartVisibility() {
  if (!state.spectrumChart) return;
  state.spectrumChart.data.datasets[0].hidden = !els.showRawSpec?.checked;
  state.spectrumChart.data.datasets[1].hidden = !els.showSmSpec?.checked;
  state.spectrumChart.update();
}

function updateSpectrumChart() {
  if (!state.spectrumChart) return;
  const raw = [], sm = [];
  for (let i = state.specStartIdx; i <= state.specEndIdx; i++) {
    const f = state.specFreqs[i];
    raw.push({ x: f, y: state.specRawDb[i] });
    sm.push({ x: f, y: state.specSmDb[i] });
  }
  state.spectrumChart.data.datasets[0].data = raw;
  state.spectrumChart.data.datasets[1].data = sm;

  // Peak
  let maxDb = -Infinity, maxIdx = -1;
  for (let i = state.specStartIdx; i <= state.specEndIdx; i++) {
    const v = state.specSmDb[i]; if (v > maxDb) { maxDb = v; maxIdx = i; }
  }
  if (maxIdx >= 0) {
    state.peakFreq = state.specFreqs[maxIdx];
    state.peakDb = maxDb;
    state.spectrumChart.$peakFreq = state.peakFreq;
    if (els.peakReadout) {
      const f = state.peakFreq >= 1000 ? (state.peakFreq/1000).toFixed(2)+' kHz' : Math.round(state.peakFreq)+' Hz';
      els.peakReadout.textContent = `Peak: ${f} @ ${Math.round(state.peakDb)} dB`;
    }
  } else {
    state.spectrumChart.$peakFreq = null;
    if (els.peakReadout) els.peakReadout.textContent = 'Peak: —';
  }
  state.spectrumChart.update();
}

/* ---------- Alert ---------- */
function setAlert(on) { document.body.classList.toggle('alert', !!on); }
function shouldAlert(smoothed, nowMs, params) {
  const onThresh  = (params.THRESH || 0) + (params.ALERT_ON_EXTRA || 0);
  const offThresh = Math.max(0, onThresh - (params.ALERT_OFF_DELTA || 0));
  if (state.alertOn) {
    if (nowMs < state.alertUntilMs) return true;
    return smoothed >= offThresh;
  } else {
    if (smoothed >= onThresh) {
      state.alertUntilMs = nowMs + (params.ALERT_HOLD_MS || 0);
      return true;
    }
    return false;
  }
}

/* ---------- Audio helpers ---------- */
function concatFloat32(a, b) { const out = new Float32Array(a.length + b.length); out.set(a, 0); out.set(b, a.length); return out; }
function resampleLinear(x, fromSr, toSr) {
  if (fromSr === toSr) return x;
  const ratio = toSr / fromSr, n = Math.max(1, Math.round(x.length * ratio)), out = new Float32Array(n);
  const dx = (x.length - 1) / (n - 1);
  for (let i=0;i<n;i++){ const pos=i*dx, i0=Math.floor(pos), i1=Math.min(i0+1, x.length-1), frac=pos-i0; out[i]=x[i0]*(1-frac)+x[i1]*frac; }
  return out;
}
function movingAvgTail(arr, k) { if (!arr.length) return 0; const start=Math.max(0, arr.length-k); let sum=0; for (let i=start;i<arr.length;i++) sum+=arr[i]; return sum/(arr.length-start); }

/* ---------- MediaPipe setup + inference ---------- */
async function ensureMPClassifierAndLabels() {
  if (!state.mpFileset) {
    state.mpFileset = await FilesetResolver.forAudioTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio@0.10.0/wasm"
    );
  }
  if (!state.mpClassifier) {
    const { MODEL_URL, SR_TARGET } = getParams();
    // Use createFromModelPath; classification is synchronous
    state.mpClassifier = await AudioClassifier.createFromModelPath(state.mpFileset, MODEL_URL);
    // Some versions let you set sampleRate via classify(..., sampleRate)
    state.mpClassifier.__sampleRate = SR_TARGET;
  }
  if (!state.labelList.length) {
    setStatus('Loading class_map.csv…');
    state.labelList = await loadLocalClassMap();
  }
  const { INCLUDE_CLASSES } = getParams();
  state.cryIdxs = findCryIdxs(state.labelList, INCLUDE_CLASSES);
  if (state.cryIdxs.length === 0) throw new Error('No matching classes found. Use ";" or new lines as separators.');
  setStatus('Ready');
}

// Classify a 0.96 s 16kHz mono clip with MediaPipe (Float32Array path)
function classifyCryScoreMP(window16k) {
  const sr = state.mpClassifier.__sampleRate || 16000;
  const results = state.mpClassifier.classify(window16k, sr); // synchronous
  if (!results || !results.length) return 0;

  // Average across returned chunks; sum selected indices per chunk
  let total = 0, chunks = 0;
  for (const res of results) {
    const head = res.classifications && res.classifications[0];
    const cats = head?.categories || [];
    let sumChunk = 0;
    for (const c of cats) if (state.cryIdxs.includes(c.index)) sumChunk += c.score;
    total += sumChunk; chunks += 1;
  }
  return chunks ? (total / chunks) : 0;
}

/* ---------- Start / Stop ---------- */
let spectrumRaf = 0;

async function start() {
  if (state.running) return;

  try {
    setStatus('Initializing…');
    await ensureMPClassifierAndLabels();

    // Reset series & plot
    state.ring16k = new Float32Array(0);
    state.lastPos = 0;
    state.timesSec = [];
    state.timesMs = [];
    state.cryRaw = [];
    state.crySm = [];
    state.alertOn = false;
    state.alertUntilMs = 0;
    setupChart();
    setAlert(false);

    // Audio
    const ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: "interactive" });
    state.audioCtx = ctx;
    await ctx.resume();

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, noiseSuppression: false, echoCancellation: false, autoGainControl: false },
      video: false
    });
    state.mediaStream = stream;

    const src = ctx.createMediaStreamSource(stream);
    const sink = ctx.createGain(); sink.gain.value = 0; sink.connect(ctx.destination);

    // Spectrum analyser
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.0;
    state.analyser = analyser;

    // Wire audio graph
    src.connect(analyser);
    let processor;
    try {
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
    } catch {
      processor = ctx.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (e) => onAudioChunk(new Float32Array(e.inputBuffer.getChannelData(0)), ctx.sampleRate);
      src.connect(processor).connect(sink);
    }
    state.processor = processor;

    // Spectrum arrays (≤ 8 kHz)
    const binCount = analyser.frequencyBinCount;
    state.specFreqs = new Array(binCount);
    for (let i = 0; i < binCount; i++) state.specFreqs[i] = i * ctx.sampleRate / analyser.fftSize;
    const nyquist = ctx.sampleRate / 2, fMax = Math.min(8000, nyquist);
    state.specStartIdx = 0; while (state.specStartIdx < binCount && state.specFreqs[state.specStartIdx] < 20) state.specStartIdx++;
    state.specEndIdx = binCount - 1; while (state.specEndIdx > 0 && state.specFreqs[state.specEndIdx] > fMax) state.specEndIdx--;
    state.specRawDb = new Array(binCount).fill(-100);
    state.specSmDb  = new Array(binCount).fill(-100);
    setupSpectrumChart(state.specFreqs.slice(state.specStartIdx, state.specEndIdx + 1));

    // Spectrum loop (~10 Hz)
    state.specLastUpdate = 0;
    function spectrumTick(ts) {
      if (!state.running || !state.analyser) return;
      if (ts - state.specLastUpdate > 100) {
        const buf = new Float32Array(binCount);
        state.analyser.getFloatFrequencyData(buf);
        for (let i = state.specStartIdx; i <= state.specEndIdx; i++) {
          const db = Math.max(-100, Math.min(0, buf[i]));
          state.specRawDb[i] = db;
          state.specSmDb[i] = 0.8 * state.specSmDb[i] + 0.2 * db;
        }
        updateSpectrumChart();
        state.specLastUpdate = ts;
      }
      spectrumRaf = requestAnimationFrame(spectrumTick);
    }
    spectrumRaf = requestAnimationFrame(spectrumTick);

    // Toggles
    els.showRawSpec?.addEventListener('change', updateSpectrumChartVisibility);
    els.showSmSpec?.addEventListener('change', updateSpectrumChartVisibility);
    updateSpectrumChartVisibility();

    state.running = true;
    els.btnStart && (els.btnStart.disabled = true);
    els.btnStop  && (els.btnStop.disabled  = false);
    setStatus(`Listening @ ${ctx.sampleRate.toFixed(0)} Hz (MediaPipe)`);
  } catch (e) {
    setStatus('Start failed');
    try { state.processor?.disconnect(); } catch {}
    try { if (state.audioCtx && state.audioCtx.state !== 'closed') await state.audioCtx.close(); } catch {}
    try { state.mediaStream?.getTracks().forEach(t => t.stop()); } catch {}
    if (spectrumRaf) cancelAnimationFrame(spectrumRaf);
    state.audioCtx = null; state.mediaStream = null; state.processor = null; state.analyser = null;
    state.running = false;
    els.btnStart && (els.btnStart.disabled = false);
    els.btnStop  && (els.btnStop.disabled  = true);
  }
}

async function stop() {
  try { state.processor?.disconnect(); } catch {}
  try { if (state.audioCtx) await state.audioCtx.close(); } catch {}
  try { state.mediaStream?.getTracks().forEach(t => t.stop()); } catch {}
  if (spectrumRaf) cancelAnimationFrame(spectrumRaf);
  state.running = false; state.analyser = null;
  els.btnStart && (els.btnStart.disabled = false);
  els.btnStop  && (els.btnStop.disabled  = true);
  setStatus('Stopped');
}

/* ---------- Audio chunk handler ---------- */
async function onAudioChunk(chunkFloat32, deviceSr) {
  const params = getParams();
  const hopMs = Math.round(params.HOP_S * 1000);

  // Resample to SR_TARGET and clamp
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

    // Classify (Float32Array path)
    let score = 0;
    try {
      // Ensure exact length expected by model (~0.96 s); our window is already right-aligned; pass as-is
      score = classifyCryScoreMP(w);
    } catch {
      // keep score = 0 on failure
    }

    state.cryRaw.push(score);
    state.crySm.push(movingAvgTail(state.cryRaw, params.SMOOTH_WIN));

    state.timesSec.push(state.timesSec.length * params.HOP_S);
    const nextMs = state.timesMs.length ? (state.timesMs[state.timesMs.length - 1] + hopMs) : Date.now();
    state.timesMs.push(nextMs);

    updateChartWindow(params);

    const smoothed = state.crySm[state.crySm.length - 1];
    const nowMs = Date.now();
    const alert = shouldAlert(smoothed, nowMs, params);
    state.alertOn = alert; setAlert(alert);

    state.lastPos += frameHop16k;

    // Bound memory
    if (state.timesSec.length % 400 === 0) {
      const keepFrom = Math.max(0, state.timesSec.length - Math.ceil(params.PLOT_WINDOW_S / params.HOP_S) - 10);
      if (keepFrom > 0) {
        state.timesSec = state.timesSec.slice(keepFrom);
        state.timesMs  = state.timesMs.slice(keepFrom);
        state.cryRaw   = state.cryRaw.slice(keepFrom);
        state.crySm    = state.crySm.slice(keepFrom);
      }
      const minNeeded = state.lastPos - frameLen16k - frameHop16k;
      if (minNeeded > 0) {
        state.ring16k = state.ring16k.slice(minNeeded);
        state.lastPos -= minNeeded;
      }
    }
  }
}

/* ---------- Wire up ---------- */
document.addEventListener('DOMContentLoaded', () => {
  bindElements();
  els.btnStart && (els.btnStart.disabled = false);
  els.btnStop  && (els.btnStop.disabled  = true);

  els.btnStart?.addEventListener('click', start);
  els.btnStop?.addEventListener('click', stop);

  Object.keys(els.inputs).forEach(k => {
    els.inputs[k]?.addEventListener('change', () => updateChartWindow(getParams()));
  });

  els.showRawSpec?.addEventListener('change', updateSpectrumChartVisibility);
  els.showSmSpec?.addEventListener('change', updateSpectrumChartVisibility);

  setStatus('Page ready. Click Start.');
});
