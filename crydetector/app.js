/* app.js — adds a real-time YAMNet-style log-mel spectrogram
   - 64 mel bins, ~25 ms window, 10 ms hop, 512-pt FFT, 125..7500 Hz
   - drawn from the SAME 0.96 s window used for YAMNet inference
   - heatmap palette: low→panel, mid→blue, high→red
   - existing cry chart (with ON/OFF lines) and spectrum upgrades retained
*/

const els = {
  btnStart: null,
  btnStop: null,
  status: null,
  chartCanvas: null,
  chartSpectrumCanvas: null,
  melCanvas: null,
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
  els.melCanvas = document.getElementById('melCanvas');
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
    MODEL_URL: document.getElementById('MODEL_URL'),
    INCLUDE_CLASSES: document.getElementById('INCLUDE_CLASSES'),
    ALERT_ON_EXTRA: document.getElementById('ALERT_ON_EXTRA'),
    ALERT_OFF_DELTA: document.getElementById('ALERT_OFF_DELTA'),
    ALERT_HOLD_MS: document.getElementById('ALERT_HOLD_MS'),
  };
}

function setStatus(msg) { if (els.status) els.status.textContent = msg; }

const state = {
  running: false,
  audioCtx: null,
  mediaStream: null,
  processor: null,
  analyser: null,
  model: null,
  labelList: [],
  cryIdxs: [],
  ring16k: new Float32Array(0),
  timesSec: [],
  timesMs: [],
  cryRaw: [],
  crySm: [],
  lastPos: 0,
  chart: null,
  spectrumChart: null,

  /* Spectrum buffers */
  specRawDb: [],
  specSmDb: [],
  specFreqs: [],
  specBinCountAll: 0,
  specStartIdx: 0,
  specEndIdx: 0,
  specLastUpdate: 0,

  /* Peak */
  peakFreq: null,
  peakDb: null,

  /* Alert state */
  alertOn: false,
  alertUntilMs: 0,

  /* Mel-spec precompute */
  melFilters: null,      // array of arrays [{k, w}, ...] per mel bin
  melWin: null,          // Hann window (winLength)
  fftN: 512,
  winLength: 400,        // ~25 ms @ 16 kHz
  hopLength: 160,        // 10 ms
  nMels: 64,
  melFmin: 125,
  melFmax: 7500,
  bitrev: null,          // bit-reversal indices for FFT
  twiddleCos: null,
  twiddleSin: null,
};

/* ===== Utils ===== */
function splitClasses(text) {
  return String(text || '').split(/[;\n\r]+/).map(s => s.trim()).filter(Boolean);
}
function getParams() {
  return {
    SR_TARGET: parseInt(els.inputs.SR_TARGET?.value, 10) || 16000,
    FRAME_LEN_S: parseFloat(els.inputs.FRAME_LEN_S?.value) || 0.96,
    HOP_S: parseFloat(els.inputs.HOP_S?.value) || 0.48,
    SMOOTH_WIN: Math.max(1, parseInt(els.inputs.SMOOTH_WIN?.value, 10) || 5),
    THRESH: parseFloat(els.inputs.THRESH?.value) || 0.25,
    PLOT_WINDOW_S: Math.max(5, parseFloat(els.inputs.PLOT_WINDOW_S?.value) || 60),
    MODEL_URL: (els.inputs.MODEL_URL?.value || 'https://tfhub.dev/google/tfjs-model/yamnet/tfjs/1').trim(),
    INCLUDE_CLASSES: splitClasses(els.inputs.INCLUDE_CLASSES?.value),
    ALERT_ON_EXTRA: parseFloat(els.inputs.ALERT_ON_EXTRA?.value ?? '0') || 0,
    ALERT_OFF_DELTA: Math.max(0, parseFloat(els.inputs.ALERT_OFF_DELTA?.value ?? '0.1') || 0.1),
    ALERT_HOLD_MS: Math.max(0, parseInt(els.inputs.ALERT_HOLD_MS?.value, 10) || 1500),
  };
}
function formatClock(ms) {
  return new Date(ms).toLocaleTimeString(undefined, { hour12: false });
}

/* ===== CSV + LABELS ===== */
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

/* ===== Charts (cry + spectrum) ===== */
const RAW_COLOR = '#60a5fa';
const SMOOTH_COLOR = '#ef4444';
const THRESH_ON_COLOR  = '#f97316';
const THRESH_OFF_COLOR = '#fb923c';

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
    // 300–3000 Hz band shading
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

  // Peak (from smoothed)
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

function setAlert(on) { document.body.classList.toggle('alert', !!on); }
function shouldAlert(smoothed, nowMs, params) {
  const onThresh  = (params.THRESH || 0) + (params.ALERT_ON_EXTRA || 0);
  const offThresh = Math.max(0, onThresh - (params.ALERT_OFF_DELTA || 0));
  if (state.alertOn) {
    if (nowMs < state.alertUntilMs) return true;
    return smoothed >= offThresh;
  } else {
    if (smoothed >= onThresh) { state.alertUntilMs = nowMs + (params.ALERT_HOLD_MS || 0); return true; }
    return false;
  }
}

/* ===== Audio utils ===== */
function concatFloat32(a, b) { const out = new Float32Array(a.length + b.length); out.set(a, 0); out.set(b, a.length); return out; }
function resampleLinear(x, fromSr, toSr) {
  if (fromSr === toSr) return x;
  const ratio = toSr / fromSr, n = Math.max(1, Math.round(x.length * ratio)), out = new Float32Array(n);
  const dx = (x.length - 1) / (n - 1);
  for (let i=0;i<n;i++){ const pos=i*dx, i0=Math.floor(pos), i1=Math.min(i0+1, x.length-1), frac=pos-i0; out[i]=x[i0]*(1-frac)+x[i1]*frac; }
  return out;
}
function movingAvgTail(arr, k) { if (!arr.length) return 0; const start=Math.max(0, arr.length-k); let sum=0; for (let i=start;i<arr.length;i++) sum+=arr[i]; return sum/(arr.length-start); }

/* ===== FFT & Mel precompute ===== */
function initFFT(n) {
  // bit-reversal
  const bitrev = new Uint32Array(n);
  let bits = 0; while ((1<<bits) < n) bits++;
  for (let i=0;i<n;i++){
    let x=i, y=0;
    for (let b=0;b<bits;b++){ y=(y<<1)|(x&1); x>>=1; }
    bitrev[i]=y;
  }
  // twiddles per stage: store cos/sin for each half-size (k)
  const twiddleCos = [], twiddleSin = [];
  for (let m=2; m<=n; m<<=1){
    const half = m>>1, step = Math.PI/half;
    const cosArr = new Float32Array(half), sinArr = new Float32Array(half);
    for (let k=0;k<half;k++){ const ang = -k*step; cosArr[k]=Math.cos(ang); sinArr[k]=Math.sin(ang); }
    twiddleCos.push(cosArr); twiddleSin.push(sinArr);
  }
  return { bitrev, twiddleCos, twiddleSin };
}
function fftRadix2(re, im, n, bitrev, twC, twS){
  // bit-reverse copy
  for (let i=0;i<n;i++){
    const j = bitrev[i];
    if (j>i){ const tr=re[i]; re[i]=re[j]; re[j]=tr; const ti=im[i]; im[i]=im[j]; im[j]=ti; }
  }
  // stages
  let stage = 0;
  for (let m=2; m<=n; m<<=1){
    const half = m>>1, cosArr = twC[stage], sinArr = twS[stage];
    for (let i=0;i<n;i+=m){
      for (let k=0;k<half;k++){
        const tcos = cosArr[k], tsin = sinArr[k];
        const j = i + k, l = j + half;
        const ur = re[j], ui = im[j];
        const vr = re[l]*tcos - im[l]*tsin;
        const vi = re[l]*tsin + im[l]*tcos;
        re[j] = ur + vr; im[j] = ui + vi;
        re[l] = ur - vr; im[l] = ui - vi;
      }
    }
    stage++;
  }
}
function hannWindow(N){ const w=new Float32Array(N); for(let n=0;n<N;n++){ w[n]=0.5*(1-Math.cos(2*Math.PI*n/(N-1))); } return w; }
function hz2mel(f){ return 2595*Math.log10(1+f/700); }
function mel2hz(m){ return 700*(Math.pow(10, m/2595)-1); }
function buildMelFilters(sr, nFft, nMels, fMin, fMax){
  const nSpec = nFft/2 + 1;
  const mMin = hz2mel(fMin), mMax = hz2mel(fMax);
  const mPts = new Float32Array(nMels+2);
  for(let i=0;i<mPts.length;i++) mPts[i] = mMin + (i*(mMax-mMin)/(nMels+1));
  const fPts = new Float32Array(nMels+2);
  for(let i=0;i<fPts.length;i++) fPts[i] = mel2hz(mPts[i]);
  const bPts = new Int32Array(nMels+2);
  for(let i=0;i<bPts.length;i++) bPts[i] = Math.floor((nFft+1)*fPts[i]/sr);

  // Sparse filters: for each mel bin, store [{k, w}, ...]
  const filters = new Array(nMels);
  for(let m=1;m<=nMels;m++){
    const a=bPts[m-1], b=bPts[m], c=bPts[m+1];
    const list = [];
    for(let k=a;k<b;k++){
      if (k>=0 && k<nSpec){
        list.push({k, w:(k-a)/(b-a+1e-9)});
      }
    }
    for(let k=b;k<=c;k++){
      if (k>=0 && k<nSpec){
        list.push({k, w:(c-k)/(c-b+1e-9)});
      }
    }
    filters[m-1]=list;
  }
  return filters;
}

/* ===== Log-mel spectrogram ===== */
function computeLogMelSpectrogram(window16k) {
  // Uses state.winLength (400), state.hopLength (160), state.fftN (512),
  // state.melFilters (64), Hann window, returns {db: frames x nMels, minDb, maxDb}
  const x = window16k;
  const N = x.length;
  const { winLength, hopLength, fftN, melFilters, melFmin, melFmax } = state;
  if (!state.melWin) state.melWin = hannWindow(winLength);
  if (!state.bitrev) {
    const { bitrev, twiddleCos, twiddleSin } = initFFT(fftN);
    state.bitrev = bitrev; state.twiddleCos = twiddleCos; state.twiddleSin = twiddleSin;
  }

  const nFrames = 1 + Math.floor((N - winLength) / hopLength);
  const nSpec = fftN/2 + 1;
  const nMels = melFilters.length;
  const melDb = new Array(nFrames);
  let globalMax = -Infinity;

  // Buffers
  const re = new Float32Array(fftN);
  const im = new Float32Array(fftN);

  for (let f = 0; f < nFrames; f++) {
    const start = f*hopLength;
    re.fill(0); im.fill(0);
    // copy + window
    for (let i=0;i<winLength;i++){ re[i] = x[start+i] * state.melWin[i]; }

    // FFT
    fftRadix2(re, im, fftN, state.bitrev, state.twiddleCos, state.twiddleSin);

    // Power spectrum
    const pow = new Float32Array(nSpec);
    for (let k=0;k<nSpec;k++){ const pr=re[k], pi=im[k]; pow[k] = pr*pr + pi*pi; }

    // Apply mel filters
    const mel = new Float32Array(nMels);
    for (let m=0;m<nMels;m++){
      let s = 0; const filt = melFilters[m];
      for (let p=0;p<filt.length;p++){ const {k,w}=filt[p]; s += w * pow[k]; }
      mel[m] = s + 1e-10; // add epsilon
    }

    // Convert to dB (relative to max within frame)
    let maxE = 1e-10; for (let m=0;m<nMels;m++) if (mel[m]>maxE) maxE = mel[m];
    const frameDb = new Float32Array(nMels);
    for (let m=0;m<nMels;m++){
      const db = 10 * Math.log10(mel[m]/maxE); // normalize per-frame (0 dB max)
      frameDb[m] = Math.max(-80, Math.min(0, db));
      if (frameDb[m] > globalMax) globalMax = frameDb[m];
    }
    melDb[f] = frameDb; // note: index = time, value = [mel]
  }

  return { db: melDb, minDb: -80, maxDb: 0 };
}

function drawMelSpectrogram(melObj) {
  const { db, minDb, maxDb } = melObj;
  if (!els.melCanvas) return;
  const ctx = els.melCanvas.getContext('2d');
  const W = els.melCanvas.width  || els.melCanvas.getBoundingClientRect().width  || 800;
  const H = els.melCanvas.height || els.melCanvas.getBoundingClientRect().height || 160;
  // force canvas bitmap to layout width for crispness
  els.melCanvas.width = Math.floor(W);
  els.melCanvas.height = Math.floor(H);

  const nFrames = db.length;
  if (!nFrames) return;
  const nMels = db[0].length;

  const cellW = Math.max(1, Math.floor(W / nFrames));
  const cellH = Math.max(1, Math.floor(H / nMels));

  // Colors
  const low = hexToRgb(getCss('--spec-low'));
  const mid = hexToRgb(getCss('--spec-mid'));
  const high= hexToRgb(getCss('--spec-high'));

  // Draw each time x mel cell
  for (let f=0; f<nFrames; f++) {
    const col = db[f]; // Float32Array (nMels)
    for (let m=0; m<nMels; m++) {
      // YAMNet displays low freqs at bottom → invert y
      const y = H - (m+1)*cellH;
      // normalize dB [-80..0] → [0..1]
      const norm = (col[m] - minDb) / (maxDb - minDb);
      const rgb = rampRgb(norm, low, mid, high);
      ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
      ctx.fillRect(f*cellW, y, cellW, cellH);
    }
  }
}

/* ===== Color helpers for spectrogram ===== */
function getCss(varName){ return getComputedStyle(document.documentElement).getPropertyValue(varName).trim(); }
function hexToRgb(hex){
  const h = hex.replace('#','');
  const bigint = parseInt(h,16);
  return h.length===6 ? [(bigint>>16)&255, (bigint>>8)&255, bigint&255] : [0,0,0];
}
function lerp(a,b,t){ return Math.round(a + (b-a)*t); }
function rampRgb(t, low, mid, high){
  // piecewise: [0..0.5] low→mid, [0.5..1] mid→high
  const u = Math.max(0, Math.min(1, t));
  if (u <= 0.5){
    const tt = u/0.5;
    return [ lerp(low[0], mid[0], tt), lerp(low[1], mid[1], tt), lerp(low[2], mid[2], tt) ];
  } else {
    const tt = (u-0.5)/0.5;
    return [ lerp(mid[0], high[0], tt), lerp(mid[1], high[1], tt), lerp(mid[2], high[2], tt) ];
  }
}

/* ===== YAMNet inference (1-D input) ===== */
function getModelInputName() {
  try { const inp = state.model.inputs?.[0]; if (inp?.name) return inp.name; } catch {}
  return 'waveform';
}
async function inferCryScore16k(wave16k, cryIdxs) {
  const { FRAME_LEN_S } = getParams();
  const targetLen = Math.round(FRAME_LEN_S * 16000);

  // Right-align to targetLen (pad/crop)
  let x = wave16k;
  if (x.length !== targetLen) {
    if (x.length < targetLen) {
      const pad = new Float32Array(targetLen);
      pad.set(x, targetLen - x.length);
      x = pad;
    } else {
      x = x.slice(x.length - targetLen);
    }
  }

  // 1-D tensor ([-1])
  const t1d = tf.tensor1d(x);
  let out;
  try { const feed = {}; feed[getModelInputName()] = t1d; out = await state.model.executeAsync(feed); }
  catch { out = await state.model.executeAsync(t1d); }
  finally { t1d.dispose(); }

  // scores: [num_patches, 521]
  let scores;
  if (Array.isArray(out)) {
    scores = out.find(o => o?.shape?.length === 2 && o.shape[1] >= 500) || out[0];
    out.forEach(o => { if (o !== scores) try { o.dispose(); } catch {} });
  } else if (out && out.shape && out.shape.length === 2) {
    scores = out;
  } else if (out && out.scores) {
    scores = out.scores;
    Object.keys(out).forEach(k => { if (k !== 'scores') try { out[k].dispose(); } catch {} });
  } else {
    throw new Error('Unexpected YAMNet output; scores tensor not found');
  }

  // Average over frames then sum selected classes (multi-label → can exceed 1)
  const score = tf.tidy(() => {
    const meanOverFrames = scores.mean(0);
    const idxs = tf.tensor1d(cryIdxs, 'int32');
    const selected = meanOverFrames.gather(idxs);
    const summed = selected.sum();
    return summed.dataSync()[0];
  });
  scores.dispose();
  return { score, x }; // return aligned window too (for mel)
}

/* ===== Model + labels ===== */
async function ensureModelAndLabels() {
  if (!state.model) {
    const { MODEL_URL } = getParams();
    state.model = await tf.loadGraphModel(MODEL_URL, { fromTFHub: true });
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

/* ===== Start / Stop ===== */
let spectrumRaf = 0;

async function start() {
  if (state.running) return;

  try {
    setStatus('Initializing…');
    await tf.ready().catch(()=>{});
    try { tf.setBackend('webgl'); } catch {}
    await ensureModelAndLabels();

    // Reset buffers/plot/alert
    state.ring16k = new Float32Array(0);
    state.timesSec = []; state.timesMs = [];
    state.cryRaw = []; state.crySm = []; state.lastPos = 0;
    state.alertOn = false; state.alertUntilMs = 0;
    setupChart(); setAlert(false);

    // Audio
    const ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: "interactive" });
    state.audioCtx = ctx; await ctx.resume();

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, noiseSuppression: false, echoCancellation: false, autoGainControl: false },
      video: false
    });
    state.mediaStream = stream;

    const src = ctx.createMediaStreamSource(stream);
    const sink = ctx.createGain(); sink.gain.value = 0; sink.connect(ctx.destination);

    // Spectrum analyser
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048; analyser.smoothingTimeConstant = 0.0;
    state.analyser = analyser;

    // Wire graph/audio paths
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

    // Prepare spectrum chart
    const binCount = analyser.frequencyBinCount;
    const nyquist = ctx.sampleRate / 2, fMax = Math.min(8000, nyquist);
    state.specFreqs = new Array(binCount);
    for (let i = 0; i < binCount; i++) state.specFreqs[i] = i * ctx.sampleRate / analyser.fftSize;
    state.specStartIdx = 0; while (state.specStartIdx < binCount && state.specFreqs[state.specStartIdx] < 20) state.specStartIdx++;
    state.specEndIdx = binCount - 1; while (state.specEndIdx > 0 && state.specFreqs[state.specEndIdx] > fMax) state.specEndIdx--;
    state.specRawDb = new Array(binCount).fill(-100);
    state.specSmDb  = new Array(binCount).fill(-100);
    setupSpectrumChart(state.specFreqs.slice(state.specStartIdx, state.specEndIdx + 1));

    // Mel filterbank precompute (SR_TARGET assumed 16 kHz)
    const { SR_TARGET } = getParams();
    state.melFilters = buildMelFilters(SR_TARGET, state.fftN, state.nMels, state.melFmin, state.melFmax);
    state.melWin = hannWindow(state.winLength);
    state.bitrev = null; state.twiddleCos = null; state.twiddleSin = null; // rebuild in compute if needed

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
    setStatus(`Listening @ ${ctx.sampleRate.toFixed(0)} Hz`);
  } catch {
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

/* ===== Audio callback ===== */
async function onAudioChunk(chunkFloat32, deviceSr) {
  const params = getParams();
  const hopMs = Math.round(params.HOP_S * 1000);

  // Resample to SR_TARGET
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

    let score = 0, aligned = w;
    try {
      const res = await inferCryScore16k(w, state.cryIdxs);
      score = res.score; aligned = res.x;
    } catch {}

    state.cryRaw.push(score);
    state.crySm.push(movingAvgTail(state.cryRaw, params.SMOOTH_WIN));

    state.timesSec.push(state.timesSec.length * params.HOP_S);
    const nextMs = state.timesMs.length ? (state.timesMs[state.timesMs.length - 1] + hopMs) : Date.now();
    state.timesMs.push(nextMs);

    updateChartWindow(params);

    // Hysteresis + hold decision
    const smoothed = state.crySm[state.crySm.length - 1];
    const nowMs = Date.now();
    const alert = shouldAlert(smoothed, nowMs, params);
    state.alertOn = alert; setAlert(alert);

    // NEW: compute + draw log-mel spectrogram for this 0.96 s window
    try {
      const mel = computeLogMelSpectrogram(aligned);
      drawMelSpectrogram(mel);
    } catch {}

    state.lastPos += frameHop16k;

    // Periodic trimming
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

/* ===== Wire up DOM ===== */
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
