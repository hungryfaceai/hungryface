// audio-resampler.js — streaming resampler for feeding YAMNet (16 kHz mono).
//
// Why this exists: AudioWorklets deliver audio in tiny 128-sample chunks. Resampling
// each chunk on its own (as the old resampleLinear did) stretches every chunk to fit
// a rounded output length, which shifts pitch by ~0.8% and adds a discontinuity every
// ~2.7 ms; it also had no low-pass filter, so content above 8 kHz aliased into the
// cry band. This class keeps state across chunks, so the output is one continuous
// signal, and low-pass filters before decimating.
//
// Usage:
//   const rs = createStreamingResampler(ctx.sampleRate, 16000);
//   const out16k = rs.process(chunk);   // call for every chunk, in order
//   rs.reset();                         // on stop/restart

// Windowed-sinc low-pass FIR (Blackman window). cutoff is a fraction of the input rate.
function designLowpass(taps, cutoff) {
  const h = new Float32Array(taps);
  const m = taps - 1;
  let sum = 0;
  for (let i = 0; i < taps; i++) {
    const x = i - m / 2;
    const sinc = x === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * x) / (Math.PI * x);
    const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / m) + 0.08 * Math.cos((4 * Math.PI * i) / m);
    h[i] = sinc * w;
    sum += h[i];
  }
  for (let i = 0; i < taps; i++) h[i] /= sum; // unity gain at DC
  return h;
}

export function createStreamingResampler(fromSr, toSr, { taps = 63 } = {}) {
  fromSr = Number(fromSr) || toSr;
  const step = fromSr / toSr; // input samples per output sample
  const passthrough = Math.abs(step - 1) < 1e-9;

  // Low-pass only when downsampling: cut at 90% of the output Nyquist.
  const down = step > 1;
  const fir = down ? designLowpass(taps, (0.5 / step) * 0.9) : null;
  const firLen = fir ? fir.length : 0;
  const H = Math.max(0, firLen - 1);
  let hist = new Float32Array(H); // last H input samples (zeros at start)

  // Filtered input not yet consumed by the interpolator. pos is the fractional read
  // position of the next output sample, relative to buf[0].
  let buf = new Float32Array(0);
  let pos = 0;
  // Initial group delay compensation is unnecessary for classification; we accept it.

  function lowpass(chunk) {
    if (!fir) return chunk;
    const n = chunk.length;
    // ext = [history | chunk]; output i uses ext[i .. i+H]
    const ext = new Float32Array(H + n);
    ext.set(hist, 0);
    ext.set(chunk, H);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (let k = 0; k < firLen; k++) acc += fir[k] * ext[i + H - k];
      out[i] = acc;
    }
    hist = ext.slice(n); // keep the last H samples for the next chunk
    return out;
  }

  return {
    process(chunk) {
      if (!chunk || !chunk.length) return new Float32Array(0);
      if (passthrough) return Float32Array.from(chunk);
      const f = lowpass(chunk);

      // append filtered samples to the interpolation buffer
      const merged = new Float32Array(buf.length + f.length);
      merged.set(buf, 0);
      merged.set(f, buf.length);
      buf = merged;

      // linear interpolation needs buf[i0] and buf[i0+1], so read positions must be
      // strictly < buf.length - 1 (a position exactly on the last sample waits for more input)
      const avail = buf.length - 1;
      const nOut = pos < avail ? Math.ceil((avail - pos) / step) : 0;
      const out = new Float32Array(Math.max(0, nOut));
      let p = pos;
      for (let i = 0; i < out.length; i++) {
        const i0 = Math.floor(p),
          t = p - i0;
        out[i] = buf[i0] * (1 - t) + buf[i0 + 1] * t;
        p += step;
      }
      // drop consumed samples, keep the fractional remainder
      const drop = Math.min(Math.floor(p), buf.length);
      buf = buf.slice(drop);
      pos = p - drop;
      return out;
    },
    reset() {
      buf = new Float32Array(0);
      pos = 0;
      hist = new Float32Array(H);
    },
    get fromSr() {
      return fromSr;
    },
    get toSr() {
      return toSr;
    },
  };
}
