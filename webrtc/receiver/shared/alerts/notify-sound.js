// /webrtc/receiver/shared/alerts/notify-sound.js
// Triple-pip alert sound (same as dashboard) with per-type switches + global cooldown.
// Debug: installAlertSound({debug:true}) OR ?notifyDebug=1 OR localStorage 'naptio:notify:debug'='1'

const KEYS = {
  audio: 'naptio:notify:audio',
  motion: 'naptio:notify:motion',
  fence: 'naptio:notify:fence',
  prone: 'naptio:notify:prone',
  cooldown: 'naptio:notifyCooldownMin',
  lastTs: 'naptio:notify:lastTs',
  tsPing: 'naptio:notify:ts',
};

function inferTypeFromPath(p = location.pathname) {
  if (/\/audio\//i.test(p))  return 'audio';
  if (/\/motion\//i.test(p)) return 'motion';
  if (/\/prone\//i.test(p))  return 'prone';
  if (/\/fence\//i.test(p))  return 'fence';
  return 'audio';
}

function settingIsOn(key, def = 'on') {
  const v = (localStorage.getItem(key) || def).toLowerCase();
  return v === 'on';
}
function getCooldownMs() {
  const n = parseInt(localStorage.getItem(KEYS.cooldown), 10);
  const minutes = Number.isFinite(n) ? n : 2;
  return Math.max(0, minutes) * 60_000;
}

let audioCtx = null;
let lastBeepWallNow = 0; // 1s safety limit (separate from minutes cooldown)
let DEBUG = false;

const dlog  = (...a) => { if (DEBUG) console.log('[notify-sound]', ...a); };
const dwarn = (...a) => { if (DEBUG) console.warn('[notify-sound]', ...a); };

function ensureCtx() {
  if (!audioCtx) {
    try {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      dlog('AudioContext created:', { sampleRate: audioCtx?.sampleRate, state: audioCtx?.state });
    } catch (e) {
      dwarn('AudioContext creation failed:', e);
    }
  }
  return audioCtx;
}

function unlockOnFirstGesture() {
  const once = async (ev) => {
    const ac = ensureCtx();
    if (!ac) { dwarn('No AudioContext on gesture'); return; }
    const before = ac.state;
    try { await ac.resume(); } catch {}
    dlog('Gesture unlock', ev?.type, 'state:', before, '→', ac.state);
  };
  ['pointerdown','touchend','keydown','click'].forEach(ev =>
    window.addEventListener(ev, once, { once:true, passive:true })
  );
}

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && audioCtx?.state === 'suspended') {
    dlog('Tab visible; resuming suspended AudioContext…');
    audioCtx.resume().then(() => dlog('AudioContext state:', audioCtx.state)).catch((e)=>dwarn('resume error', e));
  }
});

// ----- Triple pip (1.7k → 2.6k), three pips ~0.18s apart -----
async function playDashboardPips() {
  const ac = ensureCtx();
  if (!ac) { dwarn('No AudioContext; cannot play'); return; }

  if (ac.state === 'suspended') { try { await ac.resume(); } catch {} }
  if (ac.state !== 'running') { dwarn('AudioContext not running; abort. state=', ac.state); return; }

  const nowPerf = performance.now();
  if (nowPerf - lastBeepWallNow < 1000) { dlog('1s rate-limit; skip'); return; }
  lastBeepWallNow = nowPerf;

  const t0 = ac.currentTime;
  dlog('Playing triple pip @', t0);

  const pip = (start, dur = 0.12) => {
    const o = ac.createOscillator();
    const g = ac.createGain();
    o.type = 'sine';
    o.frequency.setValueAtTime(1700, start);
    o.frequency.exponentialRampToValueAtTime(2600, start + dur);
    g.gain.setValueAtTime(0.0001, start);
    g.gain.exponentialRampToValueAtTime(0.09, start + 0.012);
    g.gain.exponentialRampToValueAtTime(0.0001, start + dur);
    o.connect(g).connect(ac.destination);
    o.start(start);
    o.stop(start + dur + 0.02);
  };

  pip(t0);
  pip(t0 + 0.18);
  pip(t0 + 0.36);
}

function inMinutesCooldown(nowMs = Date.now()) {
  const last = parseInt(localStorage.getItem(KEYS.lastTs), 10) || 0;
  const inCd = (nowMs - last) < getCooldownMs();
  dlog('Cooldown check:', { nowMs, lastTs: last, cooldownMs: getCooldownMs(), inCooldown: inCd });
  return inCd;
}
function stampPlayed(nowMs = Date.now()) {
  try { localStorage.setItem(KEYS.lastTs, String(nowMs)); dlog('Stamped lastTs=', nowMs); } catch {}
}

function gateSnapshot(type) {
  return {
    type,
    perTypeOn: settingIsOn(KEYS[type], 'on'),
    cooldownMin: getCooldownMs() / 60000,
    lastTs: parseInt(localStorage.getItem(KEYS.lastTs), 10) || 0,
    audioCtxState: audioCtx?.state || 'none',
  };
}

export function installAlertSound(opts = {}) {
  const pageType = (opts.type || inferTypeFromPath());

  // Debug from opts, URL, or localStorage
  const qsDebug = new URLSearchParams(location.search).get('notifyDebug');
  DEBUG = !!opts.debug || qsDebug === '1' || (localStorage.getItem('naptio:notify:debug') === '1');
  dlog('installAlertSound:', { pageType, DEBUG });

  ensureCtx();
  unlockOnFirstGesture();

  // Optional cross-tab ping (noop here; kept for future use)
  window.addEventListener('storage', (e) => {
    if (e.key === KEYS.tsPing) { dlog('storage ping observed'); }
  });

  async function trigger(type = pageType) {
    const snap = gateSnapshot(type);
    dlog('trigger()', snap);

    if (!['audio','motion','fence','prone'].includes(type)) {
      dwarn('Invalid type for trigger:', type);
      return false;
    }
    if (!settingIsOn(KEYS[type], 'on')) {
      dlog(`Blocked by per-type toggle: ${type} is OFF`);
      return false;
    }

    const now = Date.now();
    if (inMinutesCooldown(now)) {
      dlog('Blocked by minutes cooldown.');
      return false;
    }

    await playDashboardPips();
    stampPlayed(now);
    return true;
  }

  const onEvt = (e) => {
    const t = (e?.detail?.type) || pageType;
    dlog('naptio:alert event → trigger', t);
    trigger(t);
  };
  window.addEventListener('naptio:alert', onEvt);

  // Debug helpers
  async function debugPing() { dlog('debugPing() — bypassing gates'); await playDashboardPips(); }
  function report() {
    const t = pageType;
    const snap = gateSnapshot(t);
    snap.cooldownMs = getCooldownMs();
    snap.localStorage = {
      audio: localStorage.getItem(KEYS.audio),
      motion: localStorage.getItem(KEYS.motion),
      fence: localStorage.getItem(KEYS.fence),
      prone: localStorage.getItem(KEYS.prone),
      cooldown: localStorage.getItem(KEYS.cooldown),
      lastTs: localStorage.getItem(KEYS.lastTs),
    };
    dlog('report()', snap);
    return snap;
  }

  return {
    trigger,
    emit: (type) => window.dispatchEvent(new CustomEvent('naptio:alert', { detail:{ type:type || pageType } })),
    destroy: () => window.removeEventListener('naptio:alert', onEvt),
    _debug: { ping: debugPing, report },
  };
}
