// /webrtc/receiver/shared/alerts/notify-sound.js
// Reproduces dashboard "triple pip" alert sound + respects per-type switches + global cooldown.
//
// Settings (from Receiver Settings page):
//   naptio:notify:audio|motion|fence|prone  => "on" / "off"  (default: on)
//   naptio:notifyCooldownMin                => integer minutes (default: 2)
//
// Also respects (if present) the dashboard master toggle:
//   alerts_sounds_enabled => "1" or "0" (default: allow)

const KEYS = {
  audio: 'naptio:notify:audio',
  motion: 'naptio:notify:motion',
  fence: 'naptio:notify:fence',
  prone: 'naptio:notify:prone',
  cooldown: 'naptio:notifyCooldownMin',
  lastTs: 'naptio:notify:lastTs',
  tsPing: 'naptio:notify:ts',
  dashGate: 'alerts_sounds_enabled',
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
function dashboardGateAllows() {
  const v = localStorage.getItem(KEYS.dashGate);
  return v == null || v === '1'; // default allow if unset
}
function getCooldownMs() {
  const n = parseInt(localStorage.getItem(KEYS.cooldown), 10);
  const minutes = Number.isFinite(n) ? n : 2;
  return Math.max(0, minutes) * 60_000;
}

let audioCtx = null;
let lastBeepWallNow = 0; // dashboard-like 1s rate-limit (in addition to minutes cooldown)

function ensureCtx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}
function unlockOnFirstGesture() {
  const once = async () => {
    const ac = ensureCtx();
    try { await ac.resume(); } catch {}
  };
  ['pointerdown','touchend','keydown','click'].forEach(ev =>
    window.addEventListener(ev, once, { once:true, passive:true })
  );
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && audioCtx?.state === 'suspended') {
    audioCtx.resume().catch(()=>{});
  }
});

// ----- Dashboard-like triple pip (1.7k→2.6k chirps) -----
async function playDashboardPips() {
  const ac = ensureCtx();
  if (ac.state === 'suspended') { try { await ac.resume(); } catch {} }
  if (ac.state !== 'running') return;

  const nowPerf = performance.now();
  if (nowPerf - lastBeepWallNow < 1000) return; // dashboard 1s rate-limit
  lastBeepWallNow = nowPerf;

  const t0 = ac.currentTime;
  const pip = (start, dur = 0.12) => {
    const o = ac.createOscillator();
    const g = ac.createGain();
    o.type = 'sine';
    // 1.7k → 2.6k chirp
    o.frequency.setValueAtTime(1700, start);
    o.frequency.exponentialRampToValueAtTime(2600, start + dur);
    // envelope
    g.gain.setValueAtTime(0.0001, start);
    g.gain.exponentialRampToValueAtTime(0.09, start + 0.012);
    g.gain.exponentialRampToValueAtTime(0.0001, start + dur);
    o.connect(g).connect(ac.destination);
    o.start(start);
    o.stop(start + dur + 0.02);
  };

  // three pips spaced ~0.18s, like the dashboard
  pip(t0);
  pip(t0 + 0.18);
  pip(t0 + 0.36);
}

function inMinutesCooldown(nowMs = Date.now()) {
  const last = parseInt(localStorage.getItem(KEYS.lastTs), 10) || 0;
  return (nowMs - last) < getCooldownMs();
}
function stampPlayed(nowMs = Date.now()) {
  try { localStorage.setItem(KEYS.lastTs, String(nowMs)); } catch {}
}

export function installAlertSound(opts = {}) {
  const pageType = (opts.type || inferTypeFromPath());

  ensureCtx();
  unlockOnFirstGesture();

  // optional cross-tab nudge (we just observe; values read at trigger time)
  window.addEventListener('storage', (e) => {
    if (e.key === KEYS.tsPing) { /* noop */ }
  });

  async function trigger(type = pageType) {
    // Per-type switch + dashboard master gate + per-minute cooldown
    if (!dashboardGateAllows()) return false;
    if (!['audio','motion','fence','prone'].includes(type)) return false;
    if (!settingIsOn(KEYS[type], 'on')) return false;

    const now = Date.now();
    if (inMinutesCooldown(now)) return false;

    await playDashboardPips(); // reproduces dashboard sound
    stampPlayed(now);
    return true;
  }

  // Optional event API: window.dispatchEvent(new CustomEvent('naptio:alert', { detail:{ type:'audio' }}))
  const onEvt = (e) => trigger((e?.detail?.type) || pageType);
  window.addEventListener('naptio:alert', onEvt);

  return {
    trigger,                              // call when your page raises a new alert
    emit: (type) => window.dispatchEvent(new CustomEvent('naptio:alert', { detail:{ type:type || pageType } })),
    destroy: () => window.removeEventListener('naptio:alert', onEvt),
  };
}
