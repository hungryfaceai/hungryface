// /hungryface/shared/analytics.js
// Anonymous analytics with active-time heartbeats + single-tab reporting.
// - Auto-detects ingest endpoint (localhost vs Render) unless overridden
// - Heartbeats only from one "leader" tab; custom events can be sent by any tab
// - Option: persistId=false to avoid a stored device ID until consent

export function installAnalytics(opts = {}) {
  const {
    // You can still override these:
    base,                     // e.g. 'https://signaling-server-xxx.onrender.com/a'
    endpoint,                 // full ingest URL; if set, wins over base
    app = 'naptio',
    feature = inferFeature(),
    intervalMs = 15000,
    sampleRate = 1,           // e.g., 0.5 = 50%
    persistId = true,         // false until user consents
    installIdKey = 'naptio:installId',
    onError = () => {},
    eventsFromLeaderOnly = false, // set true if you also want events gated to leader
  } = opts;

  // ---------- Resolve endpoint ----------
  const computedBase =
    base ??
    // Allow a global override if you ever need it:
    (window.__NAPTIO_ANALYTICS_BASE || window.__ANALYTICS_BASE) ??
    (isLocalHost(location.hostname)
      ? 'http://localhost:3000/a'                       // dev
      : 'https://signaling-server-f5gu.onrender.com/a'  // prod
    );

  const ingestUrl = endpoint ?? `${computedBase}/evt`;

  // Sampling: persist the coin toss per install to keep it stable
  const storage = persistId ? window.localStorage : window.sessionStorage;
  let installId = storage.getItem(installIdKey);
  if (!installId) {
    installId = (crypto.randomUUID && crypto.randomUUID()) || randId();
    if (persistId) storage.setItem(installIdKey, installId);
  }

  const sampleKey = `${installIdKey}::sample`;
  let sampled = storage.getItem(sampleKey);
  if (sampled == null) {
    sampled = Math.random() < sampleRate ? '1' : '0';
    try { storage.setItem(sampleKey, sampled); } catch {}
  }
  if (sampled !== '1') return noopApi();

  // ---------- Session + leader election ----------
  let sessionId = randId();
  let activeMs = 0;
  let hbTimer = null;
  let leaderTimer = null;
  let featureName = feature;
  const leaderKey = 'naptio:analytics:leader';
  const me = randId();
  const LEADER_TTL = Math.max(8000, intervalMs * 1.5); // stale window

  function amLeader() {
    try {
      const now = Date.now();
      const raw = localStorage.getItem(leaderKey);
      const rec = raw ? JSON.parse(raw) : null;
      if (!rec || (now - (rec.ts || 0)) > LEADER_TTL) {
        localStorage.setItem(leaderKey, JSON.stringify({ id: me, ts: now }));
        return true;
      }
      return rec.id === me;
    } catch {
      // If localStorage explodes (Safari ITP), just act as leader to keep data flowing
      return true;
    }
  }

  function renewLeader() {
    try {
      if (!amLeader()) return;
      localStorage.setItem(leaderKey, JSON.stringify({ id: me, ts: Date.now() }));
    } catch {}
  }

  window.addEventListener('storage', (e) => {
    if (e.key === leaderKey) {
      // No action needed; next tick will re-check leadership naturally
    }
  });

  // ---------- Posting ----------
  function post(payload) {
    const body = {
      ts: Date.now(),
      app,
      feature: featureName,
      page: location.pathname,
      installId,
      sessionId,
      activeMs,
      t: payload.t || 'hb',
      ev: payload.ev || null,
      props: payload.props || null
    };

    try {
      // Try sendBeacon for unloads/low-cost pings
      if (payload.beacon && navigator.sendBeacon) {
        const ok = navigator.sendBeacon(ingestUrl, jsonBlob(body));
        return ok ? Promise.resolve() : fetchJson(ingestUrl, body, onError);
      }
      return fetchJson(ingestUrl, body, onError);
    } catch (e) {
      try { onError(e); } catch {}
      return Promise.resolve();
    }
  }

  // Heartbeat loop (leader only)
  function start() {
    stop();
    // Kick a first ping immediately
    if (amLeader()) post({});
    // Accumulate active time only while visible
    const tick = () => {
      if (document.visibilityState === 'visible') activeMs += intervalMs;
      if (amLeader()) post({});
    };
    hbTimer = setInterval(tick, intervalMs);
    // Maintain leader lease
    leaderTimer = setInterval(renewLeader, Math.min(4000, intervalMs));
  }

  function stop() {
    if (hbTimer) clearInterval(hbTimer), hbTimer = null;
    if (leaderTimer) clearInterval(leaderTimer), leaderTimer = null;
  }

  // Send a final beacon on unload
  window.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') {
      if (amLeader()) post({ beacon: true });
    }
  });
  window.addEventListener('pagehide', () => {
    if (amLeader()) post({ beacon: true });
  });

  start();

  // ---------- Public API ----------
  const api = {
    event(ev, props) {
      if (eventsFromLeaderOnly && !amLeader()) return;
      post({ t: 'ev', ev, props });
    },
    newSession() {
      sessionId = randId();
      activeMs = 0;
      if (amLeader()) post({}); // record session boundary
    },
    setFeature(f) { if (f) featureName = String(f); }, // optional at runtime
    uninstall() { stop(); }
  };

  return api;
}

// --------- helpers ---------
function fetchJson(url, body, onError) {
  return fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
    keepalive: true
  }).then(() => {}).catch(e => { try { onError(e); } catch {} });
}

function jsonBlob(obj) {
  return new Blob([JSON.stringify(obj)], { type: 'application/json' });
}

function randId() {
  try { return crypto.randomUUID(); } catch { /* fallthrough */ }
  return Math.random().toString(36).slice(2) + Math.random().toString(36).slice(2);
}

function isLocalHost(h) {
  return h === 'localhost' ||
         h === '127.0.0.1' ||
         h.endsWith('.local') ||
         /^10\./.test(h) ||
         /^192\.168\./.test(h) ||
         /^172\.(1[6-9]|2\d|3[0-1])\./.test(h);
}

function inferFeature() {
  const p = location.pathname.toLowerCase();
  if (p.includes('/receiver')) return 'receiver';
  if (p.includes('/sender'))   return 'sender';
  return 'app';
}

function noopApi() {
  return {
    event() {},
    newSession() {},
    setFeature() {},
    uninstall() {}
  };
}
