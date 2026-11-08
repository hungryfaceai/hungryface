// webrtc/receiver/shared/alerts/store.js
// Centralized IndexedDB store for all alerts (Audio, Prone, Motion, Fence)

const DB = 'naptioAlerts';
const STORE = 'alerts';
// Cross-tab notifications
const _bc = 'BroadcastChannel' in self ? new BroadcastChannel('alerts-bc') : null;
// Identifies this tab, so channel listeners in the same tab can ignore its own messages
// (another BroadcastChannel object in the same tab DOES receive them).
export const TAB_ID = Math.random().toString(36).slice(2) + Date.now().toString(36);

// Broadcasts a change to this tab (document event 'alerts:changed') and to other tabs (BroadcastChannel).
function _notify(action, payload = {}) {
  try {
    // Local (same-tab) listeners
    document.dispatchEvent(
      new CustomEvent('alerts:changed', {
        detail: { action, ...payload },
      })
    );
    // Cross-tab listeners
    _bc?.postMessage({ type: 'changed', action, ...payload, tab: TAB_ID });
  } catch {}
}

// Relay changes made in OTHER tabs as a local 'alerts:changed' event, so the badge and
// drawer on every open page stay current (BroadcastChannel never echoes to the sender).
// fromOtherTab lets listeners that also watch the channel themselves skip the duplicate.
if (_bc) {
  _bc.addEventListener('message', (ev) => {
    const d = ev?.data;
    if (d?.type !== 'changed' || d.tab === TAB_ID) return;
    try {
      const { type, ...rest } = d;
      document.dispatchEvent(new CustomEvent('alerts:changed', { detail: { ...rest, fromOtherTab: true } }));
    } catch {}
  });
}

// Open (or create) the DB + object store
let _db = null,
  _dbP = null; // one shared open (parallel first calls must not open twice)
// Opens the IndexedDB once (v1: autoincrement 'id', indexes on type and startAt).
export function openDB() {
  if (_db) return Promise.resolve(_db);
  if (_dbP) return _dbP;
  return (_dbP = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const st = db.createObjectStore(STORE, { keyPath: 'id', autoIncrement: true });
        st.createIndex('type', 'type', { unique: false });
        st.createIndex('startAt', 'startAt', { unique: false });
      }
    };
    req.onsuccess = () => {
      _db = req.result;
      _db.onversionchange = () => {
        try {
          _db.close();
        } catch {}
        _db = null;
        _dbP = null;
      };
      // The browser can close the connection itself (storage cleared, low disk): open a new one next time.
      _db.onclose = () => {
        _db = null;
        _dbP = null;
      };
      resolve(_db);
    };
    req.onerror = () => {
      _dbP = null;
      reject(req.error);
    };
  }));
}

/** Create a new alert episode. Returns the new id. */
export async function addAlert(record) {
  const db = await openDB();
  const rec = {
    // fields: id (auto), type, startAt, endAt, avgScore, message
    type: record?.type || '',
    startAt: record?.startAt || new Date().toISOString(),
    endAt: record?.endAt || null,
    avgScore: Number(record?.avgScore ?? 0),
    message: record?.message || '',
  };
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    const st = tx.objectStore(STORE);
    tx.onerror = () => reject(tx.error);
    // An abort (e.g. storage full at commit) fires only "abort": without this the promise would never settle.
    tx.onabort = () => reject(tx.error || new Error('alert not saved (transaction aborted)'));
    const addReq = st.add(rec);
    addReq.onerror = () => reject(addReq.error);
    // Resolve once the transaction has committed, so callers never hold the id of a record that wasn't saved.
    tx.oncomplete = () => {
      const id = addReq.result;
      _notify('add', { id, record: { id, ...rec } });
      resolve(id);
    };
  });
}

/** Patch an existing alert episode by id. Returns true if updated. */
export async function updateAlert(id, patch) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    const st = tx.objectStore(STORE);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error || new Error('alert not updated (transaction aborted)'));
    let result = false,
      next = null;
    tx.oncomplete = () => {
      if (result) _notify('update', { id, record: next });
      resolve(result);
    };

    const getReq = st.get(id);
    getReq.onerror = () => reject(getReq.error);
    getReq.onsuccess = () => {
      const cur = getReq.result;
      if (!cur) return; // nothing to update: resolves false on complete
      next = { ...cur, ...patch };
      const putReq = st.put(next);
      putReq.onerror = () => reject(putReq.error);
      putReq.onsuccess = () => (result = true);
    };
  });
}

/** Read all alerts (newest first by startAt). */
export async function getAllAlerts() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const st = tx.objectStore(STORE);
    const req = st.getAll();
    req.onerror = () => reject(req.error);
    tx.onabort = () => reject(tx.error || new Error('alerts not read (transaction aborted)'));
    req.onsuccess = () => {
      const rows = (req.result || []).sort((a, b) => +new Date(b.startAt || 0) - +new Date(a.startAt || 0));
      resolve(rows);
    };
  });
}

/** Clear everything. */
export async function clearAllAlerts() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    const st = tx.objectStore(STORE);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error || new Error('alerts not cleared (transaction aborted)'));
    tx.oncomplete = () => {
      _notify('clear');
      resolve();
    };
    st.clear();
  });
}

// ===== Export helpers (JSON/CSV) =====

/** Build a CSV string from alert rows. */
// Fields are always quoted with doubled inner quotes; durationSec is blank if the alert is still open.
function _rowsToCSV(rows) {
  // CSV cell: quoted; a value starting with = + - @ (or tab/CR) gets a leading ' so spreadsheets show it as text
  // instead of running it as a formula.
  const esc = (s) => {
    let v = String(s ?? '');
    if (/^[=+\-@\t\r]/.test(v)) v = "'" + v;
    return `"${v.replace(/"/g, '""')}"`;
  };
  const header = ['startAt', 'endAt', 'durationSec', 'avgScore', 'type', 'message'];
  const lines = [header.join(',')];

  for (const r of rows) {
    const t0 = r.startAt ? +new Date(r.startAt) : NaN;
    const t1 = r.endAt ? +new Date(r.endAt) : NaN;
    const dur = Number.isFinite(t0) && Number.isFinite(t1) ? Math.round((t1 - t0) / 1000) : '';
    lines.push(
      [esc(r.startAt), esc(r.endAt), dur, Number(r.avgScore || 0), esc(r.type || ''), esc(r.message || '')].join(',')
    );
  }
  return lines.join('\n');
}

/**
 * Returns a pretty-printed JSON payload string.
 * If `rows` not provided, fetches all via getAllAlerts().
 */
export async function exportAlertsJSON(rows) {
  const data = rows ?? (await getAllAlerts());
  return JSON.stringify({ version: 1, exportedAt: new Date().toISOString(), rows: data }, null, 2);
}

/**
 * Returns a CSV string of alerts.
 * If `rows` not provided, fetches all via getAllAlerts().
 */
export async function exportAlertsCSV(rows) {
  const data = rows ?? (await getAllAlerts());
  return _rowsToCSV(data);
}

export const AlertTypes = Object.freeze({
  Audio: 'Audio',
  Prone: 'Prone',
  Motion: 'Motion',
  Fence: 'Fence',
});
