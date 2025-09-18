// webrtc/receiver/shared/alerts/store.js
// Centralized IndexedDB store for all alerts (Audio, Prone, Motion, Fence)

const DB = 'naptioAlerts';
const STORE = 'alerts';
// Cross-tab notifications
const _bc = ('BroadcastChannel' in self) ? new BroadcastChannel('alerts-bc') : null;

function _notify(action, payload = {}) {
  try {
    // Local (same-tab) listeners
    document.dispatchEvent(new CustomEvent('alerts:changed', {
      detail: { action, ...payload }
    }));
    // Cross-tab listeners
    _bc?.postMessage({ type: 'changed', action, ...payload });
  } catch {}
}

// Open (or create) the DB + object store
let _db = null;
export function openDB() {
  if (_db) return Promise.resolve(_db);
  return new Promise((resolve, reject) => {
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
      _db.onversionchange = () => _db.close();
      resolve(_db);
    };
    req.onerror   = () => reject(req.error);
  });
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
    message: record?.message || ''
  };
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    const st = tx.objectStore(STORE);
    tx.onerror = () => reject(tx.error);
    const addReq = st.add(rec);
    addReq.onerror = () => reject(addReq.error);
    addReq.onsuccess = () => {
      const id = addReq.result;
      const full = { id, ...rec };
      _notify('add', { id, record: full });
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

    const getReq = st.get(id);
    getReq.onerror = () => reject(getReq.error);
    getReq.onsuccess = () => {
      const cur = getReq.result;
      if (!cur) { resolve(false); return; }
      const next = { ...cur, ...patch };
      const putReq = st.put(next);
      putReq.onerror = () => reject(putReq.error);
      putReq.onsuccess = () => {
        _notify('update', { id, record: next });
        resolve(true);
      };
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
    req.onsuccess = () => {
      const rows = (req.result || [])
        .sort((a, b) => +new Date(b.startAt || 0) - +new Date(a.startAt || 0));
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
    tx.oncomplete = () => {
      _notify('clear');
      resolve();
    };
    st.clear();
  });
}

// ===== Export helpers (JSON/CSV) =====

/** Build a CSV string from alert rows. */
function _rowsToCSV(rows) {
  const esc = (s) => `"${String(s ?? '').replace(/"/g, '""')}"`;
  const header = ['startAt','endAt','durationSec','avgScore','type','message'];
  const lines = [header.join(',')];

  for (const r of rows) {
    const t0  = r.startAt ? +new Date(r.startAt) : NaN;
    const t1  = r.endAt   ? +new Date(r.endAt)   : NaN;
    const dur = (Number.isFinite(t0) && Number.isFinite(t1))
      ? Math.round((t1 - t0) / 1000)
      : '';
    lines.push([
      esc(r.startAt),
      esc(r.endAt),
      dur,
      Number(r.avgScore || 0),
      esc(r.type || ''),
      esc(r.message || '')
    ].join(','));
  }
  return lines.join('\n');
}

/**
 * Returns a pretty-printed JSON payload string.
 * If `rows` not provided, fetches all via getAllAlerts().
 */
export async function exportAlertsJSON(rows) {
  const data = rows ?? await getAllAlerts();
  return JSON.stringify(
    { version: 1, exportedAt: new Date().toISOString(), rows: data },
    null,
    2
  );
}

/**
 * Returns a CSV string of alerts.
 * If `rows` not provided, fetches all via getAllAlerts().
 */
export async function exportAlertsCSV(rows) {
  const data = rows ?? await getAllAlerts();
  return _rowsToCSV(data);
}

export const AlertTypes = Object.freeze({
  Audio:  'Audio',
  Prone:  'Prone',
  Motion: 'Motion',
  Fence:  'Fence',
});
