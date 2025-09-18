// webrtc/receiver/shared/alerts/store.js
// Centralized IndexedDB store for all alerts (Audio, Prone, Motion, Fence)

const DB = 'naptioAlerts';
const STORE = 'alerts';

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
    req.onsuccess = () => { _db = req.result; resolve(_db); };
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
      document.dispatchEvent(new CustomEvent('alerts:changed', {
        detail: { action: 'add', id, record: { id, ...rec } }
      }));
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
        document.dispatchEvent(new CustomEvent('alerts:changed', {
          detail: { action: 'update', id, record: next }
        }));
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
      document.dispatchEvent(new CustomEvent('alerts:changed', { detail: { action: 'clear' } }));
      resolve();
    };
    st.clear();
  });
}

export const AlertTypes = Object.freeze({
  Audio:  'Audio',
  Prone:  'Prone',
  Motion: 'Motion',
  Fence:  'Fence',
});
