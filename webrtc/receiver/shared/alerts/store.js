// Centralized IndexedDB store for all alerts (Audio, Prone, Motion, Fence)

const DB_NAME = 'alerts-db';
const DB_VERSION = 1;
const STORE = 'alerts';

export const AlertTypes = Object.freeze({
  Audio: 'Audio',
  Prone: 'Prone',
  Motion: 'Motion',
  Fence: 'Fence',
});

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onerror = () => reject(req.error);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const os = db.createObjectStore(STORE, { keyPath: 'id' });
        os.createIndex('startAt', 'startAt', { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
  });
}

export async function openAlertDB() {
  return openDB();
}

function genId() {
  return 'a' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 8);
}

export async function saveAlertRecord(rec) {
  const db = await openDB();
  const id = rec.id || genId();
  const toSave = { id, ...rec };

  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.onerror = () => reject(tx.error);
    const store = tx.objectStore(STORE);
    const putReq = store.put(toSave);
    putReq.onerror = () => reject(putReq.error);
    putReq.onsuccess = () => {
      document.dispatchEvent(new CustomEvent('alerts:changed', { detail: { action: 'insert', id, record: toSave } }));
      resolve(id);
    };
  });
}

export async function updateAlertById(id, patch) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    const store = tx.objectStore(STORE);
    tx.onerror = () => reject(tx.error);

    const getReq = store.get(id);
    getReq.onerror = () => reject(getReq.error);
    getReq.onsuccess = () => {
      const cur = getReq.result;
      if (!cur) { resolve(false); return; }
      const next = { ...cur, ...patch };
      const putReq = store.put(next);
      putReq.onerror = () => reject(putReq.error);
      putReq.onsuccess = () => {
        document.dispatchEvent(new CustomEvent('alerts:changed', { detail: { action: 'update', id, record: next } }));
        resolve(true);
      };
    };
  });
}

export async function getAllAlerts() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const store = tx.objectStore(STORE);
    const req = store.getAll();
    req.onerror = () => reject(req.error);
    req.onsuccess = () => {
      const rows = (req.result || []).sort((a, b) => +new Date(b.startAt || 0) - +new Date(a.startAt || 0));
      resolve(rows);
    };
  });
}

export async function clearAllAlerts() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    const store = tx.objectStore(STORE);
    tx.onerror = () => reject(tx.error);
    tx.oncomplete = () => {
      document.dispatchEvent(new CustomEvent('alerts:changed', { detail: { action: 'clear' } }));
      resolve();
    };
    store.clear();
  });
}
