// Centralized IndexedDB store for all alerts (Audio, Prone, Motion, Fence)
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
const rows = (req.result || []).sort((a,b) => +new Date(b.startAt||0) - +new Date(a.startAt||0));
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


export const AlertTypes = Object.freeze({
Audio: 'Audio',
Prone: 'Prone',
Motion: 'Motion',
Fence: 'Fence',
});
