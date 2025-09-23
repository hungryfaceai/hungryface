// /webrtc/shared/device-registry.js
const META_KEY = (fp) => `naptio:lastMeta:${fp}`;

export function saveLastMeta(fp, { room, ts = Date.now() } = {}) {
  if (!fp) return;
  try { localStorage.setItem(META_KEY(fp), JSON.stringify({ room: room || '', ts })); } catch {}
}
export function loadLastMeta(fp) {
  if (!fp) return { room: '', ts: 0 };
  try {
    const raw = localStorage.getItem(META_KEY(fp));
    return raw ? JSON.parse(raw) : { room: '', ts: 0 };
  } catch { return { room: '', ts: 0 }; }
}
export function formatFp(fp) {
  if (!fp) return '';
  return fp.length > 8 ? `${fp.slice(0,4)}…${fp.slice(-4)}` : fp;
}
export function annotateTrusted(list) {
  return (list || []).map(d => ({ ...d, meta: loadLastMeta(d.fp) }));
}
