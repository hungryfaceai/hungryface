// adopt-room.js — make a freshly paired room the current room on this device.
//
// Called whenever this device RECEIVES a pairing key from another device (QR scanned with the phone
// camera, a shared link, or the in-app "Scan QR"). Rule: the room you just paired is the room you use next,
// everywhere (Home, Camera, Viewer, Settings, Pair devices) — those screens all read these keys first.
//
// It also removes a throw-away default room this device created for itself moments before
// (opening "Get started" makes a "Baby" key before the user scans the other device's code). That room is
// removed only if ALL of: it is the default name, it was generated on this device (not installed from
// another device), and it is less than 30 minutes old — so a room that might already be shared is never
// touched.

const LAST = 'naptio:lastRoom',
  SENDER = 'bm_sender_room',
  RECEIVER = 'bm_receiver_room';
const DEFAULT_ROOM = 'Baby';
const FRESH_MS = 30 * 60 * 1000;

// Safe localStorage helpers (storage may be blocked in private modes).
function read(k) {
  try {
    return localStorage.getItem(k);
  } catch {
    return null;
  }
}
function parse(s) {
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

/**
 * True if psk:<room> is a key this device made itself and nothing else can depend on: not received from another
 * device, never rotated, and its link never copied or shared from here (sharedAt). Pairing with another device's
 * code may then replace it without asking — the normal first pairing, where both phones made their own "Baby".
 */
export function isUnusedLocalKey(room) {
  const rec = parse(read('psk:' + room));
  return !!rec && !rec.installedFrom && !rec.rotatedAt && !rec.sharedAt;
}
/** Record that this room's link was copied/shared from this device (its key may now be in use elsewhere). */
export function markShared(room) {
  try {
    const rec = parse(read('psk:' + room));
    if (rec && !rec.sharedAt) localStorage.setItem('psk:' + room, JSON.stringify({ ...rec, sharedAt: Date.now() }));
  } catch {}
}

/** True if psk:<room> was generated on this device a moment ago, never installed from elsewhere and never shared. */
function isThrowAwayDefault(room) {
  if (room !== DEFAULT_ROOM) return false;
  if (!isUnusedLocalKey(room)) return false; // received, rotated or shared → keep
  const rec = parse(read('psk:' + room));
  const created = Number(rec.createdAt) || Date.parse(rec.ts || '') || 0;
  return created > 0 && Date.now() - created < FRESH_MS;
}

/**
 * Select `room` as this device's current room (all screens).
 * Returns { room, removedDefault } — removedDefault is true when a throw-away "Baby" was cleaned up.
 */
export function adoptPairedRoom(room) {
  const r = String(room || '').trim();
  if (!r) return { room: '', removedDefault: false };
  let removedDefault = false;
  try {
    if (r !== DEFAULT_ROOM && isThrowAwayDefault(DEFAULT_ROOM)) {
      localStorage.removeItem('psk:' + DEFAULT_ROOM);
      removedDefault = true;
    }
    localStorage.setItem(LAST, r);
    localStorage.setItem(SENDER, r);
    localStorage.setItem(RECEIVER, r);
  } catch {}
  return { room: r, removedDefault };
}
