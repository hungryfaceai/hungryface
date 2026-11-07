// bootstrap.js — shared PSK + room setup for the secure receiver pages.
//
// Replaces the ~85-line block each secure receiver page used to carry. Behaviour matches
// those blocks: require a PSK (else redirect to pairing), install room scoping, pick the
// UI room (?room → lastRoom → bm_receiver_room → env.room → first available), install the
// PSK WebSocket shim keyed by the UI room label, and expose the same window globals.
//
// Usage (top of a page's module script):
//   import { bootstrapSecureReceiver } from '/hungryface/webrtc/shared/psk/bootstrap.js';
//   await bootstrapSecureReceiver({ label: 'receiver-motion' });

import { installPskShim } from '/hungryface/webrtc/shared/psk/psk-ws-shim.js';
import { requirePskOrRedirect, listValidPskRooms } from '/hungryface/webrtc/shared/psk/require-psk.js';
import { installRoomScope } from '/hungryface/webrtc/shared/psk/room-scope.js';

const DEFAULT_ROOM = 'Baby';

function readLS(key) {
  try {
    return (localStorage.getItem(key) || '').trim();
  } catch {
    return '';
  }
}
function persistRoom(label) {
  try {
    localStorage.setItem('bm_receiver_room', label);
  } catch {}
  try {
    localStorage.setItem('naptio:lastRoom', label);
  } catch {}
}
function scope(label) {
  return typeof window.__scopeRoom === 'function' ? window.__scopeRoom(label) : label;
}
function setRoomGlobals(uiRoom, scopedRoom) {
  window.__pskRoom = uiRoom; // backward-compat alias for UI room
  window.__uiRoom = uiRoom; // explicit UI label
  window.__scopedRoom = scopedRoom; // PSK-scoped signaling room ID
  window.__pskScopedRoom = scopedRoom; // alias used by the training page
}

/** Pick the UI room using the same priority chain as Settings. */
export function resolveUiRoom({ qsRoom, env, rooms = listValidPskRooms() }) {
  const lastRoom = readLS('naptio:lastRoom');
  const persisted = readLS('bm_receiver_room');
  const envRoom = (env?.room || '').trim();

  const uniq = new Set(rooms);
  if (envRoom) uniq.add(envRoom);

  const resolved = uniq.size
    ? (qsRoom && uniq.has(qsRoom) && qsRoom) ||
      (lastRoom && uniq.has(lastRoom) && lastRoom) ||
      (persisted && uniq.has(persisted) && persisted) ||
      envRoom ||
      Array.from(uniq)[0]
    : envRoom || qsRoom || DEFAULT_ROOM; // defensive fallback if list is empty

  return (resolved && String(resolved).trim()) || DEFAULT_ROOM;
}

/**
 * Run the secure-receiver bootstrap. Resolves to { env, redirected, uiRoom, scopedRoom }.
 * When redirected is true the browser is already navigating to the pairing page.
 */
export async function bootstrapSecureReceiver({ label = 'receiver', intent = 'viewer' } = {}) {
  const tag = `[PSK][${label}]`;
  const qsRoom = (new URLSearchParams(location.search).get('room') || '').trim();

  // Ensure we have a PSK; if not, redirect to pairing
  const env = await requirePskOrRedirect({
    intent,
    pairRoute: '/hungryface/webrtc/pairpsk/',
    preferRoom: qsRoom,
    fallbackRoom: DEFAULT_ROOM,
  });
  window.__pskEnv = env;

  // Install PSK-based room scoping helper (window.__scopeRoom)
  await installRoomScope(env);

  if (env.redirected) {
    console.log(tag, 'redirecting to Pair Devices…');
    return { env, redirected: true, uiRoom: null, scopedRoom: null };
  }

  const uiRoom = resolveUiRoom({ qsRoom, env });
  const scopedRoom = scope(uiRoom);

  // Persist UI label for future pages (Settings / other receivers)
  persistRoom(uiRoom);

  // Install PSK shim keyed by the UI room label
  window.__pskUndo = installPskShim({ room: uiRoom });
  console.log(tag, 'UI room:', uiRoom, 'scoped signaling room:', scopedRoom);
  setRoomGlobals(uiRoom, scopedRoom);

  // Helper if UI later switches rooms
  window.__installPskForRoom = (roomLabel) => {
    const next = (roomLabel || '').trim() || DEFAULT_ROOM;
    const scoped = scope(next);
    try {
      window.__pskUndo?.();
    } catch {}
    window.__pskUndo = installPskShim({ room: next });
    console.log(tag, 'shim re-installed for room:', next, 'scoped:', scoped);
    setRoomGlobals(next, scoped);
    persistRoom(next);
  };

  return { env, redirected: false, uiRoom, scopedRoom };
}
