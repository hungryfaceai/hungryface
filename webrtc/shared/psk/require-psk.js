// /hungryface/webrtc/shared/psk/require-psk.js
// Consume-only guard: ensure a PSK exists for a room, else redirect to Pair Devices.

import { getPsk } from './psk-ws-shim.js';

export function listValidPskRooms() {
  const out = [];
  try {
    const ls = window.localStorage;
    for (let i = 0; i < ls.length; i++) {
      const k = ls.key(i);
      if (!k || !k.startsWith('psk:')) continue;
      const room = k.slice(4);
      const rec = safeParse(ls.getItem(k));
      const token = rec && rec.tokenB64u;
      if (isValidToken(token)) out.push(room);
    }
  } catch {}
  return out.sort();
}

export async function requirePskOrRedirect({
  intent = 'unknown',                // 'sender' | 'viewer'
  pairRoute = '/hungryface/webrtc/pairpsk/',
  preferRoom = '',                   // e.g., from ?room=
  fallbackRoom = 'Baby',             // your default label
} = {}) {
  const room = chooseRoom(preferRoom, fallbackRoom);
  const rec  = getPsk(room) || {};
  const token = rec.tokenB64u || '';

  if (isValidToken(token)) {
    try { localStorage.setItem('naptio:lastRoom', room); } catch {}
    return { room, tokenB64u: token, redirected: false };
  }

  // If there is exactly one valid PSK across rooms, prefer that room to avoid mismatch.
  const rooms = listValidPskRooms();
  if (rooms.length === 1) {
    const onlyRoom = rooms[0];
    const onlyRec  = getPsk(onlyRoom) || {};
    const onlyTok  = onlyRec.tokenB64u || '';
    if (isValidToken(onlyTok)) {
      try { localStorage.setItem('naptio:lastRoom', onlyRoom); } catch {}
      return { room: onlyRoom, tokenB64u: onlyTok, redirected: false };
    }
  }

  // No valid PSK → redirect to Pair Devices with context
  const url = new URL(pairRoute, location.origin);
  if (room)  url.searchParams.set('room', room);
  if (intent) url.searchParams.set('intent', intent);
  // Optionally: url.searchParams.set('return', location.href);
  location.href = url.toString();
  return { room, tokenB64u: '', redirected: true };
}

// ---------- internal helpers ----------
function chooseRoom(prefer, fallback) {
  const p = (prefer || '').trim();
  if (p) return p;
  try {
    const last = localStorage.getItem('naptio:lastRoom');
    if (last) return last;
  } catch {}
  return fallback;
}

function isValidToken(b64u) {
  if (!b64u || typeof b64u !== 'string') return false;
  try { return b64uByteLen(b64u) >= 16; } catch { return false; }
}

function b64uByteLen(b64u) {
  const clean = String(b64u).replace(/-/g, '+').replace(/_/g, '/');
  const pad = (4 - (clean.length % 4)) % 4;
  return atob(clean + '===='.slice(0, pad)).length;
}

function safeParse(s) { try { return JSON.parse(s); } catch { return null; } }
