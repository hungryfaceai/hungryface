// /webrtc/shared/psk/psk-verify.js
// Minimal 1-RTT PSK verification over a simple relay bus.
// Bus API expected:
//   - off = signal.on('message', cb)  // cb(msg) where msg.type is 'psk-verify-*'
//   - await signal.relay(toFp, payload, { room })

import { getPsk } from '/hungryface/webrtc/shared/psk/psk-ws-shim.js';

const te = new TextEncoder();

// ---------- helpers ----------
function b64uToBytes(b64u) {
  if (!b64u) return new Uint8Array(0);
  let b64 = b64u.split('-').join('+').split('_').join('/');
  const pad = (4 - (b64.length % 4)) % 4;
  if (pad) b64 += '===='.slice(0, pad);
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
function bytesToB64u(buf) {
  let s = '';
  for (let i = 0; i < buf.length; i++) s += String.fromCharCode(buf[i]);
  let b64 = btoa(s).split('+').join('-').split('/').join('_');
  while (b64.endsWith('=')) b64 = b64.slice(0, -1);
  return b64;
}
function randomB64u(n = 16) {
  const u8 = new Uint8Array(n);
  crypto.getRandomValues(u8);
  return bytesToB64u(u8);
}
function ctEq(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  let v = 0; for (let i = 0; i < a.length; i++) v |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return v === 0;
}

// Canonical/stable stringify (keys sorted) to avoid platform key-order quirks
function stableStringify(v) {
  if (v === null || typeof v !== 'object') return JSON.stringify(v);
  if (Array.isArray(v)) return '[' + v.map(stableStringify).join(',') + ']';
  const keys = Object.keys(v).sort();
  return '{' + keys.map(k => JSON.stringify(k) + ':' + stableStringify(v[k])).join(',') + '}';
}

async function hmacB64u(pskB64u, obj) {
  const key = await crypto.subtle.importKey('raw', b64uToBytes(pskB64u), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const data = te.encode(stableStringify(obj));
  const sig = await crypto.subtle.sign('HMAC', key, data);
  return bytesToB64u(new Uint8Array(sig));
}

function waitFor(signal, { type, from }, timeoutMs = 7000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { off(); reject(new Error('TIMEOUT_' + type)); }, timeoutMs);
    const off = signal.on('message', (m) => {
      if (m && m.type === type && (!from || m.from === from)) {
        clearTimeout(timer); off(); resolve(m);
      }
    });
  });
}

// ---------- responder (host side) ----------
export function enablePskVerifyResponder(signal, { room, onStatus }) {
  const off = signal.on('message', async (msg) => {
    try {
      if (!msg || msg.type !== 'psk-verify-req') return;

      // Be room-tolerant: prefer the room in the message; else fall back
      const wantRoom = (msg.room || room || '').trim();
      let psk = getPsk(wantRoom)?.tokenB64u;
      if (!psk && room && wantRoom !== room) {
        // Fallback to the responder's current room if caller passed one
        psk = getPsk(room)?.tokenB64u;
      }
      if (!psk) return; // can't proceed

      onStatus && onStatus('verifying', { dir: 'inbound', from: msg.from });

      // Validate request MAC quickly (drops wrong-PSK or wrong-room)
      const expectReq = await hmacB64u(psk, { kind: 'req', room: wantRoom, nonceA: msg.nonceA });
      if (!ctEq(expectReq, msg.mac)) throw new Error('BAD_REQ_MAC');

      const nonceB = randomB64u(16);
      const macA = await hmacB64u(psk, { kind: 'resp', room: wantRoom, nonceA: msg.nonceA, nonceB });
      await signal.relay(msg.from, { type: 'psk-verify-resp', nonceB, macA, room: wantRoom }, { room: wantRoom });

      const ack = await waitFor(signal, { type: 'psk-verify-ack', from: msg.from }, 7000);
      const expectAck = await hmacB64u(psk, { kind: 'ack', room: wantRoom, nonceB });
      if (!ctEq(expectAck, ack.macB)) throw new Error('BAD_ACK_MAC');

      onStatus && onStatus('paired', { peerFp: msg.from });
    } catch (e) {
      onStatus && onStatus('failed', { reason: e && e.message || String(e) });
    }
  });
  return off;
}

// ---------- initiator (guest side) ----------
async function _runOnce(signal, { room, peerFp, onStatus }, timeoutMs) {
  const psk = getPsk(room)?.tokenB64u;
  if (!psk) throw new Error('NO_PSK');

  onStatus && onStatus('verifying', { dir: 'outbound', to: peerFp });

  const nonceA = randomB64u(16);
  const req = {
    type: 'psk-verify-req',
    room,
    nonceA,
    mac: await hmacB64u(psk, { kind: 'req', room, nonceA })
  };
  await signal.relay(peerFp, req, { room });

  const resp = await waitFor(signal, { type: 'psk-verify-resp', from: peerFp }, timeoutMs);
  const expectMacA = await hmacB64u(psk, { kind: 'resp', room, nonceA, nonceB: resp.nonceB });
  if (!ctEq(expectMacA, resp.macA)) throw new Error('BAD_MAC_A');

  const ack = { type: 'psk-verify-ack', macB: await hmacB64u(psk, { kind: 'ack', room, nonceB: resp.nonceB }), room };
  await signal.relay(peerFp, ack, { room });

  onStatus && onStatus('paired', { peerFp });
  return { ok: true };
}

export async function initiatePskVerify(signal, opts, timeoutMs = 7000) {
  try {
    return await _runOnce(signal, opts, timeoutMs);
  } catch (e) {
    // Single retry on timeout (host might still be arming responder)
    if (String(e && e.message || '').startsWith('TIMEOUT_')) {
      await new Promise(r => setTimeout(r, 600));
      return await _runOnce(signal, opts, timeoutMs);
    }
    throw e;
  }
}
