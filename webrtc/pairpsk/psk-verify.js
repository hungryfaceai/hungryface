// /hungryface/webrtc/shared/psk/psk-verify.js
// Minimal 1-RTT PSK verification using your SecureSignal transport.
// Assumes:
//   - signal.relay(toFp, payload, {room})
//   - const off = signal.on('message', (m) => { ... });  // returns unsubscribe
//
// Exports:
//   - listenForPskVerify(signal, { room, onStatus?, onPaired?, timeoutMs? }) -> { dispose() }
//   - initiatePskVerify(signal, { room, peerFp, onStatus?, onPaired?, timeoutMs? }) -> Promise<{ok:true}>
//
// Both sides should call listenForPskVerify(). The initiating side then calls initiatePskVerify().

import { getPsk } from '/hungryface/webrtc/shared/psk/psk-ws-shim.js';

const te = new TextEncoder();

function b64uToBytes(b64u) {
  if (!b64u) return new Uint8Array(0);
  const b64 = b64u.replace(/-/g, '+').replace(/_/g, '/') + '==='.slice((b64u.length + 3) % 4);
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
function bytesToB64u(buf) {
  let s = '';
  for (let i = 0; i < buf.length; i++) s += String.fromCharCode(buf[i]);
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/,'');
}
function randomB64u(n = 16) {
  const u8 = new Uint8Array(n);
  crypto.getRandomValues(u8);
  return bytesToB64u(u8);
}
function ctEq(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  let v = 0;
  for (let i = 0; i < a.length; i++) v |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return v === 0;
}
async function hmacB64u(pskB64u, obj) {
  const key = await crypto.subtle.importKey('raw', b64uToBytes(pskB64u), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const bytes = te.encode(typeof obj === 'string' ? obj : JSON.stringify(obj));
  const sig = await crypto.subtle.sign('HMAC', key, bytes);
  return bytesToB64u(new Uint8Array(sig));
}

function waitFor(signal, { type, from }, timeoutMs = 4000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { off(); reject(new Error('TIMEOUT_' + type)); }, timeoutMs);
    const off = signal.on('message', (m) => {
      if (m && m.type === type && (!from || m.from === from)) {
        clearTimeout(timer); off(); resolve(m);
      }
    });
  });
}

// Responder role: answer any incoming psk-verify-req, then wait for ack, call onPaired(true)
export function listenForPskVerify(signal, { room, onStatus, onPaired, timeoutMs = 4000 }) {
  const off = signal.on('message', async (m) => {
    try {
      if (!m || m.type !== 'psk-verify-req') return;
      const psk = getPsk(room)?.tokenB64u; if (!psk) return;

      onStatus && onStatus('verifying');
      // Optional: validate request MAC early
      const expectReq = await hmacB64u(psk, { kind: 'req', room, nonceA: m.nonceA });
      if (!ctEq(expectReq, m.mac)) throw new Error('BAD_REQ_MAC');

      const nonceB = randomB64u(16);
      const macA = await hmacB64u(psk, { kind: 'resp', room, nonceA: m.nonceA, nonceB });
      await signal.relay(m.from, { type:'psk-verify-resp', nonceB, macA }, { room });

      // Wait for ack
      const ack = await waitFor(signal, { type:'psk-verify-ack', from: m.from }, timeoutMs);
      const expectAck = await hmacB64u(psk, { kind:'ack', room, nonceB });
      if (!ctEq(expectAck, ack.macB)) throw new Error('BAD_ACK_MAC');

      onPaired && onPaired({ ok:true, peerFp: m.from });
    } catch (e) {
      onStatus && onStatus('failed:' + (e && e.message || e));
    }
  });
  return { dispose: off };
}

// Initiator role: start the 1-RTT handshake against a known peer fingerprint
export async function initiatePskVerify(signal, { room, peerFp, onStatus, onPaired, timeoutMs = 4000 }) {
  const psk = getPsk(room)?.tokenB64u;
  if (!psk) throw new Error('NO_PSK');

  onStatus && onStatus('verifying');
  const nonceA = randomB64u(16);
  const req = {
    type: 'psk-verify-req',
    room, nonceA,
    mac: await hmacB64u(psk, { kind: 'req', room, nonceA })
  };
  await signal.relay(peerFp, req, { room });

  const resp = await waitFor(signal, { type:'psk-verify-resp', from: peerFp }, timeoutMs);
  const expectMacA = await hmacB64u(psk, { kind:'resp', room, nonceA, nonceB: resp.nonceB });
  if (!ctEq(expectMacA, resp.macA)) throw new Error('BAD_MAC_A');

  const ack = { type:'psk-verify-ack', macB: await hmacB64u(psk, { kind:'ack', room, nonceB: resp.nonceB }) };
  await signal.relay(peerFp, ack, { room });

  onPaired && onPaired({ ok:true, peerFp });
  return { ok:true };
}
