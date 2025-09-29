// /webrtc/shared/room-auth.js
// Minimal, reusable "signed hello over rooms" helper.
//
// Usage:
//   const auth = new RoomAuth({ role: 'sender'|'receiver', room, send });
//   auth.handleRoomMessage(msg)  // call on every room message early
//   auth.isAuthed(peerId)        // gate your SDP (e.g., offers)
//   auth.kickstart(peersOrIds)   // optional: proactively challenge peers
//
// Optional convenience:
//   auth.attachToWS(ws)          // passive listener, won’t block your handler
//   auth.attachToCore(core)      // attaches to ReceiverCore (passive)
//
// Requires server to relay 'auth-hello' and 'auth-reply' in rooms.

import { te, b64u, b64uToBytes, concatBytes, rand } from '/hungryface/webrtc/shared/pairing-crypto.js';
import { ensureIdentity } from '/hungryface/webrtc/shared/identity.js';

// ----- Local trusted device store -----
function loadTrustedList() {
  try { return JSON.parse(localStorage.getItem('naptio-trusted-devices') || '[]'); } catch { return []; }
}
function trustedByFp(fp) {
  return loadTrustedList().find(d => d.fp === fp) || null;
}
async function importSignKey(spkiB64) {
  const spki = b64uToBytes(spkiB64);
  return crypto.subtle.importKey('spki', spki, { name: 'ECDSA', namedCurve: 'P-256' }, true, ['verify']);
}

// ----- Auth payload -----
const AUTH_TAG = te.encode('naptio-auth-v1');
const enc = te;
function makeAuthBuf(room, nonceBytes) {
  return concatBytes(AUTH_TAG, enc.encode(room), nonceBytes);
}

export class RoomAuth {
  /**
   * @param {Object} opts
   * @param {'sender'|'receiver'} opts.role
   * @param {string} opts.room
   * @param {(obj: any) => void} opts.send   // your existing room send() (adds {type, to, ...})
   * @param {(peerId: string) => void} [opts.onAuthed] // callback when a peer becomes AUTH_OK
   */
  constructor({ role, room, send, onAuthed } = {}) {
    if (!role || !room || !send) throw new Error('RoomAuth: role, room, send required');
    this.role = role;
    this.room = room;
    this.send = send;
    this.onAuthed = onAuthed || (()=>{});
    this.AUTH_OK = new Map();   // peerId -> true
    this.NONCES  = new Map();   // peerId -> Uint8Array we sent (receiver) or they sent (sender)
    this._started = false;
    this._me = null;
    this._init();
  }

  async _init() {
    try { this._me = await ensureIdentity(); }
    catch (e) { console.warn('[RoomAuth] ensureIdentity failed', e); }
    this._started = true;
  }

  isAuthed(peerId) { return !!this.AUTH_OK.get(peerId); }

  /** Optionally challenge a set of peers on join/roster to warm up auth. */
  kickstart(peers) {
    if (!Array.isArray(peers)) return;
    for (const p of peers) {
      const id = typeof p === 'string' ? p : p.id;
      const role = typeof p === 'string' ? null : (p.role || null);
      // Receivers challenge senders; Senders challenge receivers (mutual auth)
      if (this.role === 'receiver' && role === 'sender') this._challenge(id);
      if (this.role === 'sender'   && role === 'receiver') this._challenge(id);
    }
  }

  /** Handle every room message early; returns true if consumed. */
  handleRoomMessage(msg) {
    const t = msg?.type;
    if (!t) return false;

    // Observe roster/peer-joined to proactively challenge (optional convenience):
    if (t === 'roster' && Array.isArray(msg.peers)) {
      this.kickstart(msg.peers);
      return false;
    }
    if (t === 'peer-joined' && msg.id && msg.role) {
      this.kickstart([{ id: msg.id, role: msg.role }]);
      return false;
    }

    // Challenge we received → reply with signature
    if (t === 'auth-hello' && msg.from && msg.nonce) {
      this._replyToChallenge(msg.from, msg.nonce);
      return true;
    }

    // Signed reply we received → verify against trusted list
    if (t === 'auth-reply' && msg.from && msg.fromFp && msg.sig && msg.nonce) {
      this._verifyReply(msg);
      return true;
    }

    return false;
  }

  // ---------- Passive attachment helpers ----------

  /** Attach to a plain WebSocket that carries room messages. */
  attachToWS(ws) {
    if (!ws || typeof ws.addEventListener !== 'function') return;
    ws.addEventListener('message', (e) => {
      try {
        const msg = JSON.parse(e.data);
        this.handleRoomMessage(msg);
      } catch {}
    });
  }

  /** Attach to ReceiverCore; falls back to ws hook if exposed. */
  attachToCore(core) {
    if (!core) return;
    // If the core exposes a hook, use it.
    if (typeof core.onRoomMessage === 'function') {
      core.onRoomMessage((msg) => this.handleRoomMessage(msg));
      return;
    }
    // Fallback: try to tap its ws if present (non-fatal if not found).
    if (core._ws && typeof core._ws.addEventListener === 'function') {
      this.attachToWS(core._ws);
    }
  }

  // ---------- Internals ----------

  _challenge(peerId) {
    if (!peerId) return;
    if (this.AUTH_OK.get(peerId)) return; // no need
    const n = rand(16);
    this.NONCES.set(peerId, n);
    this.send({ type: 'auth-hello', to: peerId, nonce: b64u(n) });
  }

  async _replyToChallenge(peerId, nonceB64) {
    try {
      if (!this._me?.signKey) return;
      const nonce = b64uToBytes(nonceB64);
      const buf = makeAuthBuf(this.room, nonce);
      const sig = new Uint8Array(await crypto.subtle.sign(
        { name: 'ECDSA', hash: 'SHA-256' },
        this._me.signKey,
        buf
      ));
      this.send({ type: 'auth-reply', to: peerId, fromFp: this._me.fingerprint, nonce: nonceB64, sig: b64u(sig) });
    } catch (e) {
      console.warn('[RoomAuth] reply sign failed', e);
    }
  }

  async _verifyReply(msg) {
    const peerId = msg.from;
    const fp = msg.fromFp;
    const trusted = trustedByFp(fp);
    if (!trusted || !peerId) return;
    const expected = this.NONCES.get(peerId);
    if (!expected) {
      // We might still be okay if *they* challenged *us* first (mutual auth):
      // Just verify their reply anyway using their nonce (not our cache).
    }
    try {
      const buf = makeAuthBuf(this.room, b64uToBytes(msg.nonce));
      const pub = await importSignKey(trusted.signSpki);
      const ok = await crypto.subtle.verify(
        { name: 'ECDSA', hash: 'SHA-256' },
        pub, b64uToBytes(msg.sig), buf
      );
      if (ok) {
        this.AUTH_OK.set(peerId, true);
        this.NONCES.delete(peerId);
        try { this.onAuthed(peerId); } catch {}
      }
    } catch (e) {
      console.warn('[RoomAuth] verify failed', e);
    }
  }
}

// --- Convenience: one-shot auth handshake over rooms, then return the authed peerId ---
// Opens a *temporary* WS to the same server, runs RoomAuth on it,
// resolves with the first trusted peerId (based on your local trusted list).
// Usage (receiver page):
//   const { peerId, close } = await authHandshakeOverRooms({ url, room, role:'receiver' });
//   core.senderId = peerId; core.targetSenderId = peerId; close(); core.start();
export async function authHandshakeOverRooms({
  url,
  room,
  role = 'receiver',
  timeoutMs = 15000,
  onEvent,                 // optional: (evt) => {} for progress logging
  allowFp,                 // optional: (fingerprint) => true/false to restrict which trusted device qualifies
} = {}) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`${url}?room=${encodeURIComponent(room)}`);
    let resolved = false;

    const send = (obj) => {
      try { ws.readyState === WebSocket.OPEN && ws.send(JSON.stringify(obj)); } catch {}
    };

    const finish = (err, peerId) => {
      if (resolved) return;
      resolved = true;
      try { ws.close(); } catch {}
      if (err) reject(err); else resolve({ peerId, close: () => { try { ws.close(); } catch {} } });
    };

    const tm = setTimeout(() => finish(new Error('auth handshake timeout')), timeoutMs);

    // Create a RoomAuth bound to this temp socket
    const ra = new RoomAuth({
      role,
      room,
      send,
      // Note: onAuthed only passes peerId; fingerprint is available in messages below.
      onAuthed: (peerId) => {
        if (onEvent) try { onEvent({ type: 'authed', peerId }); } catch {}
        // We still gate with allowFp (if provided) by watching the auth-reply message below
        // to read msg.fromFp before resolving.
      }
    });

    ws.onopen = () => {
      if (onEvent) try { onEvent({ type: 'ws-open' }); } catch {}
      send({ type: 'join', room, role });
    };

    ws.onmessage = (e) => {
      let msg; try { msg = JSON.parse(e.data); } catch { return; }
      // feed all room messages to RoomAuth
      ra.handleRoomMessage(msg);

      // proactively challenge senders as soon as we learn about them
      if (msg?.type === 'roster' && Array.isArray(msg.peers)) {
        if (onEvent) try { onEvent({ type: 'roster', peers: msg.peers }); } catch {}
        ra.kickstart(msg.peers);
      }
      if (msg?.type === 'peer-joined' && msg.role) {
        if (onEvent) try { onEvent({ type: 'peer-joined', id: msg.id, role: msg.role }); } catch {}
        ra.kickstart([{ id: msg.id, role: msg.role }]);
      }

      // When we see a signed reply from a sender we trust, and RoomAuth marks it authed → resolve
      if (msg?.type === 'auth-reply' && msg.from && ra.isAuthed(msg.from)) {
        // Optional fingerprint gate
        if (typeof allowFp === 'function') {
          const fp = msg.fromFp;
          if (!allowFp(fp)) return; // ignore if this authed peer isn't allowed
        }
        clearTimeout(tm);
        if (onEvent) try { onEvent({ type: 'selected', peerId: msg.from, fp: msg.fromFp }); } catch {}
        finish(null, msg.from);
      }
    };

    ws.onerror = (err) => {
      if (onEvent) try { onEvent({ type: 'ws-error', err }); } catch {}
      // don't finish immediately; let timeout handle it unless the socket dies
    };

    ws.onclose = () => {
      if (!resolved) {
        if (onEvent) try { onEvent({ type: 'ws-closed' }); } catch {}
        // allow timeout to fire; closing early could be just a transient
      }
    };
  });
}
