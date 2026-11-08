// /webrtc/shared/psk-ws-shim.js
// Minimal PSK HMAC signing for signaling — no server change, no ReceiverCore change.
// Installs a global WebSocket proxy that auto-signs outbound JSON messages and
// verifies inbound JSON messages using an HMAC-SHA256 over key fields + a 'ctr'.
//
// Persistence: localStorage key "psk:<room>" with { tokenB64u, createdAt }.
// Token import from URL fragment: #room=Baby&token=BASE64URL (then scrubbed).

const te = new TextEncoder();
const td = new TextDecoder();

// --- small encoding/crypto/storage helpers ---
function b64uToBytes(b64u) {
  const b64 = b64u.replace(/-/g, '+').replace(/_/g, '/');
  const pad = '='.repeat((4 - (b64.length % 4)) % 4);
  const bin = atob(b64 + pad);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
function bytesToB64u(arr) {
  let s = '';
  for (let i = 0; i < arr.length; i++) s += String.fromCharCode(arr[i]);
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}
async function hmac(keyBytes, payloadBytes) {
  const key = await crypto.subtle.importKey('raw', keyBytes, { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const mac = await crypto.subtle.sign('HMAC', key, payloadBytes);
  return new Uint8Array(mac);
}
function storageKey(room) {
  return 'psk:' + String(room || '').trim();
}
function getStored(room) {
  try {
    const x = localStorage.getItem(storageKey(room));
    return x ? JSON.parse(x) : null;
  } catch {
    return null;
  }
}
function setStored(room, obj) {
  try {
    localStorage.setItem(storageKey(room), JSON.stringify(obj));
  } catch {}
}
function parseFragment() {
  const hash = (location.hash || '').replace(/^#/, '');
  if (!hash) return {};
  const qs = new URLSearchParams(hash);
  const room = qs.get('room') || qs.get('r') || null;
  const token = qs.get('token') || qs.get('psk') || qs.get('t') || null;
  return { room, token };
}
function scrubFragment() {
  if (location.hash && location.hash !== '#') {
    history.replaceState(null, '', location.pathname + location.search);
  }
}

// Constant-time byte comparison so MAC checks do not leak timing information.
function timingSafeEq(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  let v = 0;
  for (let i = 0; i < a.length; i++) v |= a[i] ^ b[i];
  return v === 0;
}

// The signed view covers only routing/SDP/ICE fields (not the whole message), so server-added fields do not break MACs.
// Build the compact view we sign/verify (stable field order)
function viewFor(obj, ctr) {
  return {
    t: obj.type || obj.op || '',
    to: obj.to || null,
    //from: obj.from || null,
    sdp: obj.sdp || null,
    cand: obj.candidate?.candidate || null,
    mid: obj.candidate?.sdpMid ?? null,
    idx: obj.candidate?.sdpMLineIndex ?? null,
    ctr: Number(ctr) || 0,
  };
}
const PASSTHRU = new Set(['hello', 'roster', 'peer-joined', 'peer-left', 'keepalive']); // never require psk

// ---- Public helpers ----
/**
 * Make sure a PSK exists for `room`. Imports a token from the URL fragment if present (then scrubs it);
 * a sender with no stored key creates a random 128-bit one, a receiver never does.
 * @returns {Promise<{room: string, tokenB64u: string|null}>} room may differ if the fragment named another one.
 */
export async function ensurePskForRoom(room, role = 'receiver') {
  // 1) import from #fragment if present (and scrub)
  const { room: fragRoom, token } = parseFragment();
  if (token) {
    const r = fragRoom || room || 'Baby';
    if (!/^[A-Za-z0-9\-_]{22,}$/.test(token)) throw new Error('Invalid token in fragment');
    // This room already has a different key on this device (an old link after a rotation, or someone else's
    // link): replacing it would silently disconnect it, so ask first; "No" keeps the working key.
    const existing = getStored(r)?.tokenB64u;
    let unused = false;
    try {
      unused = (await import('/hungryface/webrtc/shared/psk/adopt-room.js')).isUnusedLocalKey(r);
    } catch {}
    const replace =
      !existing ||
      existing === token ||
      unused || // this device's own, never shared key (first pairing): nothing depends on it
      confirm(
        `This device is already paired in room “${r}” with a different key.

` + 'Replace it with the key from this link? Devices still using the old key will stop connecting.'
      );
    if (!replace) {
      scrubFragment();
      return { room: r, tokenB64u: existing };
    }
    setStored(r, { tokenB64u: token, createdAt: Date.now(), installedFrom: 'url' });
    // If fragment specified a different room, keep it; callers may read it back via getPsk()
    if (r !== room) room = r;
    // A key received from another device: that room becomes this device's current room everywhere.
    try {
      (await import('/hungryface/webrtc/shared/psk/adopt-room.js')).adoptPairedRoom(r);
    } catch {}
    scrubFragment();
  }
  // 2) ensure present (sender may auto-create; receiver shouldn’t)
  let st = getStored(room);
  if (!st?.tokenB64u && role === 'sender') {
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    setStored(room, { tokenB64u: bytesToB64u(bytes), createdAt: Date.now() });
    st = getStored(room);
  }
  return { room, tokenB64u: st?.tokenB64u || null };
}

// Read the stored key record for a room (tokenB64u is null if none).
export function getPsk(room) {
  const st = getStored(room);
  return { room, tokenB64u: st?.tokenB64u || null };
}

// Build a pairing link carrying room and token in the URL fragment (fragments are not sent to servers).
export function makeShareUrl({ baseUrl, room }) {
  const { tokenB64u } = getPsk(room);
  const u = new URL(baseUrl, location.origin);
  const h = new URLSearchParams();
  h.set('room', room || 'Baby');
  if (tokenB64u) h.set('token', tokenB64u);
  u.hash = h.toString();
  return u.toString();
}

// ---- Install a global WebSocket proxy for this room ----
/**
 * Replace window.WebSocket with a wrapper that signs outbound and verifies inbound signalling JSON using the
 * room's PSK (HMAC-SHA256 + per-sender counter for replay protection). With no stored key it passes data through.
 * @returns {function(): void} undo function restoring the previous WebSocket.
 */
export function installPskShim({ room }) {
  const st = getStored(room);
  const tokenB64u = st?.tokenB64u || null;
  const tokenBytes = tokenB64u ? b64uToBytes(tokenB64u) : null;
  let ctr = 0;
  const lastCtrByPeer = new Map();

  const NativeWS = window.WebSocket;
  if (!NativeWS) throw new Error('WebSocket not available');
  if (NativeWS.__pskWrapped) return () => {}; // already wrapped

  // Drop-in WebSocket replacement; mirrors the native API (events, on* handlers, readyState, send, close).
  class PskWS {
    constructor(url, protocols) {
      this._inner = new NativeWS(url, protocols);

      // EventTarget shim
      this._et = document.createDocumentFragment();
      this.addEventListener = this._et.addEventListener.bind(this._et);
      this.removeEventListener = this._et.removeEventListener.bind(this._et);
      this.dispatchEvent = this._et.dispatchEvent.bind(this._et);

      // on* property handlers (to match native WebSocket usage)
      this._onopen = null;
      this._onmessage = null;
      this._onerror = null;
      this._onclose = null;

      // Inbound messages are processed strictly in arrival order. HMAC verification is
      // async; without this queue a later message could finish verifying first, and the
      // counter (replay) check would then drop the earlier, legitimate one.
      let inboundChain = Promise.resolve();
      const deliver = (data) => {
        const evOut = new MessageEvent('message', { data });
        this.dispatchEvent(evOut);
        this._onmessage && this._onmessage(evOut);
      };
      const handleInbound = async (data) => {
        if (!tokenBytes) {
          deliver(data);
          return;
        }
        let obj;
        try {
          obj = JSON.parse(data);
        } catch {
          deliver(data);
          return;
        }
        const t = obj?.type || obj?.op || '';
        if (PASSTHRU.has(t)) {
          deliver(JSON.stringify(obj));
          return;
        }

        const sig = obj?.psk;
        if (!sig || typeof sig.ctr !== 'number' || typeof sig.mac !== 'string') return; // drop

        const view = viewFor(obj, sig.ctr);
        const payload = te.encode(JSON.stringify(view));
        const expect = await hmac(tokenBytes, payload);
        let got;
        try {
          got = b64uToBytes(sig.mac);
        } catch {
          return;
        } // malformed MAC → drop
        if (!timingSafeEq(expect, got)) return; // bad MAC → drop
        const key = obj.from || 'room';
        const last = lastCtrByPeer.get(key) || 0;
        if (sig.ctr <= last) return; // replay → drop
        lastCtrByPeer.set(key, sig.ctr);
        deliver(JSON.stringify(obj));
      };

      // Relay from inner → outer
      const relay = (type, evIn) => {
        if (type === 'message') {
          const data = evIn.data;
          inboundChain = inboundChain
            .then(() => handleInbound(data))
            .catch((e) => console.warn('[PSK] inbound message dropped:', e?.message || e));
          return;
        }

        // Non-message events: forward as regular Events and also call property handler
        const evOut =
          type === 'close' ? new CloseEvent('close', evIn) : type === 'error' ? new Event('error') : new Event('open');
        this.dispatchEvent(evOut);
        if (type === 'open' && this._onopen) this._onopen(evOut);
        if (type === 'error' && this._onerror) this._onerror(evOut);
        if (type === 'close' && this._onclose) this._onclose(evOut);
      };

      this._inner.addEventListener('open', (e) => relay('open', e));
      this._inner.addEventListener('error', (e) => relay('error', e));
      this._inner.addEventListener('close', (e) => relay('close', e));
      this._inner.addEventListener('message', (e) => relay('message', e));
    }

    // on* property compatibility
    get onopen() {
      return this._onopen;
    }
    set onopen(fn) {
      this._onopen = typeof fn === 'function' ? fn : null;
    }
    get onmessage() {
      return this._onmessage;
    }
    set onmessage(fn) {
      this._onmessage = typeof fn === 'function' ? fn : null;
    }
    get onerror() {
      return this._onerror;
    }
    set onerror(fn) {
      this._onerror = typeof fn === 'function' ? fn : null;
    }
    get onclose() {
      return this._onclose;
    }
    set onclose(fn) {
      this._onclose = typeof fn === 'function' ? fn : null;
    }

    // Forwarders
    get readyState() {
      return this._inner.readyState;
    }
    get bufferedAmount() {
      return this._inner.bufferedAmount;
    }
    get url() {
      return this._inner.url;
    }
    get protocol() {
      return this._inner.protocol;
    }

    send(data) {
      if (!tokenBytes) {
        this._inner.send(data);
        return;
      }
      let obj;
      try {
        obj = typeof data === 'string' ? JSON.parse(data) : data;
      } catch {
        obj = null;
      }
      if (!obj || typeof obj !== 'object') {
        this._enqueueSend(() => data);
        return;
      }
      const t = obj?.type || obj?.op || '';
      if (PASSTHRU.has(t)) {
        this._enqueueSend(() => JSON.stringify(obj));
        return;
      }
      const nextCtr = ++ctr;
      const view = viewFor(obj, nextCtr);
      const payload = te.encode(JSON.stringify(view));
      this._enqueueSend(async () => {
        const mac = await hmac(tokenBytes, payload);
        obj.psk = { mac: bytesToB64u(mac), ctr: nextCtr };
        return JSON.stringify(obj);
      });
    }

    // Outbound messages leave in call order (signing is async; out-of-order sends would
    // make the peer's counter check drop the earlier message).
    _enqueueSend(build) {
      this._sendChain = (this._sendChain || Promise.resolve())
        .then(build)
        .then((out) => {
          if (this._inner.readyState !== NativeWS.OPEN) return; // closed meanwhile → drop (as native would error)
          this._inner.send(out);
        })
        .catch((e) => console.warn('[PSK] send failed:', e?.message || e));
    }

    // Let messages still being signed (e.g. a final "bye") go out first; never wait more than 1 s.
    close(code, reason) {
      const shut = () => {
        try {
          this._inner.close(code, reason);
        } catch {}
      };
      if (!this._sendChain) return shut();
      let done = false;
      const once = () => {
        if (!done) {
          done = true;
          shut();
        }
      };
      this._sendChain.then(once, once);
      setTimeout(once, 1000);
    }
  }
  PskWS.__pskWrapped = true;
  PskWS.CLOSING = NativeWS.CLOSING;
  PskWS.CLOSED = NativeWS.CLOSED;
  PskWS.CONNECTING = NativeWS.CONNECTING;
  PskWS.OPEN = NativeWS.OPEN;

  const prev = window.WebSocket;
  window.WebSocket = PskWS;
  return () => {
    window.WebSocket = prev;
  };
}
