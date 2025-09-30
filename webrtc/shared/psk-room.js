// psk-room.js — Lightweight PSK (pre-shared key) signaling wrapper with persistence + HMAC
// Usage (receiver or sender):
//   import { PskRoom } from '/hungryface/webrtc/shared/psk-room.js';
//   const psk = await PskRoom.init({ room: 'Baby', role: 'receiver', wsEndpoint: 'wss://.../ws' });
//   const sock = await psk.openSignedSocket(); // WebSocket-like shim
//   sock.addEventListener('message', (ev) => { /* ev.data is JSON string (verified) */ });
//   sock.send(JSON.stringify({ type:'need-offer', to: '...' })); // auto-signed
//
// Token provisioning options (any page):
//   1) URL fragment:   https://app/receiver#room=Baby&token=BASE64URL
//      -> PskRoom.init() will import & scrub location.hash automatically.
//   2) postMessage:    window.postMessage({kind:'psk-provision', room, tokenB64u}, '*')
//   3) Programmatic:   await PskRoom.setToken('Baby', 'BASE64URL');
//
// Storage: localStorage key "psk:<room>"   (value = JSON { tokenB64u, createdAt })
// Security notes:
//   - Token never sent in clear. Only HMACs (+ room, role).
//   - Adds "ctr" per peer to limit trivial replay within a session.
//   - Keep your app under HTTPS/WSS. Don’t leave tokens in query strings.
//

export class PskRoom {
  static STORAGE_PREFIX = 'psk:';
  static FRAGMENT_TOKEN_KEYS = new Set(['token','psk','t']); // accepted #token= aliases
  static FRAGMENT_ROOM_KEYS  = new Set(['room','r']);

  // ---------- helpers ----------
  static te = new TextEncoder();
  static td = new TextDecoder();

  static b64uToBytes(b64u) {
    const b64 = b64u.replace(/-/g, '+').replace(/_/g, '/');
    const pad = '='.repeat((4 - (b64.length % 4)) % 4);
    const bin = atob(b64 + pad);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  static bytesToB64u(arr) {
    let bin = '';
    for (let i = 0; i < arr.length; i++) bin += String.fromCharCode(arr[i]);
    return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/,'');
  }

  static async hmac(keyBytes, payloadBytes) {
    const key = await crypto.subtle.importKey(
      'raw', keyBytes, { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']
    );
    const mac = await crypto.subtle.sign('HMAC', key, payloadBytes);
    return new Uint8Array(mac);
  }

  // Optional: derive a sub-key for data channel crypto
  static async hkdf(tokenBytes, salt = new Uint8Array(32), info = 'dc-ctrl', length = 32) {
    // HKDF-Extract
    const ikm = await crypto.subtle.importKey('raw', tokenBytes, { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
    const prk = new Uint8Array(await crypto.subtle.sign('HMAC', ikm, salt));
    // HKDF-Expand (single block is enough for 32 bytes)
    const t1 = await this.hmac(prk, this.te.encode(info + String.fromCharCode(1)));
    return t1.slice(0, length);
  }

  static async randomTokenB64u(bits = 128) {
    const bytes = new Uint8Array(bits / 8);
    crypto.getRandomValues(bytes);
    return this.bytesToB64u(bytes);
  }

  static storageKey(room) {
    return this.STORAGE_PREFIX + String(room || '').trim();
  }
  static getStored(room) {
    try {
      const raw = localStorage.getItem(this.storageKey(room));
      return raw ? JSON.parse(raw) : null;
    } catch { return null; }
  }
  static setStored(room, obj) {
    try { localStorage.setItem(this.storageKey(room), JSON.stringify(obj)); } catch {}
  }
  static clearStored(room) {
    try { localStorage.removeItem(this.storageKey(room)); } catch {}
  }

  static parseFragment() {
    const hash = (location.hash || '').replace(/^#/, '');
    if (!hash) return {};
    const qs = new URLSearchParams(hash);
    let token = null, room = null;
    for (const [k, v] of qs.entries()) {
      const kk = k.toLowerCase();
      if (this.FRAGMENT_TOKEN_KEYS.has(kk)) token = v;
      if (this.FRAGMENT_ROOM_KEYS.has(kk))  room  = v;
    }
    return { token, room };
  }
  static scrubFragment() {
    if (location.hash && location.hash !== '#') {
      history.replaceState(null, '', location.pathname + location.search);
    }
  }

  // Consumers can call this to provision from another tab/QR overlay securely
  static async handleWindowMessage(ev) {
    const msg = ev?.data;
    if (!msg || msg.kind !== 'psk-provision') return;
    const room = String(msg.room || '').trim();
    const tokenB64u = String(msg.tokenB64u || '').trim();
    if (!room || !tokenB64u) return;
    await PskRoom.setToken(room, tokenB64u);
    // Optionally, emit a CustomEvent so the page can react
    window.dispatchEvent(new CustomEvent('psk:provisioned', { detail: { room } }));
  }

  static async getTokenBytes(room) {
    const st = this.getStored(room);
    if (!st?.tokenB64u) return null;
    return this.b64uToBytes(st.tokenB64u);
  }
  static async setToken(room, tokenB64u) {
    if (!room) throw new Error('room required');
    // Basic sanity: 128-bit = 22 char b64url (without padding)
    if (!/^[A-Za-z0-9\-_]{16,}$/.test(tokenB64u)) throw new Error('invalid token format');
    this.setStored(room, { tokenB64u, createdAt: Date.now() });
    return true;
  }

  // All-in-one init flow:
  //  - import token from #fragment (and scrub)
  //  - else use persisted token
  //  - else auto-generate (if role === 'sender') and persist
  static async init({ room = 'Baby', role = 'receiver', wsEndpoint }) {
    const p = new PskRoom(room, role, wsEndpoint);
    // (1) fragment import
    const { token, room: fragRoom } = this.parseFragment();
    if (token) {
      const r = fragRoom || room;
      await this.setToken(r, token);
      if (r !== room) p.room = r;
      this.scrubFragment();
    }
   .window?.addEventListener?.('message', this.handleWindowMessage, { passive: true });
    // (2) ensure we have a token
    let st = this.getStored(p.room);
    if (!st?.tokenB64u) {
      if (role === 'sender') {
        const gen = await this.randomTokenB64u(128);
        this.setStored(p.room, { tokenB64u: gen, createdAt: Date.now() });
      } else {
        // receiver w/o token yet — caller should show a pairing UI
      }
      st = this.getStored(p.room);
    }
    p._tokenB64u = st?.tokenB64u || null;
    p._tokenBytes = p._tokenB64u ? this.b64uToBytes(p._tokenB64u) : null;
    return p;
  }

  constructor(room, role, wsEndpoint) {
    this.room = room;
    this.role = role; // 'sender' | 'receiver'
    this.wsEndpoint = wsEndpoint;
    this._tokenB64u = null;
    this._tokenBytes = null;

    // replay/ordering guard: last ctr per (peerId or 'room') we’ve accepted
    this._lastCtrByPeer = new Map();

    // WebSocket-like shim
    this._ws = null;
    this._shim = null;
  }

  get hasToken() { return !!this._tokenBytes; }
  get tokenB64u() { return this._tokenB64u; }

  // Create a WS-like object that auto-signs outbound messages and verifies inbound.
  async openSignedSocket() {
    if (!this.wsEndpoint) throw new Error('wsEndpoint required');
    if (!this.room) throw new Error('room required');
    const url = `${this.wsEndpoint}?room=${encodeURIComponent(this.room)}`;
    const real = new WebSocket(url);
    this._ws = real;

    const shim = new EventTarget();
    // mimic WebSocket shape
    shim.readyState = real.readyState;
    shim.send = (data) => this._sendSigned(real, data);
    shim.close = (...args) => real.close(...args);
    shim.addEventListener = EventTarget.prototype.addEventListener.bind(shim);
    shim.removeEventListener = EventTarget.prototype.removeEventListener.bind(shim);
    shim.dispatchEvent = EventTarget.prototype.dispatchEvent.bind(shim);

    real.addEventListener('open', (e) => {
      shim.readyState = real.readyState;
      // join room + identify role (same as your current flow)
      try { real.send(JSON.stringify({ type: 'join', room: this.room, role: this.role })); } catch {}
      shim.dispatchEvent(new Event('open'));
    });
    real.addEventListener('close', (e) => {
      shim.readyState = real.readyState;
      shim.dispatchEvent(new CloseEvent('close', e));
    });
    real.addEventListener('error', (e) => {
      shim.dispatchEvent(new Event('error'));
    });
    real.addEventListener('message', async (e) => {
      const ok = await this._verifyInbound(e.data);
      if (!ok) return; // drop tampered/unverifiable
      // Pass through the original payload so existing code keeps working
      shim.dispatchEvent(new MessageEvent('message', { data: ok.cleanJson }));
    });

    this._shim = shim;
    return await new Promise((resolve, reject) => {
      const to = setTimeout(() => reject(new Error('WS timeout')), 15000);
      shim.addEventListener('open', () => { clearTimeout(to); resolve(shim); }, { once: true });
      shim.addEventListener('close', () => { clearTimeout(to); reject(new Error('WS closed')); }, { once: true });
      shim.addEventListener('error', () => { clearTimeout(to); reject(new Error('WS error')); }, { once: true });
    });
  }

  // ---------- signing / verifying ----------
  // We sign a compact view:
  //   view = { t: type, to, from, sdp, cand: candidate?.candidate, mid: candidate?.sdpMid, idx: candidate?.sdpMLineIndex, ctr }
  //   mac  = HMAC_SHA256(token, JSON.stringify(view))
  // Counter increases per sender; receivers track last seen per "from" (or 'room' for broadcasts).
  async _signPayload(obj) {
    const ctr = (this._ctr = (this._ctr || 0) + 1);
    const view = {
      t: obj.type || obj.op || '',
      to: obj.to || null,
      from: obj.from || null,
      sdp: obj.sdp || null,
      cand: obj.candidate?.candidate || null,
      mid: obj.candidate?.sdpMid ?? null,
      idx: obj.candidate?.sdpMLineIndex ?? null,
      ctr
    };
    const payload = PskRoom.te.encode(JSON.stringify(view));
    const mac = await PskRoom.hmac(this._tokenBytes, payload);
    obj.psk = { mac: PskRoom.bytesToB64u(mac), ctr };
    return obj;
  }

  async _verifyInbound(jsonStr) {
    // If we don't have a token yet, accept everything (page can gate UI until provisioned).
    if (!this._tokenBytes) {
      return { cleanJson: jsonStr };
    }
    let obj;
    try { obj = JSON.parse(jsonStr); } catch { return false; }
    const sig = obj?.psk;
    // Let trivially safe system messages pass (hello/roster/peer-joined/left) even before token exists
    const passthrough = new Set(['hello','roster','peer-joined','peer-left','keepalive']);
    const t = obj?.type || obj?.op || '';
    if (!sig) {
      if (passthrough.has(t)) return { cleanJson: jsonStr };
      // If this is a signaling message without psk, drop it.
      return false;
    }
    const view = {
      t,
      to: obj.to || null,
      from: obj.from || null,
      sdp: obj.sdp || null,
      cand: obj.candidate?.candidate || null,
      mid: obj.candidate?.sdpMid ?? null,
      idx: obj.candidate?.sdpMLineIndex ?? null,
      ctr: Number(sig.ctr) || 0
    };
    // replay / ordering: track by 'from' (or by room broadcast)
    const key = obj.from || 'room';
    const last = this._lastCtrByPeer.get(key) || 0;
    if (view.ctr <= last) {
      // stale/replayed packet
      return false;
    }
    const payload = PskRoom.te.encode(JSON.stringify(view));
    const expect = await PskRoom.hmac(this._tokenBytes, payload);
    const got = PskRoom.b64uToBytes(String(sig.mac || ''));
    if (!this._timingSafeEqual(expect, got)) return false;
    this._lastCtrByPeer.set(key, view.ctr);
    // Clean pass-through: you can keep obj.psk if you want; most callers don’t need it
    return { cleanJson: JSON.stringify(obj) };
  }

  _timingSafeEqual(a, b) {
    if (!a || !b || a.length !== b.length) return false;
    let acc = 0;
    for (let i = 0; i < a.length; i++) acc |= (a[i] ^ b[i]);
    return acc === 0;
  }

  async _sendSigned(realWS, data) {
    if (typeof data !== 'string') {
      // allow callers to pass objects
      data = JSON.stringify(data);
    }
    let obj;
    try { obj = JSON.parse(data); } catch { obj = null; }
    if (!obj) {
      // non-JSON → pass through
      realWS.send(data);
      return;
    }
    // Don’t sign trivial/control messages
    const passthrough = new Set(['hello','roster','peer-joined','peer-left','keepalive']);
    const t = obj?.type || obj?.op || '';
    if (!this._tokenBytes || passthrough.has(t)) {
      realWS.send(JSON.stringify(obj));
      return;
    }
    const signed = await this._signPayload(obj);
    realWS.send(JSON.stringify(signed));
  }

  // ---------- utilities you can use in your UI/QR share ----------
  // A redacted share URL that keeps token in the fragment (not sent to server)
  makeShareUrl(baseUrl) {
    const u = new URL(baseUrl, location.origin);
    const params = new URLSearchParams();
    params.set('room', this.room);
    if (this._tokenB64u) params.set('token', this._tokenB64u);
    u.hash = params.toString();
    return u.toString();
  }

  // Optional: derive a DC control key (e.g., for encrypting tiny JSON control messages)
  async deriveDataChannelKey(infoLabel = 'dc-ctrl') {
    if (!this._tokenBytes) throw new Error('no token');
    return await PskRoom.hkdf(this._tokenBytes, new Uint8Array(32), infoLabel, 32);
  }
}
