// /webrtc/shared/psk-ws-shim.js
// Minimal PSK HMAC signing for signaling — no server change, no ReceiverCore change.
// Installs a global WebSocket proxy that auto-signs outbound JSON messages and
// verifies inbound JSON messages using an HMAC-SHA256 over key fields + a 'ctr'.
//
// Persistence: localStorage key "psk:<room>" with { tokenB64u, createdAt }.
// Token import from URL fragment: #room=Baby&token=BASE64URL (then scrubbed).

const te = new TextEncoder();
const td = new TextDecoder();

function b64uToBytes(b64u) {
  const b64 = b64u.replace(/-/g, '+').replace(/_/g, '/');
  const pad = '='.repeat((4 - (b64.length % 4)) % 4);
  const bin = atob(b64 + pad);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
function bytesToB64u(arr) {
  let s = ''; for (let i = 0; i < arr.length; i++) s += String.fromCharCode(arr[i]);
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/,'');
}
async function hmac(keyBytes, payloadBytes) {
  const key = await crypto.subtle.importKey('raw', keyBytes, {name:'HMAC', hash:'SHA-256'}, false, ['sign']);
  const mac = await crypto.subtle.sign('HMAC', key, payloadBytes);
  return new Uint8Array(mac);
}
function storageKey(room) { return 'psk:' + String(room || '').trim(); }
function getStored(room) {
  try { const x = localStorage.getItem(storageKey(room)); return x ? JSON.parse(x) : null; } catch { return null; }
}
function setStored(room, obj) { try { localStorage.setItem(storageKey(room), JSON.stringify(obj)); } catch {} }
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

function timingSafeEq(a,b){ if(!a||!b||a.length!==b.length) return false; let v=0; for(let i=0;i<a.length;i++) v|=(a[i]^b[i]); return v===0; }

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
    ctr: Number(ctr) || 0
  };
}
const PASSTHRU = new Set(['hello','roster','peer-joined','peer-left','keepalive']); // never require psk

// ---- Public helpers ----
export async function ensurePskForRoom(room, role='receiver') {
  // 1) import from #fragment if present (and scrub)
  const { room: fragRoom, token } = parseFragment();
  if (token) {
    const r = (fragRoom || room || 'Baby');
    if (!/^[A-Za-z0-9\-_]{16,}$/.test(token)) throw new Error('Invalid token in fragment');
    setStored(r, { tokenB64u: token, createdAt: Date.now() });
    // If fragment specified a different room, keep it; callers may read it back via getPsk()
    if (r !== room) room = r;
    scrubFragment();
  }
  // 2) ensure present (sender may auto-create; receiver shouldn’t)
  let st = getStored(room);
  if (!st?.tokenB64u && role === 'sender') {
    const bytes = new Uint8Array(16); crypto.getRandomValues(bytes);
    setStored(room, { tokenB64u: bytesToB64u(bytes), createdAt: Date.now() });
    st = getStored(room);
  }
  return { room, tokenB64u: st?.tokenB64u || null };
}

export function getPsk(room) {
  const st = getStored(room); return { room, tokenB64u: st?.tokenB64u || null };
}

export function makeShareUrl({ baseUrl, room }) {
  const { tokenB64u } = getPsk(room);
  const u = new URL(baseUrl, location.origin);
  const h = new URLSearchParams(); h.set('room', room || 'Baby'); if (tokenB64u) h.set('token', tokenB64u);
  u.hash = h.toString(); return u.toString();
}

// ---- Install a global WebSocket proxy for this room ----
export function installPskShim({ room }) {
  const st = getStored(room);
  const tokenB64u = st?.tokenB64u || null;
  const tokenBytes = tokenB64u ? b64uToBytes(tokenB64u) : null;
  let ctr = 0;
  const lastCtrByPeer = new Map();

  const NativeWS = window.WebSocket;
  if (!NativeWS) throw new Error('WebSocket not available');
  if (NativeWS.__pskWrapped) return () => {}; // already wrapped

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
  
      // Relay from inner → outer
      const relay = (type, evIn) => {
        // Build outbound event (preserve MessageEvent data; forward original for property handler)
        if (type === 'message') {
          const data = evIn.data;
          if (!tokenBytes) {
            const evOut = new MessageEvent('message', { data });
            this.dispatchEvent(evOut);
            this._onmessage && this._onmessage(evOut);
            return;
          }
          let obj;
          try { obj = JSON.parse(data); } catch {
            const evOut = new MessageEvent('message', { data });
            this.dispatchEvent(evOut);
            this._onmessage && this._onmessage(evOut);
            return;
          }
          const t = obj?.type || obj?.op || '';
          if (PASSTHRU.has(t)) {
            const evOut = new MessageEvent('message', { data: JSON.stringify(obj) });
            this.dispatchEvent(evOut);
            this._onmessage && this._onmessage(evOut);
            return;
          }
          const sig = obj?.psk;
          if (!sig || typeof sig.ctr !== 'number' || typeof sig.mac !== 'string') return; // drop
  
          const view = viewFor(obj, sig.ctr);
          const payload = te.encode(JSON.stringify(view));
          hmac(tokenBytes, payload).then(expect => {
            const got = b64uToBytes(sig.mac);
            if (!timingSafeEq(expect, got)) return; // bad MAC → drop
            const key = obj.from || 'room';
            const last = lastCtrByPeer.get(key) || 0;
            if (sig.ctr <= last) return;           // replay → drop
            lastCtrByPeer.set(key, sig.ctr);
  
            const evOut = new MessageEvent('message', { data: JSON.stringify(obj) });
            this.dispatchEvent(evOut);
            this._onmessage && this._onmessage(evOut);
          });
          return;
        }
  
        // Non-message events: forward as regular Events and also call property handler
        const evOut =
          type === 'close' ? new CloseEvent('close', evIn) :
          type === 'error' ? new Event('error') :
                             new Event('open');
        this.dispatchEvent(evOut);
        if (type === 'open'   && this._onopen)   this._onopen(evOut);
        if (type === 'error'  && this._onerror)  this._onerror(evOut);
        if (type === 'close'  && this._onclose)  this._onclose(evOut);
      };
  
      this._inner.addEventListener('open',    (e) => relay('open', e));
      this._inner.addEventListener('error',   (e) => relay('error', e));
      this._inner.addEventListener('close',   (e) => relay('close', e));
      this._inner.addEventListener('message', (e) => relay('message', e));
    }
  
    // on* property compatibility
    get onopen()    { return this._onopen; }
    set onopen(fn)  { this._onopen = typeof fn === 'function' ? fn : null; }
    get onmessage() { return this._onmessage; }
    set onmessage(fn){ this._onmessage = typeof fn === 'function' ? fn : null; }
    get onerror()   { return this._onerror; }
    set onerror(fn) { this._onerror = typeof fn === 'function' ? fn : null; }
    get onclose()   { return this._onclose; }
    set onclose(fn) { this._onclose = typeof fn === 'function' ? fn : null; }
  
    // Forwarders
    get readyState() { return this._inner.readyState; }
    get bufferedAmount() { return this._inner.bufferedAmount; }
    get url() { return this._inner.url; }
    get protocol() { return this._inner.protocol; }
  
    send(data) {
      if (!tokenBytes) { this._inner.send(data); return; }
      let obj;
      try { obj = (typeof data === 'string') ? JSON.parse(data) : data; } catch { obj = null; }
      if (!obj || typeof obj !== 'object') { this._inner.send(data); return; }
      const t = obj?.type || obj?.op || '';
      if (PASSTHRU.has(t)) { this._inner.send(JSON.stringify(obj)); return; }
      const nextCtr = ++ctr;
      const view = viewFor(obj, nextCtr);
      const payload = te.encode(JSON.stringify(view));
      (async () => {
        const mac = await hmac(tokenBytes, payload);
        obj.psk = { mac: bytesToB64u(mac), ctr: nextCtr };
        this._inner.send(JSON.stringify(obj));
      })();
    }
  
    close(code, reason) { try { this._inner.close(code, reason); } catch {} }
  }
  PskWS.CLOSING = NativeWS.CLOSING; PskWS.CLOSED = NativeWS.CLOSED; PskWS.CONNECTING = NativeWS.CONNECTING; PskWS.OPEN = NativeWS.OPEN;
  
    
  PskWS.__pskWrapped = true;
  PskWS.CLOSING = NativeWS.CLOSING; PskWS.CLOSED = NativeWS.CLOSED; PskWS.CONNECTING = NativeWS.CONNECTING; PskWS.OPEN = NativeWS.OPEN;

  const prev = window.WebSocket;
  window.WebSocket = PskWS;
  return () => { window.WebSocket = prev; };
}
