// /webrtc/shared/secure-signal.js (WS-open wait + nonceQR + delayed-save with pair-done)
import { te, td, b64u, b64uToBytes, digestSHA256, concatBytes, rand, hkdf, aesGcmEncrypt, aesGcmDecrypt } from './pairing-crypto.js';
import { ensureIdentity, shortSAS } from './identity.js';

export class SecureSignal {
  constructor({ url, onPairPrompt, onTrustedList, onOpen, onSession } = {}) {
    this.url = url; this.ws = null;
    this.pending = new Map();
    this.trusted = [];
    this.sessions = new Map(); // sid -> { toFp, key, role, handlers: Function[], onmsg (legacy) }
    this.onPairPrompt = onPairPrompt || (()=>{});
    this.onTrustedList = onTrustedList || (()=>{});
    this.onOpen = onOpen || (()=>{});
    this.onSession = onSession || (()=>{});
    this._qrNonce = null;         // stored when we render QR
    this._openWaiters = [];       // waiters for WS "open"
    this.instanceId = null;

    // New: keep a stable, chainable per-sid dispatcher reference
    this._sidHandlers = new Map(); // sid -> dispatcher function(msg, from)
  }

  async init() {
    this.me = await ensureIdentity({});
    this.self = this.me;
    try {
      this.instanceId =
        sessionStorage.getItem('naptio:instanceId') ||
        (() => {
          const b = crypto.getRandomValues(new Uint8Array(8));
          const id = Array.from(b).map(x => x.toString(16).padStart(2,'0')).join('');
          sessionStorage.setItem('naptio:instanceId', id);
          return id;
        })();
    } catch { 
      // Fallback when sessionStorage is unavailable
      const b = new Uint8Array(8);
      crypto.getRandomValues(b);
      this.instanceId = Array.from(b).map(x => x.toString(16).padStart(2,'0')).join('');
    }
    
    this.trusted = (await this._loadTrusted()) || [];
    this._connect();
  }

  _connect() {
    const ws = new WebSocket(this.url);
    this.ws = ws;
    ws.onopen = () => {
      const w = this._openWaiters.splice(0);
      for (const res of w) try { res(); } catch {}
      ws.send(JSON.stringify({
        op: 'register',
        fp: this.me.fingerprint,
        instance: this.instanceId,   // <-- NEW
        device: this.me.deviceName
      }));
      this.onOpen();
      this._emitTrusted();
    };
    ws.onmessage = (e) => this._onmsg(JSON.parse(e.data));
    ws.onclose   = () => setTimeout(()=>this._connect(), 1200);
  }

  _whenOpen() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return Promise.resolve();
    return new Promise(res => this._openWaiters.push(res));
  }

  _emitTrusted(){ this.onTrustedList([...this.trusted]); }
  async _loadTrusted(){ const raw = localStorage.getItem('naptio-trusted-devices'); return raw ? JSON.parse(raw) : []; }
  async _saveTrusted(){ localStorage.setItem('naptio-trusted-devices', JSON.stringify(this.trusted)); }

  // ===== Pairing (QR host <-> scanner) =====

  makePairingPayload() {
    const nonce = rand(16);
    this._qrNonce = nonce; // remember
    const payload = {
      v:'n1',
      name:this.me.deviceName,
      signSpki:this.me.signSpki,
      ecdhSpki:this.me.ecdhSpki,
      fp:this.me.fingerprint,
      nonce:b64u(nonce)
    };
    return { json: payload, b64: b64u(te.encode(JSON.stringify(payload))) };
  }

  // Scanner
  async handlePairingLink(hashB64) {
    const payload = JSON.parse(td.decode(b64uToBytes(hashB64)));
    const peer = {
      fp: payload.fp,
      name: payload.name,
      signSpki: payload.signSpki,
      ecdhSpki: payload.ecdhSpki,
      nonceQR: b64uToBytes(payload.nonce)
    };
    const nonceScan = rand(16);
    const tag = te.encode('NaptioPair-v1');

    const T = concatBytes(
      tag,
      peer.nonceQR,
      nonceScan,
      b64uToBytes(peer.signSpki),
      b64uToBytes(this.me.signSpki),
      b64uToBytes(peer.ecdhSpki),
      b64uToBytes(this.me.ecdhSpki)
    );

    const sig = new Uint8Array(await crypto.subtle.sign(
      { name:'ECDSA', hash:'SHA-256' },
      this.me.signKey,
      T
    ));

    await this._whenOpen();
    this.ws.send(JSON.stringify({
      op:'pair-init',
      to:     peer.fp,
      from:   this.me.fingerprint,
      name:   this.me.deviceName,
      signSpki: this.me.signSpki,
      ecdhSpki: this.me.ecdhSpki,
      nonceQR:  payload.nonce,   // host's QR nonce
      nonceB:   b64u(nonceScan), // scanner's nonce
      sig:      b64u(sig)
    }));

    this.pending.set('pairing', { peer, T });
  }

  async _onmsg(m) {
    // Host receives scanner's init
    if (m.op === 'pair-init' && m.to === this.me.fingerprint) {
      const nonceQR = m.nonceQR ? b64uToBytes(m.nonceQR) : this._qrNonce;
      if (!nonceQR) return;

      const nonceScan = b64uToBytes(m.nonceB);
      const tag = te.encode('NaptioPair-v1');

      const T = concatBytes(
        tag,
        nonceQR,
        nonceScan,
        b64uToBytes(this.me.signSpki),
        b64uToBytes(m.signSpki),
        b64uToBytes(this.me.ecdhSpki),
        b64uToBytes(m.ecdhSpki)
      );

      const ok = await crypto.subtle.verify(
        { name:'ECDSA', hash:'SHA-256' },
        await crypto.subtle.importKey('spki', b64uToBytes(m.signSpki), { name:'ECDSA', namedCurve:'P-256' }, true, ['verify']),
        b64uToBytes(m.sig),
        T
      );
      if (!ok) return;

      const sas = shortSAS(await digestSHA256(T));
      const ctx = { fp:m.from, name:m.name, signSpki:m.signSpki, ecdhSpki:m.ecdhSpki, T };
      this.pending.set('pair-acc', ctx);

      this.onPairPrompt({
        type:'incoming',
        from:m.name, fp:m.from, sas,
        accept: () => this._acceptPair(m.from, T, ctx),
        reject: () => {}
      });
      return;
    }

    // Scanner receives host's ACK
    if (m.op === 'pair-ack' && m.to === this.me.fingerprint) {
      const st = this.pending.get('pairing'); if (!st) return;
      const T = st.T;

      const ok = await crypto.subtle.verify(
        { name:'ECDSA', hash:'SHA-256' },
        await crypto.subtle.importKey('spki', b64uToBytes(m.signSpki), { name:'ECDSA', namedCurve:'P-256' }, true, ['verify']),
        b64uToBytes(m.sig),
        T
      );
      if (!ok) return;

      const sas = shortSAS(await digestSHA256(T));
      this.onPairPrompt({
        type:'confirm',
        from:m.name, fp:m.from, sas,
        accept: () => this._finalizePair(m.from, T, m),   // <-- send pair-done AFTER saving here
        reject: () => {}
      });
      return;
    }

    // Host receives final confirmation from scanner
    if (m.op === 'pair-done' && m.to === this.me.fingerprint) {
      const key = `pair-done:${m.from}`;
      const st = this.pending.get(key);
      if (st) {
        // Optional: check transcript hash matches
        if (m.th && m.th !== st.th) { /* ignore or log */ }
        await this._storeTrusted({
          fp: st.ctx.fp, name: st.ctx.name,
          signSpki: st.ctx.signSpki, ecdhSpki: st.ctx.ecdhSpki
        });
        this.pending.delete(key);
      }
      return;
    }

    // ===== Encrypted signaling + session setup =====
    if (m.op === 'relay' && m.to === this.me.fingerprint) {
      if (m.kind === 'eph-reply') { this._resolveWaiter(`eph-reply:${m.sid}`, m); return; }
      if (m.kind === 'eph-hello') { await this._handleEphHello(m); return; }
      if (m.kind === 'enc') {
        const s = this.sessions.get(m.sid); if (!s || !s.key) return;
        try {
          const pt = await aesGcmDecrypt(s.key, b64uToBytes(m.iv), b64uToBytes(m.ct));
          const obj = JSON.parse(td.decode(pt));
          // Fan-out to all registered handlers for this sid
          const dispatcher = this._sidHandlers.get(m.sid);
          if (dispatcher) {
            try { dispatcher(obj, m.from); } catch {}
          } else if (typeof s.onmsg === 'function') {
            // Back-compat: if only legacy single handler exists
            try { s.onmsg(obj, m.from); } catch {}
          }
        } catch {}
        return;
      }
    }

    if (m.op === 'ping') this.ws.send(JSON.stringify({ op:'pong' }));
  }

  // Host accept: send ACK, then wait for scanner's 'pair-done' to save
  async _acceptPair(to, T, ctx) {
    const sig = new Uint8Array(await crypto.subtle.sign(
      { name:'ECDSA', hash:'SHA-256' },
      this.me.signKey,
      T
    ));
    const th = b64u(new Uint8Array(await digestSHA256(T)));
    this.pending.set(`pair-done:${to}`, { th, ctx });

    await this._whenOpen();
    this.ws.send(JSON.stringify({
      op:'pair-ack',
      to,
      from: this.me.fingerprint,
      name: this.me.deviceName,
      signSpki: this.me.signSpki,
      ecdhSpki: this.me.ecdhSpki,
      sig: b64u(sig)
    }));
    // Do NOT storeTrusted here; wait for 'pair-done'
  }

  // Scanner confirm: save locally, then notify host with 'pair-done'
  async _finalizePair(peerFp, T, devMsg) {
    await this._storeTrusted({
      fp: peerFp, name: devMsg.name,
      signSpki: devMsg.signSpki, ecdhSpki: devMsg.ecdhSpki
    });
    const th = b64u(new Uint8Array(await digestSHA256(T)));
    await this._whenOpen();
    this.ws.send(JSON.stringify({ op:'pair-done', to: peerFp, from: this.me.fingerprint, th }));
  }

  async _storeTrusted(dev) {
    this.trusted = this.trusted.filter(d => d.fp !== dev.fp).concat([{ ...dev, addedAt: Date.now() }]);
    await this._saveTrusted(); this._emitTrusted();
  }

  // ===== Ephemeral per-peer session key (for encrypted signaling) =====
  async startSession(toFp) {
    const sid = b64u(rand(6));
    const peer = this.trusted.find(d => d.fp === toFp); if (!peer) throw new Error('Unknown peer');

    const eph = await crypto.subtle.generateKey({ name:'ECDH', namedCurve:'P-256' }, true, ['deriveBits']);
    const ephSpki = new Uint8Array(await crypto.subtle.exportKey('spki', eph.publicKey));
    const nonce = rand(16);
    const tag = te.encode('naptio-ephemeral-v1');
    const sig = new Uint8Array(await crypto.subtle.sign(
      { name:'ECDSA', hash:'SHA-256' },
      this.me.signKey,
      concatBytes(tag, ephSpki, te.encode(peer.fp), nonce)
    ));

    // Prepare session container (no handlers yet)
    const sess = { toFp, key:null, role:'initiator', handlers: [], onmsg: null };
    this.sessions.set(sid, sess);
    // Ensure a stable dispatcher is present (even if empty) so others can chain
    this._ensureDispatcher(sid);

    const reply = await this._waitFor(`eph-reply:${sid}`, 15000, async () => {
      await this._whenOpen();
      this.ws.send(JSON.stringify({ op:'relay', to: peer.fp, from:this.me.fingerprint, fromInstance: this.instanceId, kind:'eph-hello', sid, ephSpki: b64u(ephSpki), nonce: b64u(nonce), sig: b64u(sig) }));
    });

    const ok = await crypto.subtle.verify(
      { name:'ECDSA', hash:'SHA-256' },
      await crypto.subtle.importKey('spki', b64uToBytes(peer.signSpki), { name:'ECDSA', namedCurve:'P-256' }, true, ['verify']),
      b64uToBytes(reply.sig),
      concatBytes(tag, b64uToBytes(reply.ephSpki), te.encode(this.me.fingerprint), b64uToBytes(reply.nonce))
    );
    if (!ok) throw new Error('ephemeral verify failed');

    const sessionKey = await this._deriveSessionKey(eph.privateKey, b64uToBytes(reply.ephSpki), concatBytes(nonce, b64uToBytes(reply.nonce)));
    sess.key = sessionKey;
    this.onSession({ sid, peerFp: toFp, role:'initiator' });
    return { sid, sessionKey };
  }

  async _handleEphHello(m) {
    const peer = this.trusted.find(d => d.fp === m.from); if (!peer) return;
    const tag = te.encode('naptio-ephemeral-v1');

    const ok = await crypto.subtle.verify(
      { name:'ECDSA', hash:'SHA-256' },
      await crypto.subtle.importKey('spki', b64uToBytes(peer.signSpki), { name:'ECDSA', namedCurve:'P-256' }, true, ['verify']),
      b64uToBytes(m.sig),
      concatBytes(tag, b64uToBytes(m.ephSpki), te.encode(this.me.fingerprint), b64uToBytes(m.nonce))
    );
    if (!ok) return;

    const sid = m.sid;
    const eph = await crypto.subtle.generateKey({ name:'ECDH', namedCurve:'P-256' }, true, ['deriveBits']);
    const ephSpki = new Uint8Array(await crypto.subtle.exportKey('spki', eph.publicKey));
    const nonce = rand(16);
    const sig = new Uint8Array(await crypto.subtle.sign(
      { name:'ECDSA', hash:'SHA-256' },
      this.me.signKey,
      concatBytes(tag, ephSpki, te.encode(peer.fp), nonce)
    ));
    const sessionKey = await this._deriveSessionKey(eph.privateKey, b64uToBytes(m.ephSpki), concatBytes(b64uToBytes(m.nonce), nonce));

    // Prepare session container (no handlers yet)
    this.sessions.set(sid, { toFp: m.from, key: sessionKey, role:'responder', handlers: [], onmsg: null });
    this._ensureDispatcher(sid);

    await this._whenOpen();
    this.ws.send(JSON.stringify({ op:'relay', to: peer.fp, from:this.me.fingerprint, fromInstance: this.instanceId, kind:'eph-reply', sid, ephSpki: b64u(ephSpki), nonce: b64u(nonce), sig: b64u(sig) }));
    this.onSession({ sid, peerFp: m.from, role:'responder' });
  }

  async _deriveSessionKey(privEcdh, peerSpkiBytes, salt) {
    const peerPub = await crypto.subtle.importKey('spki', peerSpkiBytes, { name:'ECDH', namedCurve:'P-256' }, true, []);
    const bits = await crypto.subtle.deriveBits({ name:'ECDH', public: peerPub }, privEcdh, 256);
    return hkdf(new Uint8Array(bits), { salt, info:'naptio-sdp-v1' });
  }

  // ===== Encrypted signaling helpers =====
  async sendJSON(sid, obj) {
    const s = this.sessions.get(sid); if (!s || !s.key) throw new Error('Missing session');
    const { iv, ct } = await aesGcmEncrypt(s.key, te.encode(JSON.stringify(obj)));
    await this._whenOpen();
    this.ws.send(JSON.stringify({
      op: 'relay',
      to: s.toFp,
      from: this.me.fingerprint,
      fromInstance: this.instanceId,   // <-- add this
      kind: 'enc',
      sid,
      iv: b64u(iv),
      ct: b64u(ct)
    }));
  }

  // New: add a handler without overwriting others (fan-out)
  onEncrypted(sid, handler) {
    const s = this.sessions.get(sid); if (!s) throw new Error('Unknown sid');
    if (!s.handlers) s.handlers = [];
    s.handlers.push(handler);
    // Keep legacy field pointing to the dispatcher for older code paths
    this._ensureDispatcher(sid);
  }

  // New: allow chaining — return the stable dispatcher currently used
  getEncryptedHandler(sid) {
    return this._sidHandlers.get(sid) || null;
  }

  // Ensure a stable per-sid dispatcher exists and is recorded in _sidHandlers (and legacy s.onmsg)
  _ensureDispatcher(sid) {
    const s = this.sessions.get(sid);
    if (!s) return null;
    const dispatcher = (msg, from) => {
      // Fan-out to all registered handlers
      if (Array.isArray(s.handlers)) {
        for (const fn of s.handlers) {
          try { fn(msg, from); } catch {}
        }
      }
      // For extreme back-compat: if someone later sets s.onmsg directly, call it too (but avoid recursion)
      if (s._legacyOnMsg && s._legacyOnMsg !== dispatcher) {
        try { s._legacyOnMsg(msg, from); } catch {}
      }
    };
    // Preserve any existing legacy handler if present before we replace s.onmsg
    if (s.onmsg && s.onmsg !== dispatcher) s._legacyOnMsg = s.onmsg;
    s.onmsg = dispatcher; // legacy entry point points to dispatcher
    this._sidHandlers.set(sid, dispatcher);
    return dispatcher;
  }

  _waitFor(key, timeout, firstSend) {
    return new Promise((resolve, reject) => {
      const t = setTimeout(() => { this.pending.delete(key); reject(new Error('timeout')); }, timeout || 10000);
      this.pending.set(key, (m) => { clearTimeout(t); this.pending.delete(key); resolve(m); });
      firstSend && firstSend();
    });
  }
  _resolveWaiter(key, payload) {
    const fn = this.pending.get(key);
    if (fn) { fn(payload); this.pending.delete(key); }
  }
}
