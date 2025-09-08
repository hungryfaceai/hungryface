// /webrtc/shared/secure-signal.js (fixed pairing + multi-session)
import { te, td, b64u, b64uToBytes, digestSHA256, concatBytes, rand, hkdf, aesGcmEncrypt, aesGcmDecrypt } from './pairing-crypto.js';
import { ensureIdentity, shortSAS } from './identity.js';

export class SecureSignal {
  /**
   * Hooks:
   *  - onPairPrompt({ type:'incoming'|'confirm', from, fp, sas, accept })
   *  - onTrustedList(trustedArray)
   *  - onOpen()
   *  - onSession({ sid, peerFp, role:'initiator'|'responder' })
   */
  constructor({ url, onPairPrompt, onTrustedList, onOpen, onSession } = {}) {
    this.url = url; this.ws = null;
    this.pending = new Map();
    this.trusted = [];
    this.sessions = new Map(); // sid -> { toFp, key, onmsg, role }
    this.onPairPrompt = onPairPrompt || (()=>{});
    this.onTrustedList = onTrustedList || (()=>{});
    this.onOpen = onOpen || (()=>{});
    this.onSession = onSession || (()=>{});
    this._qrNonce = null; // set when we render a QR
  }

  async init() {
    this.me = await ensureIdentity({});
    this.trusted = (await this._loadTrusted()) || [];
    this._connect();
  }

  _connect() {
    const ws = new WebSocket(this.url);
    this.ws = ws;
    ws.onopen = () => {
      ws.send(JSON.stringify({ op:'register', fp:this.me.fingerprint, device:this.me.deviceName }));
      this.onOpen();
      this._emitTrusted();
    };
    ws.onmessage = (e) => this._onmsg(JSON.parse(e.data));
    ws.onclose = () => setTimeout(()=>this._connect(), 1200);
  }

  _emitTrusted(){ this.onTrustedList([...this.trusted]); }
  async _loadTrusted(){ const raw = localStorage.getItem('naptio-trusted-devices'); return raw ? JSON.parse(raw) : []; }
  async _saveTrusted(){ localStorage.setItem('naptio-trusted-devices', JSON.stringify(this.trusted)); }

  // ===== Pairing (QR host <-> scanner) =====

  // Host renders a QR and REMEMBERS its nonce so it can verify the incoming request
  makePairingPayload() {
    const nonce = rand(16);
    this._qrNonce = nonce;                // <-- remember!
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

  // Scanner reads the QR and sends a signed request using a canonical transcript (QR-first order)
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

    // Canonical transcript (QR-first): tag || nonceQR || nonceScan || signQR || signSCAN || ecdhQR || ecdhSCAN
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

    this.ws.send(JSON.stringify({
      op:'pair-init',
      to: peer.fp,
      from: this.me.fingerprint,
      name: this.me.deviceName,
      signSpki: this.me.signSpki,
      ecdhSpki: this.me.ecdhSpki,
      nonceQR: payload.nonce,   // <-- include the QR host's nonce from the link
      nonceB:  b64u(nonceScan), // <-- scanner's own nonce
      sig: b64u(sig)
    }));

    // Keep T so we can verify the ACK and show SAS on the scanner too
    this.pending.set('pairing', { peer, T });
  }

  async _onmsg(m) {
    // QR host receives scanner's init
    if (m.op === 'pair-init' && m.to === this.me.fingerprint) {
    const nonceQR = m.nonceQR ? b64uToBytes(m.nonceQR) : this._qrNonce;
    if (!nonceQR) return; // no nonce to verify with

      const nonceScan = b64uToBytes(m.nonceB);
      const tag = te.encode('NaptioPair-v1');

      // Same canonical transcript (QR-first)
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
        from:m.name,
        fp:m.from,
        sas,
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
        from:m.name,
        fp:m.from,
        sas,
        accept: () => {
          this._storeTrusted({ fp:m.from, name:m.name, signSpki:m.signSpki, ecdhSpki:m.ecdhSpki });
        },
        reject: () => {}
      });
      return;
    }

    // ===== Encrypted signaling + session setup (unchanged) =====
    if (m.op === 'relay' && m.to === this.me.fingerprint) {
      if (m.kind === 'eph-reply') { this._resolveWaiter(`eph-reply:${m.sid}`, m); return; }
      if (m.kind === 'eph-hello') { await this._handleEphHello(m); return; }
      if (m.kind === 'enc') {
        const s = this.sessions.get(m.sid); if (!s) return;
        try {
          const pt = await aesGcmDecrypt(s.key, b64uToBytes(m.iv), b64uToBytes(m.ct));
          const obj = JSON.parse(td.decode(pt));
          s.onmsg && s.onmsg(obj, m.from);
        } catch {}
        return;
      }
    }

    if (m.op === 'ping') this.ws.send(JSON.stringify({ op:'pong' }));
  }

  async _acceptPair(to, T, ctx) {
    const sig = new Uint8Array(await crypto.subtle.sign(
      { name:'ECDSA', hash:'SHA-256' },
      this.me.signKey,
      T
    ));
    this.ws.send(JSON.stringify({
      op:'pair-ack',
      to,
      from: this.me.fingerprint,
      name: this.me.deviceName,
      signSpki: this.me.signSpki,
      ecdhSpki: this.me.ecdhSpki,
      sig: b64u(sig)
    }));
    await this._storeTrusted({ fp:ctx.fp, name:ctx.name, signSpki:ctx.signSpki, ecdhSpki:ctx.ecdhSpki });
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

    const sess = { toFp, key:null, onmsg:null, role:'initiator' };
    this.sessions.set(sid, sess);

    const reply = await this._waitFor(`eph-reply:${sid}`, 15000, () => {
      this.ws.send(JSON.stringify({ op:'relay', to: peer.fp, from:this.me.fingerprint, kind:'eph-hello', sid, ephSpki: b64u(ephSpki), nonce: b64u(nonce), sig: b64u(sig) }));
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

    this.sessions.set(sid, { toFp: m.from, key: sessionKey, onmsg:null, role:'responder' });
    this.ws.send(JSON.stringify({ op:'relay', to: peer.fp, from:this.me.fingerprint, kind:'eph-reply', sid, ephSpki: b64u(ephSpki), nonce: b64u(nonce), sig: b64u(sig) }));
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
    this.ws.send(JSON.stringify({ op:'relay', to: s.toFp, from:this.me.fingerprint, kind:'enc', sid, iv: b64u(iv), ct: b64u(ct) }));
  }
  onEncrypted(sid, handler) {
    const s = this.sessions.get(sid); if (!s) throw new Error('Unknown sid');
    s.onmsg = handler;
  }

  // ===== tiny wait/resolve helper =====
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
