
// /webrtc/shared/secure-signal.js (multi-session)
import { te, td, b64u, b64uToBytes, digestSHA256, concatBytes, rand, hkdf, aesGcmEncrypt, aesGcmDecrypt } from './pairing-crypto.js';
import { ensureIdentity, shortSAS } from './identity.js';

export class SecureSignal {
  constructor({ url, onPairPrompt, onTrustedList, onOpen, onSession }={}){
    this.url = url; this.ws = null;
    this.pending = new Map();
    this.trusted = [];
    this.sessions = new Map(); // sid -> { toFp, key, onmsg }
    this.onPairPrompt = onPairPrompt || (()=>{});
    this.onTrustedList = onTrustedList || (()=>{});
    this.onOpen = onOpen || (()=>{});
    this.onSession = onSession || (()=>{});
  }
  async init(){
    this.me = await ensureIdentity({});
    this.trusted = (await this._loadTrusted()) || [];
    this._connect();
  }
  _connect(){
    const ws = new WebSocket(this.url);
    this.ws = ws;
    ws.onopen = ()=>{
      ws.send(JSON.stringify({ op:'register', fp:this.me.fingerprint, device:this.me.deviceName }));
      this.onOpen();
      this._emitTrusted();
    };
    ws.onmessage = (e)=> this._onmsg(JSON.parse(e.data));
    ws.onclose = ()=> setTimeout(()=>this._connect(), 1200);
  }
  _emitTrusted(){ this.onTrustedList([...this.trusted]); }
  async _loadTrusted(){ const raw = localStorage.getItem('naptio-trusted-devices'); return raw ? JSON.parse(raw) : []; }
  async _saveTrusted(){ localStorage.setItem('naptio-trusted-devices', JSON.stringify(this.trusted)); }

  // Pairing
  makePairingPayload(){
    const nonce = rand(16);
    const payload = { v:'n1', name:this.me.deviceName, signSpki:this.me.signSpki, ecdhSpki:this.me.ecdhSpki, fp:this.me.fingerprint, nonce:b64u(nonce) };
    return { json: payload, b64: b64u(te.encode(JSON.stringify(payload))) };
  }
  async handlePairingLink(hash){
    const payload = JSON.parse(td.decode(b64uToBytes(hash)));
    const peer = { fp: payload.fp || await this._fpFromSpkis(payload.signSpki, payload.ecdhSpki), name: payload.name, signSpki: payload.signSpki, ecdhSpki: payload.ecdhSpki, nonceA: b64uToBytes(payload.nonce) };
    const nonceB = rand(16);
    const T = concatBytes(te.encode('NaptioPair-v1'), peer.nonceA, nonceB, b64uToBytes(this.me.signSpki), b64uToBytes(peer.signSpki), b64uToBytes(this.me.ecdhSpki), b64uToBytes(peer.ecdhSpki));
    const sig = new Uint8Array(await crypto.subtle.sign({name:'ECDSA', hash:'SHA-256'}, this.me.signKey, T));
    const msg = { op:'pair-init', to: peer.fp, from:this.me.fingerprint, name:this.me.deviceName, signSpki:this.me.signSpki, ecdhSpki:this.me.ecdhSpki, nonceB:b64u(nonceB), sig:b64u(sig) };
    this.ws.send(JSON.stringify(msg));
    this.pending.set('pairing', { peer, nonceB, T });
  }
  async _fpFromSpkis(signSpkiB64, ecdhSpkiB64){
    const fp = await digestSHA256(concatBytes(b64uToBytes(signSpkiB64), b64uToBytes(ecdhSpkiB64)));
    return b64u(fp.slice(0,9));
  }
  async _onmsg(m){
    if (m.op === 'pair-init' && m.to === this.me.fingerprint){
      const nonceA = rand(16);
      const T = concatBytes(te.encode('NaptioPair-v1'), b64uToBytes(m.nonceB), nonceA, b64uToBytes(m.signSpki), b64uToBytes(this.me.signSpki), b64uToBytes(m.ecdhSpki), b64uToBytes(this.me.ecdhSpki));
      const ok = await crypto.subtle.verify({name:'ECDSA', hash:'SHA-256'}, await crypto.subtle.importKey('spki', b64uToBytes(m.signSpki), {name:'ECDSA', namedCurve:'P-256'}, true, ['verify']), m.sig? b64uToBytes(m.sig):new Uint8Array(), T);
      if (!ok) return;
      const sas = shortSAS(await digestSHA256(T));
      const ctx = { fp:m.from, name:m.name, signSpki:m.signSpki, ecdhSpki:m.ecdhSpki, nonceA };
      this.pending.set('pair-acc', ctx);
      this.onPairPrompt({ type:'incoming', from:m.name, fp:m.from, sas, accept:()=>this._acceptPair(m.from, T, nonceA, ctx), reject:()=>{} });
      return;
    }
    if (m.op === 'pair-ack' && m.to === this.me.fingerprint){
      const st = this.pending.get('pairing'); if (!st) return;
      const T = st.T;
      const ok = await crypto.subtle.verify({name:'ECDSA', hash:'SHA-256'}, await crypto.subtle.importKey('spki', b64uToBytes(m.signSpki), {name:'ECDSA', namedCurve:'P-256'}, true, ['verify']), b64uToBytes(m.sig), T);
      if (!ok) return;
      const sas = shortSAS(await digestSHA256(T));
      this.onPairPrompt({ type:'confirm', from:m.name, fp:m.from, sas, accept:()=>{ this._storeTrusted({ fp:m.from, name:m.name, signSpki:m.signSpki, ecdhSpki:m.ecdhSpki }); }, reject:()=>{} });
      return;
    }
    if (m.op === 'relay' && m.to === this.me.fingerprint){
      if (m.kind === 'eph-reply'){
        const fn = this.pending.get(`eph-reply:${m.sid}`);
        if (fn){ fn(m); this.pending.delete(`eph-reply:${m.sid}`); }
        return;
      }
      if (m.kind === 'eph-hello'){
        await this._handleEphHello(m);
        return;
      }
      if (m.kind === 'enc'){
        const s = this.sessions.get(m.sid);
        if (!s) return;
        try {
          const pt = await aesGcmDecrypt(s.key, b64uToBytes(m.iv), b64uToBytes(m.ct));
          const obj = JSON.parse(td.decode(pt));
          s.onmsg && s.onmsg(obj, m.from);
        } catch {}
        return;
      }
      return;
    }
    if (m.op === 'ping') this.ws.send(JSON.stringify({op:'pong'}));
  }
  async _acceptPair(to, T, nonceA, ctx){
    const sig = new Uint8Array(await crypto.subtle.sign({name:'ECDSA', hash:'SHA-256'}, this.me.signKey, T));
    this.ws.send(JSON.stringify({ op:'pair-ack', to, from:this.me.fingerprint, name:this.me.deviceName, signSpki:this.me.signSpki, ecdhSpki:this.me.ecdhSpki, nonceA:b64u(nonceA), sig:b64u(sig) }));
    await this._storeTrusted({ fp:ctx.fp, name:ctx.name, signSpki:ctx.signSpki, ecdhSpki:ctx.ecdhSpki });
  }
  async _storeTrusted(dev){ this.trusted = this.trusted.filter(d=>d.fp!==dev.fp).concat([{...dev, addedAt:Date.now()}]); await this._saveTrusted(); this._emitTrusted(); }

  async startSession(toFp){
    const sid = b64u(rand(6));
    const peer = this.trusted.find(d=>d.fp===toFp); if (!peer) throw new Error('Unknown peer');
    const eph = await crypto.subtle.generateKey({ name:'ECDH', namedCurve:'P-256' }, true, ['deriveBits']);
    const ephSpki = new Uint8Array(await crypto.subtle.exportKey('spki', eph.publicKey));
    const nonce = rand(16);
    const tag = te.encode('naptio-ephemeral-v1');
    const sig = new Uint8Array(await crypto.subtle.sign({name:'ECDSA', hash:'SHA-256'}, this.me.signKey, concatBytes(tag, ephSpki, te.encode(peer.fp), nonce)));

    const sess = { toFp, key:null, onmsg:null, role:'initiator' };
    this.sessions.set(sid, sess);

    await new Promise((resolve, reject)=>{
      const key = `eph-reply:${sid}`;
      const t = setTimeout(()=>{ this.pending.delete(key); reject(new Error('timeout')); }, 15000);
      this.pending.set(key, (m)=>{ clearTimeout(t); resolve(m); });
      this.ws.send(JSON.stringify({ op:'relay', to: peer.fp, from:this.me.fingerprint, kind:'eph-hello', sid, ephSpki: b64u(ephSpki), nonce: b64u(nonce), sig: b64u(sig) }));
    }).then(async (reply)=>{
      const ok = await crypto.subtle.verify({name:'ECDSA', hash:'SHA-256'}, await crypto.subtle.importKey('spki', b64uToBytes(peer.signSpki), {name:'ECDSA', namedCurve:'P-256'}, true, ['verify']), b64uToBytes(reply.sig), concatBytes(tag, b64uToBytes(reply.ephSpki), te.encode(this.me.fingerprint), b64uToBytes(reply.nonce)));
      if (!ok) throw new Error('ephemeral verify failed');
      const sessionKey = await this._deriveSessionKey(eph.privateKey, b64uToBytes(reply.ephSpki), concatBytes(nonce, b64uToBytes(reply.nonce)));
      sess.key = sessionKey;
      this.onSession({ sid, peerFp: toFp, role:'initiator' });
    });

    return { sid, sessionKey: this.sessions.get(sid).key };
  }
  async _handleEphHello(m){
    const peer = this.trusted.find(d=>d.fp===m.from); if (!peer) return;
    const tag = te.encode('naptio-ephemeral-v1');
    const ok = await crypto.subtle.verify({name:'ECDSA', hash:'SHA-256'}, await crypto.subtle.importKey('spki', b64uToBytes(peer.signSpki), {name:'ECDSA', namedCurve:'P-256'}, true, ['verify']), b64uToBytes(m.sig), concatBytes(tag, b64uToBytes(m.ephSpki), te.encode(this.me.fingerprint), b64uToBytes(m.nonce)));
    if (!ok) return;
    const sid = m.sid;
    const eph = await crypto.subtle.generateKey({ name:'ECDH', namedCurve:'P-256' }, true, ['deriveBits']);
    const ephSpki = new Uint8Array(await crypto.subtle.exportKey('spki', eph.publicKey));
    const nonce = rand(16);
    const sig = new Uint8Array(await crypto.subtle.sign({name:'ECDSA', hash:'SHA-256'}, this.me.signKey, concatBytes(tag, ephSpki, te.encode(peer.fp), nonce)));
    const sessionKey = await this._deriveSessionKey(eph.privateKey, b64uToBytes(m.ephSpki), concatBytes(b64uToBytes(m.nonce), nonce));
    this.sessions.set(sid, { toFp: m.from, key: sessionKey, onmsg:null, role:'responder' });
    this.ws.send(JSON.stringify({ op:'relay', to: peer.fp, from:this.me.fingerprint, kind:'eph-reply', sid, ephSpki: b64u(ephSpki), nonce: b64u(nonce), sig: b64u(sig) }));
    this.onSession({ sid, peerFp: m.from, role:'responder' });
  }
  async _deriveSessionKey(privEcdh, peerSpkiBytes, salt){
    const peerPub = await crypto.subtle.importKey('spki', peerSpkiBytes, {name:'ECDH', namedCurve:'P-256'}, true, []);
    const bits = await crypto.subtle.deriveBits({ name:'ECDH', public: peerPub }, privEcdh, 256);
    return hkdf(new Uint8Array(bits), { salt, info:'naptio-sdp-v1' });
  }

  async sendJSON(sid, obj){
    const s = this.sessions.get(sid); if (!s || !s.key) throw new Error('Missing session');
    const { iv, ct } = await aesGcmEncrypt(s.key, te.encode(JSON.stringify(obj)));
    this.ws.send(JSON.stringify({ op:'relay', to: s.toFp, from:this.me.fingerprint, kind:'enc', sid, iv: b64u(iv), ct: b64u(ct) }));
  }
  onEncrypted(sid, handler){
    const s = this.sessions.get(sid); if (!s) throw new Error('Unknown sid');
    s.onmsg = handler;
  }
}
