// /webrtc/shared/secure-signal.js (multi-session)


// prepare session record
const sess = { toFp, key:null, onmsg:null, role:'initiator' };
this.sessions.set(sid, sess);


// send hello & wait for reply
const reply = await this._waitFor(`eph-reply:${sid}`, 15000, ()=>{
this.ws.send(JSON.stringify({ op:'relay', to: peer.fp, from:this.me.fingerprint, kind:'eph-hello', sid, ephSpki: b64u(ephSpki), nonce: b64u(nonce), sig: b64u(sig) }));
});
// verify reply
const ok = await crypto.subtle.verify({name:'ECDSA', hash:'SHA-256'}, await crypto.subtle.importKey('spki', b64uToBytes(peer.signSpki), {name:'ECDSA', namedCurve:'P-256'}, true, ['verify']), b64uToBytes(reply.sig), concatBytes(tag, b64uToBytes(reply.ephSpki), te.encode(this.me.fingerprint), b64uToBytes(reply.nonce)));
if (!ok) throw new Error('ephemeral verify failed');


const sessionKey = await this._deriveSessionKey(eph.privateKey, b64uToBytes(reply.ephSpki), concatBytes(nonce, b64uToBytes(reply.nonce)));
sess.key = sessionKey;
this.onSession({ sid, peerFp: toFp, role:'initiator' });
return { sid, sessionKey };
}
async _handleEphHello(m){
const peer = this.trusted.find(d=>d.fp===m.from); if (!peer) return; // ignore unknown
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


// === Encrypted signaling (per session) ===
async sendJSON(sid, obj){
const s = this.sessions.get(sid); if (!s || !s.key) throw new Error('Missing session');
const { iv, ct } = await aesGcmEncrypt(s.key, te.encode(JSON.stringify(obj)));
this.ws.send(JSON.stringify({ op:'relay', to: s.toFp, from:this.me.fingerprint, kind:'enc', sid, iv: b64u(iv), ct: b64u(ct) }));
}
onEncrypted(sid, handler){
const s = this.sessions.get(sid); if (!s) throw new Error('Unknown sid');
s.onmsg = handler;
}


// === tiny wait/resolve helper ===
_waitFor(key, timeout, firstSend){
return new Promise((resolve, reject)=>{
const t = setTimeout(()=>{ this.pending.delete(key); reject(new Error('timeout')); }, timeout||10000);
this.pending.set(key, (m)=>{ clearTimeout(t); this.pending.delete(key); resolve(m); });
firstSend && firstSend();
});
}
_resolveWaiter(key, payload){ const fn = this.pending.get(key); if (fn){ fn(payload); } }
}
