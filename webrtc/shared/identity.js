// /webrtc/shared/identity.js
async function dbGet(key){ const db=await openDB(); return new Promise((res,rej)=>{ const tx=db.transaction(STORE); const req=tx.objectStore(STORE).get(key); req.onsuccess=()=>res(req.result); req.onerror=()=>rej(req.error); }); }
async function dbPut(key,val){ const db=await openDB(); return new Promise((res,rej)=>{ const tx=db.transaction(STORE,'readwrite'); tx.objectStore(STORE).put(val,key); tx.oncomplete=()=>res(); tx.onerror=()=>rej(tx.error); }); }


async function exportSpki(pub){ return new Uint8Array(await crypto.subtle.exportKey('spki', pub)); }
async function exportJwk(priv){ return await crypto.subtle.exportKey('jwk', priv); }
async function importJwkECDSA(jwk){ return crypto.subtle.importKey('jwk', jwk, {name:'ECDSA', namedCurve:'P-256'}, true, ['sign']); }
async function importJwkECDH(jwk){ return crypto.subtle.importKey('jwk', jwk, {name:'ECDH', namedCurve:'P-256'}, true, ['deriveBits']); }


export async function ensureIdentity({deviceName}={}){
let rec = await dbGet('identity');
if (rec){
const signKey = await importJwkECDSA(rec.signPrivJwk);
const ecdhKey = await importJwkECDH(rec.ecdhPrivJwk);
return { ...rec, signKey, ecdhKey };
}
const signPair = await crypto.subtle.generateKey({ name:'ECDSA', namedCurve:'P-256' }, true, ['sign','verify']);
const ecdhPair = await crypto.subtle.generateKey({ name:'ECDH', namedCurve:'P-256' }, true, ['deriveBits']);
const signSpki = await exportSpki(signPair.publicKey);
const ecdhSpki = await exportSpki(ecdhPair.publicKey);
const fpBytes = await digestSHA256(concatBytes(signSpki, ecdhSpki));
const fingerprint = b64u(fpBytes.slice(0,9)); // 72 bits → short code like "cG9Ab1fF"
const createdAt = Date.now();
const recNew = {
version:'n1', deviceName: deviceName || (navigator.userAgentData?.platform || navigator.platform || 'Browser'),
fingerprint, signSpki: b64u(signSpki), ecdhSpki: b64u(ecdhSpki),
signPrivJwk: await exportJwk(signPair.privateKey),
ecdhPrivJwk: await exportJwk(ecdhPair.privateKey), createdAt
};
await dbPut('identity', recNew);
return { ...recNew, signKey: signPair.privateKey, ecdhKey: ecdhPair.privateKey };
}


export function shortSAS(bytes){
// map first 15 bits → 2 emojis (1024‑emoji set) + 6 digits (optional)
const n = (bytes[0]<<7) | (bytes[1]>>1); // 15 bits
const EMO = ['😀','😁','😂','🤣','😃','😄','😅','😆','😉','😊','😋','😎','😍','😘','😗','😙','😚','🙂','🤗','🤩','🤔','🤨','😐','😑','😶','🙄','😏','😣','😥','😮','🤐','😯','😪','😫','😴','😌','😛','😜','😝','🤤','😒','😓','😔','😕','🙃','🫠','🤑','😲','☹️','🙁','😖','😞','😟','😤','😢','😭','😦','😧','😨','😩','🤯','😬','😮‍💨','😰','😱','🥵','🥶','😳','🤪','😵','🥴','😠','😡','🤬','🤥','🤫','🤭','🫢','🫣','🤗','🤔','🤨','🫤','🤓','🧐','😇','🥳','🥸','😺','😸','😹','😻','😼','😽','🙀','😿','😾'];
const a = EMO[n % EMO.length], b = EMO[(n*7) % EMO.length];
const digits = (('000000'+(bytes[2] * 257 % 1000000)).slice(-6));
return `${a}${b} ${digits}`;
}
