
// /webrtc/shared/identity.js
import { te, td, b64u, b64uToBytes, digestSHA256, concatBytes } from './pairing-crypto.js';

const DB_NAME = 'naptio-ids';
const STORE = 'kvs';

function openDB(){
  return new Promise((res,rej)=>{
    const req = indexedDB.open(DB_NAME,1);
    req.onupgradeneeded = ()=> req.result.createObjectStore(STORE);
    req.onerror = ()=> rej(req.error);
    req.onsuccess = ()=> res(req.result);
  });
}
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
  const fingerprint = b64u(fpBytes.slice(0,9)); // 72 bits
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

// Numeric SAS: 6 digits shown as XXX-XXX (derived from the transcript hash)
export function shortSAS(hash, opts = {}) {
  const fmt = opts.format || 'numeric6';
  const b = new Uint8Array(hash);

  if (fmt === 'numeric9') {
    // 9 digits → three groups (XXX-XXX-XXX) using 5 bytes (~30 bits)
    const n = ((b[0] << 24) >>> 0) ^ ((b[1] << 16) >>> 0) ^ ((b[2] << 8) >>> 0) ^ (b[3] >>> 0) ^ ((b[4] & 0x7F) << 1);
    const s = (n % 1_000_000_000).toString().padStart(9, '0');
    return `${s.slice(0,3)}-${s.slice(3,6)}-${s.slice(6,9)}`;
  }

  // default: 6 digits → two groups (XXX-XXX) using first 3 bytes (~20 bits)
  const n = ((b[0] << 16) | (b[1] << 8) | b[2]) >>> 0;
  const s = (n % 1_000_000).toString().padStart(6, '0');
  return `${s.slice(0,3)}-${s.slice(3,6)}`;
}

