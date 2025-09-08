
// /webrtc/shared/pairing-crypto.js
export const te = new TextEncoder();
export const td = new TextDecoder();

export function b64u(bytes) {
  let s = btoa(String.fromCharCode(...bytes));
  return s.replace(/\+/g,'-').replace(/\//g,'_').replace(/=+$/,'');
}
export function b64uToBytes(s) {
  s = s.replace(/-/g,'+').replace(/_/g,'/');
  while (s.length % 4) s += '=';
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i=0;i<bin.length;i++) out[i]=bin.charCodeAt(i);
  return out;
}
export async function digestSHA256(bytes){
  const h = await crypto.subtle.digest('SHA-256', bytes);
  return new Uint8Array(h);
}
export function concatBytes(...arrs){
  const len = arrs.reduce((t,a)=>t+a.length,0);
  const out = new Uint8Array(len); let o=0;
  for (const a of arrs){ out.set(a,o); o+=a.length; }
  return out;
}
export function rand(n){ return crypto.getRandomValues(new Uint8Array(n)); }

export async function hkdf(bytes, {salt=new Uint8Array(), info=''}={}){
  const keyMat = await crypto.subtle.importKey('raw', bytes, 'HKDF', false, ['deriveKey']);
  return crypto.subtle.deriveKey(
    { name:'HKDF', hash:'SHA-256', salt, info: te.encode(info) },
    keyMat,
    { name:'AES-GCM', length:256 },
    false, ['encrypt','decrypt']
  );
}
export async function aesGcmEncrypt(aesKey, ptBytes){
  const iv = rand(12);
  const ct = await crypto.subtle.encrypt({ name:'AES-GCM', iv }, aesKey, ptBytes);
  return { iv, ct: new Uint8Array(ct) };
}
export async function aesGcmDecrypt(aesKey, iv, ctBytes){
  const pt = await crypto.subtle.decrypt({ name:'AES-GCM', iv }, aesKey, ctBytes);
  return new Uint8Array(pt);
}
