// /hungryface/webrtc/shared/psk/room-scope.js
// Derive a per-PSK namespace so that "Baby" on network A != "Baby" on network B.

const NAMESPACE_MSG = new TextEncoder().encode('naptio-room-namespace-v1');

/** base64url -> Uint8Array */
function b64uToBytes(b64u) {
  const clean = String(b64u).replace(/-/g, '+').replace(/_/g, '/');
  const pad = (4 - (clean.length % 4)) % 4;
  const withPad = clean + '===='.slice(0, pad);
  const bin = atob(withPad);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** Import the PSK token as an HMAC key */
async function importHmacKeyFromEnv(env) {
  const token = env?.tokenB64u;
  if (!token) {
    throw new Error('[room-scope] env.tokenB64u is missing – did requirePskOrRedirect succeed?');
  }

  const raw = b64uToBytes(token);
  return crypto.subtle.importKey(
    'raw',
    raw,
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign']
  );
}

/** Derive a short namespace hex string from the PSK token */
export async function deriveNamespace(env) {
  const key = await importHmacKeyFromEnv(env);
  const sig = new Uint8Array(await crypto.subtle.sign('HMAC', key, NAMESPACE_MSG));
  // use first 8 bytes => 16 hex chars
  let hex = '';
  const n = Math.min(8, sig.length);
  for (let i = 0; i < n; i++) {
    hex += sig[i].toString(16).padStart(2, '0');
  }
  return hex;
}

/** Normalise UI label into a slug */
export function slugifyRoom(uiName) {
  const raw = (uiName ?? '').toString().trim();
  if (!raw) return 'room';
  const cleaned = raw.toLowerCase().replace(/[^a-z0-9_-]+/g, '-');
  const slug = cleaned.replace(/^-+|-+$/g, '');
  return slug || 'room';
}

/** Compose final signaling room id */
export function scopeRoomId(namespaceHex, uiName) {
  const slug = slugifyRoom(uiName);
  return `${namespaceHex}:${slug}`;
}

/**
 * Main entry point:
 *   - derives namespace from env.tokenB64u
 *   - exposes window.__naptioRoomNamespace and window.__scopeRoom(uiRoomName)
 */
export async function installRoomScope(env) {
  if (env.redirected) {
    // Page is about to navigate away; nothing to do.
    return { namespace: null, scope: (x) => x };
  }

  const ns = await deriveNamespace(env);
  const scopeFn = (uiName) =>
    scopeRoomId(ns, uiName || env.room || 'Baby');

  window.__naptioRoomNamespace = ns;
  window.__scopeRoom = scopeFn;

  console.log('[room-scope] namespace =', ns, 'example scoped room =', scopeFn(env.room));

  return { namespace: ns, scope: scopeFn };
}
