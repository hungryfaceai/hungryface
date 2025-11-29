// webrtc/shared/psk/room-scope.js

const NAMESPACE_MSG = new TextEncoder().encode('naptio-room-namespace-v1');

/**
 * Pick the CryptoKey we can use for HMAC from the PSK env.
 * Adjust this if your requirePskOrRedirect(env) exposes it under a different name.
 */
function pickHmacKey(env) {
  // Try a few likely names; tweak as needed based on your real env shape.
  return (
    env?.roomHmacKey ||   // if you already have a dedicated HMAC key
    env?.hmacKey     ||   // generic HMAC key
    env?.pskKey      ||   // PSK key
    env?.key             // fallback
  );
}

/**
 * Derive a short hex namespace from the PSK (CryptoKey).
 */
export async function deriveNamespace(env) {
  const key = pickHmacKey(env);
  if (!key) {
    throw new Error('[room-scope] No HMAC CryptoKey found on env (expected env.roomHmacKey / env.hmacKey / env.pskKey)');
  }

  const sig = new Uint8Array(await crypto.subtle.sign('HMAC', key, NAMESPACE_MSG));
  // Use first 8 bytes => 16 hex chars; short but collision-safe for our use.
  let hex = '';
  const len = Math.min(8, sig.length);
  for (let i = 0; i < len; i++) {
    hex += sig[i].toString(16).padStart(2, '0');
  }
  return hex; // e.g. "a1b2c3d4e5f6a7b8"
}

/**
 * Normalise the user-facing room name into a safe slug.
 */
export function slugifyRoom(uiName) {
  const raw = (uiName ?? '').toString().trim();
  if (!raw) return 'room';
  const cleaned = raw.toLowerCase().replace(/[^a-z0-9_-]+/g, '-');
  const slug = cleaned.replace(/^-+|-+$/g, '');
  return slug || 'room';
}

/**
 * Build the actual signaling room id from namespace + UI name.
 */
export function scopeRoomId(namespaceHex, uiName) {
  const slug = slugifyRoom(uiName);
  return `${namespaceHex}:${slug}`;
}

/**
 * Main entry point for pages:
 * - derives namespace from env (PSK)
 * - installs helpers on window:
 *      window.__naptioRoomNamespace
 *      window.__scopeRoom(uiRoomName)
 */
export async function installRoomScope(env) {
  const ns = await deriveNamespace(env);
  const scopeFn = (uiName) =>
    scopeRoomId(ns, uiName || env.room || 'Baby');

  // Expose to other <script type="module"> blocks on the same page
  window.__naptioRoomNamespace = ns;
  window.__scopeRoom = scopeFn;

  return { namespace: ns, scope: scopeFn };
}
