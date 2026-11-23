// Naptio Cloudflare Worker source; deploying GitHub Pages does not deploy this Worker.
// Contract (all replies are no-store):
// 200 { status: 'ACTIVE', active: true, ..., acknowledgement: 'ACKNOWLEDGED'|'PENDING_RETRY' }
// 200 { status: 'INACTIVE'|'PENDING'|'INVALID_PURCHASE', active: false, reason, ... }
// 503 { status: 'UNAVAILABLE', active: false, retryable: true, reason }
// 429 { status: 'UNAVAILABLE', active: false, retryable: true, reason: 'RATE_LIMITED' }
// Malformed requests use 4xx and cannot grant entitlement.

const ALLOWED_ORIGIN = 'https://hungryfaceai.github.io';
const GOOGLE_TOKEN_URL = 'https://oauth2.googleapis.com/token';
const GOOGLE_SCOPE = 'https://www.googleapis.com/auth/androidpublisher';
const GOOGLE_API_ROOT = 'https://androidpublisher.googleapis.com/androidpublisher/v3';
const MAX_BODY_BYTES = 8192;
const GOOGLE_TIMEOUT_MS = 8000;

const GRANTING_STATES = new Set([
  'SUBSCRIPTION_STATE_ACTIVE',
  'SUBSCRIPTION_STATE_IN_GRACE_PERIOD',
  'SUBSCRIPTION_STATE_CANCELED',
]);
const DENYING_STATES = new Set([
  'SUBSCRIPTION_STATE_PAUSED',
  'SUBSCRIPTION_STATE_ON_HOLD',
  'SUBSCRIPTION_STATE_EXPIRED',
  'SUBSCRIPTION_STATE_PENDING_PURCHASE_CANCELED',
]);

let cachedGoogleToken = null;
let cachedGoogleTokenExpiry = 0;
let googleTokenPromise = null;

class GoogleHttpError extends Error {
  constructor(status, reason = '') {
    super(`Google API returned HTTP ${status}`);
    this.status = status;
    this.reason = reason;
  }
}

export default {
  async fetch(request, env) {
    const { pathname } = new URL(request.url);
    if (request.method === 'GET' && pathname === '/') {
      return json({ ok: true, service: 'naptio-play-verifier' });
    }
    if (pathname !== '/google-play/verify') {
      return json({ error: 'Not found' }, 404);
    }

    const origin = request.headers.get('Origin') || '';
    if (request.method === 'OPTIONS') {
      if (origin !== ALLOWED_ORIGIN) return json({ error: 'Origin not allowed' }, 403);
      return new Response(null, { status: 204, headers: corsHeaders(origin) });
    }
    if (request.method !== 'POST') {
      return json({ error: 'Method not allowed' }, 405, { Allow: 'POST, OPTIONS' });
    }
    if (origin !== ALLOWED_ORIGIN) return json({ error: 'Origin not allowed' }, 403);

    const responseHeaders = corsHeaders(origin);
    if (!/^application\/json(?:\s*;|\s*$)/i.test(request.headers.get('Content-Type') || '')) {
      return json({ error: 'Content-Type must be application/json' }, 415, responseHeaders);
    }

    let body;
    try {
      const raw = await readBodyLimited(request, MAX_BODY_BYTES);
      body = JSON.parse(raw);
    } catch (error) {
      return json(
        { error: error?.message === 'BODY_TOO_LARGE' ? 'Request too large' : 'Invalid JSON body' },
        error?.message === 'BODY_TOO_LARGE' ? 413 : 400,
        responseHeaders
      );
    }
    const purchaseToken = typeof body?.purchaseToken === 'string' ? body.purchaseToken.trim() : '';
    if (!purchaseToken || purchaseToken.length > 4096) {
      return json({ status: 'INVALID_PURCHASE', active: false, reason: 'BAD_TOKEN', retryable: false }, 400, responseHeaders);
    }

    // Optional binding; MUST be configured separately to provide rate protection.
    // Cloudflare rate limits are approximate/distributed, not a hard global Google quota.
    if (env.PLAY_VERIFY_LIMITER) {
      try {
        const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
        const limit = await env.PLAY_VERIFY_LIMITER.limit({ key: ip });
        if (!limit.success) {
          return json(unavailable('RATE_LIMITED'), 429, {
            ...responseHeaders,
            'Retry-After': '30',
          });
        }
      } catch {
        return json(unavailable('RATE_LIMITER_ERROR'), 503, responseHeaders);
      }
    }

    try {
      requireSecrets(env);
      let effectiveToken = purchaseToken;
      let subscription = await getSubscription(env, effectiveToken);
      // Google documents this special case: a cancelled pending replacement
      // purchase can point to a still-entitled earlier subscription.
      if (subscription?.subscriptionState === 'SUBSCRIPTION_STATE_PENDING_PURCHASE_CANCELED' &&
          typeof subscription.linkedPurchaseToken === 'string' &&
          subscription.linkedPurchaseToken &&
          subscription.linkedPurchaseToken !== effectiveToken) {
        effectiveToken = subscription.linkedPurchaseToken;
        try {
          subscription = await getSubscription(env, effectiveToken);
        } catch (error) {
          if (error instanceof GoogleHttpError && (error.status === 404 || error.status === 410)) {
            return json({ status: 'INACTIVE', active: false,
              reason: 'LINKED_PURCHASE_NOT_FOUND', retryable: false }, 200, responseHeaders);
          }
          throw error;
        }
      }
      const entitlement = evaluateSubscription(subscription, env.GOOGLE_PLAY_PRODUCT_ID);

      if (entitlement.status !== 'ACTIVE') {
        return json(entitlement, 200, responseHeaders);
      }

      // Acknowledgement is separate from the already-verified entitlement.
      // A failure must NOT replace ACTIVE with UNAVAILABLE, but it MUST be
      // monitored and retried on subsequent verification / via RTDN in future.
      if (entitlement.acknowledgement === 'PENDING_RETRY') {
        try {
          await acknowledgeSubscription(env, effectiveToken);
          entitlement.acknowledgement = 'ACKNOWLEDGED';
        } catch (error) {
          // No request body, token, or Google error body is logged.
          console.error('[acknowledge]', safeErrorMessage(error));
        }
      }
      return json(entitlement, 200, responseHeaders);
    } catch (error) {
      if (error instanceof GoogleHttpError && (error.status === 404 || error.status === 410)) {
        return json({
          status: 'INVALID_PURCHASE', active: false,
          reason: 'TOKEN_NOT_FOUND', retryable: false,
        }, 200, responseHeaders);
      }
      console.error('[verify]', safeErrorMessage(error));
      return json(unavailable('VERIFICATION_FAILED'), 503, {
        ...responseHeaders,
        'Retry-After': '15',
      });
    }
  },
};

function evaluateSubscription(subscription, expectedProduct) {
  if (!subscription || typeof subscription !== 'object' || Array.isArray(subscription) ||
      !Array.isArray(subscription.lineItems) ||
      typeof subscription.subscriptionState !== 'string') {
    throw new Error('UNEXPECTED_GOOGLE_RESPONSE');
  }
  const state = subscription.subscriptionState;
  if (!GRANTING_STATES.has(state) && !DENYING_STATES.has(state) &&
      state !== 'SUBSCRIPTION_STATE_PENDING') {
    throw new Error('UNKNOWN_SUBSCRIPTION_STATE');
  }

  const matching = subscription.lineItems.filter(item => item?.productId === expectedProduct);
  if (matching.length === 0) {
    return { status: 'INACTIVE', active: false, reason: 'PRODUCT_MISMATCH', state, retryable: false };
  }

  // A malformed expiry on any matching line item cannot authorize a customer.
  const expiries = matching.map(item => {
    if (typeof item.expiryTime !== 'string') return NaN;
    return Date.parse(item.expiryTime);
  });
  const validExpiries = expiries.filter(Number.isFinite);
  const latestExpiryMs = validExpiries.length ? Math.max(...validExpiries) : 0;
  const expiresAt = latestExpiryMs ? new Date(latestExpiryMs).toISOString() : null;

  const common = { state, expiresAt, testPurchase: !!subscription.testPurchase };
  if (state === 'SUBSCRIPTION_STATE_PENDING') {
    return { status: 'PENDING', active: false, reason: 'PAYMENT_PENDING', retryable: true, ...common };
  }
  if (DENYING_STATES.has(state)) {
    return { status: 'INACTIVE', active: false, reason: state, retryable: false, ...common };
  }
  // GRANTING_STATES require a trustworthy expiry and acknowledgement state.
  if (validExpiries.length !== matching.length || latestExpiryMs === 0) {
    throw new Error('INVALID_SUBSCRIPTION_EXPIRY');
  }
  const ack = subscription.acknowledgementState;
  if (ack !== 'ACKNOWLEDGEMENT_STATE_ACKNOWLEDGED' &&
      ack !== 'ACKNOWLEDGEMENT_STATE_PENDING') {
    throw new Error('UNKNOWN_ACKNOWLEDGEMENT_STATE');
  }
  if (latestExpiryMs <= Date.now()) {
    return { status: 'INACTIVE', active: false, reason: 'EXPIRED', retryable: false, ...common };
  }
  return {
    status: 'ACTIVE', active: true, reason: 'ENTITLED', retryable: false,
    acknowledgement: ack === 'ACKNOWLEDGEMENT_STATE_ACKNOWLEDGED'
      ? 'ACKNOWLEDGED' : 'PENDING_RETRY',
    ...common,
  };
}

function unavailable(reason) {
  return { status: 'UNAVAILABLE', active: false, reason, retryable: true };
}

async function readBodyLimited(request, maxBytes) {
  const length = Number(request.headers.get('Content-Length') || 0);
  if (Number.isFinite(length) && length > maxBytes) throw new Error('BODY_TOO_LARGE');
  if (!request.body) return '';
  const reader = request.body.getReader();
  const decoder = new TextDecoder();
  let received = 0, text = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      received += value.byteLength;
      if (received > maxBytes) throw new Error('BODY_TOO_LARGE');
      text += decoder.decode(value, { stream: true });
    }
    return text + decoder.decode();
  } finally {
    reader.releaseLock();
    if (received > maxBytes) request.body.cancel?.().catch?.(() => {});
  }
}

function requireSecrets(env) {
  for (const name of ['GOOGLE_SERVICE_ACCOUNT_EMAIL', 'GOOGLE_PRIVATE_KEY',
    'GOOGLE_PLAY_PACKAGE_NAME', 'GOOGLE_PLAY_PRODUCT_ID']) {
    if (!env[name]) throw new Error(`MISSING_SECRET_${name}`);
  }
}

async function getGoogleAccessToken(env) {
  const now = Math.floor(Date.now() / 1000);
  if (cachedGoogleToken && cachedGoogleTokenExpiry > now + 60) return cachedGoogleToken;
  if (!googleTokenPromise) {
    googleTokenPromise = exchangeGoogleToken(env).finally(() => { googleTokenPromise = null; });
  }
  return googleTokenPromise;
}

async function exchangeGoogleToken(env) {
  const now = Math.floor(Date.now() / 1000);
  const assertion = await createSignedJwt(env, now);
  const { response, data } = await fetchJson(GOOGLE_TOKEN_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'urn:ietf:params:oauth:grant-type:jwt-bearer', assertion,
    }),
  });
  if (!response.ok || typeof data?.access_token !== 'string' || !data.access_token) {
    throw new Error(`GOOGLE_OAUTH_HTTP_${response.status}`);
  }
  cachedGoogleToken = data.access_token;
  cachedGoogleTokenExpiry = now + (Number.isFinite(Number(data.expires_in))
    ? Number(data.expires_in) : 3600);
  return cachedGoogleToken;
}

function invalidateTokenIfSame(token) {
  if (cachedGoogleToken === token) {
    cachedGoogleToken = null;
    cachedGoogleTokenExpiry = 0;
  }
}

async function googleApiRequest(env, url, options = {}) {
  for (let attempt = 0; attempt < 2; attempt++) {
    const token = await getGoogleAccessToken(env);
    const { response, data } = await fetchJson(url, {
      ...options,
      headers: {
        ...options.headers,
        Authorization: `Bearer ${token}`,
        Accept: 'application/json',
      },
    });
    if (response.status === 401 && attempt === 0) {
      invalidateTokenIfSame(token);
      continue;
    }
    if (!response.ok) {
      const reason = typeof data?.error?.errors?.[0]?.reason === 'string'
        ? data.error.errors[0].reason : '';
      throw new GoogleHttpError(response.status, reason);
    }
    return data;
  }
  throw new Error('GOOGLE_AUTH_RETRY_EXHAUSTED');
}

async function getSubscription(env, purchaseToken) {
  const endpoint = `${GOOGLE_API_ROOT}/applications/` +
    `${encodeURIComponent(env.GOOGLE_PLAY_PACKAGE_NAME)}/purchases/` +
    `subscriptionsv2/tokens/${encodeURIComponent(purchaseToken)}`;
  return googleApiRequest(env, endpoint);
}

async function acknowledgeSubscription(env, purchaseToken) {
  const endpoint = `${GOOGLE_API_ROOT}/applications/` +
    `${encodeURIComponent(env.GOOGLE_PLAY_PACKAGE_NAME)}/purchases/subscriptions/` +
    `${encodeURIComponent(env.GOOGLE_PLAY_PRODUCT_ID)}/tokens/` +
    `${encodeURIComponent(purchaseToken)}:acknowledge`;
  await googleApiRequest(env, endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: '{}',
  });
}

async function fetchJson(url, options) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), GOOGLE_TIMEOUT_MS);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    const raw = await response.text();
    if (raw.length > 100_000) throw new Error('GOOGLE_RESPONSE_TOO_LARGE');
    let data = null;
    if (raw) {
      try { data = JSON.parse(raw); }
      catch { throw new Error('MALFORMED_GOOGLE_JSON'); }
    }
    return { response, data };
  } finally {
    clearTimeout(timer);
  }
}

async function createSignedJwt(env, now) {
  const encode = value => base64UrlFromBytes(new TextEncoder().encode(JSON.stringify(value)));
  const header = encode({ alg: 'RS256', typ: 'JWT' });
  const claims = encode({
    iss: env.GOOGLE_SERVICE_ACCOUNT_EMAIL,
    scope: GOOGLE_SCOPE,
    aud: GOOGLE_TOKEN_URL,
    iat: now,
    exp: now + 3600,
  });
  const unsigned = `${header}.${claims}`;
  const key = await importPrivateKey(env.GOOGLE_PRIVATE_KEY);
  const signature = await crypto.subtle.sign('RSASSA-PKCS1-v1_5', key,
    new TextEncoder().encode(unsigned));
  return `${unsigned}.${base64UrlFromBytes(new Uint8Array(signature))}`;
}

async function importPrivateKey(pem) {
  const cleaned = pem.replace(/\\n/g, '\n')
    .replace(/-----BEGIN PRIVATE KEY-----/g, '')
    .replace(/-----END PRIVATE KEY-----/g, '')
    .replace(/\s+/g, '');
  const bytes = Uint8Array.from(atob(cleaned), char => char.charCodeAt(0));
  return crypto.subtle.importKey('pkcs8', bytes.buffer,
    { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' }, false, ['sign']);
}

function base64UrlFromBytes(bytes) {
  let raw = '';
  for (const byte of bytes) raw += String.fromCharCode(byte);
  return btoa(raw).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
}

function corsHeaders(origin) {
  return {
    'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
    Vary: 'Origin',
  };
}

function json(value, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff',
      ...extraHeaders,
    },
  });
}

function safeErrorMessage(error) {
  // No URL (which would contain the token), request body, Google response body,
  // or secret is included in any error we construct.
  if (error instanceof GoogleHttpError) {
    return `Google API HTTP ${error.status}, reason=${error.reason || 'unspecified'}`;
  }
  return String(error?.message || 'UNKNOWN_ERROR').slice(0, 120);
}