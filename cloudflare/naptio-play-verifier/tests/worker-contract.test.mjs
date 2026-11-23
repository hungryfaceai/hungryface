import test from 'node:test';
import assert from 'node:assert/strict';
import { generateKeyPairSync } from 'node:crypto';
import worker from '../worker.js';

const { privateKey } = generateKeyPairSync('rsa', { modulusLength: 2048 });
const env = {
  GOOGLE_SERVICE_ACCOUNT_EMAIL: 'test@naptio.iam.gserviceaccount.com',
  GOOGLE_PRIVATE_KEY: privateKey.export({ type: 'pkcs8', format: 'pem' }),
  GOOGLE_PLAY_PACKAGE_NAME: 'io.github.hungryfaceai.twa',
  GOOGLE_PLAY_PRODUCT_ID: 'naptio_01',
};
const expiry = days => new Date(Date.now() + days * 86400000).toISOString();
const subscription = (state = 'SUBSCRIPTION_STATE_ACTIVE', expiresAt = expiry(10),
  acknowledgementState = 'ACKNOWLEDGEMENT_STATE_ACKNOWLEDGED', productId = 'naptio_01') => ({
  subscriptionState: state,
  acknowledgementState,
  lineItems: [{ productId, expiryTime: expiresAt }],
});
const reply = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

async function exercise(google, { lookupStatus = 200, acknowledgeStatus = 200 } = {}) {
  let acknowledged = 0;
  const oldFetch = globalThis.fetch;
  globalThis.fetch = async url => {
    const path = String(url);
    if (path.includes('oauth2.googleapis.com')) return reply({ access_token: 'test-oauth', expires_in: 3600 });
    if (path.includes(':acknowledge')) { acknowledged++; return reply({}, acknowledgeStatus); }
    if (path.includes('/subscriptionsv2/tokens/')) return reply(google, lookupStatus);
    throw new Error('Unexpected endpoint');
  };
  try {
    const request = new Request('https://test.workers.dev/google-play/verify', {
      method: 'POST',
      headers: { Origin: 'https://hungryfaceai.github.io', 'Content-Type': 'application/json' },
      body: JSON.stringify({ purchaseToken: 'test-token' }),
    });
    const http = await worker.fetch(request, env);
    return { body: await http.json(), status: http.status, acknowledged };
  } finally { globalThis.fetch = oldFetch; }
}

test('active verified subscription grants access', async () => {
  const r = await exercise(subscription());
  assert.equal(r.status, 200);
  assert.equal(r.body.status, 'ACTIVE');
  assert.equal(r.body.active, true);
  assert.equal(r.acknowledged, 0);
});
test('acknowledgement failure does not revoke verified entitlement', async () => {
  const r = await exercise(subscription('SUBSCRIPTION_STATE_ACTIVE', expiry(10),
    'ACKNOWLEDGEMENT_STATE_PENDING'), { acknowledgeStatus: 503 });
  assert.equal(r.body.status, 'ACTIVE');
  assert.equal(r.body.acknowledgement, 'PENDING_RETRY');
  assert.equal(r.body.active, true);
  assert.equal(r.acknowledged, 1);
});
test('pending acknowledgement succeeds independently of entitlement', async () => {
  const r = await exercise(subscription('SUBSCRIPTION_STATE_ACTIVE', expiry(10),
    'ACKNOWLEDGEMENT_STATE_PENDING'));
  assert.equal(r.body.status, 'ACTIVE');
  assert.equal(r.body.acknowledgement, 'ACKNOWLEDGED');
});
test('expired and mismatched subscriptions are inactive', async () => {
  assert.equal((await exercise(subscription('SUBSCRIPTION_STATE_ACTIVE', expiry(-1)))).body.status, 'INACTIVE');
  assert.equal((await exercise(subscription('SUBSCRIPTION_STATE_ACTIVE', expiry(10),
    'ACKNOWLEDGEMENT_STATE_ACKNOWLEDGED', 'other-product'))).body.status, 'INACTIVE');
});
test('pending and on-hold subscriptions do not grant access', async () => {
  assert.equal((await exercise(subscription('SUBSCRIPTION_STATE_PENDING'))).body.status, 'PENDING');
  assert.equal((await exercise(subscription('SUBSCRIPTION_STATE_ON_HOLD'))).body.status, 'INACTIVE');
});
test('permanently unknown token and Google service failure remain distinct', async () => {
  assert.equal((await exercise({}, { lookupStatus: 404 })).body.status, 'INVALID_PURCHASE');
  const unavailable = await exercise({}, { lookupStatus: 503 });
  assert.equal(unavailable.status, 503);
  assert.equal(unavailable.body.status, 'UNAVAILABLE');
});
test('malformed Google responses fail closed', async () => {
  const r = await exercise({});
  assert.equal(r.status, 503);
  assert.equal(r.body.active, false);
});
test('reject requests from unrelated origins without consulting Google', async () => {
  const request = new Request('https://test.workers.dev/google-play/verify', {
    method: 'POST', headers: { Origin: 'https://other.example', 'Content-Type': 'application/json' },
    body: '{"purchaseToken":"test-token"}',
  });
  assert.equal((await worker.fetch(request, env)).status, 403);
});
