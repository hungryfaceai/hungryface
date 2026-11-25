import test from 'node:test';
import assert from 'node:assert/strict';

let nextId = 0;
const trialProduct = {
  itemId: 'naptio_01',
  title: 'Naptio monthly subscription',
  type: 'subscription',
  price: { currency: 'GBP', value: '1.79' },
  subscriptionPeriod: 'P1M',
  freeTrialPeriod: 'P3D',
};
const active = () => ({
  status: 'ACTIVE', active: true, reason: 'ENTITLED',
  acknowledgement: 'ACKNOWLEDGED', retryable: false,
});
const inactive = () => ({
  status: 'INACTIVE', active: false, reason: 'EXPIRED', retryable: false,
});
const unavailable = () => ({
  status: 'UNAVAILABLE', active: false, reason: 'VERIFICATION_FAILED', retryable: true,
});
const invalid = () => ({
  status: 'INVALID_PURCHASE', active: false, reason: 'TOKEN_NOT_FOUND', retryable: false,
});

async function environment(t, { product = trialProduct, purchases = [],
  verifier = () => active(), responseToken = 'new-purchase-token',
  checkoutError = null, billingBridge = 'ready',
  paymentRequestAvailable = true, twa = true } = {}) {
  const original = {
    navigator: Object.getOwnPropertyDescriptor(globalThis, 'navigator'),
    window: globalThis.window,
    document: globalThis.document,
    sessionStorage: globalThis.sessionStorage,
    fetch: globalThis.fetch,
    PaymentRequest: globalThis.PaymentRequest,
  };
  t.after(() => {
    if (original.navigator) Object.defineProperty(globalThis, 'navigator', original.navigator);
    else delete globalThis.navigator;
    for (const key of ['window', 'document', 'sessionStorage', 'fetch', 'PaymentRequest']) {
      if (original[key] === undefined) delete globalThis[key];
      else globalThis[key] = original[key];
    }
  });
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true, value: { userAgent: 'Android; Chrome', language: 'en-GB' },
  });
  globalThis.document = { referrer: twa ? 'android-app://io.github.hungryfaceai.twa' : '' };
  const store = new Map();
  globalThis.sessionStorage = {
    setItem: (k, v) => store.set(k, v),
    getItem: k => store.get(k) ?? null,
  };
  const counts = { checkout: 0, completion: [], verify: 0, list: 0, service: 0 };
  let currentBridge = billingBridge;
  const fullService = {
    getDetails: async () => [product],
    listPurchases: async () => {
      counts.list++;
      return typeof purchases === 'function' ? purchases() : purchases;
    },
  };
  globalThis.window = {};
  if (currentBridge !== 'missing') {
    globalThis.window.getDigitalGoodsService = async () => {
      counts.service++;
      if (currentBridge === 'reject') throw new Error('Temporary billing bridge error');
      if (currentBridge === 'null') return null;
      if (currentBridge === 'incomplete') return { getDetails: fullService.getDetails };
      return fullService;
    };
  }
  if (paymentRequestAvailable) {
    globalThis.PaymentRequest = class {
      async show() {
        counts.checkout++;
        if (checkoutError) throw checkoutError;
        return {
          details: { purchaseToken: responseToken },
          complete: async result => { counts.completion.push(result); },
        };
      }
    };
  } else {
    globalThis.PaymentRequest = undefined;
  }
  let currentVerifier = verifier;
  globalThis.fetch = async () => {
    counts.verify++;
    const data = currentVerifier();
    return new Response(JSON.stringify(data), {
      status: data.status === 'UNAVAILABLE' ? 503 : 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };
  const module = await import(new URL('../subscription-android.js?test=' + ++nextId, import.meta.url));
  return {
    module, counts,
    setVerifier: fn => { currentVerifier = fn; },
    setBillingBridge: next => {
      currentBridge = next;
      if (next === 'missing') delete globalThis.window.getDigitalGoodsService;
      else {
        globalThis.window.getDigitalGoodsService = async () => {
          counts.service++;
          if (currentBridge === 'reject') throw new Error('Temporary billing bridge error');
          if (currentBridge === 'null') return null;
          if (currentBridge === 'incomplete') return { getDetails: fullService.getDetails };
          return fullService;
        };
      }
    },
  };
}

test('eligible Play offer shows three-day trial and localized monthly renewal', async t => {
  const { module } = await environment(t);
  const result = await module.offer();
  assert.equal(result.title, 'Start your free trial');
  assert.match(result.terms, /3 days free\nThen £1\.79\/month/);
  assert.equal(result.playDetails.subscriptionPeriod, 'P1M');
  assert.equal(result.playDetails.freeTrialPeriod, 'P3D');
});

test('non-trial Play offer never promises a trial', async t => {
  const { module } = await environment(t, { product: { ...trialProduct, freeTrialPeriod: undefined } });
  const result = await module.offer();
  assert.equal(result.title, 'Subscribe to Naptio');
  assert.doesNotMatch(result.terms, /3 days free/);
  assert.match(result.terms, /£1\.79\/month/);
});

test('unexpected billing period cannot be advertised as monthly', async t => {
  const { module } = await environment(t, { product: { ...trialProduct, subscriptionPeriod: 'P1Y' } });
  await assert.rejects(module.offer(), /UNEXPECTED_SUBSCRIPTION_PERIOD/);
});

test('verified existing purchase grants access without checkout', async t => {
  const { module, counts } = await environment(t, {
    purchases: [{ itemId: 'naptio_01', purchaseToken: 'existing-token' }],
  });
  assert.equal((await module.accessStatus()).status, 'ACTIVE');
  assert.equal(counts.checkout, 0);
});

test('verifier outage is not classified as an inactive subscription', async t => {
  const { module } = await environment(t, {
    purchases: [{ itemId: 'naptio_01', purchaseToken: 'existing-token' }],
    verifier: unavailable,
  });
  assert.equal((await module.accessStatus()).status, 'UNAVAILABLE');
});

test('completed checkout with verification outage does not reopen checkout', async t => {
  const { module, counts, setVerifier } = await environment(t, { verifier: unavailable });
  assert.equal((await module.purchaseStatus()).status, 'UNAVAILABLE');
  assert.deepEqual(counts.completion, ['unknown']);
  assert.equal((await module.purchaseStatus()).status, 'UNAVAILABLE');
  assert.equal(counts.checkout, 1);
  setVerifier(active);
  assert.equal((await module.restoreStatus()).status, 'ACTIVE');
  assert.equal((await module.purchaseStatus()).status, 'ACTIVE');
  assert.equal(counts.checkout, 1);
});

test('fresh token temporarily unknown to Google is classified as settling', async t => {
  const { module, counts, setVerifier } = await environment(t, { verifier: invalid });
  const result = await module.purchaseStatus();
  assert.equal(result.status, 'PENDING');
  assert.equal(result.reason, 'PURCHASE_SETTLING');
  assert.deepEqual(counts.completion, ['unknown']);
  assert.equal((await module.purchaseStatus()).status, 'PENDING');
  assert.equal(counts.checkout, 1);
  setVerifier(active);
  assert.equal((await module.restoreStatus()).status, 'ACTIVE');
});

test('interrupted checkout lacking a token requires explicit reconciliation', async t => {
  const { module, counts } = await environment(t, { responseToken: null });
  assert.equal((await module.purchaseStatus()).reason, 'PURCHASE_TOKEN_MISSING');
  assert.equal((await module.purchaseStatus()).reason, 'CHECK_SUBSCRIPTION_BEFORE_RETRY');
  assert.equal(counts.checkout, 1);
  assert.equal((await module.restoreStatus()).reason, 'PURCHASE_SETTLING');
  assert.equal(counts.checkout, 1);
});

test('explicit user cancellation is distinct from an unexpected checkout abort', async t => {
  const err = new Error('cancelled');
  err.name = 'AbortError';
  err.userCancelled = true;
  const { module, counts } = await environment(t, { checkoutError: err });
  const result = await module.purchaseStatus();
  assert.equal(result.status, 'CANCELLED');
  assert.equal(result.reason, 'USER_CANCELLED');
  assert.match(module.messageForStatus(result), /cancelled/i);
  assert.equal(counts.checkout, 1);
  // A real cancellation leaves retry available.
  assert.equal((await module.purchaseStatus()).status, 'CANCELLED');
  assert.equal(counts.checkout, 2);
});

test('Invalid state AbortError means checkout did not open, never a user cancellation', async t => {
  const err = new Error('Invalid state');
  err.name = 'AbortError';
  const { module, counts } = await environment(t, { checkoutError: err });
  const result = await module.purchaseStatus();
  assert.equal(result.status, 'UNAVAILABLE');
  assert.equal(result.reason, 'PLAY_CHECKOUT_DID_NOT_OPEN');
  assert.match(module.messageForStatus(result), /could not open/i);
  assert.equal(counts.checkout, 1);
  assert.equal(counts.verify, 0);
  assert.equal((await module.purchaseStatus()).reason, 'PLAY_CHECKOUT_DID_NOT_OPEN');
});

test('generic AbortError remains unconfirmed, requires checking before any second checkout', async t => {
  const err = new Error('The operation was aborted');
  err.name = 'AbortError';
  const { module, counts } = await environment(t, { checkoutError: err });
  const result = await module.purchaseStatus();
  assert.equal(result.status, 'UNAVAILABLE');
  assert.equal(result.reason, 'PLAY_CHECKOUT_ABORTED_UNCONFIRMED');
  assert.match(module.messageForStatus(result), /closed or failed to open/i);
  assert.equal((await module.purchaseStatus()).reason, 'CHECK_SUBSCRIPTION_BEFORE_RETRY');
  assert.equal(counts.checkout, 1);
  assert.equal((await module.restoreStatus()).reason, 'PURCHASE_SETTLING');
  assert.equal(counts.checkout, 1);
});

test('a generic AbortError saying cancelled is not trusted as a deliberate user cancellation', async t => {
  const err = new Error('cancelled');
  err.name = 'AbortError';
  const { module } = await environment(t, { checkoutError: err });
  assert.equal((await module.purchaseStatus()).reason, 'PLAY_CHECKOUT_ABORTED_UNCONFIRMED');
});

test('non-AbortError checkout failure is visible and requires reconciliation', async t => {
  const err = new Error('Google Play unavailable');
  err.name = 'NotSupportedError';
  const { module, counts } = await environment(t, { checkoutError: err });
  const result = await module.purchaseStatus();
  assert.equal(result.status, 'UNAVAILABLE');
  assert.equal(result.reason, 'PLAY_CHECKOUT_UNAVAILABLE');
  assert.match(module.messageForStatus(result), /did not finish normally/i);
  assert.equal((await module.purchaseStatus()).status, 'PENDING');
  assert.equal(counts.checkout, 1);
});

test('unrelated products do not grant Naptio camera access', async t => {
  const { module } = await environment(t, {
    purchases: [{ itemId: 'other_product', purchaseToken: 'unrelated-token' }],
  });
  assert.equal((await module.accessStatus()).status, 'INACTIVE');
});

test('supported Play-installed TWA is READY without starting checkout', async t => {
  const { module, counts } = await environment(t);
  assert.deepEqual(await module.billingReadiness(), { status: 'READY' });
  assert.equal(counts.checkout, 0);
  assert.equal(counts.service, 1);
});

test('missing Digital Goods API requires an update instead of opening checkout', async t => {
  const { module, counts } = await environment(t, { billingBridge: 'missing' });
  assert.deepEqual(await module.billingReadiness(), {
    status: 'UPDATE_REQUIRED', reason: 'DIGITAL_GOODS_API_MISSING',
  });
  assert.equal(counts.checkout, 0);
});

test('missing Payment Request API requires an update instead of opening checkout', async t => {
  const { module, counts } = await environment(t, { paymentRequestAvailable: false });
  assert.deepEqual(await module.billingReadiness(), {
    status: 'UPDATE_REQUIRED', reason: 'PAYMENT_REQUEST_API_MISSING',
  });
  assert.equal(counts.service, 0);
  assert.equal(counts.checkout, 0);
});

test('null billing service is not cached and can recover after an update', async t => {
  const { module, counts, setBillingBridge } = await environment(t, { billingBridge: 'null' });
  assert.equal((await module.billingReadiness()).status, 'UPDATE_REQUIRED');
  setBillingBridge('ready');
  assert.equal((await module.billingReadiness()).status, 'READY');
  assert.equal(counts.service, 2);
});

test('malformed billing service is UPDATE_REQUIRED and retryable', async t => {
  const { module, counts, setBillingBridge } = await environment(t, { billingBridge: 'incomplete' });
  assert.deepEqual(await module.billingReadiness(), {
    status: 'UPDATE_REQUIRED', reason: 'BILLING_SERVICE_UNSUPPORTED',
  });
  setBillingBridge('ready');
  assert.equal((await module.billingReadiness()).status, 'READY');
  assert.equal(counts.service, 2);
});

test('bridge initialization error is transient, never presented as update required', async t => {
  const { module, counts, setBillingBridge } = await environment(t, { billingBridge: 'reject' });
  assert.deepEqual(await module.billingReadiness(), {
    status: 'TEMPORARILY_UNAVAILABLE', reason: 'BILLING_SERVICE_INITIALIZATION_FAILED',
  });
  assert.equal(counts.checkout, 0);
  setBillingBridge('ready');
  assert.equal((await module.billingReadiness()).status, 'READY');
  assert.equal(counts.service, 2);
});

test('ordinary Android browser is not misclassified as outdated Naptio TWA', async t => {
  const { module, counts } = await environment(t, { twa: false });
  assert.deepEqual(await module.billingReadiness(), {
    status: 'NOT_TWA', reason: 'NOT_NAPTIO_TWA',
  });
  assert.equal(counts.service, 0);
  assert.equal(counts.checkout, 0);
});
