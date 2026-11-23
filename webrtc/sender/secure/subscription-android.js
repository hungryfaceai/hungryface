// Android TWA subscription adapter. Only index_inprogress.html imports the new router until cutover.
// Existing exports preserved: isNaptioTwa, needsSubscription, hasAccess,
// offer, purchase, restore.
// New exports: accessStatus, purchaseStatus, restoreStatus.
// accessStatus / purchaseStatus return {status, active, reason, retryable, ...}.

const BILLING_METHOD = 'https://play.google.com/billing';
const PRODUCT_ID = 'naptio_01';
const VERIFY_URL = 'https://naptio-play-verifier.bbmaugendre.workers.dev/google-play/verify';
const TWA_SESSION_KEY = 'naptio:android-twa-session';
const SERVICE_TIMEOUT_MS = 10000;
const VERIFY_TIMEOUT_MS = 30000;

const SERVER_STATUSES = new Set([
  'ACTIVE', 'INACTIVE', 'PENDING', 'INVALID_PURCHASE', 'UNAVAILABLE',
]);
let servicePromise = null;
let productPromise = null;
let purchaseInProgress = false;
let recentPurchaseToken = null; // In-memory only; Play remains the durable source.

const unavailable = reason => ({ status: 'UNAVAILABLE', active: false, reason, retryable: true });
const inactive = reason => ({ status: 'INACTIVE', active: false, reason, retryable: false });

const isAndroid = () => /Android/i.test(navigator.userAgent);

export function isNaptioTwa() {
  if (!isAndroid()) return false;
  const fromNaptioTwa = /^android-app:\/\/io\.github\.hungryfaceai\.twa(?:\/|$)/
    .test(document.referrer);
  if (fromNaptioTwa) {
    try { sessionStorage.setItem(TWA_SESSION_KEY, '1'); } catch {}
  }
  try {
    return fromNaptioTwa || sessionStorage.getItem(TWA_SESSION_KEY) === '1';
  } catch {
    return fromNaptioTwa;
  }
}
export const needsSubscription = () => isNaptioTwa();

function withTimeout(promise, ms, label) {
  let timer;
  return Promise.race([
    Promise.resolve(promise),
    new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(`${label} timed out`)), ms);
    }),
  ]).finally(() => clearTimeout(timer));
}

async function getService() {
  if (!isNaptioTwa()) throw new Error('NOT_NAPTIO_TWA');
  if (typeof window.getDigitalGoodsService !== 'function') {
    throw new Error('BILLING_BRIDGE_UNAVAILABLE');
  }
  if (!servicePromise) {
    servicePromise = withTimeout(
      window.getDigitalGoodsService(BILLING_METHOD),
      SERVICE_TIMEOUT_MS, 'Billing initialization'
    ).catch(error => {
      servicePromise = null;
      throw error;
    });
  }
  return servicePromise;
}

async function getProduct() {
  if (!productPromise) {
    productPromise = (async () => {
      const service = await getService();
      const items = await withTimeout(
        service.getDetails([PRODUCT_ID]), SERVICE_TIMEOUT_MS, 'Play product lookup'
      );
      const product = items?.find(item => item?.itemId === PRODUCT_ID);
      if (!product || !product.price?.currency || product.price?.value == null) {
        throw new Error('PRODUCT_DETAILS_UNAVAILABLE');
      }
      return product;
    })().catch(error => {
      productPromise = null;
      throw error;
    });
  }
  return productPromise;
}

function formatPrice(product) {
  const { value, currency } = product.price;
  const amount = Number(value);
  if (!Number.isFinite(amount) || amount < 0) throw new Error('INVALID_PRODUCT_PRICE');
  return new Intl.NumberFormat(navigator.language, {
    style: 'currency', currency,
  }).format(amount);
}

function normalizeServerReply(body, httpOk) {
  if (!body || typeof body !== 'object' || Array.isArray(body) ||
      !SERVER_STATUSES.has(body.status) || typeof body.active !== 'boolean' ||
      body.active !== (body.status === 'ACTIVE')) {
    return unavailable('INVALID_VERIFIER_RESPONSE');
  }
  if (httpOk && body.status === 'UNAVAILABLE') {
    // No successful authoritative verification may report UNAVAILABLE.
    return unavailable('INVALID_VERIFIER_RESPONSE');
  }
  if (!httpOk && body.status !== 'UNAVAILABLE' && body.status !== 'INVALID_PURCHASE') {
    return unavailable('VERIFIER_HTTP_ERROR');
  }
  if (body.status === 'ACTIVE' &&
      !['ACKNOWLEDGED', 'PENDING_RETRY'].includes(body.acknowledgement)) {
    return unavailable('INVALID_ACKNOWLEDGEMENT_RESPONSE');
  }
  return body;
}

async function verifyPurchase(purchaseToken) {
  if (typeof purchaseToken !== 'string' || !purchaseToken) {
    return { status: 'INVALID_PURCHASE', active: false, reason: 'BAD_TOKEN', retryable: false };
  }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), VERIFY_TIMEOUT_MS);
  try {
    const response = await fetch(VERIFY_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ purchaseToken }),
      signal: controller.signal,
      cache: 'no-store',
    });
    let data;
    try { data = await response.json(); }
    catch { return unavailable('INVALID_VERIFIER_JSON'); }
    return normalizeServerReply(data, response.ok);
  } catch (error) {
    console.warn('[SUB][Android] verifier unavailable', error?.name || 'fetch error');
    return unavailable('VERIFIER_NETWORK_ERROR');
  } finally {
    clearTimeout(timeout);
  }
}

function combineStatuses(results) {
  if (results.some(r => r.status === 'ACTIVE')) {
    return results.find(r => r.status === 'ACTIVE');
  }
  if (results.some(r => r.status === 'UNAVAILABLE')) {
    return results.find(r => r.status === 'UNAVAILABLE');
  }
  if (results.some(r => r.status === 'PENDING')) {
    return results.find(r => r.status === 'PENDING');
  }
  if (results.some(r => r.status === 'INACTIVE')) {
    return results.find(r => r.status === 'INACTIVE');
  }
  return results.find(r => r.status === 'INVALID_PURCHASE') || inactive('NO_PURCHASE');
}

// Rich, authoritative access result. Does NOT conflate outages with no purchase.
export async function accessStatus() {
  if (!isNaptioTwa()) return unavailable('NOT_NAPTIO_TWA');

  const results = [];
  if (recentPurchaseToken) {
    const result = await verifyPurchase(recentPurchaseToken);
    results.push(result);
    if (result.status === 'ACTIVE') return result;
    if (result.status === 'INVALID_PURCHASE' || result.status === 'INACTIVE') {
      recentPurchaseToken = null; // Confirmed not entitled; allow future checkout.
    }
  }

  let service;
  try {
    service = await getService();
    const purchases = await withTimeout(
      service.listPurchases(), SERVICE_TIMEOUT_MS, 'Play purchase lookup'
    );
    if (!Array.isArray(purchases)) return unavailable('INVALID_PURCHASE_LIST');

    // Ignore other products. Deduplicate tokens and cap work to protect quotas.
    const tokens = [...new Set(purchases
      .filter(item => item?.itemId === PRODUCT_ID && typeof item?.purchaseToken === 'string')
      .map(item => item.purchaseToken))].slice(0, 5);
    for (const token of tokens) {
      if (token === recentPurchaseToken) continue;
      const result = await verifyPurchase(token);
      results.push(result);
      if (result.status === 'ACTIVE') return result;
    }
  } catch (error) {
    console.warn('[SUB][Android] purchase lookup unavailable', error?.message || 'error');
    results.push(unavailable('PLAY_BILLING_UNAVAILABLE'));
  }
  return combineStatuses(results);
}

// Existing router contract kept for backwards compatibility. UI should use
// accessStatus() whenever it needs a reason for denial.
export async function hasAccess() {
  return (await accessStatus()).status === 'ACTIVE';
}

export async function offer() {
  const product = await getProduct();
  const price = formatPrice(product);
  // Only show a specific trial if the bridge actually reports it.
  // The TWA billing bridge may not expose all Play offers; Play checkout remains authoritative.
  const trial = product.freeTrialPeriod === 'P3D';
  return {
    title: trial ? 'Start your free trial' : 'Subscribe to Naptio',
    terms: trial
      ? `Eligible new customers: 3 days free\nThen ${price}/month\n\nRenews automatically unless cancelled. Manage your subscription in Google Play.`
      : `${price}/month\n\nRenews automatically unless cancelled. Google Play will confirm any eligible introductory offer before payment.`,
    button: 'Continue in Google Play',
  };
}

// New richer purchase API. Never represent a completed Play checkout as cancellation
// merely because Cloudflare is temporarily unavailable.
export async function purchaseStatus() {
  if (!isNaptioTwa()) return unavailable('NOT_NAPTIO_TWA');
  if (purchaseInProgress) return unavailable('PURCHASE_ALREADY_IN_PROGRESS');
  if (recentPurchaseToken) return accessStatus(); // Reconcile, do not start a second checkout.
  if (typeof PaymentRequest !== 'function') return unavailable('PAYMENT_REQUEST_UNAVAILABLE');

  purchaseInProgress = true;
  let paymentResponse = null;
  let completed = false;
  async function completeOnce(result) {
    if (!paymentResponse || completed) return;
    completed = true;
    try { await paymentResponse.complete(result); }
    catch (e) { console.warn('[SUB][Android] payment completion failed', e?.name || 'error'); }
  }

  try {
    // The offer should already have been loaded before the user presses Buy.
    const product = await getProduct();
    const request = new PaymentRequest(
      [{ supportedMethods: BILLING_METHOD, data: { sku: PRODUCT_ID } }],
      { total: { label: 'Naptio monthly subscription',
        amount: { currency: product.price.currency, value: '0' } } }
    );

    try { paymentResponse = await request.show(); }
    catch (error) {
      if (error?.name === 'AbortError') {
        return { status: 'CANCELLED', active: false, retryable: false };
      }
      return unavailable('PLAY_CHECKOUT_UNAVAILABLE');
    }

    const token = paymentResponse?.details?.purchaseToken;
    if (typeof token !== 'string' || !token) {
      await completeOnce('unknown');
      return unavailable('PURCHASE_TOKEN_MISSING');
    }
    recentPurchaseToken = token;
    const result = await verifyPurchase(token);
    await completeOnce(result.status === 'ACTIVE' ? 'success'
      : ['UNAVAILABLE', 'PENDING'].includes(result.status) ? 'unknown' : 'fail');
    return result;
  } catch (error) {
    await completeOnce('unknown');
    console.warn('[SUB][Android] purchase status unavailable', error?.message || 'error');
    return unavailable('PLAY_CHECKOUT_UNAVAILABLE');
  } finally {
    purchaseInProgress = false;
  }
}

export async function restoreStatus() {
  return accessStatus();
}

// Compatibility exports for the existing subscription-gate.js.
// Richer callers should use purchaseStatus() / restoreStatus() instead.
export async function purchase() {
  const result = await purchaseStatus();
  if (result.status === 'ACTIVE') return true;
  if (result.status === 'CANCELLED') return false;
  const error = new Error(messageForStatus(result));
  error.code = result.status;
  throw error;
}
export async function restore() {
  return (await restoreStatus()).status === 'ACTIVE';
}

export function messageForStatus(result) {
  switch (result.status) {
    case 'ACTIVE': return 'Your Naptio subscription is active.';
    case 'PENDING': return 'Google Play is still processing your payment. Check your subscription again shortly.';
    case 'UNAVAILABLE':
      if (['PLAY_BILLING_UNAVAILABLE', 'BILLING_BRIDGE_UNAVAILABLE',
           'PAYMENT_REQUEST_UNAVAILABLE'].includes(result.reason)) {
        return 'Google Play Billing is unavailable in this installation. Update Naptio and Chrome, then try again.';
      }
      if (result.reason === 'PURCHASE_ALREADY_IN_PROGRESS') {
        return 'A Google Play purchase is already in progress.';
      }
      if (result.reason === 'PLAY_CHECKOUT_UNAVAILABLE') {
        return 'Google Play checkout did not finish normally. Check subscription before trying again.';
      }
      if (result.reason === 'PURCHASE_TOKEN_MISSING') {
        return 'Google Play returned an incomplete purchase result. Use Check subscription before buying again.';
      }
      return 'Subscription verification is temporarily unavailable. If you just paid, use Check subscription before trying to buy again.';
    case 'INVALID_PURCHASE': return 'Google Play could not validate this purchase. Check your Google Play account.';
    case 'INACTIVE': return 'No active Naptio subscription was found for this Google Play account.';
    case 'CANCELLED': return 'Purchase cancelled.';
    default: return 'Subscription status is unknown. Please retry.';
  }
}