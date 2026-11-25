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
// Give a newly returned purchase time to appear consistently in Play's APIs.
const PURCHASE_SETTLEMENT_MS = 2 * 60 * 1000;

const SERVER_STATUSES = new Set([
  'ACTIVE', 'INACTIVE', 'PENDING', 'INVALID_PURCHASE', 'UNAVAILABLE',
]);
let servicePromise = null;
let productPromise = null;
let purchaseInProgress = false;
let recentPurchaseToken = null; // In-memory only; Play remains the durable source.
let recentPurchaseAt = 0;
let checkoutNeedsReconciliationAt = 0;

const unavailable = reason => ({ status: 'UNAVAILABLE', active: false, reason, retryable: true });
const inactive = reason => ({ status: 'INACTIVE', active: false, reason, retryable: false });
const pending = reason => ({ status: 'PENDING', active: false, reason, retryable: true });
const updateRequired = reason => ({ status: 'UPDATE_REQUIRED', reason });
const temporarilyUnavailable = reason => ({ status: 'TEMPORARILY_UNAVAILABLE', reason });
const isSettling = when => when > 0 && Date.now() - when < PURCHASE_SETTLEMENT_MS;

const isAndroid = () => /Android/i.test(navigator.userAgent);

// Detect the Android Play-installed TWA even when Chrome drops the initial
// android-app:// referrer and a fresh tab no longer has sessionStorage state.
// Chrome exposes the Digital Goods API for Play Billing in a TWA, not in a
// normal Chrome tab. This is a client-side compatibility signal, not an app
// identity attestation; access still requires verified Play entitlement.
export function isNaptioTwa() {
  if (!isAndroid()) return false;
  const fromNaptioTwa = /^android-app:\/\/io\.github\.hungryfaceai\.twa(?:\/|$)/
    .test(document.referrer);
  const hasPlayBillingBridge = typeof window.getDigitalGoodsService === 'function';
  let rememberedTwa = false;
  try {
    rememberedTwa = sessionStorage.getItem(TWA_SESSION_KEY) === '1';
    if (fromNaptioTwa || hasPlayBillingBridge) {
      sessionStorage.setItem(TWA_SESSION_KEY, '1');
    }
  } catch {
    // Storage can be disabled; direct referrer or billing bridge still works.
  }
  return fromNaptioTwa || rememberedTwa || hasPlayBillingBridge;
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
    ).then(service => {
      // Some older Digital Goods implementations resolve to null rather
      // than reject. Do not cache that result across an explicit retry.
      if (!service) servicePromise = null;
      return service;
    }).catch(error => {
      servicePromise = null;
      throw error;
    });
  }
  return servicePromise;
}

// Only checks whether this installed Android environment can present a Play
// Billing checkout; it does NOT claim to know the APK's version code.
// Missing APIs or an unusable service are a compatibility/update issue.
// Errors from an existing bridge, including timeouts, are treated as transient
// so an outage never sends the user to an irrelevant Play Store update.
export async function billingReadiness() {
  if (!isNaptioTwa()) return { status: 'NOT_TWA', reason: 'NOT_NAPTIO_TWA' };
  if (typeof window.getDigitalGoodsService !== 'function') {
    return updateRequired('DIGITAL_GOODS_API_MISSING');
  }
  if (typeof PaymentRequest !== 'function') {
    return updateRequired('PAYMENT_REQUEST_API_MISSING');
  }

  try {
    const service = await getService();
    if (!service || typeof service.getDetails !== 'function' ||
        typeof service.listPurchases !== 'function') {
      // Avoid caching a broken bridge if the user updates and tries again.
      servicePromise = null;
      productPromise = null;
      return updateRequired('BILLING_SERVICE_UNSUPPORTED');
    }
    return { status: 'READY' };
  } catch (error) {
    console.warn('[SUB][Android] billing readiness unavailable', error?.name || 'error');
    return temporarilyUnavailable('BILLING_SERVICE_INITIALIZATION_FAILED');
  }
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

// The Digital Goods API does not reliably report whether a Google account was
// removed. Only identify account problems when the bridge explicitly says so.
// Never turn a failed lookup into "no purchase" or claim the app is outdated.
function classifyPurchaseLookupError(error) {
  const code = typeof error?.code === 'string' ? error.code.toUpperCase() : '';
  const name = typeof error?.name === 'string' ? error.name : '';
  const message = typeof error?.message === 'string' ? error.message : '';
  const accountCodes = new Set([
    'ACCOUNT_UNAVAILABLE', 'ACCOUNT_NOT_FOUND', 'NO_GOOGLE_ACCOUNT',
    'NO_ACCOUNT', 'SIGN_IN_REQUIRED', 'NOT_SIGNED_IN',
    'AUTHENTICATION_REQUIRED', 'USER_NOT_AUTHENTICATED',
  ]);
  const accountNames = new Set([
    'AccountNotFoundError', 'AccountUnavailableError',
    'AuthenticationRequiredError', 'NotAuthenticatedError',
  ]);
  const explicitlyAccountRelated =
    accountCodes.has(code) || accountNames.has(name) ||
    /\b(?:not signed in|sign[ -]?in required|sign in to (?:google )?play|no (?:google )?accounts?(?: (?:available|found|configured|on (?:this )?device))?|(?:google play )?account (?:not found|unavailable|removed)|(?:authentication|login) required|user not authenticated)\b/i.test(message);

  return unavailable(explicitlyAccountRelated
    ? 'PLAY_ACCOUNT_UNAVAILABLE'
    : 'PLAY_PURCHASE_LOOKUP_UNAVAILABLE');
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
    let result = await verifyPurchase(recentPurchaseToken);
    if (result.status === 'ACTIVE') {
      // Keep this token for the life of the page so the Buy action cannot
      // accidentally open another checkout after a successful purchase.
      checkoutNeedsReconciliationAt = 0;
      return result;
    }
    if (result.status === 'INVALID_PURCHASE' || result.status === 'INACTIVE') {
      if (isSettling(recentPurchaseAt)) {
        // Play purchase reporting may lag checkout: do not encourage a second purchase.
        result = pending('PURCHASE_SETTLING');
      } else {
        recentPurchaseToken = null;
        recentPurchaseAt = 0;
      }
    }
    results.push(result);
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
    const classified = classifyPurchaseLookupError(error);
    // Log only diagnostic categories, never account identifiers, purchase
    // tokens, or arbitrary bridge error messages.
    console.warn('[SUB][Android] purchase lookup unavailable', {
      errorName: typeof error?.name === 'string' ? error.name : 'Error',
      reason: classified.reason,
    });
    results.push(classified);
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
  const price = formatPrice(product); // Play's recurring/base price, localized.
  const period = product.subscriptionPeriod;
  if (period && period !== 'P1M') {
    // Never advertise a monthly price if Play reports a different renewal period.
    throw new Error('UNEXPECTED_SUBSCRIPTION_PERIOD');
  }

  // Digital Goods returns one eligible Play plan/offer. A trial is advertised
  // only when that selected offer explicitly reports the correct trial period.
  // A missing freeTrialPeriod is NOT proof that the user is trial-eligible.
  const trial = product.freeTrialPeriod === 'P3D';
  const terms = trial
    ? `3 days free\nThen ${price}/month\n\nRenews automatically every month unless cancelled. Google Play confirms the offer before you subscribe.`
    : `${price}/month\n\nRenews automatically every month unless cancelled. If an introductory offer applies, Google Play will confirm it before you subscribe.`;

  return {
    title: trial ? 'Start your free trial' : 'Subscribe to Naptio',
    terms,
    button: trial ? 'Continue to 3-day free trial' : 'Continue in Google Play',
    // Non-sensitive metadata for Play-installed QA. The sender page only
    // renders title/terms/button; checkout remains the final authority.
    playDetails: {
      productId: product.itemId,
      currency: product.price.currency,
      priceValue: String(product.price.value),
      freeTrialPeriod: product.freeTrialPeriod || null,
      subscriptionPeriod: period || null,
    },
  };
}

// PaymentRequest.show() can reject with AbortError both when a customer
// cancels and when Chrome never displayed Google Play checkout. The web API
// provides no reliable "sheet opened" event: only explicit cancellation
// metadata is safe to classify as a deliberate cancellation.
function classifyCheckoutShowError(error) {
  const name = typeof error?.name === 'string' ? error.name : 'Error';
  const code = typeof error?.code === 'string' ? error.code.toUpperCase() : '';
  if (error?.userCancelled === true ||
      ['USER_CANCELLED', 'USER_CANCELED'].includes(code)) {
    return { status: 'CANCELLED', active: false, reason: 'USER_CANCELLED', retryable: false };
  }
  // A known launch-time browser error, not evidence of an opened payment sheet.
  if (name === 'InvalidStateError' ||
      (name === 'AbortError' && /invalid state/i.test(String(error?.message || '')))) {
    return unavailable('PLAY_CHECKOUT_DID_NOT_OPEN');
  }
  // A bare AbortError is ambiguous. Do not silently claim the user cancelled,
  // and do not let them accidentally purchase again before reconciliation.
  return unavailable(name === 'AbortError'
    ? 'PLAY_CHECKOUT_ABORTED_UNCONFIRMED'
    : 'PLAY_CHECKOUT_UNAVAILABLE');
}

// New richer purchase API. Never represent a completed Play checkout as cancellation
// merely because Cloudflare is temporarily unavailable.
export async function purchaseStatus() {
  if (!isNaptioTwa()) return unavailable('NOT_NAPTIO_TWA');
  if (purchaseInProgress) return unavailable('PURCHASE_ALREADY_IN_PROGRESS');
  if (recentPurchaseToken) return accessStatus(); // Reconcile, do not start a second checkout.
  if (checkoutNeedsReconciliationAt) {
    // Never reopen checkout before the user has explicitly checked for an interrupted purchase.
    return pending('CHECK_SUBSCRIPTION_BEFORE_RETRY');
  }
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
      const result = classifyCheckoutShowError(error);
      // Only record a known, non-sensitive reason; never log a purchase token,
      // checkout response, payment method, or untrusted error message.
      console.warn('[SUB][Android] checkout could not complete', {
        errorName: typeof error?.name === 'string' ? error.name : 'Error',
        reason: result.reason,
      });
      if (result.status !== 'CANCELLED' &&
          result.reason !== 'PLAY_CHECKOUT_DID_NOT_OPEN') {
        // Google Play may already have processed an ambiguous checkout.
        checkoutNeedsReconciliationAt = Date.now();
      }
      return result;
    }

    const token = paymentResponse?.details?.purchaseToken;
    if (typeof token !== 'string' || !token) {
      checkoutNeedsReconciliationAt = Date.now();
      await completeOnce('unknown');
      return unavailable('PURCHASE_TOKEN_MISSING');
    }
    recentPurchaseToken = token;
    recentPurchaseAt = Date.now();
    checkoutNeedsReconciliationAt = 0;
    let result = await verifyPurchase(token);
    if (result.status === 'INVALID_PURCHASE' || result.status === 'INACTIVE') {
      // A newly returned token may not be visible to the Developer API yet.
      result = pending('PURCHASE_SETTLING');
    }
    await completeOnce(result.status === 'ACTIVE' ? 'success'
      : ['UNAVAILABLE', 'PENDING'].includes(result.status) ? 'unknown' : 'fail');
    return result;
  } catch (error) {
    if (paymentResponse && !recentPurchaseToken) {
      checkoutNeedsReconciliationAt = Date.now();
    }
    await completeOnce('unknown');
    console.warn('[SUB][Android] purchase status unavailable', error?.name || 'error');
    return unavailable('PLAY_CHECKOUT_UNAVAILABLE');
  } finally {
    purchaseInProgress = false;
  }
}

export async function restoreStatus() {
  const result = await accessStatus();
  if (checkoutNeedsReconciliationAt &&
      (result.status === 'INACTIVE' || result.status === 'INVALID_PURCHASE')) {
    // A missing/unreturned checkout token can take time to appear in listPurchases.
    if (isSettling(checkoutNeedsReconciliationAt)) {
      return pending('PURCHASE_SETTLING');
    }
    checkoutNeedsReconciliationAt = 0;
  }
  if (result.status === 'ACTIVE') checkoutNeedsReconciliationAt = 0;
  return result;
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
    case 'PENDING':
      if (result.reason === 'PURCHASE_SETTLING') {
        return 'Your recent Google Play checkout is still being confirmed. Select Check subscription again shortly; do not purchase a second time.';
      }
      if (result.reason === 'CHECK_SUBSCRIPTION_BEFORE_RETRY') {
        return 'Google Play checkout may have completed. Select Check subscription before trying to purchase again.';
      }
      return 'Google Play is still processing your payment. Check your subscription again shortly.';
    case 'UNAVAILABLE':
      if (result.reason === 'PLAY_ACCOUNT_UNAVAILABLE') {
        return 'Google Play cannot access a signed-in account. Open the Play Store and sign in with the Google account used for Naptio, then return and select Check subscription.';
      }
      if (result.reason === 'PLAY_PURCHASE_LOOKUP_UNAVAILABLE' ||
          result.reason === 'PLAY_BILLING_UNAVAILABLE') {
        return 'Cannot retrieve subscriptions from Google Play right now. Check your internet connection and that you are signed in to the Play Store with the account used for Naptio, then select Check subscription to retry.';
      }
      if (result.reason === 'PLAY_CHECKOUT_DID_NOT_OPEN') {
        return 'Google Play checkout could not open. Check that Chrome is your default browser and update Chrome and Google Play, then select Check subscription before trying again.';
      }
      if (result.reason === 'PLAY_CHECKOUT_ABORTED_UNCONFIRMED') {
        return 'Google Play checkout closed or failed to open. We could not confirm whether it reached the payment screen. Select Check subscription before trying again.';
      }
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
    case 'INACTIVE':
      return result.reason === 'NO_PURCHASE'
        ? 'No Naptio subscription was found for the current Google Play account. Sign in with the subscribing account or subscribe to continue.'
        : 'No active Naptio subscription was found for this Google Play account.';
    case 'CANCELLED': return 'Google Play checkout was cancelled. No subscription has been confirmed; you can try again.';
    default: return 'Subscription status is unknown. Please retry.';
  }
}