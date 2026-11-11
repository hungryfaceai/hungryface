import { Purchases, LOG_LEVEL } from '@revenuecat/purchases-capacitor';

// Subscription gate for the native iOS app (RevenueCat).
// In the browser and on Android it always allows access.
//
// Test mode (browser only):
// ?submock=new     = eligible for free trial
// ?submock=expired = trial already used / expired
// ?submock=active  = active subscriber
// ?submock=off     = clear test mode

const RC_IOS_KEY = 'appl_rGJQugZEBckkaqbOCHnGCGnPgoe'; // RevenueCat public iOS API key
const ENTITLEMENT = 'premium';
const MOCK_PRICE = '£2.99';

const RENEW_TEXT =
  '\nRenews automatically unless cancelled at least 24 hours before the end of the current period. Manage or cancel in your App Store settings.';

const isNativeIos = () =>
  !!window.Capacitor?.isNativePlatform?.() &&
  window.Capacitor.getPlatform() === 'ios';

// ---- Browser test mode (never active inside the native app) ----

const MOCK_KEY = 'submock';
const qsMock = new URLSearchParams(location.search).get(MOCK_KEY);

if (!isNativeIos() && qsMock) {
  try {
    if (qsMock === 'off') {
      localStorage.removeItem(MOCK_KEY);
    } else {
      localStorage.setItem(MOCK_KEY, qsMock);
    }
  } catch {}
}

const mockState = () =>
  isNativeIos() ? null : localStorage.getItem(MOCK_KEY); // 'new' | 'expired' | 'active' | null

const setMock = (v) => {
  try {
    localStorage.setItem(MOCK_KEY, v);
  } catch {}
};

if (mockState()) {
  console.warn('[SUB] TEST MODE:', mockState());
}

export const needsSubscription = () => isNativeIos() || !!mockState();

// ---- On-screen diagnostics ----

// Sends the diagnostic both to the JS console and to index.html.
// index.html listens for "naptio:subscription-debug" and displays
// the messages in the temporary TestFlight diagnostic panel.
function diag(message) {
  const text = `[RC] ${message}`;

  console.log(text);

  try {
    window.dispatchEvent(
      new CustomEvent('naptio:subscription-debug', {
        detail: text,
      })
    );
  } catch {}
}

// ---- RevenueCat initialization ----

let ready = null;

async function rc() {
  if (!isNativeIos()) {
    throw new Error('RevenueCat is available only in the native iOS app.');
  }

  if (!ready) {
    const startedAt = performance.now();
    diag('configure START');

    ready = (async () => {
      diag('setLogLevel START');

      await Purchases.setLogLevel({
        level: LOG_LEVEL.DEBUG,
      });

      diag('setLogLevel OK');

      diag('Purchases.configure START');

      diag(
        `Purchases.configure OK after ${Math.round(
          performance.now() - startedAt
        )} ms`
      );

      diag('Purchases.configure OK');
    })().catch((err) => {
      // Do not permanently cache a failed configuration attempt.
      ready = null;

      diag(`configure ERROR: ${err?.message || String(err)}`);

      console.error('[SUB] RevenueCat configuration failed', err);

      throw err;
    });
  } else {
    diag('configure: using existing request');
  }

  await ready;

  diag('RevenueCat configured');

  return Purchases;
}

const isActive = (info) =>
  !!info?.entitlements?.active?.[ENTITLEMENT];

// ---- Timeout helper ----

function withTimeout(promise, timeoutMs, label) {
  return Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(
        () =>
          reject(
            new Error(`${label} timed out after ${timeoutMs} ms`)
          ),
        timeoutMs
      )
    ),
  ]);
}

// ---- Access check ----

// True when the user may stream.
// Outside native iOS this is always true unless browser test mode is enabled.
export async function hasAccess() {
  if (mockState()) {
    return mockState() === 'active';
  }

  if (!needsSubscription()) {
    return true;
  }

  try {
    diag('access check START');

    const purchases = await withTimeout(
      rc(),
      30000,
      'RevenueCat initialization'
    );

    diag('CustomerInfo START');

    const { customerInfo } = await withTimeout(
      purchases.getCustomerInfo(),
      10000,
      'RevenueCat customer-info request'
    );

    diag('CustomerInfo OK');

    const active = isActive(customerInfo);

    diag(`premium active: ${active}`);

    return active;
  } catch (e) {
    diag(`access ERROR: ${e?.message || String(e)}`);

    console.warn('[SUB] access check failed', e);

    return false;
  }
}

// ---- Offering / package ----

async function getPackage() {
  diag('getPackage START');

  const purchases = await rc();

  diag('Offerings START');

  const offerings = await purchases.getOfferings();

  diag('Offerings OK');

  const cur = offerings?.current;

  if (!cur) {
    diag('Offering ERROR: no current offering');

    throw new Error(
      'No current RevenueCat offering is configured.'
    );
  }

  if (!cur.monthly) {
    diag('Offering ERROR: monthly package unavailable');

    throw new Error(
      'The monthly subscription package is unavailable.'
    );
  }

  diag('Monthly package OK');

  return cur.monthly;
}

// ---- Paywall wording ----

// Paywall wording depending on whether this Apple ID
// can still receive the free trial.
function buildOffer(priceString, trialText) {
  return trialText
    ? {
        title: 'Start your free trial',
        terms:
          `${trialText} free, then ${priceString} per month. ` +
          `${RENEW_TEXT}`,
        button: 'Start free trial',
      }
    : {
        title: 'Subscribe',
        terms:
          `${priceString} per month. ` +
          `${RENEW_TEXT}`,
        button: `Subscribe for ${priceString}/month`,
      };
}

// Paywall text built from the App Store product.
// priceString is localized by Apple.
export async function offer() {
  if (mockState()) {
    // 'new' = never had the trial
    // 'expired' = trial already used
    return buildOffer(
      MOCK_PRICE,
      mockState() === 'new' ? '2 weeks' : null
    );
  }

  diag('offer START');

  const pkg = await getPackage();

  const p = pkg.product;

  diag(`Product OK: ${p?.identifier || 'unknown'}`);
  diag(`Price OK: ${p?.priceString || 'unknown'}`);

  const intro = p.introPrice;

  let trialText = null;

  if (intro && intro.price === 0) {
    diag('Introductory offer detected');
    diag('Eligibility START');

    let eligible = false;

    try {
      const purchases = await rc();

      const res =
        await purchases.checkTrialOrIntroductoryPriceEligibility({
          productIdentifiers: [p.identifier],
        });

      eligible = res?.[p.identifier]?.status === 2; // 2 = ELIGIBLE

      diag(`Eligibility OK: ${eligible}`);
    } catch (e) {
      diag(
        `Eligibility ERROR: ${e?.message || String(e)}`
      );

      console.warn(
        '[SUB] eligibility check failed',
        e
      );
    }

    if (eligible) {
      const n = intro.periodNumberOfUnits;

      trialText =
        `${n} ${intro.periodUnit.toLowerCase()}` +
        `${n > 1 ? 's' : ''}`;

      diag(`Trial available: ${trialText}`);
    }
  } else {
    diag('No free introductory offer');
  }

  diag('offer OK');

  return buildOffer(
    p.priceString,
    trialText
  );
}

// ---- Purchase ----

// Returns true if the user now has access.
// Returns false if the Apple purchase sheet was cancelled.
export async function purchase() {
  if (mockState()) {
    // Stand-in for the Apple purchase sheet:
    // OK = subscribe
    // Cancel = close
    if (
      !confirm(
        '[TEST] Simulated Apple purchase sheet.\n\n' +
          'OK = subscribe, Cancel = close'
      )
    ) {
      return false;
    }

    setMock('active');

    return true;
  }

  diag('purchase START');

  const pkg = await getPackage();

  try {
    diag('purchasePackage START');

    const purchases = await rc();

    const { customerInfo } =
      await purchases.purchasePackage({
        aPackage: pkg,
      });

    diag('purchasePackage OK');

    const active = isActive(customerInfo);

    diag(`premium active after purchase: ${active}`);

    return active;
  } catch (e) {
    if (e?.userCancelled || e?.code === '1') {
      diag('purchase CANCELLED');

      return false;
    }

    diag(
      `purchase ERROR: ${e?.message || String(e)}`
    );

    throw e;
  }
}

// ---- Restore purchases ----

export async function restore() {
  if (mockState()) {
    return mockState() === 'active';
  }

  diag('restore START');

  try {
    const purchases = await rc();

    diag('restorePurchases START');

    const { customerInfo } =
      await purchases.restorePurchases();

    diag('restorePurchases OK');

    const active = isActive(customerInfo);

    diag(`premium active after restore: ${active}`);

    return active;
  } catch (e) {
    diag(
      `restore ERROR: ${e?.message || String(e)}`
    );

    throw e;
  }
}
