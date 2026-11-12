import { Purchases } from '@revenuecat/purchases-capacitor';

// Subscription gate for the native iOS app (RevenueCat).
// In the browser and on Android it always allows access.
//
// Browser test mode:
// ?submock=new     = eligible for free trial
// ?submock=expired = trial already used / expired
// ?submock=active  = active subscriber
// ?submock=off     = clear test mode

const RC_IOS_KEY = 'appl_rGJQugZEBckkaqbOCHnGCGnPgoe';
const ENTITLEMENT = 'premium';
const MOCK_PRICE = '£2.99';

const isNativeIos = () =>
  !!window.Capacitor?.isNativePlatform?.() &&
  window.Capacitor.getPlatform() === 'ios';

// ---- Browser test mode ----

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
  isNativeIos() ? null : localStorage.getItem(MOCK_KEY);

const setMock = (value) => {
  try {
    localStorage.setItem(MOCK_KEY, value);
  } catch {}
};

export const needsSubscription = () =>
  isNativeIos() || !!mockState();

// ---- RevenueCat initialization ----

let ready = null;

async function rc() {
  if (!isNativeIos()) {
    throw new Error(
      'RevenueCat is available only in the native iOS app.'
    );
  }

  if (!ready) {
    ready = Purchases.configure({
      apiKey: RC_IOS_KEY,
    }).catch((err) => {
      // Allow a subsequent attempt if initialization fails.
      ready = null;
      throw err;
    });
  }

  await ready;

  // Do not return Purchases here.
  // Capacitor plugin proxies must not be returned through an async
  // function because they can be interpreted as Promise-like objects.
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
    await withTimeout(
      rc(),
      30000,
      'RevenueCat initialization'
    );

    const { customerInfo } = await withTimeout(
      Purchases.getCustomerInfo(),
      10000,
      'RevenueCat customer-info request'
    );

    return isActive(customerInfo);
  } catch (e) {
    console.warn('[SUB] access check failed', e);
    return false;
  }
}

// ---- Offering / package ----

async function getPackage() {
  await rc();

  const offerings = await Purchases.getOfferings();
  const current = offerings?.current;

  if (!current) {
    throw new Error(
      'No current RevenueCat offering is configured.'
    );
  }

  if (!current.monthly) {
    throw new Error(
      'The monthly subscription package is unavailable.'
    );
  }

  return current.monthly;
}

// ---- Paywall wording ----

function buildOffer(priceString, trialText) {
  return trialText
    ? {
        title: 'Start your free trial',
        terms:
          `${trialText} free\n` +
          `Then ${priceString}/month\n\n` +
          `Renews automatically unless cancelled at least 24 hours before the end of the current period. Manage or cancel in your App Store settings.`,
        button: `Start ${trialText} free trial`,
      }
    : {
        title: 'Subscribe to Naptio',
        terms:
          `${priceString}/month\n\n` +
          `Renews automatically unless cancelled at least 24 hours before the end of the current period. Manage or cancel in your App Store settings.`,
        button: `Subscribe for ${priceString}/month`,
      };
}

// Paywall text built from the App Store product.
// priceString is localized by Apple.
export async function offer() {
  if (mockState()) {
    return buildOffer(
      MOCK_PRICE,
      mockState() === 'new' ? '2 weeks' : null
    );
  }

  const pkg = await getPackage();
  const product = pkg.product;
  const intro = product.introPrice;

  let trialText = null;

  if (intro && intro.price === 0) {
    let eligible = false;

    try {
      await rc();

      const result =
        await Purchases.checkTrialOrIntroductoryPriceEligibility({
          productIdentifiers: [product.identifier],
        });

      eligible =
        result?.[product.identifier]?.status === 2;
    } catch (e) {
      console.warn('[SUB] eligibility check failed', e);
    }

    if (eligible) {
      const units = intro.periodNumberOfUnits;

      trialText =
        `${units} ${intro.periodUnit.toLowerCase()}` +
        `${units > 1 ? 's' : ''}`;
    }
  }

  return buildOffer(
    product.priceString,
    trialText
  );
}

// ---- Purchase ----

// Returns true if the user now has access.
// Returns false if the Apple purchase sheet was cancelled.
export async function purchase() {
  if (mockState()) {
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

  const pkg = await getPackage();

  try {
    await rc();

    const { customerInfo } =
      await Purchases.purchasePackage({
        aPackage: pkg,
      });

    return isActive(customerInfo);
  } catch (e) {
    if (e?.userCancelled || e?.code === '1') {
      return false;
    }

    throw e;
  }
}

// ---- Restore purchases ----

export async function restore() {
  if (mockState()) {
    return mockState() === 'active';
  }

  await rc();

  const { customerInfo } =
    await Purchases.restorePurchases();

  return isActive(customerInfo);
}
