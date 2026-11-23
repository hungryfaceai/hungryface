// Google Play subscription adapter for the Naptio Android TWA.
//
// Google Play:
//   Subscription: naptio_01
//   Base plan: monthly
//   Trial offer: 3-day-free-trial
//
// Purchase verification:
//   Cloudflare Worker -> Google Play Developer API

const BILLING_METHOD = 'https://play.google.com/billing';
const PRODUCT_ID = 'naptio_01';

const VERIFY_URL =
  'https://naptio-play-verifier.bbmaugendre.workers.dev/google-play/verify';

const TWA_SESSION_KEY = 'naptio:android-twa-session';

const isAndroid = () =>
  /Android/i.test(navigator.userAgent);

export function isNaptioTwa() {
  if (!isAndroid()) return false;

  const fromNaptioTwa =
    /^android-app:\/\/io\.github\.hungryfaceai\.twa(?:\/|$)/.test(
      document.referrer
    );

  if (fromNaptioTwa) {
    try {
      sessionStorage.setItem(TWA_SESSION_KEY, '1');
    } catch {}
  }

  try {
    return (
      fromNaptioTwa ||
      sessionStorage.getItem(TWA_SESSION_KEY) === '1'
    );
  } catch {
    return fromNaptioTwa;
  }
}

export const needsSubscription = () => isNaptioTwa();

let servicePromise = null;

function withTimeout(promise, timeoutMs, label) {
  return Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(
        () => reject(new Error(`${label} timed out after ${timeoutMs} ms`)),
        timeoutMs
      )
    ),
  ]);
}

async function getService() {
  if (!isNaptioTwa()) {
    throw new Error(
      'Google Play Billing is available only in the installed Android app.'
    );
  }

  if (typeof window.getDigitalGoodsService !== 'function') {
    throw new Error(
      'Google Play Billing is unavailable. Update Chrome and install Naptio from Google Play.'
    );
  }

  if (!servicePromise) {
    servicePromise = withTimeout(
      window.getDigitalGoodsService(BILLING_METHOD),
      10000,
      'Google Play Billing initialization'
    ).catch((err) => {
      servicePromise = null;
      throw err;
    });
  }

  return servicePromise;
}

async function getProduct() {
  const service = await getService();

  const products = await withTimeout(
    service.getDetails([PRODUCT_ID]),
    10000,
    'Google Play product request'
  );

  const product = products?.find(
    (item) => item.itemId === PRODUCT_ID
  );

  if (!product) {
    throw new Error(
      'The Naptio subscription is unavailable in Google Play.'
    );
  }

  return product;
}

function formatPrice(product) {
  const value = product?.price?.value;
  const currency = product?.price?.currency;

  if (value == null || !currency) {
    throw new Error(
      'Google Play returned incomplete pricing information.'
    );
  }

  try {
    return new Intl.NumberFormat(
      navigator.language,
      {
        style: 'currency',
        currency,
      }
    ).format(value);
  } catch {
    return `${value} ${currency}`;
  }
}

async function verifyPurchase(purchaseToken) {
  if (!purchaseToken) {
    return {
      active: false,
      state: 'NO_PURCHASE_TOKEN',
    };
  }

  const response = await withTimeout(
    fetch(VERIFY_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        purchaseToken,
      }),
    }),
    10000,
    'Subscription verification'
  );

  let result;

  try {
    result = await response.json();
  } catch {
    throw new Error(
      'Subscription verification returned an invalid response.'
    );
  }

  if (!response.ok) {
    throw new Error(
      result?.error ||
        `Subscription verification failed (${response.status}).`
    );
  }

  return result;
}

async function getVerifiedPurchase() {
  const service = await getService();

  const purchases = await withTimeout(
    service.listPurchases(),
    10000,
    'Google Play purchase check'
  );

  const matchingPurchases = (purchases || []).filter(
    (purchase) => purchase?.itemId === PRODUCT_ID
  );

  for (const purchase of matchingPurchases) {
    const result = await verifyPurchase(
      purchase.purchaseToken
    );

    if (result?.active === true) {
      return {
        purchase,
        verification: result,
      };
    }
  }

  return null;
}

export async function hasAccess() {
  if (!needsSubscription()) {
    return true;
  }

  try {
    return !!(await getVerifiedPurchase());
  } catch (error) {
    console.warn(
      '[SUB][Android] access check failed',
      error
    );

    // Fail closed when the installed Android sender cannot
    // obtain authoritative subscription status.
    return false;
  }
}

export async function offer() {
  const product = await getProduct();
  const price = formatPrice(product);

  return {
    title: 'Subscribe to Naptio',
    terms:
      `Eligible new customers: 3 days free\n` +
      `Then ${price}/month\n\n` +
      `Renews automatically every month unless cancelled. ` +
      `Manage or cancel in your Google Play subscriptions.`,
    button: 'Continue in Google Play',
  };
}

export async function purchase() {
  if (!needsSubscription()) {
    throw new Error(
      'Google Play purchases are available only in the installed Android app.'
    );
  }

  if (typeof PaymentRequest !== 'function') {
    throw new Error(
      'Google Play purchases are unavailable on this device.'
    );
  }

  // Confirm that Google Play knows the product before opening
  // the purchase sheet, and obtain the correct local currency.
  const product = await getProduct();
  const currency = product?.price?.currency || 'GBP';

  const request = new PaymentRequest(
    [
      {
        supportedMethods: BILLING_METHOD,
        data: {
          sku: PRODUCT_ID,
        },
      },
    ],
    {
      // Required by Payment Request. Google Play uses the actual
      // price configured in Play Console rather than this value.
      total: {
        label: 'Naptio monthly subscription',
        amount: {
          currency,
          value: '0',
        },
      },
    }
  );

  let response = null;

  try {
    response = await request.show();

    const purchaseToken =
      response?.details?.purchaseToken;

    if (!purchaseToken) {
      await response.complete('fail');

      throw new Error(
        'Google Play returned no purchase token.'
      );
    }

    // Access is granted only after the Worker verifies the token
    // with Google Play and acknowledges the purchase when required.
    const verification =
      await verifyPurchase(purchaseToken);

    const active =
      verification?.active === true;

    await response.complete(
      active ? 'success' : 'fail'
    );

    return active;
  } catch (error) {
    if (response) {
      try {
        await response.complete('fail');
      } catch {}
    }

    // Closing/cancelling the Google Play purchase sheet is not an error.
    if (error?.name === 'AbortError') {
      return false;
    }

    throw error;
  }
}


