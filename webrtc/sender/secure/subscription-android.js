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
