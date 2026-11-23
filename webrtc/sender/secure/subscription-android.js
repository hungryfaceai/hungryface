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
