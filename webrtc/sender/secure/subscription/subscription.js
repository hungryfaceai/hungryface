// Subscription gate for the native iOS app (RevenueCat). In the browser and on Android it always allows access.
// Test mode (browser only):
// ?submock=new     = eligible for free trial
// ?submock=expired = trial already used / expired
// ?submock=active  = active subscriber
// ?submock=off     = clear test mode
const RC_IOS_KEY = 'appl_XXXXXXXXXXXXXXXX'; // RevenueCat public iOS API key
const ENTITLEMENT = 'premium';
const RENEW_TEXT =
  'Renews automatically unless cancelled at least 24 hours before the end of the current period. Manage or cancel in your App Store settings.';
  
const cap = window.Capacitor;
const isNativeIos = () => !!cap?.isNativePlatform?.() && cap.getPlatform() === 'ios';

// ---- Browser test mode (never active inside the native app) ----
const MOCK_KEY = 'submock';
const qsMock = new URLSearchParams(location.search).get(MOCK_KEY);
if (!isNativeIos() && qsMock) {
  try {
    if (qsMock === 'off') localStorage.removeItem(MOCK_KEY);
    else localStorage.setItem(MOCK_KEY, qsMock);
  } catch {}
}
const mockState = () =>
  isNativeIos() ? null : localStorage.getItem(MOCK_KEY); // 'new' | 'expired' | 'active' | null
const setMock = (v) => {
  try {
    localStorage.setItem(MOCK_KEY, v);
  } catch {}
};
if (mockState()) console.warn('[SUB] TEST MODE:', mockState());

export const needsSubscription = () => isNativeIos() || !!mockState();

let Purchases = null;
let ready = null;
function rc() {
  Purchases ??= cap.registerPlugin('Purchases');
  ready ??= Purchases.configure({ apiKey: RC_IOS_KEY });
  return ready.then(() => Purchases);
}
const isActive = (info) => !!info?.entitlements?.active?.[ENTITLEMENT];

// True when the user may stream (always true outside the iOS app, unless test mode is on).
export async function hasAccess() {
  if (mockState()) return mockState() === 'active';
  if (!needsSubscription()) return true;
  try {
    const { customerInfo } = await (await rc()).getCustomerInfo();
    return isActive(customerInfo);
  } catch (e) {
    console.warn('[SUB] check failed', e);
    return false;
  }
}

async function getPackage() {
  const offerings = await (await rc()).getOfferings();
  const cur = offerings?.current;
  return cur?.monthly ?? cur?.availablePackages?.[0] ?? null;
}

// Paywall wording, depending on whether this Apple ID can still get the free trial.
// Returns { title, terms, button } or null if the product can't be loaded.
function buildOffer(priceString, trialText) {
  return trialText
    ? {
        title: 'Start your free trial',
        terms: `${trialText} free, then ${priceString} per month. ${RENEW_TEXT}`,
        button: 'Start free trial',
      }
    : {
        title: 'Subscribe',
        terms: `${priceString} per month. ${RENEW_TEXT}`,
        button: `Subscribe for ${priceString}/month`,
      };
}

// Paywall text built from the App Store product (price localised by Apple).
export async function offer() {
  if (mockState()) {
    // 'new' = never had the trial; 'expired' = trial already used.
    return buildOffer('£2.99', mockState() === 'new' ? '2 weeks' : null);
  }
  const pkg = await getPackage();
  if (!pkg) return null;
  const p = pkg.product;
  const intro = p.introPrice;
  let trialText = null;
  if (intro && intro.price === 0) {
    // Ask Apple (via RevenueCat) whether this Apple ID can still use the introductory offer.
    let eligible = false;
    try {
      const res = await (await rc()).checkTrialOrIntroductoryPriceEligibility({
        productIdentifiers: [p.identifier],
      });
      eligible = res?.[p.identifier]?.status === 2; // 2 = ELIGIBLE
    } catch (e) {
      console.warn('[SUB] eligibility check failed', e);
    }
    if (eligible) {
      const n = intro.periodNumberOfUnits;
      trialText = `${n} ${intro.periodUnit.toLowerCase()}${n > 1 ? 's' : ''}`;
    }
  }
  return buildOffer(p.priceString, trialText);
}

// Returns true if the user now has access, false if they closed the Apple sheet.
export async function purchase() {
  if (mockState()) {
    // Stand-in for the Apple purchase sheet: OK = buy, Cancel = user closed the sheet.
    if (!confirm('[TEST] Simulated Apple purchase sheet.\n\nOK = subscribe, Cancel = close')) return false;
    setMock('active');
    return true;
  }
  const pkg = await getPackage();
  if (!pkg) throw new Error('Subscription not available right now.');
  try {
    const { customerInfo } = await (await rc()).purchasePackage({ aPackage: pkg });
    return isActive(customerInfo);
  } catch (e) {
    if (e?.userCancelled || e?.code === '1') return false;
    throw e;
  }
}

export async function restore() {
  if (mockState()) return mockState() === 'active';
  const { customerInfo } = await (await rc()).restorePurchases();
  return isActive(customerInfo);
}
