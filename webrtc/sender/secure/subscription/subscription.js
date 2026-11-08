// Subscription gate for the native iOS app (RevenueCat). In the browser and on Android it always allows access.
// Test mode (browser only): add ?submock=expired or ?submock=active to the URL; ?submock=off clears it.
const RC_IOS_KEY = 'appl_XXXXXXXXXXXXXXXX'; // RevenueCat public iOS API key
const ENTITLEMENT = 'premium';

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
const mockState = () => (isNativeIos() ? null : localStorage.getItem(MOCK_KEY)); // 'expired' | 'active' | null
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

// Paywall text built from the App Store product (price localised by Apple).
export async function offerText() {
  if (mockState()) {
    return '[TEST] 2 weeks free, then £2.99 per month. Renews automatically unless cancelled at least 24 hours before the end of the current period. Manage or cancel in your App Store settings.';
  }
  const pkg = await getPackage();
  if (!pkg) return null;
  const p = pkg.product;
  const intro = p.introPrice;
  const trial =
    intro && intro.price === 0
      ? `${intro.periodNumberOfUnits} ${intro.periodUnit.toLowerCase()}${intro.periodNumberOfUnits > 1 ? 's' : ''} free, then `
      : '';
  return `${trial}${p.priceString} per month. Renews automatically unless cancelled at least 24 hours before the end of the current period. Manage or cancel in your App Store settings.`;
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
