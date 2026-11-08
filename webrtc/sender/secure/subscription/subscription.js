// iOS-only subscription gate (RevenueCat). Browser and Android: always allowed.
const RC_IOS_KEY = 'appl_XXXXXXXXXXXXXXXX'; // RevenueCat public iOS key
const ENTITLEMENT = 'premium';

const cap = window.Capacitor;
export const needsSubscription = () => !!cap?.isNativePlatform?.() && cap.getPlatform() === 'ios';

let Purchases = null;
let ready = null;
function rc() {
  Purchases ??= cap.registerPlugin('Purchases');
  ready ??= Purchases.configure({ apiKey: RC_IOS_KEY });
  return ready.then(() => Purchases);
}
const isActive = (info) => !!info?.entitlements?.active?.[ENTITLEMENT];

export async function hasAccess() {
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

// Text for the paywall, built from the App Store product (price is localised by Apple).
export async function offerText() {
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

// Returns true if the user now has access, false if they cancelled.
export async function purchase() {
  const pkg = await getPackage();
  if (!pkg) throw new Error('Subscription not available right now.');
  try {
    const { customerInfo } = await (await rc()).purchasePackage({ aPackage: pkg });
    return isActive(customerInfo);
  } catch (e) {
    if (e?.userCancelled || e?.code === '1') return false; // user closed the Apple sheet
    throw e;
  }
}

export async function restore() {
  const { customerInfo } = await (await rc()).restorePurchases();
  return isActive(customerInfo);
}
