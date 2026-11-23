// Platform subscriptions router. Production index.html still imports the iOS bundle directly until cutover.
import * as ios from './subscription.bundle.js';
import * as android from './subscription-android.js';

function isNativeIos() {
  const cap = window.Capacitor;
  return cap?.isNativePlatform?.() === true &&
    cap?.getPlatform?.() === 'ios' &&
    cap?.isPluginAvailable?.('Purchases') === true;
}

export function subscriptionPlatform() {
  if (isNativeIos()) return 'ios';
  if (android.isNaptioTwa()) return 'android';
  return 'unsupported';
}

function adapter() {
  const platform = subscriptionPlatform();
  if (platform === 'ios') return ios;
  if (platform === 'android') return android;
  return null;
}

export function needsSubscription() {
  return adapter()?.needsSubscription?.() ?? false;
}

// Compatibility with old callers. Browser camera blocking remains in sender page.
export async function hasAccess() {
  const selected = adapter();
  if (!selected) return true;
  return selected.hasAccess();
}

export async function offer() {
  const selected = adapter();
  if (!selected) throw new Error('Subscriptions are unavailable on this platform.');
  return selected.offer();
}

export async function purchase() {
  const selected = adapter();
  if (!selected) throw new Error('Purchases are unavailable on this platform.');
  return selected.purchase();
}

export async function restore() {
  const selected = adapter();
  if (!selected) return false;
  return selected.restore();
}

// Rich APIs for the candidate page: Android gets full verifier states;
// iOS remains on its established RevenueCat boolean flow.
export async function accessStatus() {
  if (subscriptionPlatform() === 'android') return android.accessStatus();
  if (subscriptionPlatform() === 'ios') {
    const active = await ios.hasAccess();
    return { status: active ? 'ACTIVE' : 'INACTIVE', active };
  }
  return { status: 'UNSUPPORTED', active: false };
}

export async function purchaseStatus() {
  if (subscriptionPlatform() === 'android') return android.purchaseStatus();
  throw new Error('Use purchase() for the existing iOS RevenueCat flow.');
}

export async function restoreStatus() {
  if (subscriptionPlatform() === 'android') return android.restoreStatus();
  throw new Error('Use restore() for the existing iOS RevenueCat flow.');
}

export function subscriptionStatusMessage(result) {
  return android.messageForStatus(result);
}