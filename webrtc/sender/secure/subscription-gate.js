// Platform router for Naptio sender subscriptions.
//
// iOS:
//   RevenueCat through the native Capacitor Purchases plugin.
//
// Android:
//   Google Play Billing through the Trusted Web Activity billing bridge.
//
// Browser:
//   No store adapter. Browser camera streaming is restricted separately
//   by the sender page.

import * as ios from './subscription.bundle.js';
import * as android from './subscription-android.js';

function isNativeIos() {
  const cap = window.Capacitor;

  return (
    cap?.isNativePlatform?.() === true &&
    cap?.getPlatform?.() === 'ios' &&
    cap?.isPluginAvailable?.('Purchases') === true
  );
}

function adapter() {
  if (isNativeIos()) {
    return ios;
  }

  if (android.isNaptioTwa()) {
    return android;
  }

  return null;
}

export function needsSubscription() {
  return adapter()?.needsSubscription?.() ?? false;
}

export async function hasAccess() {
  const selected = adapter();

  if (!selected) {
    return true;
  }

  return selected.hasAccess();
}

export async function offer() {
  const selected = adapter();

  if (!selected) {
    throw new Error(
      'Subscriptions are unavailable on this platform.'
    );
  }

  return selected.offer();
}

export async function purchase() {
  const selected = adapter();

  if (!selected) {
    throw new Error(
      'Purchases are unavailable on this platform.'
    );
  }

  return selected.purchase();
}

export async function restore() {
  const selected = adapter();

  if (!selected) {
    return false;
  }

  return selected.restore();
}
