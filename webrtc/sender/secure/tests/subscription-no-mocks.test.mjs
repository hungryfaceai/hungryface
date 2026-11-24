import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const canonical = new URL('../subscription.js', import.meta.url);
const duplicate = new URL('../subscription/subscription.js', import.meta.url);

test('both RevenueCat sources stay identical and contain no browser subscription mocks', async () => {
  const [a, b] = await Promise.all([readFile(canonical, 'utf8'), readFile(duplicate, 'utf8')]);
  assert.equal(a, b);
  assert.doesNotMatch(a, /submock|mockState|setMock|MOCK_PRICE|MOCK_KEY|Simulated Apple purchase sheet/i);
});

test('real iOS RevenueCat flows work; browser query/localStorage cannot impersonate entitlement', async () => {
  const src = await readFile(canonical, 'utf8');
  assert.match(src, /import \{ Purchases \} from '@revenuecat\/purchases-capacitor'/);

  let isActive = false;
  let trialEligible = true;
  let configured = 0;
  let purchases = 0;
  let restores = 0;
  const pkg = {
    product: {
      identifier: 'naptio_monthly',
      priceString: '£2.99',
      introPrice: { price: 0, periodNumberOfUnits: 3, periodUnit: 'DAY' },
    },
  };
  const fakePurchases = {
    configure: async () => { configured++; },
    getCustomerInfo: async () => ({
      customerInfo: { entitlements: { active: isActive ? { premium: {} } : {} } },
    }),
    getOfferings: async () => ({ current: { monthly: pkg } }),
    checkTrialOrIntroductoryPriceEligibility: async () => ({
      naptio_monthly: { status: trialEligible ? 2 : 0 },
    }),
    purchasePackage: async ({ aPackage }) => {
      assert.equal(aPackage, pkg);
      purchases++;
      isActive = true;
      return { customerInfo: { entitlements: { active: { premium: {} } } } };
    },
    restorePurchases: async () => {
      restores++;
      return { customerInfo: { entitlements: { active: { premium: {} } } } };
    },
  };
  const original = {
    window: globalThis.window,
    location: globalThis.location,
    localStorage: globalThis.localStorage,
    fake: globalThis.__testPurchases,
  };
  try {
    globalThis.__testPurchases = fakePurchases;
    globalThis.window = {
      Capacitor: { isNativePlatform: () => true, getPlatform: () => 'ios' },
    };
    globalThis.location = { search: '?submock=active' };
    globalThis.localStorage = {
      getItem: () => { throw Error('mock localStorage must not be read'); },
      setItem: () => { throw Error('mock localStorage must not be modified'); },
      removeItem: () => { throw Error('mock localStorage must not be modified'); },
    };

    const instrumented = src.replace(
      "import { Purchases } from '@revenuecat/purchases-capacitor';",
      'const Purchases = globalThis.__testPurchases;',
    );
    const url = 'data:text/javascript;base64,' + Buffer.from(instrumented).toString('base64');
    const subscription = await import(url);

    assert.equal(subscription.needsSubscription(), true);
    assert.equal(await subscription.hasAccess(), false);
    assert.equal(configured, 1);
    assert.match((await subscription.offer()).terms, /3 days free\nThen £2\.99\/month/);
    trialEligible = false;
    assert.doesNotMatch((await subscription.offer()).terms, /free trial|days free/i);
    assert.equal(await subscription.purchase(), true);
    assert.equal(purchases, 1);
    assert.equal(await subscription.restore(), true);
    assert.equal(restores, 1);
    assert.equal(await subscription.hasAccess(), true);

    // A browser cannot activate mock subscriptions using a URL or stale localStorage.
    globalThis.window.Capacitor = {
      isNativePlatform: () => false,
      getPlatform: () => 'web',
    };
    assert.equal(subscription.needsSubscription(), false);
    assert.equal(await subscription.hasAccess(), true);
    await assert.rejects(subscription.offer(), /RevenueCat is available only in the native iOS app/);
    await assert.rejects(subscription.purchase(), /RevenueCat is available only in the native iOS app/);
  } finally {
    for (const key of ['window', 'location', 'localStorage']) {
      if (original[key] === undefined) delete globalThis[key];
      else globalThis[key] = original[key];
    }
    if (original.fake === undefined) delete globalThis.__testPurchases;
    else globalThis.__testPurchases = original.fake;
  }
});
