import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

test('candidate Camera page contains a dedicated billing compatibility notice', async () => {
  const html = await readFile(new URL('../index_inprogress.html', import.meta.url), 'utf8');
  const ids = [
    'androidBillingNotice', 'androidBillingTitle', 'androidBillingDescription',
    'androidBillingCompatibility', 'androidBillingUpdateLink',
    'androidBillingChromeLink', 'androidBillingRetry', 'androidBillingClose',
  ];
  for (const id of ids) {
    assert.match(html, new RegExp('id="' + id + '"'), id);
  }
  assert.match(html, /https:\/\/play\.google\.com\/store\/apps\/details\?id=io\.github\.hungryfaceai\.twa/);
  assert.match(html, /https:\/\/play\.google\.com\/store\/apps\/details\?id=com\.android\.chrome/);
});

test('start preflights installed Android capability before checking entitlement', async () => {
  const html = await readFile(new URL('../index_inprogress.html', import.meta.url), 'utf8');
  const handler = html.slice(html.indexOf('async function startOrStop()'));
  const preflight = handler.indexOf('const readiness = await billingReadiness();');
  const entitlement = handler.indexOf('const result = await accessStatus();');
  const media = handler.indexOf('roomSel.disabled = true;');
  assert.ok(preflight >= 0, 'Android readiness preflight must exist');
  assert.ok(entitlement > preflight, 'entitlement checked only after capability');
  assert.ok(media > entitlement, 'media starts only after entitlement');
  assert.match(handler, /if \(readiness\.status !== 'READY'\) \{\s*showAndroidBillingNotice\(readiness\);\s*return;/);
});

test('missing capabilities request update; transient failures only present Retry', async () => {
  const html = await readFile(new URL('../index_inprogress.html', import.meta.url), 'utf8');
  const notice = html.slice(
    html.indexOf('function showAndroidBillingNotice(result)'),
    html.indexOf('// Distinct recovery state: do not open checkout')
  );
  assert.match(notice, /result\.status === 'UPDATE_REQUIRED'/);
  assert.match(notice, /androidBillingUpdateLink\.hidden = !update/);
  assert.match(notice, /androidBillingChromeLink\.hidden = !update/);
  assert.match(notice, /androidBillingRetry\.textContent = update \? 'Check again' : 'Retry'/);
  assert.match(notice, /if \(readiness\.status === 'READY'\) \{[\s\S]*await startOrStop\(\);/);
  assert.match(html, /#androidBillingNotice\[hidden\] \{\s*display: none;/);
});

test('live sender and staged sender share checkout feedback implementation', async () => {
  const [live, staged] = await Promise.all([
    readFile(new URL('../index.html', import.meta.url), 'utf8'),
    readFile(new URL('../index_inprogress.html', import.meta.url), 'utf8'),
  ]);
  assert.equal(live, staged, 'live and candidate must remain aligned for this fix');
  assert.match(live, /id="pwFeedback" role="status" hidden/);
  assert.match(live, /#pwFeedback\[hidden\] \{\s*display: none;/);
  assert.match(live, /pwFeedback\.hidden = true;\s*pwFeedback\.textContent = ''/);
});

test('explicit checkout cancellation is acknowledged without removing the subscription offer', async () => {
  const live = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const handler = live.slice(live.indexOf("pwBuy.addEventListener('click'"), live.indexOf("pwRestore.addEventListener('click'"));
  assert.match(handler, /if \(result\.status === 'CANCELLED'\) \{[\s\S]*pwFeedback\.textContent = subscriptionStatusMessage\(result\);[\s\S]*pwFeedback\.hidden = false;[\s\S]*return;/);
  assert.doesNotMatch(handler, /if \(result\.status === 'CANCELLED'\) return;/);
});

test('Android checkout errors show recovery with a specific title and no second buy button', async () => {
  const live = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const start = live.indexOf('function showAndroidRecovery(result)');
  const end = live.indexOf("pwClose.addEventListener('click'", start);
  assert.ok(start > 0 && end > start);
  const recovery = live.slice(start, end);
  assert.match(recovery, /PLAY_CHECKOUT_DID_NOT_OPEN/);
  assert.match(recovery, /PLAY_CHECKOUT_ABORTED_UNCONFIRMED/);
  assert.match(recovery, /pwBuy\.hidden = true/);
  assert.match(recovery, /pwRestore\.textContent = 'Check subscription'/);
  assert.match(recovery, /paywall\.hidden = false/);
});

test('live sender uses the shared subscription adapter for Android app detection', async () => {
  const [live, staged] = await Promise.all([
    readFile(new URL('../index.html', import.meta.url), 'utf8'),
    readFile(new URL('../index_inprogress.html', import.meta.url), 'utf8'),
  ]);
  assert.equal(live, staged, 'production and staged sender should remain aligned');
  const start = live.indexOf('function isInstalledMobileApp()');
  const end = live.indexOf('function shouldBlockBrowserStreaming()', start);
  assert.ok(start >= 0 && end > start);
  const detector = live.slice(start, end);
  assert.match(detector, /return subscriptionPlatform\(\) === 'android'/);
  assert.match(detector, /cap\?\.getPlatform\?\.\(\) === 'ios'/);
  assert.doesNotMatch(detector, /sessionStorage|document\.referrer/,
    'no second copy of Android referrer detection in the page');
  assert.match(live, /if \(shouldBlockBrowserStreaming\(\)\) \{\s*startBtn\.disabled = true;/);
  assert.match(live, /const readiness = await billingReadiness\(\);/);
  assert.match(live, /const result = await accessStatus\(\);/,
    'valid app detection must still require a subscription check');
});
