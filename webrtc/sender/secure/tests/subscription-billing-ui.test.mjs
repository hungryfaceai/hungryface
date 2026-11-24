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
