// /hungryface/shared/analytics.js
// Anonymous analytics with active-time heartbeats + single-tab reporting.
// Option: persistId=false to avoid a stored device ID until consent.

export function installAnalytics(opts = {}) {
  const {
    endpoint = '/a/evt',
    app = 'naptio',
    feature = inferFeature(),
    intervalMs = 15000,
    sampleRate = 1,          // e.g., 0.5 = 50%
    persistId = true,        // set false until user accepts analytics
    installIdKey = 'naptio:installId',
    onError = () => {},
  } = opts;

  if (Math.random() > sampleRate) return { track(){}, uninstall(){} };

  const storage = persistId ? window.localStorage : window.sessionStorage;
  let installId = storage.getItem(installIdKey);
  if (!installId) {
    installId = (crypto.randomUUID && crypto.randomUUID())
