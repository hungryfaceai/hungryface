// shared/analytics.js — RETIRED.
//
// The old self-hosted analytics (15 s heartbeats to the signaling server, with a
// persistent install ID and server-side IP logging) has been removed. Traffic is
// now measured with Umami (cookieless; see the <script> tag in each page's <head>).
//
// This stub is kept only so any cached page or older app build that still imports
// it keeps working. It sends nothing, and clears the identifiers the old version stored.

try {
  localStorage.removeItem('naptio:installId');
  localStorage.removeItem('naptio:analytics:leader');
  sessionStorage.removeItem('naptio:installId');
} catch {}

export function installAnalytics() {
  return {
    event() {},
    newSession() {},
    setFeature() {},
    uninstall() {},
  };
}
