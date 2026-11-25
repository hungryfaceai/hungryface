# Naptio Android subscription — Play-installed validation (required before cutover)

**Current state:** Production `webrtc/sender/secure/index.html` now contains the staged Android billing integration (copied from `index_inprogress.html`). This does not establish a successful checkout. A Cloudflare health check alone does **not** prove that Google Play checkout, offer selection or acknowledgement works.

## Scope and decision
Naptio deliberately uses **client-side subscription deterrence**, not server-admitted WebRTC streaming. The Cloudflare Worker remains the authority for the subscription token but does not enforce admission at the signaling server. Legacy sender pages are intentionally untouched and will be dealt with separately.

## Test prerequisites

- A physical Android device with Naptio **version 2**, installed from Google Play via a valid internal/closed/production track. Do not sideload the AAB/APK for this test.
- Updated Chrome, Play Store and Play Services, signed in to the intended Google Play test account.
- Two Google Play test accounts, where possible: **trial eligible** and **trial already used**. Configure appropriate Play license testers.
- A paired free Viewer; do not modify the production sender URL merely to run this test.
- The Cloudflare Worker `naptio-play-verifier` deployed with its Google secrets and `PLAY_VERIFY_LIMITER` binding; health endpoint must return `ok:true`.
- Open `/hungryface/webrtc/sender/secure/index_inprogress.html` within the **same verified TWA session** (for example through a temporary internal test navigation or Chrome remote debugging). Opening this URL in an ordinary browser is not equivalent to the TWA.
- Use the Chrome remote debugger to observe the page/console, without saving or sharing a purchase token, Google credentials or purchase receipts.

## Pre-purchase checks (must be observed in a genuine Play-installed TWA)

1. Inspect the value returned by `offer()`. The offered product must be `naptio_01`, the base price must be localized and the period should be `P1M` if reported by the bridge. The `playDetails` metadata contains only non-sensitive product pricing and offer fields.
2. For an eligible UK tester, confirm whether `getDetails()` actually exposes `freeTrialPeriod: "P3D"`. Only then should Naptio display **3 days free, then £1.79/month**.
3. For the trial-ineligible tester, confirm the paywall does **not** falsely promise a trial.
4. Confirm the Google Play checkout itself displays the correct up-front charge, trial period (if eligible), monthly renewal amount/currency, and auto-renewal terms. **The store checkout is the final authority.**
5. For accounts outside the UK, compare the product's `price.currency` and `price.value` with Google's localized checkout; do not hard-code £1.79 globally.
6. Confirm `Manage or cancel subscription in Google Play` is visible from the Camera screen even when the paywall is dismissed; its linked subscription-management screen opens in Google Play.

Important: Android Browser Helper 1.2.0 maps Google Play Billing Library 8 subscriptions into one flat Digital Goods API item. When multiple plans/offers are eligible, `getDetails()` exposes only the **first** eligible plan/offer, not all offers. Missing `freeTrialPeriod` is not proof the Play account was ineligible; investigate the actual checkout.

## Transaction and verification checks

| Case | Expected UI and access |
|---|---|
| Active existing subscription | Camera starts; no purchase sheet |
| New purchase succeeds and Worker returns ACTIVE | Payment response completes; Camera can start |
| Explicitly identified user cancellation | Keep the existing offer visible with a small “checkout cancelled” message; no subscription granted |
| PaymentRequest.show() immediately rejects with AbortError: Invalid state | Show “Google Play checkout did not open”, Check subscription; never claim the user cancelled |
| PaymentRequest.show() rejects with an ambiguous AbortError | Show “Google Play checkout could not be confirmed”; Check subscription before a second purchase; no silent return |
| Other PaymentRequest.show() failures | Show a recovery message and Check subscription; no duplicate checkout |
| Payment remains pending | Recovery view shows Check subscription; Buy hidden, no second purchase prompted |
| Checkout returns token; Google initially says token invalid/inactive | A brief settling state; Check subscription rather than immediate repurchase |
| Worker temporarily unreachable / HTTP 503 | UNAVAILABLE, not “not subscribed”; no streaming until verified |
| Worker rate limited (429) | Recoverable UNAVAILABLE; user can retry after cooldown |
| Checkout returns no token / bridge errors after checkout | Check subscription before offering a second checkout |
| Ack API fails after valid Google lookup | ACTIVE may be granted with `acknowledgement: "PENDING_RETRY"`; retry must be monitored |
| Cancelled, paid-until-expiry subscription | ACTIVE until entitlement expiry |
| Expired, paused or account on hold | INACTIVE; Buy available only after Play status reconciliation |
| Returning subscriber after reinstall | Restore/Check subscription uses `listPurchases()` and Worker verification |
| Same user taps Buy repeatedly while checking | Only one PaymentRequest checkout at a time |
| Google account changed | No entitlement unless the active account has a valid matching purchase |

## iOS and production nonregression

- On iOS, the **existing RevenueCat** trial-eligibility, `purchasePackage()`, restore and cancellation behaviors must match production.
- The new Android management links must remain hidden on iOS and in ordinary browsers.
- An ordinary browser remains blocked from Camera streaming when monetization is on.
- **TWA restart detection:** `subscription-android.js` now recognises Android Play TWA sessions from either the Naptio `android-app://` referrer, the existing session marker, **or** Chrome's Play Billing `window.getDigitalGoodsService` function. Both live and staged Camera pages delegate installed-Android detection to `subscriptionPlatform()`, avoiding divergent logic. This is client-side detection, **not** authoritative app identity; a verified Google Play subscription is still required.
- After purchasing on a real Play-installed device, close and reopen Naptio from its **installed app icon**, and verify that the Start streaming control does not change to “Available in the mobile app” even if the initial referrer and sessionStorage marker are absent.
- A normal Android Chrome tab with no TWA referrer, stored TWA marker or Digital Goods API must remain blocked. An older TWA with its app referrer but no billing bridge must continue to display the update-required notice, not stream for free.
- Free Viewer pairing and the PSK/WebRTC camera-to-viewer stream remain operational.
- The staged Camera page now runs `billingReadiness()` before Android subscription lookup. Missing `getDigitalGoodsService`, missing `PaymentRequest`, null/invalid Digital Goods service => a dedicated **Update Naptio to continue** dialog. It links directly to the Naptio listing and Chrome on Google Play, shows an incompatibility note for older devices, and permits a fresh capability check.
- A billing bridge that exists but fails to initialize or times out => **Google Play Billing is temporarily unavailable**, with a **Retry** button. A transient outage must never be presented as an outdated installation.
- The update/retry dialog is separate from the paywall. It never offers a purchase, never grants entitlement, and after a successful retry it re-enters the normal access check.
- Verify the missing-API, missing-PaymentRequest, null-service, partial-service, initialization-error, and recovered-service scenarios in a genuine TWA where possible. A missing API does not prove the exact installed APK version; check Chrome and Play certification too.
- Older Android TWA builds without the billing bridge require an explicit update/compatibility plan **before** `index_inprogress.html` is promoted. This UI addresses the failure mode but real Play-installed version 1/version 2 tests remain required.

## Cutover blocker

**The live `index.html` has already been replaced with the Android-enabled candidate.** This rollout is not yet validated end-to-end: real Play-installed checkout, free-trial eligibility, purchase acknowledgement, and older-build handling must still be verified before considering Android monetisation production-ready. No removal of legacy routes or server-side enforcement is included in this work.

**AbortError limitation:** The PaymentRequest API does not reliably report whether Google Play's native checkout UI was actually displayed. Only an explicit cancellation indicator is treated as a confirmed user cancellation. `Invalid state` is treated as a launch failure; all other `AbortError` cases are treated as unconfirmed, with a visible message and purchase reconciliation before retry. The console records non-sensitive error names and reason codes, never purchase tokens or payment data.

## Local mocked tests

With Node.js 22:

```sh
node --test cloudflare/naptio-play-verifier/tests/worker-contract.test.mjs
node --test webrtc/sender/secure/tests/subscription-android.test.mjs
```

These tests use fake Play/Worker responses and never charge a customer. They cannot prove real Play billing configuration or free-trial eligibility.
