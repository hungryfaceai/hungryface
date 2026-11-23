# Naptio Android subscription — Play-installed validation (required before cutover)

**Current state:** Production `webrtc/sender/secure/index.html` still uses the iOS-only RevenueCat subscription gate. Android is free in production. The candidate Android flow is in `index_inprogress.html`. A Cloudflare health check alone does **not** prove that Google Play checkout, offer selection or acknowledgement works.

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
| User cancels Play checkout | No subscription granted; no scary infrastructure-error message |
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
- Free Viewer pairing and the PSK/WebRTC camera-to-viewer stream remain operational.
- Older Android TWA builds without the billing bridge require an explicit update/compatibility plan **before** `index_inprogress.html` is promoted.

## Cutover blocker

**Do not** replace the live `index.html` merely because the mocked Node tests pass or the Worker health endpoint responds. Real Play-installed trial/price/checkout verification and older-build handling remain required. No removal of legacy routes or server-side enforcement is included in this work.

## Local mocked tests

With Node.js 22:

```sh
node --test cloudflare/naptio-play-verifier/tests/worker-contract.test.mjs
node --test webrtc/sender/secure/tests/subscription-android.test.mjs
```

These tests use fake Play/Worker responses and never charge a customer. They cannot prove real Play billing configuration or free-trial eligibility.
