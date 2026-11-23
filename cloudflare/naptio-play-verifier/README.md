# Naptio — Google Play subscription verification Worker

**Source:** [worker.js](./worker.js). **Deployed service:** `naptio-play-verifier`.

This folder contains deployable Cloudflare Worker source, but **GitHub Pages does not deploy it**. It must be copied or deployed separately in Cloudflare after tests. The current production sender page (`webrtc/sender/secure/index.html`) continues using the existing iOS-only subscription gate; this change only stages the Android web code in `index_inprogress.html`.

## Required Cloudflare configuration

Keep these settings on the **existing** Cloudflare Worker; do not put the private key in GitHub or browser code:

| Variable or secret | Value |
|---|---|
| `GOOGLE_SERVICE_ACCOUNT_EMAIL` | Existing service account email |
| `GOOGLE_PRIVATE_KEY` | Existing private service-account PEM key (encrypted Worker secret) |
| `GOOGLE_PLAY_PACKAGE_NAME` | `io.github.hungryfaceai.twa` |
| `GOOGLE_PLAY_PRODUCT_ID` | `naptio_01` |

Google Cloud must continue to have Android Publisher API enabled and the Play Console service-account permissions configured. The Worker needs no subscription database. Treat all four values as server-only configuration and keep the email/key in encrypted secrets.

**Endpoint:** `POST https://naptio-play-verifier.bbmaugendre.workers.dev/google-play/verify`. Request JSON is `{"purchaseToken":"<token from Google Play>"}`. The Worker only allows browser requests with origin `https://hungryfaceai.github.io`. `GET /` remains a safe, unauthenticated health check.

## Paste into Cloudflare

1. Open **Cloudflare Dashboard → Workers & Pages → naptio-play-verifier → Edit code**.
2. Save a copy of the currently deployed Worker source or record the prior deployment version for rollback.
3. Copy **all contents of [worker.js](./worker.js)** into the Cloudflare Worker editor, replacing the old Worker code. Do not paste the README or Android JavaScript into Cloudflare.
4. Keep the same Worker name, hostname, URL path and encrypted secrets. Test the health route. Deploy only after the existing Google Play test purchase can be verified.
5. Exercise Android testing for active/unexpired, cancelled/unexpired, expired, pending, invalid token, Google API outage, and failed acknowledgement. An acknowledgement failure should produce `ACTIVE` with `acknowledgement: "PENDING_RETRY"` once Google has confirmed access.
6. If the Worker misbehaves, roll back in Cloudflare's deployment history. Restoring a web commit does **not** restore a Cloudflare deployment.

For a CLI deployment, use Wrangler with `main = "worker.js"` and the same Worker name. Reconfigure secrets through `wrangler secret put` rather than storing secret values in `wrangler.toml`.

## Optional request rate limiter

The Worker supports an optional binding named `PLAY_VERIFY_LIMITER`. Without it, the proposed Worker itself does **not** limit verification requests. When using Wrangler, add (after choosing a namespace ID valid for the Cloudflare account):

```toml
[[ratelimits]]
name = "PLAY_VERIFY_LIMITER"
namespace_id = "1001"

[ratelimits.simple]
limit = 30
period = 60
```

This is an approximate **per-client IP, per-Cloudflare-location** safeguard, *not* a precise global Google API quota. Choose a threshold that allows legitimate shared-network users to restore and check purchases.

## Response contract

| HTTP | `status` | `active` | Meaning |
|---|---|---|---|
| 200 | `ACTIVE` | true | Valid, unexpired purchase; `acknowledgement` is `ACKNOWLEDGED` or `PENDING_RETRY` |
| 200 | `INACTIVE` | false | No valid entitlement for the required subscription |
| 200 | `PENDING` | false | Payment remains pending |
| 200 | `INVALID_PURCHASE` | false | Definitively unrecognized/revoked token (e.g., Google 404/410) |
| 400 | `INVALID_PURCHASE` | false | Empty/invalid request token |
| 429 / 503 | `UNAVAILABLE` | false | Rate limited or temporarily unable to verify; retryable |

The Android frontend uses the `status` field to avoid interpreting infrastructure downtime as a cancelled purchase. Existing clients that only inspect `active === true` remain compatible.

**Acknowledgement limitation:** `PENDING_RETRY` is retried when the Worker next verifies that purchase. Without a durable queue or Google Play Real-time Developer Notifications, purchases that never trigger another verification could still be unacknowledged and refunded. Configure monitoring and consider RTDN for a stronger guarantee.

## Staged app files

- `webrtc/sender/secure/subscription-android.js` — richer Android access, checkout and restore statuses.
- `webrtc/sender/secure/subscription-gate.js` — platform router, retaining the iOS RevenueCat adapter.
- `webrtc/sender/secure/index_inprogress.html` — **candidate only**, rich-state UI; not the live sender.

No changes to `webrtc/sender/secure/index.html` or `subscription.bundle.js` are part of this release. Promotion of the candidate sender requires separate review, a Play-installed TWA billing test, and confirmation that older Android builds are handled.
