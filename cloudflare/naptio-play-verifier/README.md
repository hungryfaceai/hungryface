# Naptio subscription verification Worker

This directory stores the proposed Cloudflare Worker source. Committing to GitHub **does not deploy** this Worker. Keep the live Cloudflare Worker unchanged until independently tested.

## Files

- `naptio-play-verifier.proposed.js`: replacement Cloudflare Worker implementation.
- `subscription-android.proposed.js`: replacement Android TWA subscription adapter.
- `subscription-gate.proposed.js`: drop-in platform-router expansion preserving the old iOS calls.
- `index_inprogress.integration.proposed.md`: candidate-page changes needed to display states. Do not edit live `index.html` at this stage.

When implementing, rename the proposal files to the actual deployed names. The import in `subscription-gate.proposed.js` intentionally refers to `./subscription-android.js`, the production filename.

## Worker HTTP response contract

| HTTP | `status` | `active` | Meaning |
|---|---|---|---|
| 200 | `ACTIVE` | true | Google verified matching Naptio subscription and unexpired entitlement; acknowledgement can independently be `ACKNOWLEDGED` or `PENDING_RETRY`. |
| 200 | `INACTIVE` | false | Authoritatively no current entitlement, including expired, on hold, paused, cancelled pending purchase, or product mismatch. |
| 200 | `PENDING` | false | Payment is still pending in Google Play. |
| 200 | `INVALID_PURCHASE` | false | Purchase lookup definitively returned 404/410. |
| 400 | `INVALID_PURCHASE` | false | Empty/invalid supplied token. |
| 429 | `UNAVAILABLE` | false | Optional Worker limiter rejected request. |
| 503 | `UNAVAILABLE` | false | Google auth/API, malformed response, timeout, Worker configuration, or rate-limiter failure. Retryable. |

Do **not** use `active` alone when showing purchase failure UI; use `status`.
Any non-`ACTIVE` response is unauthorized for the camera sender.

## Example: active but acknowledgement retry required

```json
{
  "status": "ACTIVE",
  "active": true,
  "reason": "ENTITLED",
  "retryable": false,
  "state": "SUBSCRIPTION_STATE_ACTIVE",
  "acknowledgement": "PENDING_RETRY",
  "expiresAt": "2026-12-01T00:00:00Z",
  "testPurchase": false
}
```

The date above is an example only, never a fabricated live entitlement.
A failed acknowledgement no longer denies an independently verified active purchase.
However, no database means acknowledgement is only retried on later verification requests; if the user closes the application and never returns, the purchase may still be refunded after the Google acknowledgement deadline. RTDN or a durable retry mechanism is recommended for stronger guarantees.

## Config notes

Existing required Worker environment variables stay unchanged:

```
GOOGLE_SERVICE_ACCOUNT_EMAIL
GOOGLE_PRIVATE_KEY
GOOGLE_PLAY_PACKAGE_NAME  (must be io.github.hungryfaceai.twa)
GOOGLE_PLAY_PRODUCT_ID   (must be naptio_01)
```

Optional Cloudflare Workers rate-limiting binding name:
`PLAY_VERIFY_LIMITER`.

For example, a Wrangler configuration can include the following binding. Choose the namespace ID and threshold for your deployment, not blindly this example:

```toml
[[ratelimits]]
name = "PLAY_VERIFY_LIMITER"
namespace_id = "1001" # choose a unique namespace ID for your account
[ratelimits.simple]
limit = 30
period = 60
```

This is **per IP and Cloudflare location**, can affect users sharing a NAT, and is not a guaranteed global Google quota guard. The binding is intentionally optional, so the proposed Worker still runs without altering Cloudflare configuration, but abuse protection then remains absent.

## Important limitations and test requirements

1. The Worker still authenticates the Google purchase, not the user/device. CORS is not cryptographic TWA authentication and token sharing is possible.
2. The native Android billing bridge cannot be verified by unit tests; test `getDetails()`, trial disclosures and `PaymentRequest.show()` on the actual Play-installed TWA.
3. The proposal preserves iOS RevenueCat and existing `subscription.bundle.js` untouched.
4. The candidate sender page must switch to `accessStatus()`, `purchaseStatus()` and `restoreStatus()` to surface richer states; merely replacing the Android adapter and Worker keeps the old boolean-only UI.
5. Validate old Android versions before the web cutover; they may not have Play Billing support.
6. This is an unmerged proposal, not production-tested software. A local simulated test covered worker active, acknowledgement failure, pending, expired, product mismatch, invalid token, malformed Google response and auth error, plus Android adapter pending/recovery and single-checkout behavior.
## Manual Cloudflare deployment

1. Cloudflare Dashboard → Workers & Pages → `naptio-play-verifier` → Edit code. Copy the **full contents of [`worker.js`](./worker.js)**, replacing the existing Worker code only after backing up the previous deployed version.
2. Keep the existing route `/google-play/verify` and Worker hostname unchanged; the Android adapter sends POST JSON to that route. The `/` GET health endpoint should continue to return `ok: true`.
3. In the Worker's Settings → Variables and Secrets, retain the four existing server-side values: `GOOGLE_SERVICE_ACCOUNT_EMAIL`, `GOOGLE_PRIVATE_KEY`, `GOOGLE_PLAY_PACKAGE_NAME=io.github.hungryfaceai.twa`, and `GOOGLE_PLAY_PRODUCT_ID=naptio_01`. Use **Secret** (encrypted), not public browser variables, for the private key. Do not paste credentials into this repository.
4. Deploy the Worker separately from GitHub Pages. If desired, configure a Cloudflare `PLAY_VERIFY_LIMITER` rate-limit binding; without this optional binding, requests are *not* limited by the Worker.
5. Validate with Google Play test purchases: active, pending, cancelled-unexpired, expired, product mismatch, transient Google authentication/API failures, and acknowledgement failure. Never use an invented token to assert an active purchase.
6. Rollback: use Cloudflare deployment history to restore the preceding Worker version independently of the GitHub web code.

The production `webrtc/sender/secure/index.html` must **not** be replaced by `index_inprogress.html` as part of this change. Android enforcement remains off until the native version-compatibility and Play checkout tests pass.
