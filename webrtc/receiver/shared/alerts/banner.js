// webrtc/receiver/shared/alerts/banner.js
// Generic alert banner styled like the audio page (red, top-centered)
// + free-form "dismiss for N minutes" input (default 10m).

const DEFAULT_SNOOZE_MIN = 10;                      // <— default 10 minutes
const LS_KEY_SNOOZE_MIN  = 'naptio-alerts-snooze-mins';

let bannerEl = null;
let timeEl = null;
let labelEl = null;
let dismissBtn = null;
let minsInput = null;

let suppressUntil = 0;

function injectStyleOnce(id, css) {
  if (document.getElementById(id)) return;
  const s = document.createElement('style');
  s.id = id;
  s.textContent = css;
  document.head.appendChild(s);
}

function fmtHM(ms) {
  return new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function getSavedSnoozeMinutes() {
  const raw = localStorage.getItem(LS_KEY_SNOOZE_MIN);
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : DEFAULT_SNOOZE_MIN;
}
function setSavedSnoozeMinutes(mins) {
  const m = Math.max(1, Math.round(Number(mins) || DEFAULT_SNOOZE_MIN));
  localStorage.setItem(LS_KEY_SNOOZE_MIN, String(m));
  return m;
}

export function setupAlertBanner() {
  // Styles copied to match audio banner look & feel
  injectStyleOnce('shared-alert-banner-css', `
    .alert-banner{
      position: fixed;
      top: 12px; left: 50%; transform: translateX(-50%);
      background: #b91c1c; /* red-700 */
      color: #fff;
      padding: 10px 14px; border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.25);
      box-shadow: 0 10px 20px rgba(0,0,0,.35);
      z-index: 10000;
      font-size: 14px; text-align: center;
      max-width: min(92vw, 720px);
    }
    .alert-banner.hidden { display: none; }
    .alert-banner__actions { margin-top: 6px; display: inline-flex; gap: 10px; align-items: center; flex-wrap: wrap; justify-content: center; }
    .banner-link {
      background: none; border: 0; color: #fff; padding: 0;
      font: inherit; cursor: pointer; text-decoration: underline;
    }
    .banner-link:active { transform: scale(0.98); }
    .banner-inline { display: inline-flex; gap: 6px; align-items: center; }
    .alert-banner input.banner-input[type="number"] {
      width: 64px; padding: 4px 6px; border-radius: 8px;
      border: 1px solid #222;
      background: #7f1d1d;  /* #0b0b0b solid dark, matches audio page */
      color: #fff;
      font: inherit; -moz-appearance: textfield;
    }
    .banner-input::-webkit-outer-spin-button,
    .banner-input::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
  `);

  // Mount HTML
  const host = document.getElementById('alertBannerHost') || document.body;
  const wrap = document.createElement('div');
  wrap.innerHTML = `
    <div id="alertBanner" class="alert-banner hidden" role="alert" aria-live="assertive">
      <div class="alert-banner__text">
        Alert — <span id="alertBannerLabel">Motion detected</span>
        at <span id="alertBannerTime">--:--</span>.
      </div>
      <div class="alert-banner__actions">
        <label class="banner-inline" for="alertsDismissMins">
          Dismiss for
          <input id="alertsDismissMins" class="banner-input" type="number" min="1" step="1" inputmode="numeric" pattern="[0-9]*">
          min
        </label>
        <button id="alertDismissBtn" class="banner-link" type="button">Dismiss</button>
      </div>
    </div>
  `.trim();
  const node = wrap.firstChild;
  host.appendChild(node);

  bannerEl = node;
  timeEl = node.querySelector('#alertBannerTime');
  labelEl = node.querySelector('#alertBannerLabel');
  dismissBtn = node.querySelector('#alertDismissBtn');
  minsInput = node.querySelector('#alertsDismissMins');

  // Initialize snooze minutes (default 10)
  minsInput.value = String(getSavedSnoozeMinutes());
  minsInput.addEventListener('change', () => {
    const newVal = Math.max(1, Math.round(Number(minsInput.value) || DEFAULT_SNOOZE_MIN));
    minsInput.value = String(setSavedSnoozeMinutes(newVal));
  });

  dismissBtn?.addEventListener('click', () => {
    const mins = Math.max(1, Math.round(Number(minsInput.value) || getSavedSnoozeMinutes()));
    setSavedSnoozeMinutes(mins);            // persist the latest choice
    suppressUntil = Date.now() + mins * 60 * 1000;
    hideAlertBanner();
  });
}

export function showAlertBanner(whenMs = Date.now(), text) {
  if (!bannerEl) setupAlertBanner();
  if (Date.now() < suppressUntil) return; // snoozed
  if (timeEl) timeEl.textContent = fmtHM(whenMs);
  if (text && labelEl) labelEl.textContent = text;
  bannerEl?.classList.remove('hidden');
}
export function setAlertBannerText(t) {
  if (!bannerEl) setupAlertBanner();
  if (labelEl) labelEl.textContent = String(t || '');
}

export function hideAlertBanner() {
  bannerEl?.classList.add('hidden');
}

// Optional external helpers (kept for compatibility with prior code)
export function setSnoozeMinutes(mins) {
  const m = setSavedSnoozeMinutes(mins);
  if (minsInput) minsInput.value = String(m);
}
export function getSnoozeMinutes() {
  return getSavedSnoozeMinutes();
}
