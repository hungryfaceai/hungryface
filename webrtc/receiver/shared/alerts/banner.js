// Generic alert banner with persistent, tunable snooze (dismiss) duration.

let bannerEl, timeEl, dismissBtn, minsSelect;
let suppressUntil = 0;

const LS_SNOOZE_MINS = 'alerts:snoozeMins';

function fmtHM(ms) {
  const d = new Date(ms);
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  return `${hh}:${mm}`;
}

function getSavedSnoozeMinutes() {
  const n = Number(localStorage.getItem(LS_SNOOZE_MINS));
  return Number.isFinite(n) && n > 0 ? n : 15;
}

function setSavedSnoozeMinutes(mins) {
  try { localStorage.setItem(LS_SNOOZE_MINS, String(Math.max(1, Math.floor(mins)))) } catch {}
}

function setupAlertBanner(host = document.body) {
  if (bannerEl) return;
  const wrap = document.createElement('div');
  wrap.innerHTML = `
<div id="alertBanner" class="alert-banner hidden" style="position:fixed;left:16px;bottom:16px;z-index:9999;background:rgba(20,20,20,.92);border:1px solid rgba(255,255,255,.15);color:#fff;padding:10px 12px;border-radius:12px;backdrop-filter:blur(6px);font:13px/1.3 -apple-system,system-ui,Segoe UI,Roboto,sans-serif;">
  <div>
    <strong>Motion detected</strong>
    <span style="opacity:.8">at <span id="alertBannerTime"></span></span>
  </div>
  <div style="margin-top:8px;display:flex;gap:10px;align-items:center;">
    <label style="opacity:.9">Snooze
      <select id="alertsDismissMins" style="margin-left:6px;">
        <option value="5">5 min</option>
        <option value="15">15 min</option>
        <option value="30">30 min</option>
        <option value="60">60 min</option>
      </select>
    </label>
    <button id="alertDismissBtn" type="button" style="appearance:none;border:0;background:#2a2a2a;color:#fff;border-radius:10px;padding:6px 10px;cursor:pointer;">Dismiss</button>
  </div>
</div>
`.trim();
  bannerEl = wrap.firstChild;
  host.appendChild(bannerEl);

  timeEl = bannerEl.querySelector('#alertBannerTime');
  dismissBtn = bannerEl.querySelector('#alertDismissBtn');
  minsSelect = bannerEl.querySelector('#alertsDismissMins');

  const saved = getSavedSnoozeMinutes();
  if (minsSelect) minsSelect.value = String(saved);

  minsSelect?.addEventListener('change', () => setSavedSnoozeMinutes(Number(minsSelect.value)));

  dismissBtn?.addEventListener('click', () => {
    const mins = Number(minsSelect?.value) || getSavedSnoozeMinutes();
    setSavedSnoozeMinutes(mins);
    suppressUntil = Date.now() + mins * 60 * 1000;
    hideAlertBanner();
  });
}

export function showAlertBanner(whenMs = Date.now()) {
  if (!bannerEl) setupAlertBanner();
  if (Date.now() < suppressUntil) return; // currently snoozed
  if (timeEl) timeEl.textContent = fmtHM(whenMs);
  bannerEl?.classList.remove('hidden');
}

export function hideAlertBanner() {
  bannerEl?.classList.add('hidden');
}

export function setSnoozeMinutes(mins) {
  setSavedSnoozeMinutes(mins);
  if (minsSelect) minsSelect.value = String(mins);
}
export function getSnoozeMinutes() { return getSavedSnoozeMinutes(); }
