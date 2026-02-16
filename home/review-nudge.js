// /hungryface/home/review-nudge.js
// Naptio review nudge (native-only), shown on Home when returning from Camera/Viewer.

export function installReviewNudge(userOptions = {}) {
  const options = {
    daysAfterFirstSeen: 0, //7,
    snoozeDays: 7, //14,
    maxDeferrals: 3, //2,

    // Debug/testing: bypass all checks and always show the nudge card.
    // In force mode, we do NOT persist state (no deferrals/snooze/done written).
    force: false,

    // Where to send users:
    appStoreUrl: 'https://apps.apple.com/us/app/naptio/id6756505573',
    playStoreUrl: 'https://play.google.com/store/apps/details?id=io.github.hungryfaceai.twa',

    // Insert the card right after this element if found:
    insertAfterSelector: '.hero-ctas',

    // The routes we consider "Camera/Viewer"
    fromPatterns: [
      '/webrtc/sender/',
      '/webrtc/receiver/',
    ],

    // Simple daily cap (recommended). Set to false if you truly want it multiple times/day.
    oncePerDay: true,

    ...userOptions,
  };

  const persist = !options.force; // QA mode: no localStorage writes / no counters

  const runtime = getRuntime();
  if (!options.force && !runtime.allowed) return; // do nothing on web unless forced

  const state = loadState();

  // Set first_native_home_seen_at once (native-only) — only when persisting
  if (persist && !state.first_native_home_seen_at) {
    state.first_native_home_seen_at = Date.now();
    saveState(state);
  }

  // Must be returning from Camera/Viewer (unless forced)
  if (!options.force && !cameFromCameraOrViewer(options.fromPatterns)) return;

  // Stop conditions / eligibility (skipped in force mode)
  if (!options.force) {
    if (state.done) return;
    if ((state.deferral_count || 0) >= options.maxDeferrals) return;

    if (state.snooze_until && Date.now() < state.snooze_until) return;

    const eligibleAt =
      state.first_native_home_seen_at +
      options.daysAfterFirstSeen * 24 * 60 * 60 * 1000;

    if (Date.now() < eligibleAt) return;

    if (options.oncePerDay) {
      const today = dayStamp();
      if (state.last_shown_day === today) return;
    }
  }

  // Show UI
  const card = renderCard();
  injectCard(card, options.insertAfterSelector);

  // Mark shown (only when persisting)
  if (persist) {
    state.last_shown_day = options.oncePerDay ? dayStamp() : state.last_shown_day;
    saveState(state);
  }

  // Track "ignore" (user leaves Home without clicking)
  const visit = {
    visible: true,
    actionTaken: false,
    ignoreCounted: false,
  };

  function finalizeIgnoreIfNeeded() {
    if (!visit.visible) return;
    if (visit.actionTaken) return;
    if (visit.ignoreCounted) return;
    visit.ignoreCounted = true;

    if (!persist) return; // force mode: do not persist/counter anything

    incrementDeferralAndMaybeDone(state, options);
    // no snooze for ignore
    saveState(state);
  }

  // Count ignore on leaving Home (covers back, navigation, app background)
  window.addEventListener('pagehide', finalizeIgnoreIfNeeded, { capture: true });

  // Also count ignore on link navigation from Home (more deterministic)
  document.addEventListener(
    'click',
    (e) => {
      if (!visit.visible || visit.actionTaken || visit.ignoreCounted) return;

      const a = e.target?.closest?.('a[href]');
      if (!a) return;

      // Ignore clicks inside the card itself (buttons/close)
      if (a.closest && a.closest('#naptio-review-nudge')) return;

      const href = a.getAttribute('href') || '';
      if (!href || href === '#' || href.startsWith('javascript:')) return;

      // Links that open in new tab shouldn't count as "leaving"
      const target = (a.getAttribute('target') || '').toLowerCase();
      if (target === '_blank') return;

      // We're leaving Home -> count ignore once
      finalizeIgnoreIfNeeded();
    },
    { capture: true }
  );

  // Wire actions
  const btnReview = card.querySelector('[data-action="review"]');
  const btnNotNow = card.querySelector('[data-action="not-now"]');
  const btnClose  = card.querySelector('[data-action="close"]');

  btnReview?.addEventListener('click', async () => {
    visit.actionTaken = true;
    visit.visible = false;

    if (persist) {
      state.done = true; // stop forever immediately
      saveState(state);
    }

    removeCard(card);

    // Best-effort: if a native in-app review plugin exists, use it; otherwise open the store listing.
    try {
      await requestNativeReviewOrOpenStore(runtime, options);
    } catch {
      // ignore
    }
  });

  btnNotNow?.addEventListener('click', () => {
    visit.actionTaken = true;
    visit.visible = false;

    if (persist) {
      state.deferral_count = (state.deferral_count || 0) + 1;
      state.snooze_until = Date.now() + options.snoozeDays * 24 * 60 * 60 * 1000;

      if (state.deferral_count >= options.maxDeferrals) state.done = true;
      saveState(state);
    }

    removeCard(card);
  });

  btnClose?.addEventListener('click', () => {
    visit.actionTaken = true;
    visit.visible = false;

    if (persist) {
      // Dismiss counts as deferral, no snooze
      state.deferral_count = (state.deferral_count || 0) + 1;
      if (state.deferral_count >= options.maxDeferrals) state.done = true;
      saveState(state);
    }

    removeCard(card);
  });

  // ---- helpers ----

  function incrementDeferralAndMaybeDone(st, opt) {
    st.deferral_count = (st.deferral_count || 0) + 1;
    if (st.deferral_count >= opt.maxDeferrals) st.done = true;
  }

  function renderCard() {
    ensureStyles();

    const el = document.createElement('section');
    el.id = 'naptio-review-nudge';
    el.className = 'review-nudge';
    el.setAttribute('role', 'region');
    el.setAttribute('aria-label', 'Naptio review request');

    el.innerHTML = `
      <div class="review-nudge__inner">
        <button class="review-nudge__close" type="button" aria-label="Close" data-action="close">×</button>

        <div class="review-nudge__text">
          <div class="review-nudge__title">Keeping Naptio free for parents</div>
          <div class="review-nudge__body">
            If you have a minute, a store review helps other parents find Naptio and helps us keep it free.
          </div>
        </div>

        <div class="review-nudge__actions" role="group" aria-label="Review actions">
          <button class="review-nudge__btn review-nudge__btn--primary" type="button" data-action="review">
            Leave a review
          </button>
          <button class="review-nudge__btn" type="button" data-action="not-now">
            Not now
          </button>
        </div>
      </div>
    `;

    return el;
  }

  function injectCard(cardEl, afterSelector) {
    const anchor = document.querySelector(afterSelector);
    if (anchor && anchor.parentNode) {
      anchor.insertAdjacentElement('afterend', cardEl);
    } else {
      // fallback
      (document.querySelector('main') || document.body).appendChild(cardEl);
    }
  }

  function removeCard(cardEl) {
    try { cardEl.remove(); } catch {}
  }

  function ensureStyles() {
    if (document.getElementById('review-nudge-styles')) return;
  
    const style = document.createElement('style');
    style.id = 'review-nudge-styles';
    style.textContent = `
      .review-nudge{
        margin: 14px auto 0;
        width: min(920px, 100%);
        padding: 0 6px;
        box-sizing: border-box;
      }
  
      .review-nudge__inner{
        position: relative;
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 12px;
        align-items: center;
      
        /* Make it stand out more (Option A) */
        background: #101a33; /* slightly brighter tint */
        border: 1px solid rgba(122,162,255,0.45);
        border-radius: 16px;
      
        /* IMPORTANT: reserve space for the close button so it never overlaps */
        padding: 16px 56px 16px 16px;
      
        box-shadow:
          0 18px 48px rgba(0,0,0,0.65),
          0 0 0 1px rgba(122,162,255,0.10) inset;
      
        filter: drop-shadow(0 0 12px rgba(122,162,255,0.20));
      }
  
      /* Accent stripe to increase salience */
      .review-nudge__inner::before{
        content: "";
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
        border-radius: 16px 0 0 16px;
        background: #7aa2ff;
      }
  
      @media (max-width: 560px){
        .review-nudge__inner{
          grid-template-columns: 1fr;
          gap: 10px;
  
          /* Keep right padding for the close button on small screens too */
          padding: 16px 56px 14px 12px;
        }
      }
  
      .review-nudge__close{
        position: absolute;
        top: 10px; right: 10px;
        width: 34px; height: 34px;
        border-radius: 9999px;
  
        border: 1px solid rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.08);
        color: #fff;
  
        font-size: 22px;
        line-height: 1;
        cursor: pointer;
        z-index: 2;
      }
      .review-nudge__close:hover{ background: rgba(255,255,255,0.12); }
      .review-nudge__close:focus-visible{ outline: 2px solid #fff; outline-offset: 2px; }
  
      .review-nudge__title{
        font-weight: 800;
        font-size: 0.95rem;
        margin-bottom: 4px;
      }
      .review-nudge__body{
        opacity: 0.92;
        font-size: 0.9rem;
        line-height: 1.25;
      }
  
      .review-nudge__actions{
        display: inline-flex;
        gap: 10px;
        align-items: center;
        justify-content: flex-end;
        flex-wrap: wrap;
      }
      @media (max-width: 560px){
        .review-nudge__actions{
          justify-content: flex-start;
        }
      }
  
      .review-nudge__btn{
        appearance: none;
        border: 1px solid rgba(255,255,255,0.22);
        background: rgba(255,255,255,0.10);
        color: #fff;
        border-radius: 9999px;
        padding: 10px 12px;
        font-weight: 700;
        cursor: pointer;
        line-height: 1;
        font-size: 0.9rem;
      }
      .review-nudge__btn:hover{
        background: rgba(255,255,255,0.14);
      }
      .review-nudge__btn:focus-visible{
        outline: 2px solid #7aa2ff;
        outline-offset: 2px;
      }
  
      .review-nudge__btn--primary{
        background: #fff;
        color: #000;
        border-color: #fff;
      }
      .review-nudge__btn--primary:hover{
        background: #eaeaea;
      }
  
      @media (prefers-reduced-motion: reduce){
        .review-nudge__btn, .review-nudge__close { transition: none !important; }
      }
    `;
  
    document.head.appendChild(style);
  }
} // end installReviewNudge

/* ---------- State + runtime detection ---------- */

const LS_KEY = 'naptio_review_nudge_v1';

function loadState() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return { deferral_count: 0, done: false, snooze_until: null, last_shown_day: null, first_native_home_seen_at: null };
    const parsed = JSON.parse(raw);
    return {
      deferral_count: Number(parsed.deferral_count || 0),
      done: !!parsed.done,
      snooze_until: parsed.snooze_until ? Number(parsed.snooze_until) : null,
      last_shown_day: parsed.last_shown_day || null,
      first_native_home_seen_at: parsed.first_native_home_seen_at ? Number(parsed.first_native_home_seen_at) : null,
    };
  } catch {
    return { deferral_count: 0, done: false, snooze_until: null, last_shown_day: null, first_native_home_seen_at: null };
  }
}

function saveState(state) {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(state));
  } catch {
    // ignore storage errors
  }
}

function dayStamp() {
  // local date stamp
  const d = new Date();
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  return `${yyyy}-${mm}-${dd}`;
}

function cameFromCameraOrViewer(patterns) {
  try {
    const ref = (document.referrer || '').toString();
    if (!ref) return false;
    return patterns.some((p) => ref.includes(p));
  } catch {
    return false;
  }
}

function getRuntime() {
  const Cap = window.Capacitor;
  const isCapNative = !!Cap?.isNativePlatform?.();

  if (isCapNative) {
    const platform = Cap?.getPlatform?.(); // 'ios' | 'android' | 'web'
    const allowed = platform === 'ios' || platform === 'android';
    return { allowed, platform: allowed ? platform : 'web', via: 'capacitor' };
  }

  // No Capacitor: allow Android *installed* (TWA/PWA standalone), but not browser tab.
  const ua = (navigator.userAgent || '').toLowerCase();
  const isAndroid = ua.includes('android');
  const isStandalone =
    (window.matchMedia && (
      window.matchMedia('(display-mode: standalone)').matches ||
      window.matchMedia('(display-mode: fullscreen)').matches ||
      window.matchMedia('(display-mode: minimal-ui)').matches
    ));

  if (isAndroid && isStandalone) {
    return { allowed: true, platform: 'android', via: 'standalone' };
  }

  // iOS standalone should NOT count (you asked: iOS native only)
  return { allowed: false, platform: 'web', via: 'web' };
}

/* ---------- Review request / store open ---------- */

async function requestNativeReviewOrOpenStore(runtime, options) {
  const Cap = window.Capacitor;
  const url = runtime.platform === 'ios' ? options.appStoreUrl : options.playStoreUrl;

  // If you later add a dedicated in-app-review plugin, this will use it automatically if present.
  // Examples of possible plugin shapes:
  // - Cap.Plugins.InAppReview.requestReview()
  // - Cap.Plugins.RateApp.requestReview()
  // (We won't hard-depend on any plugin.)

  const plugins = Cap?.Plugins || {};

  // Best-effort: try any known method if present
  const maybeFns = [
    plugins?.InAppReview?.requestReview,
    plugins?.InAppReview?.requestReviewFlow,
    plugins?.RateApp?.requestReview,
    plugins?.StoreReview?.requestReview,
  ].filter(Boolean);

  for (const fn of maybeFns) {
    try {
      const res = fn.call(plugins);
      if (res && typeof res.then === 'function') await res;
      return;
    } catch {
      // try next
    }
  }

  // Fallback: open store listing
  // Prefer Capacitor Browser plugin if available (opens external browser / store more cleanly)
  if (plugins?.Browser?.open) {
    try {
      await plugins.Browser.open({ url });
      return;
    } catch {
      // fall through
    }
  }

  // Last resort
  try {
    window.open(url, '_blank', 'noopener');
  } catch {
    location.href = url;
  }
}
