// alert-banner.js
// Mounts a red "Cry alert" banner and exposes a small API on window.alertBanner.
// Also provides IndexedDB helpers to persist alert episodes.

(function () {
  const STYLE_ID = 'audioBannerStyles';
  const BANNER_ID = 'audioBanner';
  const DISMISS_MS = 10 * 60 * 1000; // 10 minutes

  // --- inject CSS (once) ---
  function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const css = `
.alert-banner{
  position:fixed; top:12px; left:50%; transform:translateX(-50%);
  background:#b91c1c; color:#fff;
  padding:10px 14px; border-radius:12px;
  border:1px solid rgba(255,255,255,0.25);
  box-shadow:0 10px 20px rgba(0,0,0,.35);
  z-index:10000; font-size:14px; text-align:center;
  max-width:min(92vw,720px);
}
.alert-banner.hidden{ display:none; }
.alert-banner__actions{ margin-top:6px; display:inline-flex; gap:8px; align-items:center; }
.banner-link{ background:none; border:0; color:#fff; padding:0; font:inherit; cursor:pointer; text-decoration:underline; }
.banner-link:active{ transform:scale(0.98); }

/* lightweight badge style used in header */
.badge{
  display:inline-flex; align-items:center; justify-content:center;
  min-width:18px; height:18px; padding:0 6px; margin-left:6px;
  border-radius:999px; background:#374151; color:#fff; font-size:12px; font-weight:700;
}
`.trim();
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = css;
    document.head.appendChild(style);
  }

  // --- mount banner into host ---
  let bannerEl, bannerTimeEl, dismissBtn;
  let suppressUntilMs = 0;
  let keepReminding = true;

  function fmtHM(ms) {
    return new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function mountAudioBanner(hostId = 'audioBannerHost') {
    injectStyles();
    if (bannerEl) return; // already mounted
    let host = document.getElementById(hostId);
    if (!host) {
      host = document.createElement('div');
      host.id = hostId;
      document.body.appendChild(host);
    }
    host.innerHTML = `
      <div id="${BANNER_ID}" class="alert-banner hidden" role="alert" aria-live="assertive">
        <div class="alert-banner__text">
          Cry alert — Audio detection triggered at <span id="audioBannerTime">--:--</span>.
        </div>
        <div class="alert-banner__actions">
          <button id="audioDismissBtn" class="banner-link" type="button">Dismiss</button>
        </div>
      </div>
    `;
    bannerEl = document.getElementById(BANNER_ID);
    bannerTimeEl = document.getElementById('audioBannerTime');
    dismissBtn = document.getElementById('audioDismissBtn');
    dismissBtn?.addEventListener('click', () => {
      keepReminding = false;
      suppressUntilMs = Date.now() + DISMISS_MS;
      hideAudioBanner();
    });
  }

  function showAudioBanner(whenMs) {
    if (!bannerEl) mountAudioBanner();
    if (!keepReminding && Date.now() < suppressUntilMs) return;
    if (bannerTimeEl) bannerTimeEl.textContent = fmtHM(whenMs || Date.now());
    bannerEl?.classList.remove('hidden');
  }

  function hideAudioBanner() {
    bannerEl?.classList.add('hidden');
  }

  // --- IndexedDB helpers ---
  let _db = null;
  function openAlertDB() {
    if (_db) return Promise.resolve(_db);
    return new Promise((resolve, reject) => {
      const req = indexedDB.open('naptioAlerts', 1);
      req.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains('alerts')) {
          const st = db.createObjectStore('alerts', { keyPath: 'id', autoIncrement: true });
          st.createIndex('type', 'type', { unique: false });
          st.createIndex('startAt', 'startAt', { unique: false });
        }
      };
      req.onsuccess = () => { _db = req.result; resolve(_db); };
      req.onerror = () => reject(req.error);
    });
  }

  async function saveAlertRecord(rec) {
    const db = await openAlertDB();
    await new Promise((resolve, reject) => {
      const tx = db.transaction('alerts', 'readwrite');
      const store = tx.objectStore('alerts');
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error);
      // normalize + guard
      const row = {
        type: rec?.type || 'audio detection',
        startAt: rec?.startAt,
        endAt: rec?.endAt,
        avgScore: Number(rec?.avgScore ?? 0)
      };
      store.add(row);
    });
    document.dispatchEvent(new CustomEvent('alerts:changed', { detail: { action: 'add', record: rec } }));
  }

  async function getAllAlerts() {
    const db = await openAlertDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction('alerts', 'readonly');
      const store = tx.objectStore('alerts');
      const req = store.getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  }

  // public API
  const api = {
    mountAudioBanner,
    showAudioBanner,
    hideAudioBanner,
    openAlertDB,
    saveAlertRecord,
    getAllAlerts
  };

  // attach to window for easy access from inline module
  window.alertBanner = Object.freeze(api);

  // auto-mount once DOM is ready (in case host exists already)
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => mountAudioBanner());
  } else {
    mountAudioBanner();
  }
})();
