// alert-banner.js
// Renders/dismisses the Cry Alert banner and provides IndexedDB helpers.

const AUDIO_DISMISS_SUPPRESS_MS = 1 * 60 * 1000; // 1 minute

let audioBanner = null;
let audioBannerTime = null;
let audioDismissBtn = null;

let audioSuppressUntilMs = 0;
let audioKeepReminding = true;

function injectStyleOnce(id, css) {
  if (document.getElementById(id)) return;
  const s = document.createElement('style');
  s.id = id;
  s.textContent = css;
  document.head.appendChild(s);
}

function fmtTimeHM(ms) {
  return new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export function setupAudioBanner() {
  // Inject styles for the banner
  injectStyleOnce('audio-alert-banner-css', `
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
    .alert-banner__actions { margin-top: 6px; display: inline-flex; gap: 8px; align-items: center; }
    .banner-link {
      background: none; border: 0; color: #fff; padding: 0;
      font: inherit; cursor: pointer; text-decoration: underline;
    }
    .banner-link:active { transform: scale(0.98); }
  `);

  // Mount HTML
  const host = document.getElementById('audioBannerHost') || document.body;
  const wrap = document.createElement('div');
  wrap.innerHTML = `
    <div id="audioBanner" class="alert-banner hidden" role="alert" aria-live="assertive">
      <div class="alert-banner__text">
        Cry alert — Audio detection triggered at
        <span id="audioBannerTime">--:--</span>.
      </div>
      <div class="alert-banner__actions">
        <button id="audioDismissBtn" class="banner-link" type="button">Dismiss</button>
      </div>
    </div>
  `.trim();
  const node = wrap.firstChild;
  host.appendChild(node);

  audioBanner = node;
  audioBannerTime = node.querySelector('#audioBannerTime');
  audioDismissBtn = node.querySelector('#audioDismissBtn');

  audioDismissBtn?.addEventListener('click', () => {
    audioKeepReminding = false;
    audioSuppressUntilMs = Date.now() + AUDIO_DISMISS_SUPPRESS_MS;
    hideAudioBanner();
  });
}

export function showAudioBanner(whenMs) {
  if (!audioBanner) setupAudioBanner();
  if (!audioKeepReminding && Date.now() < audioSuppressUntilMs) return;
  if (audioBannerTime) audioBannerTime.textContent = fmtTimeHM(whenMs);
  audioBanner?.classList.remove('hidden');
}

export function hideAudioBanner() {
  audioBanner?.classList.add('hidden');
}

/* ===== IndexedDB helpers (shared with alerts-drawer) ===== */
let _alertDB = null;

export function openAlertDB() {
  if (_alertDB) return Promise.resolve(_alertDB);
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
    req.onsuccess = () => { _alertDB = req.result; resolve(_alertDB); };
    req.onerror   = () => reject(req.error);
  });
}

export async function saveAlertRecord(rec) {
  const db = await openAlertDB();
  await new Promise((resolve, reject) => {
    const tx = db.transaction('alerts', 'readwrite');
    tx.oncomplete = resolve;
    tx.onerror    = () => reject(tx.error);
    tx.objectStore('alerts').add(rec);
  });
  document.dispatchEvent(new CustomEvent('alerts:changed', { detail: { action: 'add', record: rec } }));
}

export async function getAllAlerts() {
  const db = await openAlertDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('alerts', 'readonly');
    const store = tx.objectStore('alerts');
    const req = store.getAll();
    req.onsuccess = () => {
      const rows = (req.result || []).sort((a,b) => +new Date(b.startAt||0) - +new Date(a.startAt||0));
      resolve(rows);
    };
    req.onerror = () => reject(req.error);
  });
}
