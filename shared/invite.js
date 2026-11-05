// invite.js — the "Invite a friend" / "Share Naptio" window, shared by every page.
//
// Usage:
//   <script src="/hungryface/shared/invite.js" defer></script>
//   …then any element with id="menu-invite" or [data-naptio-invite] opens it.
//   Optional on the trigger: data-invite-title="Share Naptio".
//   Pages that inject their trigger later (the sidebar) call window.NaptioInvite.attach(el).
//
// The QR code is drawn on this device (QRious): the link is never sent to a QR service.
(() => {
  if (window.NaptioInvite) { window.NaptioInvite.scan(); return; }

  const CSS_URL = '/hungryface/shared/invite.css';
  const QRIOUS_URL = 'https://cdn.jsdelivr.net/npm/qrious@4.0.2/dist/qrious.min.js';
  const DEFAULT_URL = 'https://hungryfaceai.github.io/hungryface/home/';
  const SHARE_TEXT = 'Naptio turns two phones into a private baby monitor. Take a look:';
  const DEFAULT_TITLE = 'Invite a friend';

  let modal, canvas, urlText, btnCopy, btnShare, btnClose, statusEl, titleEl;
  let lastFocus = null, qrReady = null, drawnFor = null;

  const inviteUrl = () => window.HF_INVITE_URL || document.body?.dataset?.inviteUrl || DEFAULT_URL;

  function ensureCss() {
    if (document.querySelector('link[data-naptio-invite-css]')) return;
    const l = document.createElement('link');
    l.rel = 'stylesheet'; l.href = CSS_URL; l.setAttribute('data-naptio-invite-css', '');
    document.head.appendChild(l);
  }

  function build() {
    if (modal) return;
    ensureCss();
    modal = document.createElement('div');
    modal.id = 'invite-modal'; modal.className = 'invite-modal';
    modal.setAttribute('role', 'dialog'); modal.setAttribute('aria-modal', 'true'); modal.setAttribute('aria-hidden', 'true');
    modal.setAttribute('aria-labelledby', 'invite-title'); modal.setAttribute('aria-describedby', 'invite-desc');
    modal.innerHTML =
      '<div class="invite-sheet">' +
        '<button type="button" class="invite-close" aria-label="Close">' +
          '<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>' +
        '</button>' +
        '<h2 id="invite-title" class="invite-title"></h2>' +
        '<p id="invite-desc" class="invite-desc">Share Naptio with another parent. They can scan this code with their phone’s camera, or you can send them the link.</p>' +
        '<div class="invite-qr-wrap"><canvas id="invite-qr" class="invite-qr" width="440" height="440" role="img" aria-label="QR code linking to the Naptio home page"></canvas></div>' +
        '<p class="invite-url"><span class="invite-url-label">Link</span> <span id="invite-url" class="invite-url-text"></span></p>' +
        '<div class="invite-actions">' +
          '<button id="invite-share" type="button" class="invite-btn invite-btn-primary">Share</button>' +
          '<button id="invite-copy" type="button" class="invite-btn">Copy link</button>' +
        '</div>' +
        '<p class="invite-status" id="invite-status" role="status" aria-live="polite"></p>' +
      '</div>';
    document.body.appendChild(modal);
    canvas = modal.querySelector('#invite-qr'); urlText = modal.querySelector('#invite-url');
    btnCopy = modal.querySelector('#invite-copy'); btnShare = modal.querySelector('#invite-share');
    btnClose = modal.querySelector('.invite-close'); statusEl = modal.querySelector('#invite-status');
    titleEl = modal.querySelector('#invite-title');

    modal.addEventListener('click', (e) => { if (e.target === modal) close(); });
    btnClose.addEventListener('click', close);
    btnCopy.addEventListener('click', copyLink);
    btnShare.addEventListener('click', async () => {
      try { await navigator.share({ title: 'Naptio', text: SHARE_TEXT, url: inviteUrl() }); }
      catch (err) { if (!err || err.name !== 'AbortError') copyLink(); }   // cancelled: do nothing; failed: copy instead
    });
  }

  function loadQRious() {
    if (window.QRious) return Promise.resolve();
    return qrReady || (qrReady = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = QRIOUS_URL;
      s.onload = resolve; s.onerror = () => { qrReady = null; reject(new Error('QR library failed to load')); };
      document.head.appendChild(s);
    }));
  }
  async function drawQR(url) {
    if (drawnFor === url) return;
    try {
      await loadQRious();
      // Dark modules on white, with a quiet zone: the most reliable for phone cameras.
      new window.QRious({ element: canvas, value: url, size: 440, padding: 24, level: 'M', background: '#ffffff', foreground: '#000000' });
      canvas.classList.remove('invite-qr-failed'); drawnFor = url;
    } catch {
      canvas.classList.add('invite-qr-failed');
      canvas.setAttribute('aria-label', 'QR code unavailable — use Share or Copy link');
    }
  }

  function say(msg) { statusEl.textContent = msg; clearTimeout(say.t); say.t = setTimeout(() => { statusEl.textContent = ''; }, 2500); }

  function focusables() { return [...modal.querySelectorAll('button:not([hidden])')].filter(b => b.offsetParent !== null); }
  function onKey(e) {
    if (e.key === 'Escape') { e.preventDefault(); close(); return; }
    if (e.key !== 'Tab') return;
    const f = focusables(); if (!f.length) return;
    const first = f[0], last = f[f.length - 1];
    if (!modal.contains(document.activeElement)) { e.preventDefault(); first.focus(); }
    else if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }

  function open(trigger) {
    build();
    const url = inviteUrl();
    lastFocus = trigger || document.activeElement;
    titleEl.textContent = (trigger && trigger.dataset && trigger.dataset.inviteTitle) || DEFAULT_TITLE;
    // Share where the device supports it (phones); otherwise Copy is the main action.
    const canShare = 'share' in navigator && (!navigator.canShare || navigator.canShare({ url }));
    btnShare.hidden = !canShare; btnCopy.classList.toggle('invite-btn-primary', !canShare);
    urlText.textContent = url.replace(/^https?:\/\//, '').replace(/\/$/, '');
    statusEl.textContent = '';
    drawQR(url);
    modal.classList.add('visible');
    modal.setAttribute('aria-hidden', 'false');
    document.addEventListener('keydown', onKey);
    setTimeout(() => (focusables().find(b => b.classList.contains('invite-btn-primary')) || btnClose).focus(), 0);
  }
  function close() {
    if (!modal || !modal.classList.contains('visible')) return;
    modal.classList.remove('visible');
    modal.setAttribute('aria-hidden', 'true');
    document.removeEventListener('keydown', onKey);
    if (lastFocus && lastFocus.focus && document.contains(lastFocus)) lastFocus.focus();
  }

  async function copyLink() {
    try {
      await navigator.clipboard.writeText(inviteUrl());
      say('Link copied'); btnCopy.textContent = 'Copied';
      setTimeout(() => { btnCopy.textContent = 'Copy link'; }, 1500);
    } catch {
      const r = document.createRange(); r.selectNodeContents(urlText);
      const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r);
      say('Press and hold the link to copy it');
    }
  }

  function attach(el) {
    if (!el || el.dataset.naptioInviteBound) return;
    el.dataset.naptioInviteBound = '1';
    el.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); open(el); });
  }
  function scan() { document.querySelectorAll('#menu-invite, [data-naptio-invite]').forEach(attach); }

  window.NaptioInvite = { attach, open, close, scan };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', scan, { once: true });
  else scan();
})();
