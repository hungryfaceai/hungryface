// landing-pair-gate.js
// Minimal dependency-free gate for first-time users (0 PSKs).
// Shows a bottom sheet prompting to pair, then routes to your Pair Devices page.

export function installLandingPairGate(userOptions = {}) {
  // Guard against double-init
  if (window.__NaptioPairGateInstalled) return;
  window.__NaptioPairGateInstalled = true;

  const defaults = {
    cameraSelector: 'a[href*="/webrtc/sender/secure/"]',
    viewerSelector: 'a[href*="/webrtc/receiver/shared/alerts/dashboard/"]',
    pairRoute: '/hungryface/webrtc/pairpsk/',
    durations: { sheetInMs: 260, sheetOutMs: 220 },
    onEvent: () => {},
    // Optional overrides
    detectHasAnyPsk: null,
    validateToken: null,
    copy: null,
  };

  const opts = { ...defaults, ...userOptions };
  const onEvent = (name, data) => { try { opts.onEvent(name, data); } catch {} };

  // ---------- Copy with device nouns ----------
  const nouns = detectDeviceNouns();
  const COPY = normalizeCopy(opts.copy, nouns);

  // ---------- Styles (scoped) ----------
  injectStyles();

  // ---------- DOM (created once) ----------
  const dom = buildSheetDOM(COPY, opts.durations);
  let currentIntent = null; // 'camera' | 'viewer'
  let lastFocused = null;
  let isOpen = false;

  // ---------- Public-ish helpers ----------
  function openSheet(intent) {
    if (isOpen) return;
    currentIntent = intent;
    lastFocused = document.activeElement || null;

    // Context subline
    dom.subline.textContent = intent === 'camera'
      ? COPY.sublineCamera
      : COPY.sublineViewer;

    document.documentElement.classList.add('lpg-open');
    dom.scrim.style.display = 'block';
    dom.root.style.display = 'block';
    requestAnimationFrame(() => {
      dom.scrim.classList.add('lpg-visible');
      dom.root.classList.add('lpg-visible');
      dom.sheet.classList.add('lpg-in');
      isOpen = true;
      focusFirst(dom);
    });
    onEvent('sheet_open', { intent });
  }

  function closeSheet(reason = 'dismiss') {
    if (!isOpen) return;
    dom.scrim.classList.remove('lpg-visible');
    dom.sheet.classList.remove('lpg-in');
    setTimeout(() => {
      dom.root.classList.remove('lpg-visible');
      dom.root.style.display = 'none';
      dom.scrim.style.display = 'none';
      document.documentElement.classList.remove('lpg-open');
      isOpen = false;
      try { lastFocused && lastFocused.focus(); } catch {}
    }, opts.durations.sheetOutMs);
    onEvent('sheet_close', { reason, intent: currentIntent });
  }

  // ---------- Focus trap ----------
  function onKeydown(e) {
    if (!isOpen) return;
    if (e.key === 'Escape') {
      e.preventDefault();
      closeSheet('esc');
      return;
    }
    if (e.key === 'Tab') {
      trapTab(e, dom.sheet);
    }
  }

  // ---------- Wiring ----------
  //dom.scrim.addEventListener('click', () => closeSheet('scrim'));
  //dom.btnClose.addEventListener('click', () => closeSheet('close'));
  // Close when clicking outside the sheet (anywhere on the overlay that isn’t the sheet)
  dom.root.addEventListener('click', (e) => {
    if (!dom.sheet.contains(e.target)) closeSheet('outside');
  });
  dom.btnPrimary.addEventListener('click', () => {
    onEvent('pair_cta', { intent: currentIntent });
    // Route to Pair Devices
    try { window.location.href = opts.pairRoute; }
    catch { closeSheet('navigate-failed'); }
  });

  dom.howLink.addEventListener('click', (e) => {
    e.preventDefault();
    const expanded = dom.howBody.getAttribute('aria-hidden') === 'false';
    if (expanded) {
      dom.howBody.setAttribute('aria-hidden', 'true');
      dom.howLink.setAttribute('aria-expanded', 'false');
      dom.howBody.classList.remove('lpg-how-open');
      onEvent('how_collapsed', {});
    } else {
      dom.howBody.setAttribute('aria-hidden', 'false');
      dom.howLink.setAttribute('aria-expanded', 'true');
      dom.howBody.classList.add('lpg-how-open');
      onEvent('how_expanded', {});
    }
  });

  document.addEventListener('keydown', onKeydown);

  // Attach to Camera / Viewer links (progressive enhancement)
  bindLink(opts.cameraSelector, 'camera');
  bindLink(opts.viewerSelector, 'viewer');

  function bindLink(selector, intent) {
    const el = document.querySelector(selector);
    if (!el) return;
    el.addEventListener('click', (e) => {
      // If we already have at least one PSK, let navigation proceed.
      if (hasAnyPsk(opts)) return;
      // First-time: intercept and show the sheet.
      e.preventDefault();
      e.stopPropagation();
      openSheet(intent);
    });
  }

  // ---------- PSK detection ----------
  function hasAnyPsk(options) {
    if (typeof options.detectHasAnyPsk === 'function') {
      try { return !!options.detectHasAnyPsk(window.localStorage); }
      catch { return false; }
    }
    try {
      const ls = window.localStorage;
      for (let i = 0; i < ls.length; i++) {
        const key = ls.key(i);
        if (!key || !key.startsWith('psk:')) continue;
        try {
          const obj = JSON.parse(ls.getItem(key) || 'null');
          const token = obj && obj.tokenB64u;
          if (token && isValidToken(token, options.validateToken)) return true;
        } catch { /* ignore bad json */ }
      }
    } catch { /* private mode or blocked */ }
    return false;
  }

  function isValidToken(b64u, externalValidator) {
    if (typeof externalValidator === 'function') {
      try { return !!externalValidator(b64u); } catch {}
    }
    // Default: base64url decode to >=16 bytes
    try {
      const len = b64uByteLen(b64u);
      return len >= 16;
    } catch {
      return false;
    }
  }

  // Base64url -> byte length (no allocation of large arrays)
  function b64uByteLen(b64u) {
    const clean = String(b64u).replace(/-/g, '+').replace(/_/g, '/');
    const pad = (4 - (clean.length % 4)) % 4;
    const b64 = clean + '===='.slice(0, pad);
    // atob returns string length == byte length
    return atob(b64).length;
  }

  // ---------- Utils ----------
  function focusFirst(dom) {
    const els = dom.sheet.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    if (els.length) try { els[0].focus(); } catch {}
  }

  function trapTab(e, container) {
    const focusables = Array.from(container.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    )).filter(el => !el.hasAttribute('disabled') && el.offsetParent !== null);
    if (!focusables.length) return;
    const first = focusables[0];
    const last  = focusables[focusables.length - 1];
    const active = document.activeElement;
    const forward = !e.shiftKey;

    if (forward && active === last) {
      e.preventDefault(); first.focus();
    } else if (!forward && active === first) {
      e.preventDefault(); last.focus();
    }
  }

  function injectStyles() {
    if (document.getElementById('lpg-styles')) return;
    const css = `
      .lpg-open body { overflow: hidden; }

      .lpg-scrim {
        position: fixed; inset: 0; background: rgba(0,0,0,.55);
        opacity: 0; transition: opacity ${opts.durations.sheetInMs}ms ease;
        z-index: 1001; display: none;
      }
      .lpg-scrim.lpg-visible { opacity: 1; }

      .lpg-root {
        position: fixed; inset: 0; z-index: 1002; display: none;
        pointer-events: none;
      }
      .lpg-root.lpg-visible { pointer-events: auto; }

      .lpg-sheet {
        position: absolute; left: 50%; bottom: 0; transform: translate(-50%, 18px);
        width: min(92vw, 520px);
        background: #0b0b0b; border: 1px solid #141414; border-radius: 16px 16px 0 0;
        box-shadow: 0 16px 40px rgba(0,0,0,.45);
        color: #fff; padding: 14px 16px 16px;
        opacity: 0; transition: transform ${opts.durations.sheetInMs}ms ease, opacity ${opts.durations.sheetInMs}ms ease;
      }
      .lpg-sheet.lpg-in { transform: translate(-50%, 0); opacity: 1; }

      @media (min-width: 860px) {
        .lpg-sheet {
          bottom: auto; top: 12%; border-radius: 16px;
          transform: translate(-50%, -6px);
        }
        .lpg-sheet.lpg-in { transform: translate(-50%, 0); }
      }

      .lpg-row { display: grid; gap: 10px; }
      .lpg-hdr { display: flex; align-items: center; justify-content: center; }
      .lpg-title { margin: 0; font-weight: 800; font-size: 1.05rem; }
      .lpg-close {
        appearance: none; background: #0d0d0d; color: #fff;
        border: 1px solid #2a2a2a; border-radius: 9999px; cursor: pointer;
        width: 34px; height: 34px; font-size: 18px; line-height: 1;
      }
      .lpg-close:hover { background: #111; }

      .lpg-subline { font-size: .95rem; opacity: .95; margin: 2px 0 2px; }
      .lpg-bullets { margin: 6px 0 0 18px; padding: 0; }
      .lpg-bullets li { margin: 4px 0; opacity: .95; }

      .lpg-actions { display: flex; gap: 10px; align-items: center; margin-top: 8px; }
      .lpg-cta {
        appearance: none; cursor: pointer;
        color: #000; background: #fff; border: 1px solid #fff; border-radius: 9999px;
        font-weight: 700; padding: 8px 14px;
      }
      .lpg-cta:hover { background: #eaeaea; }

      .lpg-how {
        display: inline-flex; align-items: center; gap: 8px;
        font-size: .92rem; opacity: .9; text-decoration: underline; cursor: pointer;
        background: none; border: none; color: #fff; padding: 0;
      }

      .lpg-how-body {
        margin-top: 10px; border-top: 1px solid #1a1a1a; padding-top: 10px;
        font-size: .92rem; line-height: 1.45; display: none;
      }
      .lpg-how-body.lpg-how-open { display: block; }
      .lpg-how-body small { opacity: .8; display: block; margin-top: 8px; }
    `;
    const style = document.createElement('style');
    style.id = 'lpg-styles';
    style.textContent = css;
    document.head.appendChild(style);
  }

  function buildSheetDOM(copy, durations) {
    const scrim = el('div', 'lpg-scrim', { role: 'presentation' });
    const root  = el('div', 'lpg-root', { role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'lpg-title' });
    const sheet = el('div', 'lpg-sheet');

    const hdr   = el('div', 'lpg-hdr');
    const title = el('h2', 'lpg-title', { id: 'lpg-title' }, copy.title);
    // No close button
    hdr.append(title);

    const subline = el('div', 'lpg-subline');

    const bullets = el('ul', 'lpg-bullets');
    bullets.append(el('li', null, {}, copy.bullet1), el('li', null, {}, copy.bullet2), el('li', null, {}, copy.bullet3));

    const actions = el('div', 'lpg-actions');
    const btnPrimary = el('button', 'lpg-cta', { type: 'button' }, copy.primaryCta);
    const howLink = el('button', 'lpg-how', { type: 'button', 'aria-expanded': 'false' }, copy.howLink);
    actions.append(btnPrimary, howLink);

    const howBody = el('div', 'lpg-how-body', { 'aria-hidden': 'true' });
    copy.howBody.forEach(p => howBody.append(el('div', null, {}, p)));
    if (copy.privacyTip) howBody.append(el('small', null, {}, copy.privacyTip));

    const body = el('div', 'lpg-row');
    body.append(hdr, subline, bullets, actions, howBody);

    sheet.append(body);
    root.append(sheet);
    document.body.append(scrim, root);

    return { scrim, root, sheet, btnPrimary, howLink, howBody, subline };
  }

  function el(tag, className, attrs = {}, text = null) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text != null) node.textContent = text;
    return node;
  }

  function normalizeCopy(override, nouns) {
    const base = {
      title: 'Set up a secure link',
      closeAria: 'Close',
      sublineCamera: `Before you use Camera mode, let’s pair your ${nouns.singular}.`,
      sublineViewer: `Before you use Viewer mode, let’s pair your ${nouns.singular}.`,
      bullet1: 'Generate a one-time secret',
      bullet2: `Install it on the other device via QR/Share`,
      bullet3: `If you have already generated a secret on another device, go to Pair devices on that device and scan QR or Share`,
      primaryCta: 'Pair devices',
      howLink: 'How pairing works',
      howBody: [
        `We create a one-time secret on this ${nouns.singular} and save it locally.`,
        `You share that secret to the other device (QR code or share link).`,
        `The other device installs the secret locally too.`,
        `From then on, only devices that hold the same secret can connect.`,
        `You can rotate or forget the secret anytime. The secret never leaves your devices.`,
      ],
      //privacyTip: 'The share link keeps the secret in the URL fragment (#…), which isn’t sent to servers.',
    };
    if (!override) return base;

    // Shallow override + token replace if author passes {nounSingular}/{nounPlural}
    const merged = { ...base, ...override };
    const tokens = {
      '{nounSingular}': nouns.singular,
      '{nounPlural}': nouns.plural,
    };
    for (const k of Object.keys(merged)) {
      if (typeof merged[k] === 'string') merged[k] = replaceTokens(merged[k], tokens);
      if (Array.isArray(merged[k])) merged[k] = merged[k].map(s => replaceTokens(s, tokens));
    }
    return merged;
  }

  function replaceTokens(str, map) {
    return String(str).replace(/\{nounSingular\}|\{nounPlural\}/g, m => map[m] || m);
  }

  function detectDeviceNouns() {
    const ua = (navigator.userAgent || '').toLowerCase();
    const isIPhone = /\biphone\b/.test(ua);
    const isIPad   = /\bipad\b/.test(ua) || (/\bmacintosh\b/.test(ua) && 'ontouchend' in window);
    const isAndroid = /\bandroid\b/.test(ua);
    const isMobile = /\bmobi\b/.test(ua);
    const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
    const vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);

    // Coarse, friendly inferences — never mention OS or model to the user.
    if (isIPhone || (isAndroid && isMobile)) return { singular: 'phone', plural: 'phones' };
    if (isIPad || (isAndroid && !isMobile))  return { singular: 'tablet', plural: 'tablets' };

    // Large screens with fine pointer → "computer"
    const finePointer = window.matchMedia && matchMedia('(pointer: fine)').matches;
    if ((vw >= 1024 || vh >= 1024) && finePointer) return { singular: 'computer', plural: 'computers' };

    // Friendly default (prefer this over “device”)
    return { singular: 'phone or tablet', plural: 'phones or tablets' };
  }
}

export default installLandingPairGate;
