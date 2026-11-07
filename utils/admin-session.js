// utils/admin-session.js — password sign-in for the admin tools (rooms, roster).
//
// The password is sent once to the signalling server's /admin/login, which checks it against a
// salted scrypt hash and returns a session token. The password is then cleared; only the token is
// kept and sent as "Authorization: Bearer …". How long a sign-in lasts is set on the server
// (ADMIN_SESSION_HOURS on Render).
//
// "Keep me signed in on this device" (on by default) keeps the token in localStorage, so it
// survives closing the tab or browser; unticked, it lives only in this tab (sessionStorage).
// "Sign out" forgets it on this device. To sign out EVERY device, change ADMIN_SESSION_SECRET
// (or the password) on Render.
//
// Usage (classic script, after the page's toolbar exists):
//   const admin = NaptioAdmin.mount({ toolbar, loginUrl, onChange });
//   fetch(url, { headers: admin.headers() })

(function () {
  // Session record is JSON { token, expiresAt }. LEGACY_KEY held the old static admin key and is wiped.
  const KEY = 'naptio:adminSession';
  const LEGACY_KEY = 'naptio:adminKey';
  const MAX_TIMER_MS = 2147483647; // setTimeout limit (~24.8 days); longer delays fire immediately
  try {
    sessionStorage.removeItem(LEGACY_KEY);
  } catch {} // old static key: no longer kept

  // Available web storages (either can throw in private modes), localStorage first.
  const stores = () =>
    [
      (() => {
        try {
          return localStorage;
        } catch {
          return null;
        }
      })(),
      (() => {
        try {
          return sessionStorage;
        } catch {
          return null;
        }
      })(),
    ].filter(Boolean);
  // Returns the stored, unexpired session or null; expired entries are removed.
  function load() {
    for (const st of stores()) {
      try {
        const s = JSON.parse(st.getItem(KEY) || 'null');
        if (s && s.token && Date.parse(s.expiresAt) > Date.now()) return s;
        st.removeItem(KEY); // expired or unreadable
      } catch {}
    }
    return null;
  }
  // Replaces the stored session: clears both storages, then writes to local or session storage per s.remember.
  function save(s) {
    for (const st of stores()) {
      try {
        st.removeItem(KEY);
      } catch {}
    }
    if (!s) return;
    try {
      (s.remember ? localStorage : sessionStorage).setItem(KEY, JSON.stringify(s));
    } catch {}
  }
  // Expiry text for the UI: a clock time if under 24 h away (864e5 ms), else a date.
  function until(expiresAt) {
    const d = new Date(expiresAt),
      soon = d - Date.now() < 864e5;
    return soon
      ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      : d.toLocaleDateString([], { day: 'numeric', month: 'short', year: 'numeric' });
  }

  /**
   * Adds the sign-in form to the toolbar.
   * @param {{toolbar: HTMLElement, loginUrl: string, onChange?: (signedIn: boolean) => void}} opts
   * @returns {{headers: Function, signedIn: Function, expired: Function}}
   */
  function mount({ toolbar, loginUrl, onChange = () => {} }) {
    let session = load();
    let expiryTimer = null;

    const wrap = document.createElement('form');
    wrap.className = 'admin-signin';
    wrap.setAttribute('autocomplete', 'on');
    wrap.style.cssText = 'display:inline-flex;gap:6px;align-items:center;margin:0;flex-wrap:wrap';
    wrap.innerHTML =
      '<input name="username" autocomplete="username" value="naptio-admin" hidden>' + // lets password managers file it
      '<input type="password" name="password" autocomplete="current-password" placeholder="Admin password" aria-label="Admin password" aria-describedby="adminSigninError" style="max-width:12rem">' +
      '<p class="admin-error" id="adminSigninError" role="alert" hidden><svg aria-hidden="true" viewBox="0 0 24 24" width="18" height="18"><circle cx="12" cy="12" r="10" fill="currentColor"/><path d="M12 7v6M12 16.5v.5" stroke="#1a0606" stroke-width="2.4" stroke-linecap="round"/></svg><span></span></p>' +
      '<label class="admin-remember" style="font-size:12px;gap:4px;align-items:center;cursor:pointer">' +
      '<input type="checkbox" checked> Keep me signed in on this device</label>' +
      '<button type="submit">Sign in</button>' +
      '<span class="admin-state" role="status" aria-live="polite" style="font-size:12px;opacity:.85"></span>' +
      '<button type="button" class="admin-signout" hidden>Sign out</button>';
    toolbar.appendChild(wrap);
    const pw = wrap.querySelector('input[type="password"]');
    const remember = wrap.querySelector('.admin-remember input');
    const rememberLabel = wrap.querySelector('.admin-remember');
    const signIn = wrap.querySelector('button[type="submit"]');
    const state = wrap.querySelector('.admin-state');
    const signOut = wrap.querySelector('.admin-signout');
    const errBox = wrap.querySelector('.admin-error');
    const errText = errBox.querySelector('span');

    // Error styling lives here so it looks the same on every page that uses this module.
    if (!document.getElementById('admin-session-css')) {
      const css = document.createElement('style');
      css.id = 'admin-session-css';
      css.textContent =
        '.admin-signin .admin-error{flex:1 0 100%;box-sizing:border-box;display:flex;align-items:center;gap:8px;margin:2px 0 0;padding:10px 12px;' +
        'border-radius:10px;background:#2a1212;border:1px solid #7f1d1d;color:#fecaca;font-size:15px;font-weight:600;line-height:1.35}' +
        '.admin-signin .admin-error[hidden]{display:none}' +
        '.admin-signin .admin-error svg{flex:0 0 18px;color:#ef4444}' +
        '.admin-signin input[aria-invalid="true"]{border-color:#ef4444 !important;box-shadow:0 0 0 3px rgba(239,68,68,.25) !important}' +
        '@media (prefers-reduced-motion:no-preference){.admin-signin .admin-error.shake{animation:adminShake .3s}}' +
        '@keyframes adminShake{0%,100%{transform:none}25%{transform:translateX(-4px)}75%{transform:translateX(4px)}}';
      document.head.appendChild(css);
    }
    // Keep the error box exactly as wide as the password field (it would otherwise span the whole row).
    function fitError() {
      if (!errBox.hidden && pw.offsetWidth) errBox.style.maxWidth = pw.offsetWidth + 'px';
    }
    if ('ResizeObserver' in window) new ResizeObserver(fitError).observe(pw);
    else addEventListener('resize', fitError);
    const ERROR_VISIBLE_MS = 1000;
    let errTimer = null;
    function showError(msg) {
      errText.textContent = msg === 'Wrong password.' ? 'Wrong password' : msg;
      errBox.hidden = false;
      pw.setAttribute('aria-invalid', 'true');
      fitError();
      errBox.classList.remove('shake');
      void errBox.offsetWidth;
      errBox.classList.add('shake'); // replay on each failure
      clearTimeout(errTimer);
      errTimer = setTimeout(clearError, ERROR_VISIBLE_MS); // brief: vanishes after 1 s
      // ...or straight away on the next tap / click anywhere (passive, capture: never blocks what was tapped)
      setTimeout(() => {
        if (!errBox.hidden) document.addEventListener('pointerdown', clearError, { capture: true, passive: true });
      }, 0);
    }
    function clearError() {
      clearTimeout(errTimer);
      errTimer = null;
      document.removeEventListener('pointerdown', clearError, true);
      errBox.hidden = true;
      errText.textContent = '';
      pw.removeAttribute('aria-invalid');
    }
    pw.addEventListener('input', clearError); // start typing again -> error goes away

    // Forgets the session on this device and notifies the page.
    function end(msg) {
      session = null;
      save(null);
      render(msg);
      onChange(false);
    }
    // Signs out automatically when the session expires.
    function scheduleExpiry() {
      clearTimeout(expiryTimer);
      if (!session) return;
      const left = Date.parse(session.expiresAt) - Date.now();
      if (left <= 0) return end('Session expired — sign in again');
      // Long sessions: re-check periodically instead of one huge (overflowing) timeout.
      expiryTimer = setTimeout(scheduleExpiry, Math.min(left, MAX_TIMER_MS));
    }
    // Shows the form when signed out, the Sign out button when signed in.
    function render(msg) {
      const on = !!session;
      // Set display directly: an inline/page `display` value would override the `hidden` attribute.
      const show = (el, yes) => {
        el.hidden = !yes;
        el.style.display = yes ? '' : 'none';
      };
      show(pw, !on);
      show(signIn, !on);
      show(rememberLabel, !on);
      show(signOut, on);
      if (on) clearError();
      state.textContent = msg || ''; // no "signed in" text: the Sign out button says it
      scheduleExpiry();
    }

    // POST the password once to loginUrl; on success keep only the returned token.
    wrap.addEventListener('submit', async (e) => {
      e.preventDefault();
      const password = pw.value;
      if (!password) {
        pw.focus();
        return;
      }
      signIn.disabled = true;
      clearError();
      state.textContent = 'Signing in…';
      try {
        const resp = await fetch(loginUrl, {
          method: 'POST',
          mode: 'cors',
          cache: 'no-store',
          credentials: 'omit',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ password }),
        });
        const body = await resp.json().catch(() => ({}));
        if (!resp.ok || !body.token) throw new Error(body.error || 'Sign-in failed (HTTP ' + resp.status + ')');
        session = { token: body.token, expiresAt: body.expiresAt, remember: remember.checked };
        clearError();
        save(session);
        render();
        onChange(true);
      } catch (err) {
        render('');
        showError((err && err.message) || 'Sign-in failed.');
      } finally {
        pw.value = ''; // never keep the password around
        signIn.disabled = false;
        if (!session) pw.focus(); // ready to type again
      }
    });
    signOut.addEventListener('click', () => end('Signed out on this device'));
    // Signing in/out in another tab of this browser: follow it.
    window.addEventListener('storage', (ev) => {
      if (ev.key !== KEY) return;
      const next = load();
      if (!!next !== !!session || (next && session && next.token !== session.token)) {
        session = next;
        render();
        onChange(!!session);
      }
    });

    render();
    return {
      headers() {
        return session ? { Authorization: 'Bearer ' + session.token } : {};
      },
      signedIn() {
        return !!session;
      },
      /** Call when the server answers 401: the token was revoked (password or secret changed). */
      expired() {
        if (session) end('Session ended — sign in again');
      },
    };
  }

  window.NaptioAdmin = { mount };
})();
