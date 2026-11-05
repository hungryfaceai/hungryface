// utils/admin-session.js — password sign-in for the admin tools (rooms, roster).
//
// The password is sent once to the signalling server's /admin/login, which checks it against a
// salted scrypt hash and returns a short-lived session token. The password is then cleared; only
// the token is kept, for this tab only (sessionStorage), and sent as "Authorization: Bearer …".
// The token expires on its own (8 h by default); "Sign out" forgets it immediately.
//
// Usage (classic script, after the page's toolbar exists):
//   const admin = NaptioAdmin.mount({ toolbar, loginUrl, onChange });
//   fetch(url, { headers: admin.headers() })

(function () {
  const KEY = 'naptio:adminSession';
  const LEGACY_KEY = 'naptio:adminKey';
  try { sessionStorage.removeItem(LEGACY_KEY); } catch {}   // old static key: no longer kept

  function load() {
    try {
      const s = JSON.parse(sessionStorage.getItem(KEY) || 'null');
      if (s && s.token && Date.parse(s.expiresAt) > Date.now()) return s;
    } catch {}
    try { sessionStorage.removeItem(KEY); } catch {}
    return null;
  }
  function save(s) { try { s ? sessionStorage.setItem(KEY, JSON.stringify(s)) : sessionStorage.removeItem(KEY); } catch {} }

  function mount({ toolbar, loginUrl, onChange = () => {} }) {
    let session = load();
    let expiryTimer = null;

    const wrap = document.createElement('form');
    wrap.className = 'admin-signin';
    wrap.setAttribute('autocomplete', 'on');
    wrap.style.cssText = 'display:inline-flex;gap:6px;align-items:center;margin:0';
    wrap.innerHTML =
      '<input name="username" autocomplete="username" value="naptio-admin" hidden>' +   // lets password managers file it
      '<input type="password" name="password" autocomplete="current-password" placeholder="Admin password" aria-label="Admin password" style="max-width:12rem">' +
      '<button type="submit">Sign in</button>' +
      '<span class="admin-state" role="status" aria-live="polite" style="font-size:12px;opacity:.85"></span>' +
      '<button type="button" class="admin-signout" hidden>Sign out</button>';
    toolbar.appendChild(wrap);
    const pw = wrap.querySelector('input[type="password"]');
    const signIn = wrap.querySelector('button[type="submit"]');
    const state = wrap.querySelector('.admin-state');
    const signOut = wrap.querySelector('.admin-signout');

    function render(msg) {
      const on = !!session;
      pw.hidden = signIn.hidden = on;
      signOut.hidden = !on;
      state.textContent = msg || (on ? 'Signed in until ' + new Date(session.expiresAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '');
      clearTimeout(expiryTimer);
      if (on) expiryTimer = setTimeout(() => { session = null; save(null); render('Session expired — sign in again'); onChange(false); },
                                       Math.max(0, Date.parse(session.expiresAt) - Date.now()));
    }

    wrap.addEventListener('submit', async (e) => {
      e.preventDefault();
      const password = pw.value;
      if (!password) { pw.focus(); return; }
      signIn.disabled = true; state.textContent = 'Signing in…';
      try {
        const resp = await fetch(loginUrl, {
          method: 'POST', mode: 'cors', cache: 'no-store', credentials: 'omit',
          headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ password }),
        });
        const body = await resp.json().catch(() => ({}));
        if (!resp.ok || !body.token) throw new Error(body.error || ('Sign-in failed (HTTP ' + resp.status + ')'));
        session = { token: body.token, expiresAt: body.expiresAt };
        save(session); render(); onChange(true);
      } catch (err) {
        render((err && err.message) || 'Sign-in failed');
      } finally {
        pw.value = '';                      // never keep the password around
        signIn.disabled = false;
      }
    });
    signOut.addEventListener('click', () => { session = null; save(null); render('Signed out'); onChange(false); });

    render();
    return {
      headers() { return session ? { Authorization: 'Bearer ' + session.token } : {}; },
      signedIn() { return !!session; },
      /** Call when the server answers 401: the token was revoked (e.g. password changed). */
      expired() { if (session) { session = null; save(null); render('Session ended — sign in again'); onChange(false); } },
    };
  }

  window.NaptioAdmin = { mount };
})();
