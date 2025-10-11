// /webrtc/shared/terms/require-terms.js
export const TERMS_VERSION = '2025-10-11'; // match the "Effective" pill on your Legal page
const STORAGE_KEY = `naptio:terms:${TERMS_VERSION}`;

export function hasAcceptedTerms() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return false;
    const { acceptedAt } = JSON.parse(raw);
    return Boolean(acceptedAt);
  } catch {
    return false;
  }
}

export async function ensureTermsAccepted(opts = {}) {
  const role = opts.role ?? 'pair'; // 'sender' | 'receiver' | 'pair'
  const locale = opts.locale ?? (navigator.language || 'en-GB');

  if (hasAcceptedTerms()) return true;

  // Build modal (self-contained styles; reuses .btn-pill if present)
  const overlay = document.createElement('div');
  overlay.id = 'tos-overlay';
  overlay.innerHTML = `
    <div class="tos-backdrop" part="backdrop"></div>
    <div class="tos-dialog" role="dialog" aria-modal="true" aria-labelledby="tos-title" aria-describedby="tos-desc">
      <div class="tos-body">
        <h2 id="tos-title">Before we pair: a quick heads-up</h2>
        <p id="tos-desc" class="tos-lead">
          Naptio is a helpful tool, <strong>not a medical device</strong>. Keep a caregiver nearby.
          Alerts can miss things due to device limits or noise. Always use your judgment.
        </p>
        <p class="tos-meta">
          By continuing, you agree to our
          <a class="tos-link" href="/hungryface/legal/" target="_blank" rel="noopener">Terms of Use & Privacy Notice</a>
          <span class="tos-pill" aria-label="Effective date">Effective: 11 Oct 2025</span>
        </p>

        <label class="tos-check">
          <input id="tos-checkbox" type="checkbox" aria-describedby="tos-desc" />
          <span>I’ve read and accept the Terms of Use and Privacy Notice.</span>
        </label>

        <div class="tos-actions">
          <button id="tos-accept" class="btn-pill tos-primary" disabled>Accept & continue</button>
          <a id="tos-decline" class="tos-link" href="/hungryface/home/">Decline</a>
          <a class="tos-link" href="/hungryface/legal/" target="_blank" rel="noopener">View full terms</a>
        </div>
      </div>
    </div>
  `;

  // Minimal, page-safe CSS (prefers your existing tokens if present)
  const style = document.createElement('style');
  style.textContent = `
  #tos-overlay{position:fixed;inset:0;z-index:2147483647;display:grid;place-items:center}
  .tos-backdrop{position:absolute;inset:0;background:rgba(0,0,0,.6);backdrop-filter:saturate(120%) blur(2px)}
  .tos-dialog{position:relative;max-width:min(92vw,520px);margin:16px;border-radius:16px;background:#111;color:#fff;box-shadow:0 10px 30px rgba(0,0,0,.5);outline:none}
  .tos-body{padding:20px}
  .tos-lead{opacity:.95;line-height:1.45}
  .tos-meta{opacity:.8;margin:.5rem 0 1rem 0}
  .tos-pill{display:inline-block;margin-left:.5rem;padding:.15rem .5rem;border-radius:999px;border:1px solid #444;font-size:12px;opacity:.9}
  .tos-check{display:flex;gap:.6rem;align-items:flex-start;margin:1rem 0}
  .tos-check input{transform:translateY(2px)}
  .tos-actions{display:flex;flex-wrap:wrap;gap:.8rem;align-items:center}
  .tos-link{color:#fff;opacity:.9;text-decoration:underline}
  .tos-primary{background:#111;border:1px solid #444;color:#fff;padding:.6rem 1rem;border-radius:999px}
  .btn-pill.tos-primary{ /* if your .btn-pill is present it takes over visual style */ }
  @media (max-width:380px){ .tos-body{padding:16px} }
  `;
  document.head.appendChild(style);
  document.body.appendChild(overlay);

  // Accessibility: focus trap
  const dialog = overlay.querySelector('.tos-dialog');
  const checkbox = overlay.querySelector('#tos-checkbox');
  const acceptBtn = overlay.querySelector('#tos-accept');
  const decline = overlay.querySelector('#tos-decline');

  const focusables = () => [...overlay.querySelectorAll('a,button,input,[tabindex]:not([tabindex="-1"])')].filter(el => !el.disabled);
  let first, last;
  function setTrap() { [first] = focusables(); last = focusables().slice(-1)[0]; }
  setTrap();
  overlay.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { // treat Escape as decline (just navigate away)
      decline.click();
    } else if (e.key === 'Tab') {
      setTrap();
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    }
  });

  // Logic
  checkbox.addEventListener('change', () => { acceptBtn.disabled = !checkbox.checked; });
  setTimeout(() => (checkbox.focus({ preventScroll: true })), 0);

  return new Promise((resolve) => {
    acceptBtn.addEventListener('click', () => {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({
          acceptedAt: new Date().toISOString(),
          locale,
          role
        }));
      } catch {
        // If storage blocked, we still let them proceed after explicit acceptance
      }
      overlay.remove();
      style.remove();
      resolve(true);
    }, { once: true });
  });
}
