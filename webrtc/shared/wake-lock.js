// wake-lock.js — keep the screen awake while monitoring.
//
// Uses the Screen Wake Lock API where available; silently does nothing elsewhere.
// The browser drops the lock whenever the page is hidden, so we re-acquire it when
// the page becomes visible again. Some browsers (notably Safari) only grant the lock
// after a user gesture, so we also retry on the first tap/click/keypress.
//
// Usage:
//   import { keepAwake } from '/hungryface/webrtc/shared/wake-lock.js';
//   const awake = keepAwake();   // start
//   awake.release();             // stop (e.g. when streaming stops)

const supported = typeof navigator !== 'undefined' && 'wakeLock' in navigator;

export function keepAwake({ log = () => {} } = {}) {
  let sentinel = null;
  let wanted = true;
  let requesting = false;

  async function acquire() {
    if (!supported || !wanted || sentinel || requesting) return;
    if (document.visibilityState !== 'visible') return;
    requesting = true;
    try {
      const s = await navigator.wakeLock.request('screen');
      if (!wanted) {
        try {
          await s.release();
        } catch {}
        return;
      }
      sentinel = s;
      s.addEventListener('release', () => {
        if (sentinel === s) sentinel = null;
      });
      log('wake lock acquired');
    } catch (e) {
      // NotAllowedError (no gesture yet / battery saver) etc. — retry later on gesture/visibility
      log('wake lock unavailable: ' + (e?.name || e));
    } finally {
      requesting = false;
    }
  }

  const onVisible = () => {
    if (document.visibilityState === 'visible') acquire();
  };
  const onGesture = () => acquire();
  const gestureEvents = ['pointerdown', 'keydown', 'touchend'];

  if (supported) {
    document.addEventListener('visibilitychange', onVisible);
    gestureEvents.forEach((ev) => document.addEventListener(ev, onGesture, { passive: true }));
    acquire();
  }

  return {
    get active() {
      return !!sentinel;
    },
    supported,
    async release() {
      wanted = false;
      document.removeEventListener('visibilitychange', onVisible);
      gestureEvents.forEach((ev) => document.removeEventListener(ev, onGesture));
      const s = sentinel;
      sentinel = null;
      if (s) {
        try {
          await s.release();
        } catch {}
        log('wake lock released');
      }
    },
  };
}
