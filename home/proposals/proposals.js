// TEMPORARY — shared behaviour for the landing-page proposals. Delete this folder after choosing.
// 1) Paired devices see the "returning" layout; others see the "first-time" layout.
// 2) Camera/Viewer links on unpaired devices go to pairing (same rule as the current home page's pair gate).
// 3) Preview either state on any device with ?state=paired or ?state=new (links stay safe: an unpaired
//    device still goes to pairing).
// 4) Remembers which role this device last opened from here (naptio:homeRole), so the paired screen can
//    put it first. Pages opt in with data-role="viewer|camera" on the links and a [data-role-slot] container.
import { listValidPskRooms } from '/hungryface/webrtc/shared/psk/require-psk.js';

let rooms = [];
try { rooms = listValidPskRooms(); } catch {}
const reallyPaired = rooms.length > 0;
const forced = new URLSearchParams(location.search).get('state');
const showPaired = forced === 'paired' ? true : forced === 'new' ? false : reallyPaired;
document.body.classList.toggle('is-paired', showPaired);

if (!reallyPaired) {
  document.querySelectorAll('a[data-needs-pairing]').forEach(a => { a.href = '/hungryface/webrtc/pairpsk/'; });
}

// Room name(s), for pages that show "Room: Baby"
const roomName = rooms[0] || (forced === 'paired' ? 'Baby' : '');
document.querySelectorAll('[data-room-name]').forEach(el => { el.textContent = roomName; });
document.querySelectorAll('[data-room-count]').forEach(el => { el.hidden = rooms.length < 2; el.textContent = rooms.length + ' rooms'; });

// Last role used on this device: put it first and make it the main button.
const ROLE_KEY = 'naptio:homeRole';
let role = null; try { role = localStorage.getItem(ROLE_KEY); } catch {}
document.querySelectorAll('a[data-role]').forEach(a => a.addEventListener('click', () => {
  try { localStorage.setItem(ROLE_KEY, a.dataset.role); } catch {}
}));
if (role === 'camera') {
  document.querySelectorAll('[data-role-slot]').forEach(slot => {
    const cam = slot.querySelector('a[data-role="camera"]'), view = slot.querySelector('a[data-role="viewer"]');
    if (!cam || !view) return;
    slot.insertBefore(cam, view);
    cam.classList.add('is-main'); view.classList.remove('is-main');
  });
}
document.body.dataset.lastRole = role || '';

// Installed app (home-screen web app or store app): no store badges.
const standalone = matchMedia('(display-mode: standalone)').matches || navigator.standalone === true || !!window.Capacitor?.isNativePlatform?.();
if (standalone) document.querySelectorAll('.stores').forEach(el => el.remove());
