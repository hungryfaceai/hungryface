// TEMPORARY — shared behaviour for the landing-page proposals. Delete this folder after choosing.
// 1) Returning visitors (already paired) see "Open Camera / Open Viewer" instead of "Get started".
// 2) Camera/Viewer links on unpaired devices go to pairing (same rule as the current home page's pair gate).
import { listValidPskRooms } from '/hungryface/webrtc/shared/psk/require-psk.js';

let paired = false;
try { paired = listValidPskRooms().length > 0; } catch {}
document.body.classList.toggle('is-paired', paired);

if (!paired) {
  document.querySelectorAll('a[data-needs-pairing]').forEach(a => { a.href = '/hungryface/webrtc/pairpsk/'; });
}
