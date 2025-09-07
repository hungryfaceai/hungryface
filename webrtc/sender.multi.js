// /webrtc/sender.multi.js
import { SecureSignal } from '/hungryface/webrtc/shared/secure-signal.js';


const WS_ENDPOINT = 'wss://signaling-server-f5gu.onrender.com/ws';
const ss = new SecureSignal({ url: WS_ENDPOINT, onSession: ({sid, peerFp, role})=>{
if (role==='initiator') {
// We create and send offer right after startSession resolves; handled below
}
}});
await ss.init();


// Get local media once
const localStream = await navigator.mediaDevices.getUserMedia({ audio:true, video:true });
const pcs = new Map(); // sid -> RTCPeerConnection


async function connectTo(fp){
// 1) Create encrypted signaling session to this peer
const { sid } = await ss.startSession(fp);
// 2) Create RTCPeerConnection
const pc = new RTCPeerConnection({ iceServers: [] });
pcs.set(sid, pc);
// 3) Add local tracks
localStream.getTracks().forEach(t => pc.addTrack(t, localStream));
// 4) Send ICE via encrypted signaling
pc.onicecandidate = (e)=>{ if (e.candidate) ss.sendJSON(sid, { type:'ice', candidate:e.candidate }); };
// 5) Handle encrypted messages for this session
ss.onEncrypted(sid, async (msg)=>{
if (msg.type==='answer') {
await pc.setRemoteDescription(msg);
} else if (msg.type==='ice') {
try { await pc.addIceCandidate(msg.candidate); } catch {}
}
});
// 6) Create and send offer
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);
await ss.sendJSON(sid, offer);
}


// Example UI: connect to all trusted receivers selected in a list
window.naptioConnectSelected = async function(selectedFingerprints){
for (const fp of selectedFingerprints) {
try { await connectTo(fp); } catch (e) { console.warn('connect fail', fp, e); }
}
};
