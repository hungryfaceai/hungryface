// /webrtc/receiver.multi.js
import { SecureSignal } from '/hungryface/webrtc/shared/secure-signal.js';


const WS_ENDPOINT = 'wss://signaling-server-f5gu.onrender.com/ws';
const ss = new SecureSignal({ url: WS_ENDPOINT, onSession: ({sid, peerFp, role})=>{
if (role==='responder') {
// Once session key is ready, we expect an offer next on this sid.
// We attach a handler that will build a PC on first offer.
ss.onEncrypted(sid, async (msg)=>{
let pc = pcs.get(sid);
if (!pc) {
pc = new RTCPeerConnection({ iceServers: [] });
pcs.set(sid, pc);
pc.ontrack = (e)=>{
const stream = e.streams[0];
attachRemoteStream(stream, sid); // your UI hook to render video/audio
};
pc.onicecandidate = (e)=>{ if (e.candidate) ss.sendJSON(sid, { type:'ice', candidate:e.candidate }); };
}
if (msg.type==='offer') {
await pc.setRemoteDescription(msg);
const answer = await pc.createAnswer();
await pc.setLocalDescription(answer);
await ss.sendJSON(sid, answer);
} else if (msg.type==='ice') {
try { await pc.addIceCandidate(msg.candidate); } catch {}
}
});
}
}});
await ss.init();


const pcs = new Map(); // sid -> RTCPeerConnection


function attachRemoteStream(stream, sid){
// Minimal DOM hook; replace with your actual video element mapping
let v = document.querySelector(`#remote-${sid}`);
if (!v){
v = document.createElement('video'); v.autoplay = true; v.playsInline = true; v.muted = true; v.id = `remote-${sid}`; v.style.width='320px'; v.style.margin='8px';
document.body.appendChild(v);
}
v.srcObject = stream;
}
