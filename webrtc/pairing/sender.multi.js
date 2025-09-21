
// /webrtc/sender.multi.js
import { SecureSignal } from '/hungryface/webrtc/shared/secure-signal.js';

const WS_ENDPOINT = 'wss://signaling-server-f5gu.onrender.com/ws';
const ss = new SecureSignal({ url: WS_ENDPOINT });
await ss.init();

const localStream = await navigator.mediaDevices.getUserMedia({ audio:true, video:true });
const pcs = new Map(); // sid -> RTCPeerConnection

async function connectTo(fp){
  const { sid } = await ss.startSession(fp);
  const pc = new RTCPeerConnection({ iceServers: [] });
  pcs.set(sid, pc);
  localStream.getTracks().forEach(t => pc.addTrack(t, localStream));
  pc.onicecandidate = (e)=>{ if (e.candidate) ss.sendJSON(sid, { type:'ice', candidate:e.candidate }); };
  ss.onEncrypted(sid, async (msg)=>{
    if (msg.type==='answer') { await pc.setRemoteDescription(msg); }
    else if (msg.type==='ice') { try { await pc.addIceCandidate(msg.candidate); } catch {} }
  });
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  await ss.sendJSON(sid, offer);
}

window.naptioConnectSelected = async function(selectedFingerprints){
  for (const fp of selectedFingerprints) {
    try { await connectTo(fp); } catch (e) { console.warn('connect fail', fp, e); }
  }
};
