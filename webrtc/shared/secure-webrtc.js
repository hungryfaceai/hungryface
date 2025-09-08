// secure-webrtc.js — Gate WebRTC on pairing + encrypted signaling via SecureSignal
// Usage (initiator): const {pc, dc} = await connectToPeer(ss, peerFp, {initiator:true});
// Usage (responder): waits for ss.onSession({role:'responder'}) and auto-accepts only for that peer.

export async function connectToPeer(ss, peerFp, {
  initiator = false,
  rtcConfig = { iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] },
  onTrack,           // (event) => {}
  onData,            // (message, dc) => {}
  label = 'app',     // datachannel label (initiator creates it)
  verifyDataChannel = false, // optional DC proof (see note)
} = {}) {
  // 1) Make sure we're paired
  const peer = ss.trusted.find(d => d.fp === peerFp);
  if (!peer) throw new Error('Peer not in Trusted devices. Pair first.');

  // 2) Establish an encrypted signaling session (sid) via SecureSignal
  const sid = initiator
    ? (await ss.startSession(peerFp)).sid
    : await waitForSessionFrom(ss, peerFp); // waits for role:'responder' with this peer

  // 3) Wire encrypted handler for WebRTC messages on this sid
  const pc = new RTCPeerConnection(rtcConfig);

  // Forward local ICE to the peer via encrypted signaling
  pc.onicecandidate = ({ candidate }) => {
    ss.sendJSON(sid, { webrtc: 'ice', candidate });
  };

  // Optional media
  if (onTrack) pc.ontrack = onTrack;

  // Data channel setup
  let dc;
  if (initiator) {
    dc = pc.createDataChannel(label, { ordered: true });
    attachDcHandlers(dc, onData, verifyDataChannel);
  } else {
    pc.ondatachannel = (e) => {
      dc = e.channel;
      attachDcHandlers(dc, onData, verifyDataChannel);
    };
  }

  // Handle incoming encrypted signaling on this sid only
  ss.onEncrypted(sid, async (msg) => {
    if (msg.webrtc === 'offer') {
      await pc.setRemoteDescription(new RTCSessionDescription(msg.desc));
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      await ss.sendJSON(sid, { webrtc: 'answer', desc: pc.localDescription });
      return;
    }
    if (msg.webrtc === 'answer') {
      await pc.setRemoteDescription(new RTCSessionDescription(msg.desc));
      return;
    }
    if (msg.webrtc === 'ice') {
      try {
        if (msg.candidate) await pc.addIceCandidate(msg.candidate);
        else await pc.addIceCandidate(null); // end-of-candidates
      } catch (err) {
        // benign if it arrives early / after close
        console.debug('ICE add error:', err);
      }
      return;
    }
  });

  // 4) If we're the initiator, create the offer and send it (encrypted)
  if (initiator) {
    const offer = await pc.createOffer({ offerToReceiveAudio: true, offerToReceiveVideo: true });
    await pc.setLocalDescription(offer);
    await ss.sendJSON(sid, { webrtc: 'offer', desc: pc.localDescription });
  }

  return { pc, dc, sid };
}

// Wait for SecureSignal to create a responder session from a trusted peer
function waitForSessionFrom(ss, peerFp, timeoutMs = 15000) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => {
      cleanup();
      reject(new Error('Timed out waiting for session'));
    }, timeoutMs);

    function onSession(e) {
      if (e.role === 'responder' && e.peerFp === peerFp) {
        cleanup();
        resolve(e.sid);
      }
    }
    function cleanup() {
      clearTimeout(t);
      // Re-wrap onSession while preserving any existing hook:
      const prev = ss.onSession;
      ss.onSession = prev; // no-op if you bound directly; otherwise app should re-attach
    }

    // Wrap existing ss.onSession so we don't clobber app behavior
    const prev = ss.onSession;
    ss.onSession = (...args) => { try { prev && prev(...args); } catch {} ; onSession(...args[0] || args); };
  });
}

// Attach basic handlers to a data channel (and optional proof)
function attachDcHandlers(dc, onData, verifyDataChannel) {
  dc.onopen = () => {
    if (!verifyDataChannel) return;
    // Simple liveness proof: both sides echo a small token before sending real data
    const token = Math.random().toString(36).slice(2, 10);
    const timer = setTimeout(() => { try { dc.close(); } catch {} }, 5000);
    function handler(ev) {
      try {
        const obj = JSON.parse(ev.data);
        if (obj?.type === 'dc-echo' && obj?.token === token) {
          clearTimeout(timer);
          dc.removeEventListener('message', handler);
        }
      } catch {}
    }
    dc.addEventListener('message', handler);
    dc.send(JSON.stringify({ type: 'dc-echo', token }));
  };

  dc.onmessage = (ev) => {
    if (onData) {
      try { onData(JSON.parse(ev.data), dc); }
      catch { onData(ev.data, dc); }
    }
  };
}
