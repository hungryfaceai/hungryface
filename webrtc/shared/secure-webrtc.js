// secure-webrtc.js — Gate WebRTC on pairing + encrypted signaling via SecureSignal
// Usage (initiator): const {pc, dc} = await connectToPeer(ss, peerFp, { initiator:true, localStream });
// Usage (responder): waits for ss.onSession({role:'responder'}) and auto-accepts only for that peer.

export async function connectToPeer(ss, peerFp, {
  initiator = false,
  rtcConfig = { iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] },
  onTrack,           // (event) => {}
  onData,            // (message, dc) => {}
  label = 'app',     // datachannel label (initiator creates it)
  verifyDataChannel = false, // optional DC proof (see note)
  localStream = null,        // add tracks before first offer
  localTracks = null,        // optional alternative to localStream
} = {}) {
  // 1) Make sure we're paired
  const peer = ss.trusted.find(d => d.fp === peerFp);
  if (!peer) throw new Error('Peer not in Trusted devices. Pair first.');

  // 2) Establish an encrypted signaling session (sid) via SecureSignal
  const sid = initiator
    ? (await ss.startSession(peerFp)).sid
    : await waitForSessionFrom(ss, peerFp); // waits for role:'responder' with this peer

  // 3) Create RTCPeerConnection
  const pc = new RTCPeerConnection(rtcConfig);

  // --- Perfect negotiation helpers ---
  let makingOffer = false;
  const polite = !initiator; // responder is polite; initiator is impolite

  // Optional media (remote)
  if (onTrack) pc.ontrack = onTrack;

  // --- Add local media BEFORE any offer is created (prevents glare) ---
  if (initiator) {
    const tracks = Array.isArray(localTracks) ? localTracks
                  : localStream ? localStream.getTracks()
                  : [];
    if (tracks.length) {
      const streamForSender = localStream || new MediaStream(tracks);
      for (const t of tracks) pc.addTrack(t, streamForSender);
    }
  }

  // Drive negotiation from the initiator only
  pc.onnegotiationneeded = async () => {
    if (!initiator) return;
    try {
      makingOffer = true;
      const offer = await pc.createOffer({
        offerToReceiveAudio: true,
        offerToReceiveVideo: true
      });
      await pc.setLocalDescription(offer);
      await ss.sendJSON(sid, { webrtc: 'offer', desc: pc.localDescription });
    } finally {
      makingOffer = false;
    }
  };

  // Forward local ICE to the peer via encrypted signaling
  pc.onicecandidate = ({ candidate }) => {
    ss.sendJSON(sid, { webrtc: 'ice', candidate });
  };

  // Data channel setup
  let dc;
  if (initiator) {
    dc = pc.createDataChannel(label, { ordered: true });
    attachDcHandlers(dc, onData, verifyDataChannel);
    // Creating a DC will also fire onnegotiationneeded → initiator will send the first offer.
  } else {
    pc.ondatachannel = (e) => {
      dc = e.channel;
      attachDcHandlers(dc, onData, verifyDataChannel);
    };
  }

  // Handle incoming encrypted signaling on this sid only (role-gated + rollback)
  ss.onEncrypted(sid, async (msg) => {
    if (msg.webrtc === 'offer') {
      if (initiator) return;

      const offer = new RTCSessionDescription(msg.desc);
      const offerCollision = makingOffer || pc.signalingState !== 'stable';

      if (offerCollision) {
        if (!polite) return;                 // impolite side ignores glare
        try { await pc.setLocalDescription({ type: 'rollback' }); } catch {}
      }

      try { await setRemoteWithTimeout(pc, offer); } catch { return; }

      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      await ss.sendJSON(sid, { webrtc: 'answer', desc: pc.localDescription });
      return;
    }

    if (msg.webrtc === 'answer') {
      if (!initiator) return;
      if (pc.signalingState !== 'have-local-offer') return; // ignore late/dup
      try { await setRemoteWithTimeout(pc, new RTCSessionDescription(msg.desc)); } catch {}
      return;
    }

    if (msg.webrtc === 'ice') {
      try {
        if (msg.candidate) await pc.addIceCandidate(msg.candidate);
        else await pc.addIceCandidate(null); // end-of-candidates
      } catch (err) {
        console.debug('ICE add error:', err);
      }
      return;
    }
  });

  // NOTE: No manual "kick off" offer; onnegotiationneeded drives offers.

  return { pc, dc, sid };
}

// Defensive timeout wrapper for applying remote descriptions
async function setRemoteWithTimeout(pc, desc, ms = 5000) {
  return await Promise.race([
    pc.setRemoteDescription(desc),
    new Promise((_, rej) => setTimeout(() => rej(new Error('setRemoteDescription timeout')), ms)),
  ]);
}

// Wait for SecureSignal to create *or already have* a responder session from a trusted peer
function waitForSessionFrom(ss, peerFp, timeoutMs = 15000) {
  // 1) Resolve immediately if a responder session for this peer already exists.
  try {
    if (ss && ss.sessions && typeof ss.sessions.forEach === 'function') {
      let existingSid = null;
      ss.sessions.forEach((v, k) => {
        if (v && v.role === 'responder' && (v.toFp === peerFp || v.peerFp === peerFp)) existingSid = k;
      });
      if (existingSid) return Promise.resolve(existingSid);
    }
  } catch {}

  // 2) Otherwise, await the next matching onSession event.
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error('Timed out waiting for session'));
    }, timeoutMs);

    const originalOnSession = ss.onSession; // save original

    function handle(e) {
      const evt = e && e.sid ? e : (e && e[0]) ? e[0] : e; // guard against odd callers
      if (evt && evt.role === 'responder' && evt.peerFp === peerFp) {
        cleanup();
        resolve(evt.sid);
      }
    }

    function cleanup() {
      clearTimeout(timer);
      // restore the original handler
      ss.onSession = originalOnSession;
    }

    // Wrap to preserve app behavior, then call our handler
    ss.onSession = (...args) => {
      try { originalOnSession && originalOnSession(...args); } catch {}
      try { handle(args[0]); } catch {}
    };
  });
}

// Attach basic handlers to a data channel (and optional proof)
function attachDcHandlers(dc, onData, verifyDataChannel) {
  dc.onopen = () => {
    if (!verifyDataChannel) return;
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
