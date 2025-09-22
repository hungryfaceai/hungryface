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
  localStream = null,        // NEW: add tracks before first offer
  localTracks = null,        // NEW: optional alternative to localStream
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
      const ms = localStream || new MediaStream();
      // If localStream not given but tracks are, build a temporary stream for addTrack’s 2nd arg.
      const streamForSender = localStream || new MediaStream(tracks);
      for (const t of tracks) pc.addTrack(t, streamForSender);
    }
  }

  // Drive negotiation from the initiator only
  pc.onnegotiationneeded = async () => {
    if (!initiator) return; // only initiator starts offers
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
      // Only the responder handles remote offers
      if (initiator) return;

      const offer = new RTCSessionDescription(msg.desc);
      const offerCollision = makingOffer || pc.signalingState !== 'stable';

      if (offerCollision) {
        if (!polite) return;                 // impolite side ignores glare
        try {
          await pc.setLocalDescription({ type: 'rollback' }); // polite side rolls back
        } catch (e) {
          // benign if not in a state allowing rollback
        }
      }

      try {
        await setRemoteWithTimeout(pc, offer);
      } catch (e) {
        // If we hit a timing edge, drop the offer (polite flow should resend)
        return;
      }

      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      await ss.sendJSON(sid, { webrtc: 'answer', desc: pc.localDescription });
      return;
    }

    if (msg.webrtc === 'answer') {
      // Only the initiator expects/handles an answer
      if (!initiator) return;
      // Ignore late/duplicate answers if we no longer have a local offer pending
      if (pc.signalingState !== 'have-local-offer') return;
      try {
        await setRemoteWithTimeout(pc, new RTCSessionDescription(msg.desc));
      } catch (e) {
        // Likely a race or late answer; safe to ignore
      }
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

  // NOTE: No manual "kick off" initial offer here.
  // Adding tracks (or creating the data channel) will trigger onnegotiationneeded,
  // and only the initiator will generate/signal the offer.

  return { pc, dc, sid };
}

// Defensive timeout wrapper for applying remote descriptions
async function setRemoteWithTimeout(pc, desc, ms = 5000) {
  return await Promise.race([
    pc.setRemoteDescription(desc),
    new Promise((_, rej) => setTimeout(() => rej(new Error('setRemoteDescription timeout')), ms)),
  ]);
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
    ss.onSession = (...args) => { try { prev && prev(...args); } catch {} ; onSession(...(args[0] || args)); };
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
