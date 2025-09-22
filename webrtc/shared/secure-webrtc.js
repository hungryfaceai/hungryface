// secure-webrtc.js — Gate WebRTC on pairing + encrypted signaling via SecureSignal
// Usage (initiator):
//   const { pc, dc } = await connectToPeer(ss, peerFp, {
//     initiator: true,
//     localStreams: [myMediaStream],   // <-- add tracks before first offer
//     rtcConfig
//   });
// Usage (responder): waits for ss.onSession({role:'responder'}) for that peer.

export async function connectToPeer(
  ss,
  peerFp,
  {
    initiator = false,
    rtcConfig = { iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] },
    localStreams = [],              // NEW: streams whose tracks will be added before creating an offer
    onTrack,                        // (event) => {}
    onData,                         // (message, dc) => {}
    label = 'app',                  // datachannel label (initiator creates it)
    verifyDataChannel = false,      // optional DC proof
    onBeforeAnswer,
    sid: knownSid,
  } = {}
) {
  // 1) Make sure we're paired
  const peer = ss.trusted.find(d => d.fp === peerFp);
  if (!peer) throw new Error('Peer not in Trusted devices. Pair first.');

  // 2) Establish encrypted signaling session
  const sid = initiator
    ? (await ss.startSession(peerFp)).sid
    : (knownSid || await waitForSessionFrom(ss, peerFp));

  // 3) Create RTCPeerConnection
  const pc = new RTCPeerConnection(rtcConfig);

  // --- Perfect negotiation helpers ---
  let makingOffer = false;
  const polite = !initiator; // responder is polite

  // Forward local ICE via encrypted signaling
  pc.onicecandidate = ({ candidate }) => {
    ss.sendJSON(sid, { webrtc: 'ice', candidate });
  };

  // Optional media
  if (onTrack) pc.ontrack = onTrack;

  // Data channel
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

  // ---- Add tracks BEFORE any offer is created ----
  // Adding tracks triggers onnegotiationneeded on the initiator.
  try {
    for (const stream of (localStreams || [])) {
      for (const track of stream.getTracks()) {
        pc.addTrack(track, stream);
      }
    }
  } catch (e) {
    // non-fatal if no streams provided
  }

  // Initiator only: drive negotiation from here
  pc.onnegotiationneeded = async () => {
    if (!initiator) return;
    try {
      makingOffer = true;
      const offer = await pc.createOffer({ offerToReceiveAudio: true, offerToReceiveVideo: true });
      await pc.setLocalDescription(offer);
      await ss.sendJSON(sid, { webrtc: 'offer', desc: pc.localDescription });
    } finally {
      makingOffer = false;
    }
  };

  // Encrypted signaling handler (role-gated + glare-safe)
  ss.onEncrypted(sid, async (msg) => {
    if (msg.webrtc === 'offer') {
      // Only responder handles remote offers
      if (initiator) return;

      const offer = new RTCSessionDescription(msg.desc);
      const offerCollision = makingOffer || pc.signalingState !== 'stable';

      if (offerCollision) {
        if (!polite) return; // impolite side ignores glare
        try { await pc.setLocalDescription({ type: 'rollback' }); } catch {}
      }

      await pc.setRemoteDescription(offer);
      // Allow responder to add transceivers/tracks before answering
      if (typeof onBeforeAnswer === 'function') {
        try { await onBeforeAnswer(pc); } catch {}
      }
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      await ss.sendJSON(sid, { webrtc: 'answer', desc: pc.localDescription });
      return;
    }

    if (msg.webrtc === 'answer') {
      // Only the initiator expects/handles an answer
      if (!initiator) return;

      // Hardening: ignore duplicates/out-of-order answers
      if (pc.signalingState !== 'have-local-offer') return;

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

  // 4) Kick off initial negotiation from initiator if no tracks were provided yet.
  // (If tracks were added above, onnegotiationneeded already fired.)
  if (initiator && (!localStreams || localStreams.length === 0)) {
    try {
      makingOffer = true;
      const offer = await pc.createOffer({ offerToReceiveAudio: true, offerToReceiveVideo: true });
      await pc.setLocalDescription(offer);
      await ss.sendJSON(sid, { webrtc: 'offer', desc: pc.localDescription });
    } finally {
      makingOffer = false;
    }
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
      const prev = ss.onSession;
      ss.onSession = prev;
    }

    const prev = ss.onSession;
    ss.onSession = (...args) => {
      try { prev && prev(...args); } catch {}
      onSession(...(args[0] || args));
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
