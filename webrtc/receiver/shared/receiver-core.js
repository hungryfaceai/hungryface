//https://chatgpt.com/c/68a91295-fee8-832c-b2b3-0a48defff90f
// receiver-core.js — shared signaling + WebRTC for receivers (targeting a specific sender)
export class ReceiverCore {
  constructor(cfg) {
    this.wsEndpoint = cfg.wsEndpoint;                 // wss://.../ws
    this.room       = cfg.room || 'Baby';
    this.iceServers = cfg.iceServers || [{ urls: 'stun:stun.l.google.com:19302' }];

    // UI / lifecycle hooks (all optional)
    this.onStatus        = cfg.onStatus        || (()=>{});
    this.onStream        = cfg.onStream        || (()=>{});
    this.onIceState      = cfg.onIceState      || (()=>{});
    this.onDataChannel   = cfg.onDataChannel   || (()=>{});
    this.onSenderChange  = cfg.onSenderChange  || (()=>{});
    this.onSignal        = cfg.onSignal        || (()=>{});
    this.onHello         = cfg.onHello         || (()=>{});
    this.onBye           = cfg.onBye           || (()=>{});
    this.onRoster        = cfg.onRoster        || (()=>{});
    this.onNeedOffer     = cfg.onNeedOffer     || (()=>{});
    this.onCreatePC      = cfg.onCreatePC      || (()=>{});

    // Timers / backoff (overridable)
    this.keepaliveMs     = cfg.keepaliveMs     || 25000;
    this.reconnectMinMs  = cfg.reconnectMinMs  || 1000;
    this.reconnectMaxMs  = cfg.reconnectMaxMs  || 10000;

    // Internals
    this.ws = null;
    this.wsKeepalive = null;
    this.wsRetryMs = this.reconnectMinMs;

    this.pc = null;
    this.remoteDescriptionSet = false;
    this.lastAnswerSdp = null;
    this.candidateQueue = [];
    this.offerResendTimer = null;
    this.iceDisconnectedSince = null;

    // Targeted routing
    this.targetSenderId = null;
    this.senderId = null;
  }

  /* ---------- Public API ---------- */
  async start() {
    await this.#ensureWS();
    await this.#ensurePC();
    await this.#negotiate();
  }

  close() {
    try { this.#stopOfferResendLoop(); } catch {}
    try { this.ws?.close(); } catch {}
    try { this.pc?.close(); } catch {}
    if (this.wsKeepalive) clearInterval(this.wsKeepalive);
  }

  getPeerConnection() { return this.pc; }

  /* ---------- Internals ---------- */
  async #ensureWS() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return;
    await new Promise((resolve, reject) => {
      const url = `${this.wsEndpoint}?room=${encodeURIComponent(this.room)}`;
      this.#status('Connecting to signaling…');
      const sock = new WebSocket(url);
      this.ws = sock;

      let opened = false;
      const t = setTimeout(() => {
        if (!opened) { try { sock.close(); } catch{}; reject(new Error('WS timeout')); }
      }, 15000);

      sock.onopen = () => {
        opened = true; clearTimeout(t);
        this.#status('Signaling: connected');
        this.wsRetryMs = this.reconnectMinMs;
        this.#send({ type: 'join', room: this.room });

        if (this.wsKeepalive) clearInterval(this.wsKeepalive);
        this.wsKeepalive = setInterval(() => {
          if (this.ws?.readyState === WebSocket.OPEN) this.#send({ type: 'keepalive', ts: Date.now() });
        }, this.keepaliveMs);

        resolve();
      };
      sock.onmessage = (evt) => this.#onSignal(evt);
      sock.onerror = (e) => console.warn('[WS error]', e);
      sock.onclose  = () => {
        this.#status('Signaling: closed');
        if (this.wsKeepalive) { clearInterval(this.wsKeepalive); this.wsKeepalive = null; }
        this.#scheduleReconnect();
      };
    });
  }

  #scheduleReconnect() {
    const delay = Math.min(this.wsRetryMs, this.reconnectMaxMs);
    setTimeout(() => {
      this.wsRetryMs = Math.min(this.wsRetryMs * 2, this.reconnectMaxMs);
      this.start().catch(()=>{}); // re-enter
    }, delay);
  }

  #send(obj) {
    try {
      if (this.ws?.readyState !== WebSocket.OPEN) return;
      const needsTo = (obj?.type === 'offer' || obj?.type === 'candidate') && this.targetSenderId;
      this.ws.send(JSON.stringify(needsTo ? { ...obj, to: this.targetSenderId } : obj));
    } catch {}
  }

  async #ensurePC(force = false) {
    if (this.pc && !force) return;

    if (this.pc) { try { this.pc.close(); } catch{}; }
    this.pc = null;
    this.remoteDescriptionSet = false;
    this.lastAnswerSdp = null;
    this.candidateQueue.length = 0;

    const pc = new RTCPeerConnection({ iceServers: this.iceServers });
    this.pc = pc;

    // Default recvonly AV
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });

    // Let the page add custom DCs (e.g., 'pose')
    try { this.onCreatePC(pc); } catch (e) { console.warn('[onCreatePC] error', e); }

    // Leave inbound DCs open
    pc.ondatachannel = (e) => {
      this.onDataChannel?.(e);
    };

    pc.onicecandidate = (ev) => { if (ev.candidate) this.#send({ type:'candidate', candidate: ev.candidate }); };
    pc.ontrack = (ev) => {
      const stream = ev.streams?.[0];
      if (stream) this.onStream?.(stream, ev.track);
    };

    pc.oniceconnectionstatechange = () => {
      const s = pc.iceConnectionState;
      this.onIceState?.(s);
      if (s === 'connected' || s === 'completed') {
        this.#status('Connected');
        this.iceDisconnectedSince = null;
      } else if (s === 'failed') {
        this.#status('ICE: failed');
        this.#ensurePC(true); // rebuild pc
        this.#negotiate().catch(()=>{});
      } else if (s === 'disconnected') {
        if (!this.iceDisconnectedSince) this.iceDisconnectedSince = Date.now();
        setTimeout(() => {
          if (this.iceDisconnectedSince && Date.now() - this.iceDisconnectedSince >= 15000) {
            this.#status('ICE: disconnected >15s (renegotiating)');
            this.#ensurePC(true);
            this.#negotiate().catch(()=>{});
          }
        }, 15100);
        this.#status('ICE: disconnected');
      } else if (s === 'closed') {
        this.#status('ICE: closed');
      } else {
        this.#status('ICE: ' + s);
      }
    };
  }

  async #negotiate() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) await this.#ensureWS();
    if (!this.pc) await this.#ensurePC();

    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);
    await this.#sendCurrentOffer();
    this.#startOfferResendLoop();
    this.#status('Offer sent, waiting for answer…');
  }

  async #sendCurrentOffer() {
    if (this.pc?.localDescription?.type === 'offer') {
      const payload = { type: this.pc.localDescription.type, sdp: this.pc.localDescription.sdp };
      if (this.senderId) payload.to = this.senderId;              // narrow to known sender
      this.#send(payload);
    }
  }

  #startOfferResendLoop() {
    this.#stopOfferResendLoop();
    this.offerResendTimer = setInterval(() => {
      if (!this.remoteDescriptionSet) this.#sendCurrentOffer();
      else this.#stopOfferResendLoop();
    }, 2000);
  }
  #stopOfferResendLoop() {
    if (this.offerResendTimer) clearInterval(this.offerResendTimer);
    this.offerResendTimer = null;
  }

  async #onSignal(evt) {
    let msg; try { msg = JSON.parse(evt.data); } catch { return; }
    this.onSignal?.(msg);

    if (msg.type === 'roster' && Array.isArray(msg.peers)) {
      const snd = msg.peers.find(p => p.role === 'sender');
      if (snd) this.senderId = snd.id;
      const newId = snd?.id || null;
      if (newId && newId !== this.targetSenderId) {
        this.targetSenderId = newId;
        await this.#sendCurrentOffer();
      }
      this.onRoster?.(msg);
      return;
    }

    if (msg.type === 'hello') {
      await this.#sendCurrentOffer();
      this.onHello?.(msg);
      return;
    }

    if (msg.type === 'need-offer') {
      await this.#sendCurrentOffer();
      this.onNeedOffer?.(msg);
      return;
    }

    if (msg.type === 'peer-joined' && msg.role === 'sender' && msg.id) {
      this.senderId = msg.id;
      await this.#sendCurrentOffer();
      return;
    }

    if (msg.type === 'bye') {
      this.remoteDescriptionSet = false;
      this.#stopOfferResendLoop();
      this.onBye?.(msg);
      return;
    }

    if (msg.type === 'answer') {
      if (!this.targetSenderId && msg.from) this.targetSenderId = msg.from;
      if (msg.from && this.targetSenderId && msg.from !== this.targetSenderId) return;

      if (this.pc.signalingState !== 'have-local-offer') {
        if (this.pc.signalingState === 'stable') this.#stopOfferResendLoop();
        return;
      }
      if (this.lastAnswerSdp === msg.sdp) return;
      this.lastAnswerSdp = msg.sdp;

      this.#status('Got answer, applying…');
      await this.pc.setRemoteDescription(new RTCSessionDescription(msg));
      this.remoteDescriptionSet = true;
      this.#stopOfferResendLoop();

      for (const c of this.candidateQueue) {
        try { await this.pc.addIceCandidate(c); } catch (e) { console.warn('late ICE add failed', e); }
      }
      this.candidateQueue.length = 0;
      return;
    }

    if (msg.type === 'candidate' && msg.candidate) {
      if (!this.targetSenderId && msg.from) this.targetSenderId = msg.from;
      if (msg.from && this.targetSenderId && msg.from !== this.targetSenderId) return;

      const cand = new RTCIceCandidate(msg.candidate);
      if (this.remoteDescriptionSet) {
        try { await this.pc.addIceCandidate(cand); } catch (e) { console.warn('ICE add failed', e); }
      } else {
        this.candidateQueue.push(cand);
      }
    }
  }

  #status(s) { try { this.onStatus?.(s); } catch {} }
}
