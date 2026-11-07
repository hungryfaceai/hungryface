
// receiver-core.js — shared signaling + WebRTC for receivers (targets a specific sender)
import { ICE_SERVERS } from '/hungryface/webrtc/shared/config.js';

export class ReceiverCore {
  constructor(cfg) {
    this.wsEndpoint = cfg.wsEndpoint;                 // wss://.../ws
    this.room       = cfg.room || 'Baby';
    this.iceServers = cfg.iceServers || ICE_SERVERS;

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
    this.keepaliveMs          = cfg.keepaliveMs          || 25000;
    this.reconnectMinMs       = cfg.reconnectMinMs       || 1000;
    this.reconnectMaxMs       = cfg.reconnectMaxMs       || 10000;

    // NEW: make 15s constants configurable
    this.wsConnectTimeoutMs   = cfg.wsConnectTimeoutMs   || 10000; // WS open timeout
    this.iceDisconnectGraceMs = cfg.iceDisconnectGraceMs || 9000; // ICE disconnected grace

    // Internals
    this.ws = null;
    this.wsKeepalive = null;
    this.wsRetryMs = this.reconnectMinMs;

    this.pc = null;
    this.remoteDescriptionSet = false;
    this.lastAnswerSdp = null;
    this.candidateQueue = [];
    this.processingAnswer = false;
    this.offerResendTimer = null;
    this.iceDisconnectedSince = null;

    // Lifecycle guards (prevent overlapping reconnect chains / zombie timers)
    this.closed = false;
    this.reconnectTimer = null;
    this.iceGraceTimer = null;
    this.startPromise = null;
    this.wsConnectPromise = null;

    // Targeted routing
    this.targetSenderId = null;
    this.senderId = null;
  }

  /* ---------- Public API ---------- */
  async start() {
    this.closed = false;
    // Single-flight: concurrent callers share one in-progress start
    if (this.startPromise) return this.startPromise;
    this.startPromise = (async () => {
      await this.#ensureWS();
      await this.#ensurePC();
      await this.#negotiate();
    })();
    try { return await this.startPromise; }
    finally { this.startPromise = null; }
  }

  close() {
    this.closed = true;
    if (this.reconnectTimer) { clearTimeout(this.reconnectTimer); this.reconnectTimer = null; }
    this.#clearIceGraceTimer();
    try { this.#stopOfferResendLoop(); } catch {}
    try { this.ws?.close(); } catch {}
    try { this.pc?.close(); } catch {}
    if (this.wsKeepalive) { clearInterval(this.wsKeepalive); this.wsKeepalive = null; }
  }

  getPeerConnection() { return this.pc; }

  /* ---------- Internals ---------- */
  async #ensureWS() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return;
    if (this.closed) throw new Error('ReceiverCore closed');
    // A connect attempt is already in flight — share it instead of opening a second socket
    if (this.wsConnectPromise) return this.wsConnectPromise;

    // Retire a dead socket (CLOSING/CLOSED) so its late onclose can't schedule another reconnect
    const old = this.ws;
    if (old) {
      old.onopen = old.onmessage = old.onerror = old.onclose = null;
      try { old.close(); } catch {}
    }

    this.wsConnectPromise = new Promise((resolve, reject) => {
      const url = `${this.wsEndpoint}?room=${encodeURIComponent(this.room)}`;
      this.#status('Connecting to signaling…');
      const sock = new WebSocket(url);
      this.ws = sock;

      let opened = false;
      const t = setTimeout(() => {
        if (!opened) { try { sock.close(); } catch{}; reject(new Error('WS timeout')); }
      }, this.wsConnectTimeoutMs); // ← was 15000

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
        clearTimeout(t);
        if (sock !== this.ws) return;              // stale socket — ignore
        this.#status('Signaling: closed');
        if (this.wsKeepalive) { clearInterval(this.wsKeepalive); this.wsKeepalive = null; }
        if (!opened) reject(new Error('WS closed before open'));
        this.#scheduleReconnect();
      };
    });
    try { await this.wsConnectPromise; }
    finally { this.wsConnectPromise = null; }
  }

  #scheduleReconnect() {
    if (this.closed || this.reconnectTimer) return; // one pending reconnect at a time
    const base  = Math.min(this.wsRetryMs, this.reconnectMaxMs);
    const delay = Math.round(base * (0.8 + Math.random() * 0.4)); // ±20% jitter
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (this.closed) return;
      this.wsRetryMs = Math.min(this.wsRetryMs * 2, this.reconnectMaxMs);
      this.start().catch(() => {
        // start() failed (e.g. WS timeout without onclose) — keep trying
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) this.#scheduleReconnect();
      });
    }, delay);
  }

  #clearIceGraceTimer() {
    if (this.iceGraceTimer) { clearTimeout(this.iceGraceTimer); this.iceGraceTimer = null; }
  }

  async #restartPC() {
    if (this.closed) return;
    this.#clearIceGraceTimer();
    this.iceDisconnectedSince = null;
    await this.#ensurePC(true);
    await this.#negotiate();
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

    if (this.pc) {
      const oldPc = this.pc;
      // Detach handlers so the retired PC's state events can't trigger another rebuild
      oldPc.oniceconnectionstatechange = oldPc.onicecandidate = oldPc.ontrack = oldPc.ondatachannel = null;
      try { oldPc.close(); } catch{};
    }
    this.#clearIceGraceTimer();
    this.#stopOfferResendLoop();   // kill any old resend timer
    this.processingAnswer = false; // clear any stale lock
    this.pc = null;
    this.remoteDescriptionSet = false;
    this.lastAnswerSdp = null;
    this.candidateQueue.length = 0;

    const pc = new RTCPeerConnection({ iceServers: this.iceServers });
    this.pc = pc;

    // Default recvonly AV
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });

    // Allow page to add custom DCs (e.g., 'pose')
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
      if (pc !== this.pc) return;                  // event from a retired PC
      const s = pc.iceConnectionState;
      this.onIceState?.(s);
      //if (s === 'connected' || s === 'completed') {
      if (s === 'connected' || s === 'completed') {
        //this.#status('Connected'); //// UI logging handled by onIceState in the page
        this.iceDisconnectedSince = null;
        this.#clearIceGraceTimer();
      } else if (s === 'failed') {
        this.#status('ICE: failed');
        this.#restartPC().catch(()=>{});
      } else if (s === 'disconnected') {
        if (!this.iceDisconnectedSince) this.iceDisconnectedSince = Date.now();
        const grace = this.iceDisconnectGraceMs;
        if (!this.iceGraceTimer) {
          this.iceGraceTimer = setTimeout(() => {
            this.iceGraceTimer = null;
            if (pc !== this.pc) return;
            if (this.iceDisconnectedSince && Date.now() - this.iceDisconnectedSince >= grace) {
              //this.#status(`ICE: disconnected >${Math.round(grace/1000)}s (renegotiating)`);
              this.#restartPC().catch(()=>{});
            }
          }, grace + 100); // ← was 15100 to exceed the ≥ check reliably
        }
        this.#status('ICE: disconnected');
      } else if (s === 'closed') {
        this.#clearIceGraceTimer();
        this.#status('ICE: closed');
      } else {
        //this.#status('ICE: ' + s);
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
    //this.#status('Offer sent, waiting for answer…');
  }

  async #sendCurrentOffer() {
    if (this.pc?.localDescription?.type === 'offer') {
      const payload = { type: this.pc.localDescription.type, sdp: this.pc.localDescription.sdp };
      if (this.senderId) payload.to = this.senderId;
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
      this.targetSenderId = msg.id;   // the Camera re-joined with a new id: talk to the new one
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

      /*if (this.pc.signalingState !== 'have-local-offer') {
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
      return;*/
    
      if (this.processingAnswer) return;           // drop concurrent duplicate
      this.processingAnswer = true;
      try {
        // state may change while we were awaiting earlier tasks; check *now*
        if (this.pc.signalingState !== 'have-local-offer') {
          if (this.pc.signalingState === 'stable') this.#stopOfferResendLoop();
          return;
        }
        if (this.lastAnswerSdp === msg.sdp) return;  // exact dup?
      
        // Apply; if another answer raced and got in first, this may flip to stable mid-flight
        await this.pc.setRemoteDescription(new RTCSessionDescription(msg));
        this.remoteDescriptionSet = true;
        this.lastAnswerSdp = msg.sdp;
        this.#stopOfferResendLoop();
        //this.#status('Got answer, applied.');
      
        for (const c of this.candidateQueue) {
          try { await this.pc.addIceCandidate(c); } catch (e) { console.warn('late ICE add failed', e); }
        }
        this.candidateQueue.length = 0;
      } catch (e) {
        // Benign duplicate: someone else already applied an answer
        if (this.pc.signalingState === 'stable') {
          this.remoteDescriptionSet = true;
          this.#stopOfferResendLoop();
          console.debug('[ReceiverCore] duplicate answer ignored');
        } else {
          console.warn('[ReceiverCore] setRemoteDescription failed:', e);
        }
      } finally {
        this.processingAnswer = false;
      }
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

