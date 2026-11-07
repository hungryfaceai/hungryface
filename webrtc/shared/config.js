// webrtc/shared/config.js — single place for connection settings used by sender/receivers.
//
// Change the signalling server or ICE (STUN/TURN) servers here instead of in each page.
// Pages import what they need:
//   import { WS_ENDPOINT, ICE_SERVERS } from '/hungryface/webrtc/shared/config.js';

// Signalling server (WebSocket). Override for testing with window.__NAPTIO_WS_ENDPOINT
// before this module loads, e.g. a local server: 'ws://localhost:3000/ws'.
export const WS_ENDPOINT =
  (typeof window !== 'undefined' && window.__NAPTIO_WS_ENDPOINT) || 'wss://signaling-server-f5gu.onrender.com/ws';

// ICE servers for RTCPeerConnection. A TURN server can be added here later so devices
// on different networks (e.g. mobile data) can connect.
export const ICE_SERVERS = [{ urls: 'stun:stun.l.google.com:19302' }];
