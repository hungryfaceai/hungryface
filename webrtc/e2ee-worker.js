// App-level E2EE worker for the spec API path (RTCRtpScriptTransform).
// AES-GCM key is derived from (secret, room) via PBKDF2.
// IV = first 4 bytes of SHA-256("iv:"+room+":"+kind+":"+role) + 8 bytes of frame.timestamp.
// Payload is wrapped as: MAGIC(5 bytes = "E2EE1") + AES-GCM(ciphertext+tag).

const MAGIC = new Uint8Array([0x45, 0x32, 0x45, 0x45, 0x31]); // "E2EE1"

async function deriveKey(secret, room) {
  const enc = new TextEncoder();
  const keyMaterial = await crypto.subtle.importKey(
    "raw", enc.encode(secret), { name: "PBKDF2" }, false, ["deriveKey"]
  );
  const salt = enc.encode("webrtc-e2ee:" + room);
  return crypto.subtle.deriveKey(
    { name: "PBKDF2", salt, iterations: 100_000, hash: "SHA-256" },
    keyMaterial, { name: "AES-GCM", length: 256 }, false, ["encrypt", "decrypt"]
  );
}

async function ivSalt(kind, role, room) {
  const enc = new TextEncoder();
  const data = enc.encode("iv:" + room + ":" + kind + ":" + role);
  const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", data));
  return hash.slice(0, 4); // 4 bytes
}

function ivFrom(ts, salt4) {
  // ts is a Number (µs). Build 12-byte IV: [salt4][ts as 8-byte BE]
  const iv = new Uint8Array(12);
  iv.set(salt4, 0);
  const view = new DataView(iv.buffer, 4, 8);
  // Convert to BigInt safely (ts may exceed 2^31)
  const big = BigInt(Math.floor(ts));
  view.setBigUint64(0, big, false);
  return iv;
}

function xorIfNoKey(u8) { for (let i=0;i<u8.length;i++) u8[i]^=0x5a; return u8; } // unreachable fallback

self.addEventListener("rtctransform", async (ev) => {
  // Options were passed from the main thread when creating RTCRtpScriptTransform
  const { secret, room, role, kind } = ev.transformer.options || {};
  const key = await deriveKey(secret, room);
  const salt4 = await ivSalt(kind || "video", role || "sender", room || "default");

  const t = new TransformStream({
    transform: async (frame, controller) => {
      try {
        const ts = (frame.timestamp ?? 0);
        const iv = ivFrom(ts, salt4);
        const data = new Uint8Array(frame.data);

        if (role === "sender") {
          const ct = new Uint8Array(await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, data));
          const out = new Uint8Array(MAGIC.length + ct.length);
          out.set(MAGIC, 0); out.set(ct, MAGIC.length);
          frame.data = out.buffer;
        } else { // receiver
          // If peer didn't enable E2EE yet, pass-through
          if (data.length < MAGIC.length) { controller.enqueue(frame); return; }
          let isMagic = true;
          for (let i=0;i<MAGIC.length;i++) if (data[i] !== MAGIC[i]) { isMagic = false; break; }
          if (!isMagic) { controller.enqueue(frame); return; }
          const pt = new Uint8Array(await crypto.subtle.decrypt({ name: "AES-GCM", iv }, key, data.slice(MAGIC.length)));
          frame.data = pt.buffer;
        }
      } catch (e) {
        // If anything goes wrong, best effort pass-through (so call doesn't collapse)
        try { controller.enqueue(frame); } catch {}
        return;
      }
      controller.enqueue(frame);
    }
  });

  ev.transformer.readable
    .pipeThrough(t)
    .pipeTo(ev.transformer.writable)
    .catch(() => {});
});
