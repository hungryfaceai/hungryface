// webrtc/receiver/shared/switcher.js
export function initTrustedSwitcher({
  wsEndpoint,
  currentPeer = '',
  desiredRoom = '',
  onPick,
}) {
  // --- Button (top-right) ---
  const btn = document.createElement('button');
  btn.textContent = 'Switch device';
  btn.style.cssText = `
    position: fixed; top: 12px; right: 12px; z-index: 1000;
    padding: 8px 12px; border-radius: 10px; border: 1px solid #444;
    background:#111; color:#fff; font-size:14px; font-weight:600; cursor:pointer;
    box-shadow: 0 2px 10px rgba(0,0,0,.35);
  `;
  document.body.appendChild(btn);

  // --- Dropdown panel (top-right) ---
  const panel = document.createElement('div');
  panel.style.cssText = `
    position: fixed; top: 50px; right: 12px; z-index: 1001;
    width: min(92vw, 420px); max-height: 60vh; overflow:auto;
    background: rgba(15,15,15,.95); backdrop-filter: blur(4px);
    border: 1px solid #333; border-radius: 12px; padding: 10px; display: none;
    box-shadow: 0 10px 24px rgba(0,0,0,.5);
  `;
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-label', 'Trusted devices');
  document.body.appendChild(panel);

  const header = document.createElement('div');
  header.style.cssText = `display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:8px;`;
  header.innerHTML = `
    <div style="font-weight:700;color:#fff;">Trusted devices</div>
    <div style="display:flex; gap:6px; align-items:center;">
      ${desiredRoom ? `<span style="font-size:12px; color:#aaa;">desired room:</span>
      <span style="font-size:12px; color:#ddd; background:#212121; border:1px solid #2a2a2a; padding:2px 6px; border-radius:8px;">${escapeHtml(desiredRoom)}</span>` : ''}
      <button id="hfSwitchClose" style="border:1px solid #444;background:#1a1a1a;color:#ddd;padding:4px 8px;border-radius:8px;cursor:pointer;">Close</button>
    </div>
  `;
  panel.appendChild(header);

  const list = document.createElement('div');
  list.style.cssText = `display:flex; flex-direction:column; gap:8px;`;
  panel.appendChild(list);

  btn.addEventListener('click', async (e) => {
    e.stopPropagation();
    if (panel.style.display === 'none') {
      await populate();
      panel.style.display = 'block';
    } else {
      panel.style.display = 'none';
    }
  });

  document.getElementById('hfSwitchClose')?.addEventListener('click', () => {
    panel.style.display = 'none';
  });

  // Hide panel when clicking outside
  document.addEventListener('click', (e) => {
    if (panel.style.display === 'none') return;
    if (!panel.contains(e.target) && e.target !== btn) panel.style.display = 'none';
  });

  async function populate() {
    list.innerHTML = '<div style="color:#aaa; font-size:13px;">Loading…</div>';
    let devices = [];
    try {
      // This endpoint is just an example; replace with your real source if needed.
      const res = await fetch(wsEndpoint + '?trusted=list', { cache: 'no-cache' });
      devices = await res.json(); // expected [{ fp, name?, lastMeta?: { room? } }, ...]
      if (!Array.isArray(devices)) devices = [];
    } catch (e) {
      list.innerHTML = `<div style="color:#f88; font-size:13px;">Failed to load trusted devices: ${escapeHtml(String(e))}</div>`;
      return;
    }

    if (!devices.length) {
      list.innerHTML = `<div style="color:#aaa; font-size:13px;">No trusted devices found.</div>`;
      return;
    }

    list.innerHTML = '';
    for (const d of devices) {
      const fp = String(d.fp || '');
      const name = (d.name || '').trim();
      const room = d.lastMeta?.room || '';

      const item = document.createElement('div');
      item.style.cssText = `
        border: 1px solid ${fp === currentPeer ? '#3a6aff' : '#2a2a2a'};
        background: ${fp === currentPeer ? 'rgba(58,106,255,.08)' : '#121212'};
        border-radius: 10px; padding: 10px; display:flex; gap:10px; align-items:flex-start;
      `;

      const meta = document.createElement('div');
      meta.style.cssText = `flex:1 1 auto; min-width:0;`;

      const nameLine = document.createElement('div');
      nameLine.style.cssText = `display:flex; align-items:center; gap:8px; flex-wrap:wrap;`;
      nameLine.innerHTML = `
        <span style="color:#fff; font-weight:600;">${escapeHtml(name || 'Unnamed device')}</span>
        ${fp === currentPeer ? `<span style="font-size:11px; color:#9ad; border:1px solid #2a3d76; padding:1px 6px; border-radius:999px;">current</span>` : ''}
        ${room ? `<span style="font-size:11px; color:#ddd; background:#1b1b1b; border:1px solid #2a2a2a; padding:1px 6px; border-radius:999px;">room: ${escapeHtml(room)}</span>` : ''}
      `;
      meta.appendChild(nameLine);

      const fpLine = document.createElement('div');
      fpLine.style.cssText = `
        margin-top:6px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size:12px; color:#cfcfcf; line-height:1.3;
        word-break: break-all; overflow-wrap: anywhere; /* full FP visible, wraps as needed */
        background:#0e0e0e; border:1px solid #232323; border-radius:8px; padding:6px 8px;
      `;
      fpLine.textContent = fp || '—';
      meta.appendChild(fpLine);

      item.appendChild(meta);

      const side = document.createElement('div');
      side.style.cssText = `display:flex; flex-direction:column; gap:6px;`;

      const copyBtn = document.createElement('button');
      copyBtn.textContent = 'Copy';
      copyBtn.title = 'Copy fingerprint';
      copyBtn.style.cssText = `
        border:1px solid #444;background:#1a1a1a;color:#ddd;
        padding:6px 10px;border-radius:8px;cursor:pointer;font-size:12px;
      `;
      copyBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        try {
          await navigator.clipboard.writeText(fp);
          copyBtn.textContent = 'Copied!';
          setTimeout(() => (copyBtn.textContent = 'Copy'), 1200);
        } catch {}
      });
      side.appendChild(copyBtn);

      const pickBtn = document.createElement('button');
      pickBtn.textContent = fp === currentPeer ? 'Selected' : 'Switch';
      pickBtn.disabled = fp === currentPeer;
      pickBtn.style.cssText = `
        border:1px solid ${fp === currentPeer ? '#2a2a2a' : '#3a6aff'};
        background:${fp === currentPeer ? '#171717' : '#1b2a55'};
        color:${fp === currentPeer ? '#aaa' : '#dbe3ff'};
        padding:6px 10px;border-radius:8px;cursor:${fp === currentPeer ? 'default' : 'pointer'};
        font-size:12px;
      `;
      pickBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (fp && fp !== currentPeer) {
          panel.style.display = 'none';
          onPick?.(fp);
        }
      });
      side.appendChild(pickBtn);

      item.appendChild(side);
      list.appendChild(item);
    }
  }
}

function escapeHtml(s) {
  return String(s)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}
