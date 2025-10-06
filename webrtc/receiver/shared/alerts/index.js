// webrtc/receiver/shared/alerts/index.js
// Public surface for shared Alerts UI + store + banner

import { setupAlertBanner, showAlertBanner as _showAlertBanner, hideAlertBanner, setAlertBannerText } from './banner.js';
import { setupAlertsDrawer, refreshAlertsBadge } from './drawer.js';
import { getAllAlerts, addAlert, updateAlert, clearAllAlerts, AlertTypes } from './store.js';
import { mountAlertsPill } from './pill.js';

/** One-shot UI init: pill + drawer + banner */
export function initAlertsUI(opts = {}) {
  // Banner (top red)
  setupAlertBanner();

  // Pill (button + badge). If you already have your own markup, pass ids:
  //   { drawerOpenBtnId: 'btnOpenAlerts', drawerBadgeId:'alertsBadge' }
  const { openBtn, badge } = mountAlertsPill({
    drawerOpenBtnId: opts.drawerOpenBtnId || 'btnOpenAlerts',
    drawerBadgeId:   opts.drawerBadgeId   || 'alertsBadge',
    injectIfMissing: true
  });

  // Drawer (history modal). You can pass a host to render into:
  setupAlertsDrawer({
    openBtnEl: openBtn,
    badgeEl: badge,
    hostId: opts.drawerHostId || 'alertsModalHost'
  });

  // Initial badge load
  refreshAlertsBadge();
}

/** Start a new alert episode; returns the record id. */
export async function beginAlert({ type, message }) {
  // type must be from AlertTypes; message is free text
  const rec = {
    type: String(type || 'Motion'),
    message: String(message || ''),
    startAt: new Date().toISOString(),
    endAt: null,
    avgScore: 0
  };
  const id = await addAlert(rec);
  // Notify listeners (drawer & badge refresh themselves)
  document.dispatchEvent(new CustomEvent('alerts:changed', { detail: { action: 'add', id, record: { id, ...rec } } }));
  return id;
}

/** Finish an existing alert episode (by id). */
export async function finishAlert(id, { avgScore = 0, message = '' } = {}) {
  const patch = {
    endAt: new Date().toISOString(),
    avgScore: Number(avgScore) || 0
  };
  if (message) patch.message = message;
  const ok = await updateAlert(id, patch);
  if (ok) document.dispatchEvent(new CustomEvent('alerts:changed', { detail: { action: 'update', id, record: patch } }));
  return ok;
}

/** Red banner (styled like audio page) */
export function showAlertBanner(whenMs = Date.now(), text) {
  _showAlertBanner(whenMs, text);
}

// re-exports for convenience / compatibility
export { refreshAlertsBadge, getAllAlerts, clearAllAlerts, AlertTypes, hideAlertBanner, setAlertBannerText, };
