// Public entrypoint that re-exports alerts functionality
// Used by all receivers (Audio / Motion / Prone / Fence)

import {
  showAlertBanner,
  hideAlertBanner,
  setSnoozeMinutes,
  getSnoozeMinutes
} from './banner.js';

import { refreshAlertsBadge } from './drawer.js';

import {
  openAlertDB,
  saveAlertRecord,
  updateAlertById,
  getAllAlerts,
  clearAllAlerts,
  AlertTypes
} from './store.js';

export {
  // banner
  showAlertBanner,
  hideAlertBanner,
  setSnoozeMinutes,
  getSnoozeMinutes,
  // drawer
  refreshAlertsBadge,
  // store
  openAlertDB,
  getAllAlerts,
  clearAllAlerts,
  // enum
  AlertTypes
};

// Begin/finish lifecycle helpers used by receiver pages
export async function beginAlert({ type, message }) {
  return saveAlertRecord({
    type,
    message: message || '',
    startAt: new Date().toISOString()
  }); // returns the record id
}

export async function finishAlert(id, extra = {}) {
  if (!id) return false;
  const patch = {
    endAt: new Date().toISOString()
  };
  if (extra.avgScore != null) patch.avgScore = Number(extra.avgScore) || 0;
  if (extra.message) patch.message = extra.message;
  return updateAlertById(id, patch);
}

// Lightweight UI initializer (auto-refresh badge if present)
export function initAlertsUI({ drawerBadgeId = 'alertsBadge' } = {}) {
  const badge =
    document.getElementById(drawerBadgeId) ||
    document.querySelector('#alertsBadge');

  const refresh = () => refreshAlertsBadge(badge || '#alertsBadge');
  document.addEventListener('alerts:changed', refresh);
  refresh();
}
