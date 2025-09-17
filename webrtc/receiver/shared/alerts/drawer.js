import { getAllAlerts } from './store.js';

// Update a badge element with the current count.
// Accepts an element, an id, or a selector (defaults to '#alertsBadge').
export async function refreshAlertsBadge(target = '#alertsBadge') {
  const el =
    typeof target === 'string'
      ? (target.startsWith('#') ? document.querySelector(target) : document.getElementById(target))
      : target;

  if (!el) return;
  try {
    const rows = await getAllAlerts();
    el.textContent = String(rows.length);
  } catch {
    el.textContent = '0';
  }
}

// Keep badge live as alerts change
document.addEventListener('alerts:changed', () => refreshAlertsBadge());
