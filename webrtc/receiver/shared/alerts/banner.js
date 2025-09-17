// Generic alert banner with persistent, tunable snooze (dismiss) duration.
<option value="15">15 min</option>
<option value="30">30 min</option>
<option value="60">60 min</option>
</select>
</label>
<button id="alertDismissBtn" class="banner-link" type="button">Dismiss</button>
</div>
</div>
`.trim();
const node = wrap.firstChild;
host.appendChild(node);


bannerEl = node;
timeEl = node.querySelector('#alertBannerTime');
dismissBtn = node.querySelector('#alertDismissBtn');
minsSelect = node.querySelector('#alertsDismissMins');


const saved = getSavedSnoozeMinutes();
minsSelect.value = String(saved);


minsSelect.addEventListener('change', () => setSavedSnoozeMinutes(Number(minsSelect.value)));


dismissBtn?.addEventListener('click', () => {
const mins = Number(minsSelect.value) || getSavedSnoozeMinutes();
setSavedSnoozeMinutes(mins);
suppressUntil = Date.now() + mins * 60 * 1000;
hideAlertBanner();
});
}


export function showAlertBanner(whenMs = Date.now()) {
if (!bannerEl) setupAlertBanner();
if (Date.now() < suppressUntil) return; // snoozed
if (timeEl) timeEl.textContent = fmtHM(whenMs);
bannerEl?.classList.remove('hidden');
}


export function hideAlertBanner() {
bannerEl?.classList.add('hidden');
}


export function setSnoozeMinutes(mins) { setSavedSnoozeMinutes(mins); if (minsSelect) minsSelect.value = String(mins); }
export function getSnoozeMinutes() { return getSavedSnoozeMinutes(); }
