/**
 * ═══════════════════════════════════════════════════════════════
 *  AccessEye — popup.js
 *  Controls the popup UI — communicates with background.js
 * ═══════════════════════════════════════════════════════════════
 */

const btnToggle  = document.getElementById('btnToggle');
const statusDot  = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const micDenied  = document.getElementById('micDenied');

let listening = false;

// ── Sync UI state ─────────────────────────────────────────────
function setUI(isListening) {
  listening = isListening;
  if (isListening) {
    btnToggle.textContent  = '⏹ Stop Listening';
    btnToggle.classList.add('active');
    statusDot.classList.add('active');
    statusText.textContent = 'Listening';
  } else {
    btnToggle.textContent  = '▶ Start Listening';
    btnToggle.classList.remove('active');
    statusDot.classList.remove('active');
    statusText.textContent = 'Idle';
  }
}

// ── Get current status from background ───────────────────────
chrome.runtime.sendMessage({ type: 'GET_STATUS' }, (response) => {
  if (chrome.runtime.lastError) return;
  if (response) setUI(response.listening);
});

// ── Toggle button ─────────────────────────────────────────────
btnToggle.addEventListener('click', () => {
  const type = listening ? 'STOP_LISTENING' : 'START_LISTENING';
  chrome.runtime.sendMessage({ type }, (response) => {
    if (chrome.runtime.lastError) return;
    if (response) setUI(response.listening);
  });
});

// ── Listen for mic permission denial ─────────────────────────
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'MIC_PERMISSION_DENIED') {
    micDenied.classList.add('visible');
    setUI(false);
  }
});
