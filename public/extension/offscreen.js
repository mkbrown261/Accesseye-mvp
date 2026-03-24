/**
 * ═══════════════════════════════════════════════════════════════
 *  AccessEye — offscreen.js
 *  Runs in an offscreen document — owns Web Speech API
 *  Relays results to background.js
 * ═══════════════════════════════════════════════════════════════
 *
 *  Chrome MV3 service workers cannot use Web Speech API.
 *  Offscreen documents CAN. This file bridges that gap.
 * ═══════════════════════════════════════════════════════════════
 */

let recognition = null;
let active = false;

function initRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    console.error('[AccessEye] SpeechRecognition not available');
    return null;
  }

  const r = new SpeechRecognition();
  r.continuous = true;
  r.interimResults = false;
  r.lang = 'en-US';
  r.maxAlternatives = 1;

  r.onresult = (event) => {
    const last = event.results[event.results.length - 1];
    if (last.isFinal) {
      const text = last[0].transcript.trim();
      if (text) {
        chrome.runtime.sendMessage({ type: 'VOICE_RESULT', text });
      }
    }
  };

  r.onend = () => {
    // Auto-restart if still supposed to be active
    if (active) {
      setTimeout(() => {
        try { r.start(); } catch (e) { /* already started */ }
      }, 200);
    }
  };

  r.onerror = (e) => {
    console.warn('[AccessEye] SpeechRecognition error:', e.error);
    if (e.error === 'not-allowed') {
      active = false;
      chrome.runtime.sendMessage({ type: 'MIC_PERMISSION_DENIED' });
    }
  };

  return r;
}

function startListening() {
  if (!recognition) recognition = initRecognition();
  if (!recognition) return;
  active = true;
  try { recognition.start(); } catch (e) { /* already started */ }
}

function stopListening() {
  active = false;
  if (recognition) {
    try { recognition.stop(); } catch (e) {}
  }
}

// Listen for commands from background.js
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'START_RECOGNITION') startListening();
  if (message.type === 'STOP_RECOGNITION') stopListening();
});
