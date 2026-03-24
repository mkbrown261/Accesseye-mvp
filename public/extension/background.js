/**
 * ═══════════════════════════════════════════════════════════════
 *  AccessEye — background.js (Service Worker)
 *  Voice engine + tab control + message dispatcher
 * ═══════════════════════════════════════════════════════════════
 *
 *  NOTE: Chrome MV3 service workers do not support Web Speech API
 *  directly. Voice recognition is handled in an offscreen document
 *  (Chrome 109+) and messages are relayed here for processing.
 *  This background script owns ALL tab logic and command routing.
 * ═══════════════════════════════════════════════════════════════
 */

import { processVoiceInput, INTENTS } from './voiceEngine.js';

// ── State ────────────────────────────────────────────────────
let isListening = false;
let accessEyeTabId = null; // Track the extension's own tab

// ── Offscreen document (voice recognition lives here) ────────
const OFFSCREEN_URL = chrome.runtime.getURL('offscreen.html');

async function ensureOffscreenDocument() {
  const existing = await chrome.offscreen.getContexts?.({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
  }).catch(() => []);
  if (existing && existing.length > 0) return;

  await chrome.offscreen.createDocument({
    url: OFFSCREEN_URL,
    reasons: ['USER_MEDIA'],
    justification: 'Voice recognition for AccessEye commands',
  }).catch(() => {
    // Fallback: offscreen API may not be available in older Chrome
    console.warn('[AccessEye] Offscreen document unavailable. Voice runs in popup only.');
  });
}

// ── Tab Helpers ───────────────────────────────────────────────
async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

async function getAllTabs() {
  return chrome.tabs.query({ currentWindow: true });
}

async function focusTab(tabId) {
  await chrome.tabs.update(tabId, { active: true });
}

// ── Command Handlers ──────────────────────────────────────────
const handlers = {
  async [INTENTS.ACTION_NEW_TAB]() {
    const tab = await chrome.tabs.create({ url: 'about:blank', active: true });
    speak('New tab opened');
    return { success: true, tabId: tab.id };
  },

  async [INTENTS.ACTION_NEXT_TAB]() {
    const tabs = await getAllTabs();
    const active = tabs.find(t => t.active);
    if (!active) return;
    const idx = tabs.findIndex(t => t.id === active.id);
    const nextTab = tabs[(idx + 1) % tabs.length];
    await focusTab(nextTab.id);
    speak('Next tab');
    return { success: true };
  },

  async [INTENTS.ACTION_PREV_TAB]() {
    const tabs = await getAllTabs();
    const active = tabs.find(t => t.active);
    if (!active) return;
    const idx = tabs.findIndex(t => t.id === active.id);
    const prevTab = tabs[(idx - 1 + tabs.length) % tabs.length];
    await focusTab(prevTab.id);
    speak('Previous tab');
    return { success: true };
  },

  async [INTENTS.ACTION_CLOSE_TAB]() {
    const tab = await getActiveTab();
    if (tab) await chrome.tabs.remove(tab.id);
    return { success: true };
  },

  async [INTENTS.ACTION_ACCESSEYE_TAB]() {
    if (accessEyeTabId) {
      await focusTab(accessEyeTabId).catch(() => { accessEyeTabId = null; });
      if (accessEyeTabId) return { success: true };
    }
    // Fallback: open extension page
    const tab = await chrome.tabs.create({ url: chrome.runtime.getURL('popup.html'), active: true });
    accessEyeTabId = tab.id;
    return { success: true };
  },

  async [INTENTS.ACTION_SCROLL_DOWN]() {
    return sendToContent({ type: 'SCROLL', direction: 'down' });
  },

  async [INTENTS.ACTION_SCROLL_UP]() {
    return sendToContent({ type: 'SCROLL', direction: 'up' });
  },

  async [INTENTS.ACTION_SCROLL_TOP]() {
    return sendToContent({ type: 'SCROLL', direction: 'top' });
  },

  async [INTENTS.ACTION_SCROLL_BOTTOM]() {
    return sendToContent({ type: 'SCROLL', direction: 'bottom' });
  },

  async [INTENTS.ACTION_GO_BACK]() {
    const tab = await getActiveTab();
    if (tab) await chrome.tabs.goBack(tab.id).catch(() => {});
    return { success: true };
  },

  async [INTENTS.ACTION_GO_FORWARD]() {
    const tab = await getActiveTab();
    if (tab) await chrome.tabs.goForward(tab.id).catch(() => {});
    return { success: true };
  },

  async [INTENTS.ACTION_RELOAD]() {
    const tab = await getActiveTab();
    if (tab) await chrome.tabs.reload(tab.id);
    speak('Reloading');
    return { success: true };
  },

  async [INTENTS.ACTION_CLICK](params) {
    return sendToContent({ type: 'CLICK', target: params.target });
  },

  [INTENTS.ACTION_UNKNOWN](params) {
    console.log('[AccessEye] Unknown command:', params.raw);
    return { success: false, reason: 'unknown command' };
  },
};

// ── Send message to active tab's content script ───────────────
async function sendToContent(message) {
  const tab = await getActiveTab();
  if (!tab) return { success: false, reason: 'no active tab' };
  try {
    const response = await chrome.tabs.sendMessage(tab.id, message);
    return response || { success: true };
  } catch (e) {
    // Content script may not be injected on chrome:// pages etc.
    console.warn('[AccessEye] Content script unreachable:', e.message);
    return { success: false, reason: 'content script unreachable' };
  }
}

// ── TTS Feedback ──────────────────────────────────────────────
function speak(text) {
  chrome.tts.speak(text, { rate: 1.2, volume: 0.8 });
}

// ── Main voice command processor ─────────────────────────────
async function handleVoiceCommand(rawText) {
  const result = processVoiceInput(rawText);
  console.log(`[AccessEye] Voice: "${rawText}" → ${result.action}`);

  const handler = handlers[result.action];
  if (handler) {
    return handler(result.params);
  }
  return { success: false };
}

// ── Message listener (from popup.js + offscreen + content) ───
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  switch (message.type) {
    case 'VOICE_RESULT':
      // From offscreen voice recognizer
      handleVoiceCommand(message.text).then(sendResponse);
      return true; // async

    case 'START_LISTENING':
      isListening = true;
      ensureOffscreenDocument().then(() => {
        chrome.runtime.sendMessage({ type: 'START_RECOGNITION' }).catch(() => {});
        sendResponse({ success: true, listening: true });
      });
      return true;

    case 'STOP_LISTENING':
      isListening = false;
      chrome.runtime.sendMessage({ type: 'STOP_RECOGNITION' }).catch(() => {});
      sendResponse({ success: true, listening: false });
      break;

    case 'GET_STATUS':
      sendResponse({ listening: isListening });
      break;

    case 'COMMAND':
      // Direct command from popup (e.g. button click)
      handleVoiceCommand(message.text).then(sendResponse);
      return true;

    default:
      break;
  }
});

// ── Track extension tab lifecycle ────────────────────────────
chrome.tabs.onRemoved.addListener((tabId) => {
  if (tabId === accessEyeTabId) accessEyeTabId = null;
});

// ── Startup ───────────────────────────────────────────────────
chrome.runtime.onInstalled.addListener(() => {
  console.log('[AccessEye] Extension installed. Voice control ready.');
});

chrome.runtime.onStartup.addListener(() => {
  console.log('[AccessEye] Service worker started.');
});
