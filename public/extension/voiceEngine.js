/**
 * ═══════════════════════════════════════════════════════════════
 *  AccessEye — voiceEngine.js
 *  Centralized command parser + intent mapper
 * ═══════════════════════════════════════════════════════════════
 */

// ── Intent constants ──────────────────────────────────────────
export const INTENTS = {
  ACTION_NEW_TAB:       'ACTION_NEW_TAB',
  ACTION_NEXT_TAB:      'ACTION_NEXT_TAB',
  ACTION_PREV_TAB:      'ACTION_PREV_TAB',
  ACTION_ACCESSEYE_TAB: 'ACTION_ACCESSEYE_TAB',
  ACTION_SCROLL_DOWN:   'ACTION_SCROLL_DOWN',
  ACTION_SCROLL_UP:     'ACTION_SCROLL_UP',
  ACTION_SCROLL_TOP:    'ACTION_SCROLL_TOP',
  ACTION_SCROLL_BOTTOM: 'ACTION_SCROLL_BOTTOM',
  ACTION_CLICK:         'ACTION_CLICK',
  ACTION_GO_BACK:       'ACTION_GO_BACK',
  ACTION_GO_FORWARD:    'ACTION_GO_FORWARD',
  ACTION_RELOAD:        'ACTION_RELOAD',
  ACTION_CLOSE_TAB:     'ACTION_CLOSE_TAB',
  ACTION_UNKNOWN:       'ACTION_UNKNOWN',
};

// ── Filler words to strip ─────────────────────────────────────
const FILLERS = [
  'please', 'can you', 'could you', 'would you',
  'hey', 'okay', 'ok', 'alright', 'now', 'just',
  'accesseye', 'computer',
];

/**
 * Normalize raw speech input.
 * - Lowercase
 * - Strip punctuation
 * - Remove filler words
 */
export function normalizeInput(raw) {
  let text = raw.toLowerCase().trim();
  text = text.replace(/[.,!?;:]/g, '');
  for (const filler of FILLERS) {
    text = text.replace(new RegExp(`\\b${filler}\\b`, 'gi'), '');
  }
  return text.replace(/\s+/g, ' ').trim();
}

// ── Intent map: phrase fragments → intent ────────────────────
const INTENT_MAP = [
  // New tab
  { patterns: ['new tab', 'open tab', 'open a tab', 'create tab', 'open new tab'], intent: INTENTS.ACTION_NEW_TAB },

  // Next tab
  { patterns: ['next tab', 'go to next tab', 'switch tab', 'switch to next tab', 'tab right', 'move tab right'], intent: INTENTS.ACTION_NEXT_TAB },

  // Previous tab
  { patterns: ['previous tab', 'last tab', 'go back tab', 'prior tab', 'tab left', 'move tab left', 'switch to previous tab'], intent: INTENTS.ACTION_PREV_TAB },

  // AccessEye tab
  { patterns: ['go to eye tab', 'eye tab', 'go to extension tab', 'extension tab', 'home tab'], intent: INTENTS.ACTION_ACCESSEYE_TAB },

  // Close tab
  { patterns: ['close tab', 'close this tab', 'close current tab'], intent: INTENTS.ACTION_CLOSE_TAB },

  // Scroll down
  { patterns: ['scroll down', 'go down', 'move down', 'page down'], intent: INTENTS.ACTION_SCROLL_DOWN },

  // Scroll up
  { patterns: ['scroll up', 'go up', 'move up', 'page up'], intent: INTENTS.ACTION_SCROLL_UP },

  // Scroll to top
  { patterns: ['scroll to top', 'go to top', 'top of page', 'jump to top', 'back to top'], intent: INTENTS.ACTION_SCROLL_TOP },

  // Scroll to bottom
  { patterns: ['scroll to bottom', 'go to bottom', 'bottom of page', 'jump to bottom', 'end of page'], intent: INTENTS.ACTION_SCROLL_BOTTOM },

  // Navigation
  { patterns: ['go back', 'back', 'navigate back', 'previous page'], intent: INTENTS.ACTION_GO_BACK },
  { patterns: ['go forward', 'forward', 'navigate forward', 'next page'], intent: INTENTS.ACTION_GO_FORWARD },
  { patterns: ['reload', 'refresh', 'reload page', 'refresh page'], intent: INTENTS.ACTION_RELOAD },

  // Click
  { patterns: ['click', 'press', 'tap', 'select'], intent: INTENTS.ACTION_CLICK },
];

/**
 * Parse normalized input into an intent object.
 * Returns { action, params }
 */
export function parseIntent(normalized) {
  for (const { patterns, intent } of INTENT_MAP) {
    for (const pattern of patterns) {
      if (normalized.includes(pattern)) {
        // Extract params for click commands
        let params = {};
        if (intent === INTENTS.ACTION_CLICK) {
          params.target = normalized.replace(/^(click|press|tap|select)\s+/, '').trim();
        }
        return { action: intent, params };
      }
    }
  }
  return { action: INTENTS.ACTION_UNKNOWN, params: { raw: normalized } };
}

/**
 * Full pipeline: raw speech → intent object
 */
export function processVoiceInput(raw) {
  const normalized = normalizeInput(raw);
  const intent = parseIntent(normalized);
  return { raw, normalized, ...intent };
}
