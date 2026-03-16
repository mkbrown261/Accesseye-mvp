/**
 * ═══════════════════════════════════════════════════════════════════════════
 *  AccessEye — Voice Navigation + Intent Fusion System
 *  voice-nav.js  (completely standalone — NEVER modifies cursor or tracking)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  System 1 – Voice Control Navigation
 *    • Web Speech API continuous recognition
 *    • Scans DOM for interactive elements → NavigationElementList
 *    • Matches speech to element text/label → highlights → clicks
 *
 *  System 2 – Intent Fusion (Gaze + Voice)
 *    • Reads window.app._lastScreenX / _lastScreenY (cursor output only)
 *    • On action command ("click", "open", "select" …) applies to gaze target
 *    • Separate from all cursor/tracking code — read-only access to gaze position
 *
 *  Safety: this module ONLY adds event listeners and reads cursor position.
 *          It does NOT modify, replace, or wrap any phase2/phase3 functions.
 * ═══════════════════════════════════════════════════════════════════════════
 */

'use strict';

/* ─────────────────────────────────────────────────────────────────────────
   CONSTANTS
───────────────────────────────────────────────────────────────────────── */
const VN_VERSION = '1.0.0';

// Words to strip before matching
const FILLER_WORDS = new Set([
  'please','go','to','can','you','would','the','a','an','and','or','now',
  'just','hey','um','uh','like','that','this','it','on','at','in','with'
]);

// Voice commands → action keys
const ACTION_COMMANDS = {
  click    : 'click',
  press    : 'click',
  tap      : 'click',
  open     : 'open',
  select   : 'select',
  choose   : 'select',
  scroll   : 'scroll',
  'scroll up'           : 'scrollUp',
  'scroll down'         : 'scrollDown',
  'stop scrolling'      : 'stopScrolling',
  'scroll to top'       : 'scrollTop',
  'scroll to bottom'    : 'scrollBottom',
  'go back'             : 'navBack',
  'go forward'          : 'navForward',
  'reload page'         : 'reloadPage',
  'open new tab'        : 'newTab',
  'close tab'           : 'closeTab',
  'zoom in'             : 'zoomIn',
  'zoom out'            : 'zoomOut',
  'reset zoom'          : 'resetZoom',
  'double click'        : 'dblclick',
  'right click'         : 'rightClick',
  'next item'           : 'focusNext',
  'previous item'       : 'focusPrev',
  'pause control'       : 'pauseControl',
  'resume control'      : 'resumeControl',
  'reset cursor'        : 'resetCursor',
  'clear selection'     : 'clearSelection',
  'exit mode'           : 'exitMode',
  // FIX VOICE-4: Discovery commands — were never in ACTION_COMMANDS, causing
  // fallthrough to navList.findBest which matched "Show User Guide" button instead
  'show clickable items': 'showClickable',
  'show clickable'      : 'showClickable',
  'hide clickable items': 'hideClickable',
  'hide clickable'      : 'hideClickable',
  'what can i click'    : 'showClickable',
  'show interactive'    : 'showClickable',
  'show all clickable'  : 'showClickable',
  'highlight clickable' : 'showClickable',
  'focus on'            : 'focusOn',
  // Editing commands
  'select all'          : 'selectAll',
  'open settings'       : 'openSettings',
  'focus search'        : 'focusSearch',
  copy     : 'copyText',
  paste    : 'pasteText',
  cut      : 'cutText',
  // Pause/resume voice itself
  'pause voice'         : 'pauseVoice',
  'resume voice'        : 'resumeVoice',
  play     : 'play',
  pause    : 'pause',
  stop     : 'stopControl',
  submit   : 'submit',
  send     : 'submit',
  back     : 'navBack',
  cancel   : 'cancel',
  close    : 'cancel',
  home     : 'nav-home',
  demo     : 'nav-demo',
  architecture : 'nav-architecture',
  docs     : 'nav-docs',
  studio   : 'nav-studio',
  calibrate : 'mode-calibrate',
  gaze     : 'mode-gaze',
  mouse    : 'mode-mouse',
  start    : 'start-camera-btn',
  restart  : 'start-camera-btn',
  camera   : 'start-camera-btn',
};

/* Actions that execute directly without needing a gaze/element target */
const NO_TARGET_ACTIONS = new Set([
  'scrollUp','scrollDown','scroll','stopScrolling','scrollTop','scrollBottom',
  'navBack','navForward','reloadPage','newTab','closeTab',
  'zoomIn','zoomOut','resetZoom',
  'focusNext','focusPrev',
  'pauseControl','resumeControl','stopControl',
  'resetCursor','clearSelection','exitMode',
  'play','pause',
  // FIX VOICE-4: Discovery + editing commands need no gaze target
  'showClickable','hideClickable','focusOn',
  'selectAll','copyText','pasteText','cutText','openSettings','focusSearch',
  'pauseVoice','resumeVoice',
]);

/* FIX VOICE-1: Nav/mode/camera commands must fire directly — they target a
   specific element by ID and do NOT need a gaze target.  Routing them through
   intent fusion caused silent failures whenever no element was near the cursor. */
const DIRECT_ID_ACTIONS = new Set([
  'nav-home','nav-demo','nav-architecture','nav-docs','nav-studio',
  'mode-calibrate','mode-gaze','mode-mouse',
  'start-camera-btn',
]);

/* ─────────────────────────────────────────────────────────────────────────
   NAVIGATION ELEMENT LIST
   Scans the DOM for interactive elements and caches them.
───────────────────────────────────────────────────────────────────────── */
class NavigationElementList {
  constructor() {
    this.elements = [];
    this._scanTimer = null;
  }

  /** Full DOM scan — call on page navigation or when voice is enabled */
  scan() {
    this.elements = [];
    const selectors = [
      'button:not([disabled])',
      '[role="button"]:not([disabled])',
      'a[href]',
      '[data-id]',
      '.nav-btn',
      '.mode-tab',
      '.gaze-target',
    ];

    const seen = new Set();
    document.querySelectorAll(selectors.join(',')).forEach(el => {
      if (seen.has(el)) return;
      // Skip hidden elements
      const rect = el.getBoundingClientRect();
      if (rect.width === 0 && rect.height === 0) return;
      if (getComputedStyle(el).display === 'none') return;
      if (getComputedStyle(el).visibility === 'hidden') return;

      seen.add(el);

      const text = this._extractText(el);
      if (!text) return;

      const entry = {
        el,
        text,
        normalised: this._normalise(text),
        type: el.tagName.toLowerCase(),
        id: el.id || el.dataset?.id || '',
        rect: () => el.getBoundingClientRect(),   // live rect
      };
      this.elements.push(entry);
    });

    return this.elements.length;
  }

  _extractText(el) {
    // Priority: data-label → aria-label → title → innerText
    return (
      el.dataset?.label ||
      el.getAttribute('aria-label') ||
      el.getAttribute('title') ||
      el.textContent?.trim().replace(/\s+/g, ' ').slice(0, 60) ||
      ''
    ).trim();
  }

  _normalise(str) {
    return str.toLowerCase().replace(/[^a-z0-9\s]/g, '').trim();
  }

  /**
   * Find best match for spoken words.
   * Returns { entry, score } or null.
   */
  findBest(words) {
    const needle = words.join(' ');
    let best = null, bestScore = 0;

    for (const entry of this.elements) {
      const score = this._matchScore(needle, entry.normalised);
      if (score > bestScore) {
        bestScore = score;
        best = entry;
      }
    }
    return bestScore >= 0.4 ? { entry: best, score: bestScore } : null;
  }

  _matchScore(needle, haystack) {
    if (haystack === needle) return 1.0;
    if (haystack.includes(needle)) return 0.9;
    if (needle.includes(haystack)) return 0.85;

    // Word overlap score
    const nWords = needle.split(' ');
    const hWords = new Set(haystack.split(' '));
    const overlap = nWords.filter(w => hWords.has(w)).length;
    if (overlap > 0) return 0.5 + (overlap / nWords.length) * 0.4;

    // Levenshtein-lite: check shortest word similarity
    for (const nw of nWords) {
      if (nw.length < 3) continue;
      for (const hw of hWords) {
        if (hw.length < 3) continue;
        if (hw.startsWith(nw) || nw.startsWith(hw)) return 0.55;
      }
    }
    return 0;
  }

  /** Find the element closest to screen position (px) */
  findNearestToPoint(x, y, maxDistPx = 250) {
    let best = null, bestDist = Infinity;
    for (const entry of this.elements) {
      const r = entry.rect();
      const cx = r.left + r.width / 2;
      const cy = r.top + r.height / 2;
      const dist = Math.hypot(cx - x, cy - y);
      if (dist < bestDist && dist < maxDistPx) {
        bestDist = dist;
        best = entry;
      }
    }
    return best ? { entry: best, dist: bestDist } : null;
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   VOICE NAVIGATION CONTROLLER
───────────────────────────────────────────────────────────────────────── */
class VoiceNavigationController {
  constructor() {
    this.enabled      = false;
    this.recognition  = null;
    this.navList      = new NavigationElementList();
    this._lastGazeTarget = null;
    this._confirmTimer   = null;
    this._scanScheduled  = false;
    // FIX VOICE-4: pause/resume voice flag
    this._paused = false;
    // FIX VOICE-4: overlay tracking for showClickable / hideClickable
    this._clickableOverlays = [];

    // UI refs (set after DOM ready)
    this._transcriptEl = null;
    this._badgeEl      = null;
    this._toggleBtn    = null;
    this._toggleLabel  = null;

    this._initUI();
  }

  /* ── UI Setup ── */
  _initUI() {
    const ready = () => {
      this._transcriptEl = document.getElementById('voice-transcript-bar');
      this._badgeEl      = document.getElementById('voice-status-badge');
      this._toggleBtn    = document.getElementById('voice-toggle-btn');
      this._toggleLabel  = document.getElementById('voice-toggle-label');

      if (this._toggleBtn) {
        this._toggleBtn.addEventListener('click', () => this.toggle());
      }

      // Periodically re-scan on page navigation (debounced)
      document.addEventListener('click', () => this._scheduleScan());
    };

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', ready);
    } else {
      setTimeout(ready, 500); // give app.js time to render
    }
  }

  _scheduleScan() {
    if (this._scanScheduled) return;
    this._scanScheduled = true;
    setTimeout(() => {
      this.navList.scan();
      this._scanScheduled = false;
    }, 300);
  }

  /* ── Toggle ── */
  toggle() {
    if (this.enabled) {
      this.disable();
    } else {
      this.enable();
    }
  }

  enable() {
    if (!('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)) {
      this._setStatus('Not supported in this browser', '#e53e3e');
      this._showToast('Voice Not Supported',
        'Your browser does not support the Web Speech API. Try Chrome or Edge.',
        'warn');
      return;
    }

    this.enabled = true;
    this._updateToggleUI(true);
    this.navList.scan();
    this._startRecognition();
    this._showToast('Voice Navigation ON', 'Say a button name or "click" / "scroll"', 'success');
  }

  disable() {
    this.enabled = false;
    this._updateToggleUI(false);
    if (this.recognition) {
      try { this.recognition.stop(); } catch (_) {}
      this.recognition = null;
    }
    this._setTranscript('Say a button name or "click", "scroll"…');
    this._setStatus('Inactive', '#546e7a');
    this._showToast('Voice Navigation OFF', '', 'info');
  }

  _updateToggleUI(on) {
    if (this._toggleBtn) {
      this._toggleBtn.classList.toggle('active', on);
      const icon = this._toggleBtn.querySelector('i');
      if (icon) icon.className = on ? 'fas fa-microphone' : 'fas fa-microphone-slash';
    }
    if (this._toggleLabel) this._toggleLabel.textContent = on ? 'ON' : 'OFF';
    this._setStatus(on ? 'Listening…' : 'Inactive', on ? '#00d4ff' : '#546e7a');
  }

  /* ── Speech Recognition ── */
  _startRecognition() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return;

    this.recognition = new SR();
    this.recognition.continuous     = true;
    this.recognition.interimResults = true;
    this.recognition.lang           = 'en-US';
    this.recognition.maxAlternatives = 3;

    this.recognition.onstart = () => {
      this._setStatus('Listening…', '#00d4ff');
    };

    this.recognition.onresult = (event) => {
      let interim = '', final = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          final += t;
        } else {
          interim += t;
        }
      }

      const display = final || interim;
      if (display) this._setTranscript(display, !!interim);

      if (final.trim()) {
        this._processUtterance(final.trim(), event.results[event.results.length - 1][0].confidence);
      }
    };

    this.recognition.onerror = (e) => {
      if (e.error === 'not-allowed') {
        this._setStatus('Mic blocked', '#e53e3e');
        this._showToast('Microphone Blocked', 'Allow mic access to use voice navigation.', 'warn');
        this.disable();
      } else if (e.error !== 'no-speech' && e.error !== 'aborted') {
        console.warn('[VoiceNav] recognition error:', e.error);
        this._setStatus(`Error: ${e.error}`, '#e53e3e');
      }
    };

    this.recognition.onend = () => {
      // Auto-restart if still enabled
      if (this.enabled) {
        setTimeout(() => {
          if (this.enabled && this.recognition) {
            try { this.recognition.start(); } catch (_) {}
          }
        }, 300);
      }
    };

    this.recognition.start();
  }

  /* ── Utterance Processing ── */
  _processUtterance(text, confidence = 1) {
    // FIX VOICE-4: honour pause/resume voice state
    if (this._paused) {
      const lower = text.toLowerCase();
      if (lower.includes('resume voice') || lower.includes('resume control')) {
        this._paused = false;
        this._setStatus('Listening…', '#00d4ff');
        this._showToast('Voice Resumed', 'Voice navigation active', 'success');
      }
      return;
    }

    const lower = text.toLowerCase();
    const words = this._tokenise(text);
    if (!words.length) return;

    console.log(`[VoiceNav] heard: "${text}" (words: [${words.join(', ')}])`);

    // 0. FIX VOICE-4: Multi-word action check against raw lowercase text FIRST
    //    This ensures "show clickable items", "focus on", "select all" etc. are
    //    caught before the tokenised single-word pipeline even runs.
    const multiAction = this._extractMultiWordAction(lower);
    if (multiAction) {
      // "focus on <name>" needs special handling — extract the target name
      if (multiAction === 'focusOn') {
        const afterFocus = lower.replace(/focus on\s*/i, '').trim();
        if (afterFocus) {
          this._handleFocusOn(afterFocus);
        } else {
          this._setStatus('Say: focus on [element name]', '#f59e0b');
          setTimeout(() => { if (this.enabled) this._setStatus('Listening…', '#00d4ff'); }, 2000);
        }
        return;
      }
      this._setStatus(`▶ ${text}`, '#00ff88');
      this._performAction({ el: null, text }, multiAction);
      return;
    }

    // 1. Check for action commands
    const action = this._extractAction(words);
    if (action) {
      // Control-recovery and element-free navigation commands execute directly —
      // they do NOT need a gaze target so must not go through intent fusion.
      if (NO_TARGET_ACTIONS.has(action)) {
        this._setStatus(`▶ ${text}`, '#00ff88');
        this._performAction({ el: null, text }, action);
        return;
      }
      // FIX VOICE-1: Nav/mode/camera commands fire directly — no gaze target needed
      if (DIRECT_ID_ACTIONS.has(action)) {
        this._setStatus(`▶ ${text}`, '#00ff88');
        this._performAction({ el: null, text }, action);
        return;
      }
      // All other actions route through intent fusion (gaze target required)
      this._handleIntentFusion(action, confidence, text);
      return;
    }

    // 2. Check ACTION + ELEMENT combinations ("click home", "open demo")
    const embeddedAction = this._extractEmbeddedAction(words);
    if (embeddedAction) {
      const { action: act, remaining } = embeddedAction;
      if (remaining.length > 0) {
        const match = this.navList.findBest(remaining);
        if (match) {
          this._executeOnElement(match.entry, act, text);
          return;
        }
      }
      // No element specified — fall through to gaze fusion
      this._handleIntentFusion(act, confidence, text);
      return;
    }

    // 3. Named element match → click it
    const match = this.navList.findBest(words);
    if (match) {
      this._executeOnElement(match.entry, 'click', text);
      return;
    }

    // 4. No match
    this._setStatus('No match', '#f59e0b');
    setTimeout(() => {
      if (this.enabled) this._setStatus('Listening…', '#00d4ff');
    }, 1500);
  }

  /* FIX VOICE-4: Match raw lowercase text against multi-word ACTION_COMMANDS keys.
     Sorts by descending length so longest phrase wins over shorter sub-phrases. */
  _extractMultiWordAction(lower) {
    const multiKeys = Object.keys(ACTION_COMMANDS)
      .filter(k => k.includes(' '))
      .sort((a, b) => b.length - a.length);
    for (const key of multiKeys) {
      if (lower.includes(key)) return ACTION_COMMANDS[key];
    }
    return null;
  }

  _tokenise(text) {
    return text.toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .split(/\s+/)
      .filter(w => w.length > 0 && !FILLER_WORDS.has(w));
  }

  _extractAction(words) {
    const joined = words.join(' ');
    // Check multi-word commands first (longest match wins)
    const multiWord = [
      'stop scrolling','scroll to top','scroll to bottom',
      'scroll up','scroll down',
      'go back','go forward',
      'reload page','open new tab','close tab',
      'zoom in','zoom out','reset zoom',
      'double click','right click',
      'next item','previous item',
      'pause control','resume control',
      'reset cursor','clear selection','exit mode',
    ];
    for (const phrase of multiWord) {
      if (joined.includes(phrase)) return ACTION_COMMANDS[phrase];
    }
    // Single-word — exclude nav/mode words that need an element target
    const NAV_WORDS = new Set(['home','demo','architecture','docs','studio','calibrate','gaze','mouse','start','restart','camera']);
    for (const w of words) {
      if (ACTION_COMMANDS[w] &&
          !ACTION_COMMANDS[w].startsWith('nav-') &&
          !ACTION_COMMANDS[w].startsWith('mode-') &&
          !NAV_WORDS.has(w)) {
        return ACTION_COMMANDS[w];
      }
    }
    return null;
  }

  _extractEmbeddedAction(words) {
    // FIX VOICE-2: Skip nav/mode/camera words here — they are not action verbs.
    // "click home" should NOT set action='click' with remaining=[] falling to fusion;
    // it should fall through so path-3 named-element match finds the nav button.
    const skipAsVerb = new Set(['home','demo','architecture','docs','studio',
                                 'calibrate','gaze','mouse','start','restart','camera']);
    for (let i = 0; i < words.length; i++) {
      const w = words[i];
      if (ACTION_COMMANDS[w] && !skipAsVerb.has(w)) {
        return {
          action: ACTION_COMMANDS[w],
          remaining: [...words.slice(0, i), ...words.slice(i + 1)],
        };
      }
    }
    return null;
  }

  /* ── Intent Fusion: apply command to gaze target ── */
  _handleIntentFusion(action, confidence, rawText) {
    // Read gaze cursor position from app (read-only, no modification)
    const app = window.app;
    const gx = app?._lastScreenX ?? (window.innerWidth  / 2);
    const gy = app?._lastScreenY ?? (window.innerHeight / 2);

    const nearest = this.navList.findNearestToPoint(gx, gy, 300);

    if (!nearest) {
      this._setStatus('No target at gaze point', '#f59e0b');
      this._setTranscript(`⚠ No element near gaze for "${rawText}"`);
      setTimeout(() => { if (this.enabled) this._setStatus('Listening…', '#00d4ff'); }, 2000);
      return;
    }

    const { entry } = nearest;
    this._setStatus(`Intent: "${rawText}" → ${entry.text}`, '#00d4ff');
    this._executeOnElement(entry, action, rawText);
  }

  /* ── Element Execution ── */
  _executeOnElement(entry, action, rawText) {
    // Highlight
    this._highlightElement(entry.el);
    this._setStatus(`▶ "${rawText}" → ${entry.text}`, '#00ff88');

    // Confirm delay then act
    clearTimeout(this._confirmTimer);
    this._confirmTimer = setTimeout(() => {
      this._performAction(entry, action);
    }, 280);
  }

  _performAction(entry, action) {
    const el = entry.el;

    switch (action) {
      case 'click':
      case 'select':
      case 'open':
      case 'submit': {
        // Dispatch real click — same as manual click
        el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
        this._log(`Voice activated: ${entry.text} (${action})`);
        break;
      }
      case 'scrollUp': {
        const container = this._findScrollable(el) || document.documentElement;
        container.scrollBy({ top: -200, behavior: 'smooth' });
        this._log('Voice: Scroll Up');
        break;
      }
      case 'scrollDown':
      case 'scroll': {
        const container = this._findScrollable(el) || document.documentElement;
        container.scrollBy({ top: 200, behavior: 'smooth' });
        this._log('Voice: Scroll Down');
        break;
      }
      case 'play':
      case 'pause': {
        const media = document.querySelector('video, audio');
        if (media) {
          action === 'play' ? media.play() : media.pause();
          this._log(`Voice: ${action}`);
        }
        break;
      }
      case 'navBack':
      case 'back':
      case 'cancel': {
        const cancelBtn = document.getElementById('cancel-calib-btn');
        if (cancelBtn && getComputedStyle(cancelBtn.closest('.calibration-overlay') || cancelBtn).display !== 'none') {
          cancelBtn.click();
        } else {
          history.back();
        }
        this._log('Voice: Back');
        break;
      }
      case 'navForward': {
        history.forward();
        this._log('Voice: Forward');
        break;
      }
      case 'reloadPage': {
        this._log('Voice: Reload');
        setTimeout(() => location.reload(), 300);
        break;
      }
      case 'newTab': {
        window.open('', '_blank');
        this._log('Voice: New Tab');
        break;
      }
      case 'closeTab': {
        this._log('Voice: Close Tab');
        setTimeout(() => window.close(), 300);
        break;
      }
      case 'scrollTop': {
        window.scrollTo({ top: 0, behavior: 'smooth' });
        this._log('Voice: Scroll to Top');
        break;
      }
      case 'scrollBottom': {
        window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
        this._log('Voice: Scroll to Bottom');
        break;
      }
      case 'stopScrolling': {
        // Clear any pending scroll by triggering a zero-scroll
        window.scrollBy({ top: 0, behavior: 'instant' });
        this._log('Voice: Stop Scrolling');
        break;
      }
      case 'zoomIn': {
        const cur = parseFloat(document.body.style.zoom || '1');
        document.body.style.zoom = Math.min(cur + 0.15, 3.0);
        this._log('Voice: Zoom In');
        break;
      }
      case 'zoomOut': {
        const curZ = parseFloat(document.body.style.zoom || '1');
        document.body.style.zoom = Math.max(curZ - 0.15, 0.5);
        this._log('Voice: Zoom Out');
        break;
      }
      case 'resetZoom': {
        document.body.style.zoom = '1';
        this._log('Voice: Reset Zoom');
        break;
      }
      case 'dblclick': {
        if (el) el.dispatchEvent(new MouseEvent('dblclick', { bubbles: true, cancelable: true }));
        this._log('Voice: Double Click');
        break;
      }
      case 'rightClick': {
        if (el) el.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
        this._log('Voice: Right Click');
        break;
      }
      case 'focusNext': {
        const focusables = [...document.querySelectorAll(
          'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])'
        )].filter(e => e.offsetParent !== null);
        const idx = focusables.indexOf(document.activeElement);
        const next = focusables[idx + 1] || focusables[0];
        if (next) { next.focus(); this._log('Voice: Next Item'); }
        break;
      }
      case 'focusPrev': {
        const fps = [...document.querySelectorAll(
          'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])'
        )].filter(e => e.offsetParent !== null);
        const pi = fps.indexOf(document.activeElement);
        const prev = fps[pi - 1] || fps[fps.length - 1];
        if (prev) { prev.focus(); this._log('Voice: Previous Item'); }
        break;
      }
      // FIX VOICE-4: Discovery actions
      case 'showClickable': {
        this._showClickableOverlays();
        this._log('Voice: Show Clickable Items');
        break;
      }
      case 'hideClickable': {
        this._hideClickableOverlays();
        this._log('Voice: Hide Clickable Items');
        break;
      }
      case 'openSettings': {
        const settingsBtn = document.querySelector('[data-id="btn-settings"], #btn-settings, [aria-label*="setting" i], [data-label*="setting" i]');
        if (settingsBtn) {
          settingsBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
          this._log('Voice: Open Settings');
        } else {
          this._showToast('Open Settings', 'No settings button found', 'warn');
        }
        break;
      }
      case 'selectAll': {
        try { document.execCommand('selectAll'); } catch (_) {}
        this._log('Voice: Select All');
        break;
      }
      case 'copyText': {
        try { document.execCommand('copy'); } catch (_) {}
        this._log('Voice: Copy');
        break;
      }
      case 'pasteText': {
        try { document.execCommand('paste'); } catch (_) {}
        this._log('Voice: Paste');
        break;
      }
      case 'cutText': {
        try { document.execCommand('cut'); } catch (_) {}
        this._log('Voice: Cut');
        break;
      }
      case 'focusSearch': {
        const searchEl = document.querySelector('input[type="search"], input[type="text"], input:not([type]), [role="searchbox"]');
        if (searchEl) {
          searchEl.focus();
          this._highlightElement(searchEl);
          setTimeout(() => this._unhighlightElement(searchEl), 1200);
          this._log('Voice: Focus Search');
        } else {
          this._showToast('Focus Search', 'No search input found', 'warn');
        }
        break;
      }
      case 'focusOn': {
        // Handled upstream in _processUtterance — no-op here
        break;
      }
      case 'pauseVoice': {
        this._paused = true;
        this._setStatus('⏸ Voice Paused', '#f59e0b');
        this._setTranscript('Voice paused — say "Resume Voice" to re-enable');
        this._showToast('Voice Paused', 'Say "Resume Voice" to re-enable', 'info');
        this._log('Voice: Pause Voice');
        break;
      }
      case 'resumeVoice': {
        this._paused = false;
        this._setStatus('Listening…', '#00d4ff');
        this._showToast('Voice Resumed', 'Voice navigation active', 'success');
        this._log('Voice: Resume Voice');
        break;
      }
      /* ── Control Recovery ── */
      case 'pauseControl': {
        // Stop eye-tracking — _stopCamera already resets mode to 'mouse' internally
        if (window.app?.cameraOn) {
          window.app._stopCamera();         // resets mode to 'mouse' + starts sim
          this._log('Voice: Pause Control — eye tracking paused');
          this._showToast('Pause Control', 'Eye tracking paused', 'info');
        } else {
          this._showToast('Pause Control', 'Eye tracking already off', 'info');
        }
        break;
      }
      case 'resumeControl': {
        // Restart eye-tracking camera then immediately switch to gaze mode
        // so the simulation is stopped and the eye cursor takes back control.
        if (!window.app?.cameraOn) {
          this._log('Voice: Resume Control — restarting eye tracking');
          this._showToast('Resume Control', 'Restarting eye tracking…', 'success');
          const app = window.app;
          app._startCamera().then(() => {
            // _startCamera sets mode back to 'mouse' on failure and leaves it
            // unchanged on success — we must explicitly switch to gaze mode so
            // the simulation stops and the eye cursor takes over.
            if (app.cameraOn) {
              app._setMode('gaze');
              this._log('Voice: Resume Control — gaze mode restored');
            } else {
              this._showToast('Resume Control', 'Camera failed — still in mouse mode', 'warn');
            }
          }).catch(() => {
            this._showToast('Resume Control', 'Camera error', 'error');
          });
        } else {
          // Camera already on but mode may be wrong — restore gaze mode
          window.app._setMode('gaze');
          this._showToast('Resume Control', 'Gaze control restored', 'success');
        }
        break;
      }
      case 'stopControl': {
        // Hard stop — same as pause control but labelled 'stop'
        if (window.app?.cameraOn) {
          window.app._stopCamera();
          this._log('Voice: Stop — eye tracking stopped');
          this._showToast('Stop', 'Eye tracking stopped', 'info');
        }
        break;
      }
      case 'resetCursor': {
        // Snap gaze cursor back to screen centre
        // FIX VOICE-3: The live cursor is #global-gaze-cursor, not #gaze-cursor
        // (#gaze-cursor is a static element inside the camera preview feed)
        if (window.app) {
          window.app._lastScreenX = window.innerWidth  / 2;
          window.app._lastScreenY = window.innerHeight / 2;
          const cursorEl = document.getElementById('global-gaze-cursor');
          if (cursorEl) {
            cursorEl.style.left = (window.innerWidth  / 2) + 'px';
            cursorEl.style.top  = (window.innerHeight / 2) + 'px';
          }
        }
        this._log('Voice: Reset Cursor');
        this._showToast('Reset Cursor', 'Gaze cursor centred', 'info');
        break;
      }
      case 'clearSelection': {
        window.getSelection()?.removeAllRanges();
        if (document.activeElement && document.activeElement !== document.body) {
          document.activeElement.blur();
        }
        this._log('Voice: Clear Selection');
        break;
      }
      case 'exitMode': {
        // Return to mouse simulation mode
        if (window.app) {
          window.app.mode = 'mouse';
          document.querySelectorAll('.mode-tab').forEach(t =>
            t.classList.toggle('active', t.dataset.mode === 'mouse'));
        }
        this._log('Voice: Exit Mode');
        this._showToast('Exit Mode', 'Returned to mouse mode', 'info');
        break;
      }
      default: {
        // Treat special action IDs as direct element IDs (nav-*, mode-*)
        const directEl = document.getElementById(action) || document.querySelector(`[data-id="${action}"]`);
        if (directEl) {
          directEl.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
          this._log(`Voice direct: ${action}`);
        }
      }
    }

    // Brief "executed" badge
    setTimeout(() => {
      if (this.enabled) this._setStatus('Listening…', '#00d4ff');
    }, 1500);

    // Remove highlight after action
    setTimeout(() => this._unhighlightElement(el), 1200);
  }

  /* ── Show / Hide Clickable Overlays (FIX VOICE-4) ── */
  _showClickableOverlays() {
    this._hideClickableOverlays(); // clear any existing

    // Re-scan to get the freshest list
    const count = this.navList.scan();

    this.navList.elements.forEach((entry, idx) => {
      const rect = entry.rect();
      if (rect.width === 0 && rect.height === 0) return;

      // Numbered badge overlay
      const badge = document.createElement('div');
      badge.className = 'vn-clickable-badge';
      badge.dataset.vnBadge = '1';
      badge.textContent = String(idx + 1);
      Object.assign(badge.style, {
        position: 'fixed',
        left:   (rect.left + rect.width  / 2 - 10) + 'px',
        top:    (rect.top  + rect.height / 2 - 10) + 'px',
        width:  '20px',
        height: '20px',
        lineHeight: '20px',
        textAlign: 'center',
        borderRadius: '50%',
        background: 'rgba(0,212,255,0.9)',
        color: '#000',
        fontSize: '11px',
        fontWeight: 'bold',
        zIndex: '999999',
        pointerEvents: 'none',
        boxShadow: '0 0 6px 2px rgba(0,212,255,0.5)',
      });
      document.body.appendChild(badge);
      this._clickableOverlays.push(badge);

      // Ring highlight on the element itself
      entry.el._vnRingOrig = entry.el.style.outline;
      entry.el.style.outline = '2px solid rgba(0,212,255,0.7)';
    });

    this._setStatus(`✅ ${count} clickable items highlighted`, '#00ff88');
    this._setTranscript(`${count} clickable items — say a name or "hide clickable"`);
    this._showToast('Show Clickable', `${count} interactive elements highlighted`, 'success');

    // Auto-clear after 8 seconds
    setTimeout(() => this._hideClickableOverlays(), 8000);
  }

  _hideClickableOverlays() {
    this._clickableOverlays.forEach(el => el.remove());
    this._clickableOverlays = [];
    // Restore element outlines
    this.navList.elements.forEach(entry => {
      if ('_vnRingOrig' in entry.el) {
        entry.el.style.outline = entry.el._vnRingOrig;
        delete entry.el._vnRingOrig;
      }
    });
    if (this.enabled && !this._paused) {
      this._setStatus('Listening…', '#00d4ff');
    }
  }

  /* ── Focus On <name> handler (FIX VOICE-4) ── */
  _handleFocusOn(nameStr) {
    const words = nameStr.toLowerCase().replace(/[^a-z0-9\s]/g, '').split(/\s+/).filter(Boolean);
    const match = this.navList.findBest(words);
    if (!match) {
      this._setStatus(`No match for "${nameStr}"`, '#f59e0b');
      this._setTranscript(`⚠ Could not find element "${nameStr}"`);
      setTimeout(() => { if (this.enabled) this._setStatus('Listening…', '#00d4ff'); }, 2000);
      return;
    }
    const el = match.entry.el;
    this._highlightElement(el);
    el.focus?.();
    this._setStatus(`Focused: ${match.entry.text}`, '#00ff88');
    this._setTranscript(`Focused on "${match.entry.text}"`);
    this._log(`Voice: Focus On — ${match.entry.text}`);
    setTimeout(() => this._unhighlightElement(el), 2000);
    setTimeout(() => { if (this.enabled) this._setStatus('Listening…', '#00d4ff'); }, 2200);
  }

  _findScrollable(startEl) {
    let el = startEl?.parentElement;
    while (el && el !== document.body) {
      const { overflow, overflowY } = getComputedStyle(el);
      if (/auto|scroll/.test(overflow + overflowY) && el.scrollHeight > el.clientHeight) {
        return el;
      }
      el = el.parentElement;
    }
    return null;
  }

  /* ── Visual Confirmation ── */
  _highlightElement(el) {
    this._unhighlightElement(el);
    el._vnOrigOutline    = el.style.outline;
    el._vnOrigBoxShadow  = el.style.boxShadow;
    el._vnOrigTransition = el.style.transition;
    el.style.transition  = 'box-shadow 0.1s, outline 0.1s';
    el.style.outline     = '2px solid #00d4ff';
    el.style.boxShadow   = '0 0 12px 4px rgba(0,212,255,0.6)';
  }

  _unhighlightElement(el) {
    if (!el) return;
    el.style.outline    = el._vnOrigOutline    ?? '';
    el.style.boxShadow  = el._vnOrigBoxShadow  ?? '';
    el.style.transition = el._vnOrigTransition ?? '';
  }

  /* ── UI Helpers ── */
  _setTranscript(text, isInterim = false) {
    if (!this._transcriptEl) return;
    this._transcriptEl.textContent = text;
    this._transcriptEl.style.color = isInterim ? '#7c4dff' : '#e2e8f0';
    this._transcriptEl.style.fontStyle = isInterim ? 'italic' : 'normal';
  }

  _setStatus(text, color = '#546e7a') {
    if (!this._badgeEl) return;
    this._badgeEl.textContent = text;
    this._badgeEl.style.color = color;
  }

  _log(msg) {
    console.log(`[VoiceNav] ${msg}`);
    window.app?.log?.add?.(msg, 'info');
    window.app?.toast?.show?.('Voice', msg, 'success', 'fas fa-microphone', 2000);
  }

  _showToast(title, msg, type = 'info') {
    window.app?.toast?.show?.(title, msg, type, 'fas fa-microphone', 3000);
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   BOOTSTRAP — wait for AccessEyeApp, then attach
───────────────────────────────────────────────────────────────────────── */
(function bootstrap() {
  const controller = new VoiceNavigationController();

  const attach = () => {
    if (!window.app) {
      setTimeout(attach, 200);
      return;
    }

    // Expose globally
    window.voiceNav = controller;
    window.AccessEye = window.AccessEye || {};
    window.AccessEye.voiceNav = controller;

    // Re-scan on every page navigation
    const origNavigateTo = window.app._navigateTo?.bind(window.app);
    if (origNavigateTo) {
      window.app._navigateTo = function(page) {
        origNavigateTo(page);
        if (controller.enabled) {
          setTimeout(() => controller.navList.scan(), 400);
        }
      };
    }

    console.log(`%c Voice Navigation + Intent Fusion ✅ v${VN_VERSION}`,
                'color:#00d4ff;font-weight:bold;font-size:12px;');
  };

  setTimeout(attach, 600);
})();
