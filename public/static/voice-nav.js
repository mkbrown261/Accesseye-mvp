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
const VN_VERSION = '2.0.0';

// Words to strip before matching
const FILLER_WORDS = new Set([
  'please','go','to','can','you','would','the','a','an','and','or','now',
  'just','hey','um','uh','like','that','this','it','on','at','in','with'
]);

// Voice commands → action keys
const ACTION_COMMANDS = {
  // ── existing ──────────────────────────────────────────────────
  click    : 'click',
  press    : 'click',
  tap      : 'click',
  open     : 'open',
  select   : 'select',
  choose   : 'select',
  scroll   : 'scroll',
  'scroll up'      : 'scrollUp',
  'scroll down'    : 'scrollDown',
  'stop scrolling' : 'stopScrolling',
  'scroll top'     : 'scrollTop',
  'scroll bottom'  : 'scrollBottom',
  'scroll to top'  : 'scrollTop',
  'scroll to bottom': 'scrollBottom',
  play     : 'play',
  pause    : 'pause',
  stop     : 'stop',
  submit   : 'submit',
  send     : 'submit',
  back     : 'back',
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

  // ── new: navigation ───────────────────────────────────────────
  forward         : 'navForward',
  'go forward'    : 'navForward',
  reload          : 'reloadPage',
  refresh         : 'reloadPage',
  'reload page'   : 'reloadPage',
  'refresh page'  : 'reloadPage',
  'go home'       : 'nav-home',
  'open new tab'  : 'newTab',
  'new tab'       : 'newTab',
  'close tab'     : 'closeTab',

  // ── new: click variants ───────────────────────────────────────
  'double click'  : 'dblclick',
  'right click'   : 'rightClick',
  'context menu'  : 'rightClick',

  // ── new: focus traversal ──────────────────────────────────────
  'next item'     : 'focusNext',
  'previous item' : 'focusPrev',
  next            : 'focusNext',
  previous        : 'focusPrev',
  prev            : 'focusPrev',

  // ── new: zoom ─────────────────────────────────────────────────
  'zoom in'       : 'zoomIn',
  'zoom out'      : 'zoomOut',
  'reset zoom'    : 'zoomReset',
  zoom            : 'zoomIn',

  // ── new: settings ─────────────────────────────────────────────
  'open settings' : 'openSettings',
  settings        : 'openSettings',

  // ── new: text editing ─────────────────────────────────────────
  search          : 'focusSearch',
  'select all'    : 'selectAll',
  copy            : 'copyText',
  paste           : 'pasteText',
  cut             : 'cutText',
  'new folder'    : 'newFolder',
  rename          : 'renameItem',
  delete          : 'deleteItem',
  'open file'     : 'openFile',

  // ── new: discovery ────────────────────────────────────────────
  'show clickable'       : 'showClickable',
  'show clickable items' : 'showClickable',
  'hide clickable'       : 'hideClickable',
  'hide clickable items' : 'hideClickable',
  'what can i click'     : 'showClickable',
  'show interactive'     : 'showClickable',
  'focus on'             : 'focusOn',

  // ── new: control recovery ─────────────────────────────────────
  'pause control'  : 'pauseControl',
  'pause voice'    : 'pauseControl',
  'resume control' : 'resumeControl',
  'resume voice'   : 'resumeControl',
  resume           : 'resumeControl',
  'reset cursor'   : 'resetCursor',
  'clear selection': 'clearSelection',
  'exit mode'      : 'exitMode',
  exit             : 'exitMode',
};

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
    this._paused      = false;          // NEW: pause/resume control
    this._scrollTimer = null;           // NEW: for stop-scrolling
    this._clickableOverlays = [];       // NEW: discovery overlays
    this.recognition  = null;
    this.navList      = new NavigationElementList();
    this._lastGazeTarget = null;
    this._confirmTimer   = null;
    this._scanScheduled  = false;

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
    const lower = text.toLowerCase().replace(/[^a-z0-9\s]/g, '').trim();

    // Always allow resume even when paused
    if (this._paused) {
      if (lower.includes('resume') || lower.includes('resume control') || lower.includes('resume voice')) {
        this._performAction(null, 'resumeControl');
        this._setTranscript('▶ Resumed voice control');
      } else {
        this._setTranscript(`⏸ Paused — say "Resume Control" to re-enable`);
      }
      return;
    }

    const words = this._tokenise(text);
    if (!words.length) return;

    console.log(`[VoiceNav] heard: "${text}" (words: [${words.join(', ')}])`);

    // 0. Check multi-word ACTION_COMMANDS against the raw lowercase text first
    const multiAction = this._extractMultiWordAction(lower);
    if (multiAction) {
      // Special case: "focus on [name]" — extract what follows
      if (multiAction === 'focusOn') {
        const afterFocus = lower.replace(/focus on\s*/i, '').trim();
        if (afterFocus) {
          this._handleFocusOn(afterFocus);
        } else {
          this._setStatus('Focus on — say element name', '#f59e0b');
        }
        return;
      }
      this._handleIntentFusion(multiAction, confidence, text);
      return;
    }

    // 1. Check for pure action commands that apply to gaze target (Intent Fusion)
    const action = this._extractAction(words);
    if (action) {
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

  /** Check raw lowercase text against all multi-word ACTION_COMMANDS keys */
  _extractMultiWordAction(lower) {
    // Sort by length descending so longest match wins
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
    // Check multi-word commands first
    if (joined.includes('scroll up'))   return 'scrollUp';
    if (joined.includes('scroll down')) return 'scrollDown';
    // Single-word
    for (const w of words) {
      if (ACTION_COMMANDS[w] && ACTION_COMMANDS[w].startsWith('nav-') === false &&
          ACTION_COMMANDS[w].startsWith('mode-') === false &&
          !['home','demo','architecture','docs','studio','calibrate','gaze','mouse','start','restart','camera'].includes(w)) {
        return ACTION_COMMANDS[w];
      }
    }
    return null;
  }

  _extractEmbeddedAction(words) {
    for (let i = 0; i < words.length; i++) {
      const w = words[i];
      if (ACTION_COMMANDS[w]) {
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
    // Actions that don't need a DOM element target
    const noTargetActions = new Set([
      'scrollUp','scrollDown','scroll','stopScrolling','scrollTop','scrollBottom',
      'navForward','reloadPage','newTab','closeTab',
      'zoomIn','zoomOut','zoomReset',
      'openSettings','focusSearch','selectAll','copyText','pasteText','cutText',
      'newFolder','renameItem','deleteItem','openFile',
      'showClickable','hideClickable',
      'pauseControl','resumeControl','resetCursor','clearSelection','exitMode',
      'stop','focusNext','focusPrev',
    ]);

    if (noTargetActions.has(action)) {
      this._setStatus(`▶ ${rawText}`, '#00d4ff');
      this._performAction(null, action);
      return;
    }

    // Read gaze cursor position from app (read-only, no modification)
    const app = window.app;
    const gx = app?._lastScreenX ?? (window.innerWidth  / 2);
    const gy = app?._lastScreenY ?? (window.innerHeight / 2);

    const nearest = this.navList.findNearestToPoint(gx, gy, 300);

    if (!nearest) {
      this._setStatus('No target at gaze point', '#f59e0b');
      this._setTranscript(`⚠ No element near gaze for "${rawText}"`);
      setTimeout(() => { if (this.enabled && !this._paused) this._setStatus('Listening…', '#00d4ff'); }, 2000);
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
    const el = entry?.el;

    switch (action) {
      // ── existing ────────────────────────────────────────────────
      case 'click':
      case 'select':
      case 'open':
      case 'submit': {
        if (!el) break;
        el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
        this._log(`Voice activated: ${entry.text} (${action})`);
        break;
      }
      case 'scrollUp': {
        const cu = el ? (this._findScrollable(el) || document.documentElement) : document.documentElement;
        this._activeScrollEl = cu;
        cu.scrollBy({ top: -200, behavior: 'smooth' });
        this._log('Voice: Scroll Up');
        break;
      }
      case 'scrollDown':
      case 'scroll': {
        const cd = el ? (this._findScrollable(el) || document.documentElement) : document.documentElement;
        this._activeScrollEl = cd;
        cd.scrollBy({ top: 200, behavior: 'smooth' });
        this._log('Voice: Scroll Down');
        break;
      }
      case 'play':
      case 'pause': {
        const media = document.querySelector('video, audio');
        if (media) { action === 'play' ? media.play() : media.pause(); }
        this._log(`Voice: ${action}`);
        break;
      }
      case 'stop': {
        clearTimeout(this._confirmTimer);
        this._log('Voice: Stop');
        break;
      }
      case 'back':
      case 'cancel': {
        const cancelBtn = document.getElementById('cancel-calib-btn');
        if (cancelBtn && getComputedStyle(cancelBtn.closest('.calibration-overlay') || cancelBtn).display !== 'none') {
          cancelBtn.click();
        } else {
          history.back();
        }
        this._log('Voice: Back/Cancel');
        break;
      }

      // ── new navigation ──────────────────────────────────────────
      case 'navForward':
        history.forward();
        this._log('Voice: Go Forward');
        break;

      case 'reloadPage':
        this._log('Voice: Reload Page');
        setTimeout(() => location.reload(), 300);
        break;

      case 'newTab':
        window.open('about:blank', '_blank');
        this._log('Voice: Open New Tab');
        break;

      case 'closeTab':
        this._log('Voice: Close Tab (browser may block)');
        window.close();
        break;

      // ── new click variants ───────────────────────────────────────
      case 'dblclick': {
        if (!el) break;
        el.dispatchEvent(new MouseEvent('dblclick', { bubbles: true, cancelable: true }));
        this._log(`Voice: Double Click → ${entry.text}`);
        break;
      }
      case 'rightClick': {
        if (!el) break;
        el.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
        this._log(`Voice: Right Click → ${entry.text}`);
        break;
      }

      // ── new scroll variants ──────────────────────────────────────
      case 'stopScrolling':
        clearInterval(this._scrollTimer);
        this._scrollTimer = null;
        this._log('Voice: Stop Scrolling');
        break;

      case 'scrollTop':
        (this._activeScrollEl || document.documentElement).scrollTo({ top: 0, behavior: 'smooth' });
        this._log('Voice: Scroll To Top');
        break;

      case 'scrollBottom': {
        const sc = this._activeScrollEl || document.documentElement;
        sc.scrollTo({ top: sc.scrollHeight, behavior: 'smooth' });
        this._log('Voice: Scroll To Bottom');
        break;
      }

      // ── focus traversal ──────────────────────────────────────────
      case 'focusNext': {
        const focusable = Array.from(document.querySelectorAll(
          'button:not([disabled]),a[href],input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])'
        )).filter(e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; });
        const cur = document.activeElement;
        const idx = focusable.indexOf(cur);
        const next = focusable[idx + 1] || focusable[0];
        if (next) { next.focus(); this._highlightElement(next); setTimeout(() => this._unhighlightElement(next), 1000); }
        this._log('Voice: Next Item');
        break;
      }
      case 'focusPrev': {
        const focusable2 = Array.from(document.querySelectorAll(
          'button:not([disabled]),a[href],input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])'
        )).filter(e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; });
        const cur2 = document.activeElement;
        const idx2 = focusable2.indexOf(cur2);
        const prev = focusable2[idx2 - 1] || focusable2[focusable2.length - 1];
        if (prev) { prev.focus(); this._highlightElement(prev); setTimeout(() => this._unhighlightElement(prev), 1000); }
        this._log('Voice: Previous Item');
        break;
      }

      // ── zoom ─────────────────────────────────────────────────────
      case 'zoomIn': {
        const cur3 = parseFloat(document.body.style.zoom || '1');
        document.body.style.zoom = Math.min(cur3 + 0.1, 3).toFixed(1);
        this._log(`Voice: Zoom In → ${document.body.style.zoom}`);
        break;
      }
      case 'zoomOut': {
        const cur4 = parseFloat(document.body.style.zoom || '1');
        document.body.style.zoom = Math.max(cur4 - 0.1, 0.5).toFixed(1);
        this._log(`Voice: Zoom Out → ${document.body.style.zoom}`);
        break;
      }
      case 'zoomReset':
        document.body.style.zoom = '1';
        this._log('Voice: Reset Zoom');
        break;

      // ── settings ─────────────────────────────────────────────────
      case 'openSettings': {
        const settingsBtn = document.querySelector('[data-page="settings"], [data-id="settings"], #settings-btn');
        if (settingsBtn) settingsBtn.click();
        else this._showToast('Settings', 'No settings panel found', 'warn');
        this._log('Voice: Open Settings');
        break;
      }

      // ── text editing ─────────────────────────────────────────────
      case 'focusSearch': {
        const searchEl = document.querySelector('input[type="search"],input[type="text"],[role="searchbox"],#search-input');
        if (searchEl) { searchEl.focus(); this._highlightElement(searchEl); setTimeout(() => this._unhighlightElement(searchEl), 1500); }
        this._log('Voice: Focus Search');
        break;
      }
      case 'selectAll':
        document.execCommand('selectAll');
        this._log('Voice: Select All');
        break;

      case 'copyText':
        document.execCommand('copy');
        this._log('Voice: Copy');
        break;

      case 'pasteText':
        document.execCommand('paste');
        this._log('Voice: Paste');
        break;

      case 'cutText':
        document.execCommand('cut');
        this._log('Voice: Cut');
        break;

      case 'newFolder':
      case 'renameItem':
      case 'deleteItem':
      case 'openFile':
        this._showToast('Voice Command', `"${action}" requires a file manager context`, 'info');
        this._log(`Voice: ${action} (no-op in browser context)`);
        break;

      // ── discovery ────────────────────────────────────────────────
      case 'showClickable':
        this._showClickableOverlays();
        break;

      case 'hideClickable':
        this._hideClickableOverlays();
        this._log('Voice: Hide Clickable Items');
        break;

      case 'focusOn':
        // handled upstream in _processUtterance
        break;

      // ── control recovery ─────────────────────────────────────────
      case 'pauseControl':
        this._paused = true;
        this._setStatus('⏸ Paused', '#f59e0b');
        this._setTranscript('Voice control paused — say "Resume Control"');
        this._showToast('Voice Paused', 'Say "Resume Control" to re-enable', 'info');
        this._log('Voice: Pause Control');
        break;

      case 'resumeControl':
        this._paused = false;
        this._setStatus('Listening…', '#00d4ff');
        this._setTranscript('Voice control resumed');
        this._showToast('Voice Resumed', 'Listening for commands', 'success');
        this._log('Voice: Resume Control');
        break;

      case 'resetCursor': {
        // Move gaze cursor to screen center via app API (read-write on cursor only)
        const cx = window.innerWidth / 2;
        const cy = window.innerHeight / 2;
        const gazeEl = document.getElementById('gaze-cursor');
        if (gazeEl) {
          gazeEl.style.left = cx + 'px';
          gazeEl.style.top  = cy + 'px';
        }
        this._log('Voice: Reset Cursor to center');
        break;
      }

      case 'clearSelection':
        window.getSelection?.()?.removeAllRanges();
        document.querySelectorAll('.gaze-active,.snap-active').forEach(e => e.classList.remove('gaze-active','snap-active'));
        this._log('Voice: Clear Selection');
        break;

      case 'exitMode': {
        // Close any visible overlay (calibration, modals)
        const calibOverlay = document.getElementById('calibration-overlay');
        if (calibOverlay && calibOverlay.style.display !== 'none') {
          document.getElementById('cancel-calib-btn')?.click();
        }
        // Remove any highlighted elements
        this._hideClickableOverlays();
        window.getSelection?.()?.removeAllRanges();
        this._log('Voice: Exit Mode');
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
      if (this.enabled && !this._paused) this._setStatus('Listening…', '#00d4ff');
    }, 1500);

    // Remove highlight after action
    if (el) setTimeout(() => this._unhighlightElement(el), 1200);
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

  /* ── Discovery: Focus On [name] ── */
  _handleFocusOn(nameStr) {
    this.navList.scan();
    const words = nameStr.split(/\s+/).filter(Boolean);
    const match = this.navList.findBest(words);
    if (match) {
      this._highlightElement(match.entry.el);
      match.entry.el.focus?.();
      this._setStatus(`Focus → ${match.entry.text}`, '#00ff88');
      this._setTranscript(`Focused: ${match.entry.text}`);
      this._log(`Focus on: ${match.entry.text}`);
      setTimeout(() => this._unhighlightElement(match.entry.el), 2000);
    } else {
      this._setStatus(`No element matching "${nameStr}"`, '#f59e0b');
      this._setTranscript(`⚠ No element found: "${nameStr}"`);
    }
    setTimeout(() => { if (this.enabled && !this._paused) this._setStatus('Listening…', '#00d4ff'); }, 2500);
  }

  /* ── Discovery: Show / Hide Clickable Items ── */
  _showClickableOverlays() {
    this._hideClickableOverlays();
    this.navList.scan();
    this.navList.elements.forEach(entry => {
      const r = entry.el.getBoundingClientRect();
      if (r.width === 0) return;
      const badge = document.createElement('div');
      badge.className = 'vn-clickable-badge';
      badge.textContent = entry.text.slice(0, 18);
      badge.style.cssText = `
        position:fixed;
        left:${r.left + r.width / 2}px;
        top:${r.top - 2}px;
        transform:translate(-50%,-100%);
        background:rgba(0,212,255,0.85);
        color:#000;
        font-size:10px;
        font-weight:700;
        padding:2px 5px;
        border-radius:3px;
        pointer-events:none;
        z-index:99999;
        white-space:nowrap;
        max-width:120px;
        overflow:hidden;
        text-overflow:ellipsis;
      `;
      document.body.appendChild(badge);
      this._clickableOverlays.push(badge);
      // Also ring the element
      entry.el._vnRingOrig = entry.el.style.outline;
      entry.el.style.outline = '1px dashed rgba(0,212,255,0.6)';
    });
    this._setStatus(`${this._clickableOverlays.length} clickable items`, '#00d4ff');
    this._setTranscript(`Showing ${this._clickableOverlays.length} clickable items — say "Hide Clickable Items"`);
    this._log(`Showing ${this._clickableOverlays.length} clickable items`);
    // Auto-hide after 8 seconds
    setTimeout(() => this._hideClickableOverlays(), 8000);
  }

  _hideClickableOverlays() {
    this._clickableOverlays.forEach(b => b.remove());
    this._clickableOverlays = [];
    // Remove rings
    this.navList.elements.forEach(entry => {
      if (entry.el._vnRingOrig !== undefined) {
        entry.el.style.outline = entry.el._vnRingOrig;
        delete entry.el._vnRingOrig;
      }
    });
    if (this.enabled && !this._paused) this._setStatus('Listening…', '#00d4ff');
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

    console.log(`%c Voice Navigation + Intent Fusion ✅ v${VN_VERSION} — 40 commands`,
                'color:#00d4ff;font-weight:bold;font-size:12px;');
  };

  setTimeout(attach, 600);
})();
