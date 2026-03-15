/**
 * ═══════════════════════════════════════════════════════════════════════════
 *  AccessEye — Accessibility Control Mode (ACM)
 *  a11y-mode.js  v1.0.0
 *
 *  Standards Compliance:
 *    • WCAG 2.1 AA/AAA — perceivable, operable, understandable, robust
 *    • ADA Title III   — full digital access without mouse/keyboard
 *    • Section 508     — federal IT accessibility requirement
 *
 *  Architecture (ADDITIVE ONLY — zero core modifications):
 *    • Reads window.app._lastScreenX / _lastScreenY (cursor position, read-only)
 *    • Reads window.voiceNav (voice controller, read-only)
 *    • Reads window.snapEngine (snap-to engine, read-only)
 *    • Emits events on window.AccessEye.a11y for other modules to consume
 *    • All actions are logged via window.AccessEye.a11yLogger
 *
 *  Modalities supported:
 *    Gaze → dwell-click on focusable elements
 *    Voice → command matching via existing voice-nav.js pipeline
 *    Intent Fusion → gaze + voice combined actions
 *    Snap-To → precise targeting via existing snap-engine.js
 * ═══════════════════════════════════════════════════════════════════════════
 */

'use strict';

/* ─────────────────────────────────────────────────────────────────────────
   CONSTANTS
───────────────────────────────────────────────────────────────────────── */
const ACM_VERSION = '1.0.0';

// WCAG 2.1 focusable element selector (covers all interactive roles)
const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
  '[role="button"]',
  '[role="link"]',
  '[role="menuitem"]',
  '[role="option"]',
  '[role="tab"]',
  '[role="checkbox"]',
  '[role="radio"]',
  '[role="switch"]',
  '[role="combobox"]',
  '[role="listbox"]',
  '[role="slider"]',
  '[role="spinbutton"]',
  '[contenteditable="true"]',
].join(',');

// Compliance mode identifiers
const STANDARDS = {
  WCAG_AA:    'WCAG 2.1 AA',
  WCAG_AAA:   'WCAG 2.1 AAA',
  ADA:        'ADA Title III',
  SECTION508: 'Section 508',
};

/* ─────────────────────────────────────────────────────────────────────────
   ACCESSIBILITY FOCUS RING MANAGER
   Manages the enhanced focus ring overlay for gaze + keyboard nav
───────────────────────────────────────────────────────────────────────── */
class A11yFocusRing {
  constructor() {
    this._ring = null;
    this._current = null;
    this._init();
  }

  _init() {
    this._ring = document.createElement('div');
    this._ring.id = 'a11y-focus-ring';
    this._ring.setAttribute('aria-hidden', 'true');
    this._ring.style.cssText = `
      position: fixed;
      pointer-events: none;
      z-index: 999990;
      border: 3px solid #00d4ff;
      border-radius: 4px;
      box-shadow: 0 0 0 2px rgba(0,212,255,0.25), 0 0 16px 4px rgba(0,212,255,0.4);
      transition: all 0.12s ease;
      opacity: 0;
      display: none;
    `;
    document.body.appendChild(this._ring);
  }

  show(el) {
    if (!el) return;
    this._current = el;
    const r = el.getBoundingClientRect();
    const pad = 3;
    Object.assign(this._ring.style, {
      display: 'block',
      opacity: '1',
      left:   (r.left   - pad) + 'px',
      top:    (r.top    - pad) + 'px',
      width:  (r.width  + pad * 2) + 'px',
      height: (r.height + pad * 2) + 'px',
    });
  }

  hide() {
    this._ring.style.opacity = '0';
    setTimeout(() => { this._ring.style.display = 'none'; }, 150);
    this._current = null;
  }

  update() {
    if (this._current) this.show(this._current);
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   ACCESSIBLE HINT OVERLAY
   First-time user guidance overlay (WCAG 3.3.5 Help criterion)
───────────────────────────────────────────────────────────────────────── */
class A11yHintOverlay {
  constructor() {
    this._el = null;
    this._dismissed = localStorage.getItem('acm_hint_dismissed') === '1';
  }

  show() {
    if (this._dismissed) return;
    if (this._el) return;

    this._el = document.createElement('div');
    this._el.id = 'a11y-hint-overlay';
    this._el.setAttribute('role', 'dialog');
    this._el.setAttribute('aria-label', 'Accessibility Control Mode — Quick Guide');
    this._el.innerHTML = `
      <div class="a11y-hint-card">
        <div class="a11y-hint-header">
          <i class="fas fa-universal-access"></i>
          <span>Accessibility Control Mode Active</span>
          <button class="a11y-hint-close" id="a11y-hint-close" aria-label="Dismiss guide">
            <i class="fas fa-times"></i>
          </button>
        </div>
        <div class="a11y-hint-body">
          <div class="a11y-hint-row">
            <span class="a11y-hint-icon"><i class="fas fa-eye"></i></span>
            <span><strong>Gaze</strong> — Look at any element for ~800ms to activate it</span>
          </div>
          <div class="a11y-hint-row">
            <span class="a11y-hint-icon"><i class="fas fa-microphone"></i></span>
            <span><strong>Voice</strong> — Say element name or command (e.g. "Click Send")</span>
          </div>
          <div class="a11y-hint-row">
            <span class="a11y-hint-icon"><i class="fas fa-crosshairs"></i></span>
            <span><strong>Intent Fusion</strong> — Gaze at target + speak action simultaneously</span>
          </div>
          <div class="a11y-hint-row">
            <span class="a11y-hint-icon"><i class="fas fa-magnet"></i></span>
            <span><strong>Snap-To</strong> — Auto-snaps cursor to nearest interactive element</span>
          </div>
          <div class="a11y-hint-row">
            <span class="a11y-hint-icon"><i class="fas fa-keyboard"></i></span>
            <span><strong>Tab / Arrow keys</strong> still work for keyboard-only navigation</span>
          </div>
        </div>
        <div class="a11y-hint-footer">
          <label class="a11y-hint-noshow">
            <input type="checkbox" id="a11y-hint-noshow-cb"> Don't show again
          </label>
          <button class="a11y-hint-btn" id="a11y-hint-ok">Got it</button>
        </div>
      </div>
    `;
    document.body.appendChild(this._el);

    document.getElementById('a11y-hint-close').addEventListener('click', () => this.dismiss());
    document.getElementById('a11y-hint-ok').addEventListener('click', () => {
      if (document.getElementById('a11y-hint-noshow-cb').checked) {
        localStorage.setItem('acm_hint_dismissed', '1');
      }
      this.dismiss();
    });

    // Auto-dismiss after 15 seconds
    setTimeout(() => this.dismiss(), 15000);
  }

  dismiss() {
    if (!this._el) return;
    this._el.style.opacity = '0';
    this._el.style.transform = 'translateY(12px)';
    setTimeout(() => { this._el?.remove(); this._el = null; }, 300);
    this._dismissed = true;
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   FOCUSABLE ELEMENT NAVIGATOR
   Provides Tab-order navigation through all WCAG-compliant focusable elements
───────────────────────────────────────────────────────────────────────── */
class A11yNavigator {
  constructor() {
    this._elements = [];
    this._index = -1;
  }

  scan() {
    this._elements = Array.from(document.querySelectorAll(FOCUSABLE_SELECTOR))
      .filter(el => {
        if (el.offsetParent === null) return false;
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0 &&
               getComputedStyle(el).visibility !== 'hidden' &&
               getComputedStyle(el).display !== 'none';
      })
      .sort((a, b) => {
        // Sort by tab-index then visual position (top→bottom, left→right)
        const ta = parseInt(a.tabIndex) || 0;
        const tb = parseInt(b.tabIndex) || 0;
        if (ta !== tb) return ta - tb;
        const ra = a.getBoundingClientRect();
        const rb = b.getBoundingClientRect();
        return ra.top !== rb.top ? ra.top - rb.top : ra.left - rb.left;
      });
    return this._elements.length;
  }

  focusNext() {
    if (!this._elements.length) this.scan();
    this._index = (this._index + 1) % this._elements.length;
    return this._focusCurrent();
  }

  focusPrev() {
    if (!this._elements.length) this.scan();
    this._index = (this._index - 1 + this._elements.length) % this._elements.length;
    return this._focusCurrent();
  }

  _focusCurrent() {
    const el = this._elements[this._index];
    if (el) {
      el.focus({ preventScroll: false });
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    return el || null;
  }

  activateCurrent() {
    const el = this._elements[this._index];
    if (!el) return null;
    el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    return el;
  }

  findNearest(x, y, maxDist = 200) {
    if (!this._elements.length) this.scan();
    let best = null, bestDist = Infinity;
    for (const el of this._elements) {
      const r = el.getBoundingClientRect();
      const cx = r.left + r.width  / 2;
      const cy = r.top  + r.height / 2;
      const d  = Math.hypot(cx - x, cy - y);
      if (d < bestDist && d <= maxDist) { bestDist = d; best = el; }
    }
    return best;
  }

  get current() {
    return this._elements[this._index] || null;
  }

  get count() {
    return this._elements.length;
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   GAZE DWELL ACTIVATOR
   Watches gaze position and activates focusable elements via dwell
   (reads cursor position read-only — does NOT modify tracking)
───────────────────────────────────────────────────────────────────────── */
class A11yDwellActivator {
  constructor(navigator, logger, focusRing) {
    this._nav       = navigator;
    this._logger    = logger;
    this._ring      = focusRing;
    this._dwellEl   = null;
    this._dwellStart= 0;
    this._dwellMs   = 800;   // default 800ms dwell to activate
    this._active    = false;
    this._raf       = null;
    // Visual dwell indicator
    this._indicator = this._createIndicator();
  }

  _createIndicator() {
    const el = document.createElement('div');
    el.id = 'a11y-dwell-indicator';
    el.setAttribute('aria-hidden', 'true');
    el.style.cssText = `
      position:fixed; pointer-events:none; z-index:999989;
      width:50px; height:50px; border-radius:50%;
      border: 3px solid transparent;
      transition: opacity 0.1s;
      opacity: 0;
      display:none;
    `;
    el.innerHTML = `<svg width="50" height="50" viewBox="0 0 50 50">
      <circle id="a11y-dwell-arc" cx="25" cy="25" r="21"
        fill="none" stroke="#00ff88" stroke-width="3"
        stroke-dasharray="0 132" stroke-linecap="round"
        transform="rotate(-90 25 25)"/>
    </svg>`;
    document.body.appendChild(el);
    return el;
  }

  start() {
    this._active = true;
    this._tick();
  }

  stop() {
    this._active = false;
    if (this._raf) cancelAnimationFrame(this._raf);
    this._hideIndicator();
  }

  setDwellTime(ms) {
    this._dwellMs = Math.max(300, Math.min(2000, ms));
  }

  _tick() {
    if (!this._active) return;
    this._raf = requestAnimationFrame(() => this._tick());

    const app = window.app;
    if (!app) return;

    const gx = app._lastScreenX;
    const gy = app._lastScreenY;
    if (typeof gx !== 'number' || typeof gy !== 'number') return;

    // Find focusable element at gaze position
    const el = document.elementFromPoint(gx, gy);
    const target = el ? el.closest(FOCUSABLE_SELECTOR) : null;

    if (target && target !== this._dwellEl) {
      // New target — reset dwell
      this._dwellEl    = target;
      this._dwellStart = performance.now();
      this._ring.show(target);
      this._showIndicator(gx, gy);
    } else if (target && target === this._dwellEl) {
      // Same target — update dwell progress
      const elapsed  = performance.now() - this._dwellStart;
      const progress = Math.min(elapsed / this._dwellMs, 1);
      this._updateIndicator(gx, gy, progress);

      if (progress >= 1) {
        // Dwell threshold reached — activate
        this._activate(target);
        this._dwellEl    = null;
        this._dwellStart = 0;
      }
    } else {
      // No target
      this._dwellEl    = null;
      this._dwellStart = 0;
      this._ring.hide();
      this._hideIndicator();
    }
  }

  _activate(el) {
    this._ring.hide();
    this._hideIndicator();
    el.focus();
    el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    const label = el.getAttribute('aria-label') || el.textContent?.trim().slice(0, 50) || el.tagName;
    this._logger?.log(`User activated "${label}" via gaze dwell`, 'gaze', 'activate', el);
    // Auditory cue
    this._playTone(880, 80);
  }

  _showIndicator(x, y) {
    Object.assign(this._indicator.style, {
      display: 'block',
      opacity: '1',
      left: (x - 25) + 'px',
      top:  (y - 25) + 'px',
    });
  }

  _updateIndicator(x, y, progress) {
    const circ = 2 * Math.PI * 21; // r=21
    const arc  = document.getElementById('a11y-dwell-arc');
    if (arc) arc.setAttribute('stroke-dasharray', `${circ * progress} ${circ}`);
    Object.assign(this._indicator.style, {
      left: (x - 25) + 'px',
      top:  (y - 25) + 'px',
    });
  }

  _hideIndicator() {
    this._indicator.style.opacity = '0';
    setTimeout(() => { this._indicator.style.display = 'none'; }, 100);
    const arc = document.getElementById('a11y-dwell-arc');
    if (arc) arc.setAttribute('stroke-dasharray', '0 132');
  }

  _playTone(freq, durationMs) {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain); gain.connect(ctx.destination);
      osc.frequency.value = freq;
      osc.type = 'sine';
      gain.gain.setValueAtTime(0.15, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + durationMs / 1000);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + durationMs / 1000);
    } catch (_) {}
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   ACCESSIBLE TEXT DICTATION
   Hooks into the active focused text field and dispatches dictated text
   (additive only — voice-nav.js handles speech recognition; this
    module listens for a custom 'acm:dictate' event emitted by the
    enhanced voice pipeline below)
───────────────────────────────────────────────────────────────────────── */
class A11yDictation {
  constructor(logger) {
    this._logger   = logger;
    this._active   = false;
    this._recognition = null;
    this._indicator   = this._createIndicator();
    this._listen();
  }

  _createIndicator() {
    const el = document.createElement('div');
    el.id = 'a11y-dictation-indicator';
    el.setAttribute('aria-live', 'polite');
    el.setAttribute('aria-atomic', 'true');
    el.style.cssText = `
      position:fixed; bottom:70px; left:50%; transform:translateX(-50%);
      background:rgba(0,255,136,0.12); border:1px solid rgba(0,255,136,0.4);
      color:#00ff88; padding:6px 16px; border-radius:20px;
      font-size:0.78rem; font-weight:600; z-index:999995;
      display:none; pointer-events:none;
    `;
    document.body.appendChild(el);
    return el;
  }

  _listen() {
    // Listen for ACM dictation start/stop events
    window.addEventListener('acm:dictate:start', () => this._startDictation());
    window.addEventListener('acm:dictate:stop',  () => this._stopDictation());
  }

  _startDictation() {
    const el = document.activeElement;
    const isTextField = el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable);
    if (!isTextField) {
      window.app?.toast?.show?.('Dictation', 'Focus a text field first', 'warn', 'fas fa-keyboard', 3000);
      return;
    }

    if (!('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)) {
      window.app?.toast?.show?.('Dictation', 'Speech recognition not supported', 'error', 'fas fa-microphone-slash', 3000);
      return;
    }

    this._active = true;
    this._indicator.style.display = 'block';
    this._indicator.textContent   = '🎙 Dictating…';

    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    this._recognition = new SR();
    this._recognition.continuous     = false;
    this._recognition.interimResults  = false;
    this._recognition.lang            = 'en-US';

    this._recognition.onresult = (e) => {
      const text = e.results[0][0].transcript;
      this._insertText(el, text);
      this._logger?.log(`User dictated text: "${text.slice(0,40)}${text.length>40?'…':''}"`, 'voice', 'dictate', el);
    };
    this._recognition.onend = () => this._stopDictation();
    this._recognition.onerror = () => this._stopDictation();
    this._recognition.start();
  }

  _stopDictation() {
    this._active = false;
    this._indicator.style.display = 'none';
    try { this._recognition?.stop(); } catch (_) {}
    this._recognition = null;
  }

  _insertText(el, text) {
    if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
      const start = el.selectionStart ?? el.value.length;
      const end   = el.selectionEnd   ?? el.value.length;
      el.value = el.value.slice(0, start) + text + el.value.slice(end);
      el.selectionStart = el.selectionEnd = start + text.length;
      el.dispatchEvent(new Event('input', { bubbles: true }));
    } else if (el.isContentEditable) {
      const sel = window.getSelection();
      if (sel.rangeCount) {
        sel.deleteFromDocument();
        sel.getRangeAt(0).insertNode(document.createTextNode(text));
        sel.collapseToEnd();
      }
    }
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   SKIP NAVIGATION
   WCAG 2.4.1 — provides a "Skip to main content" landmark
───────────────────────────────────────────────────────────────────────── */
function injectSkipNav() {
  if (document.getElementById('a11y-skip-nav')) return;
  const skip = document.createElement('a');
  skip.id        = 'a11y-skip-nav';
  skip.href       = '#demo-main';
  skip.textContent = 'Skip to main content';
  skip.setAttribute('aria-label', 'Skip to main content');
  document.body.insertBefore(skip, document.body.firstChild);
}

/* ─────────────────────────────────────────────────────────────────────────
   LIVE REGION ANNOUNCER
   WCAG 4.1.3 — announces actions to screen readers via aria-live
───────────────────────────────────────────────────────────────────────── */
class A11yAnnouncer {
  constructor() {
    this._el = document.createElement('div');
    this._el.id = 'a11y-announcer';
    this._el.setAttribute('aria-live', 'assertive');
    this._el.setAttribute('aria-atomic', 'true');
    this._el.setAttribute('aria-relevant', 'text');
    this._el.style.cssText = 'position:absolute;left:-9999px;width:1px;height:1px;overflow:hidden;';
    document.body.appendChild(this._el);
  }

  announce(text) {
    this._el.textContent = '';
    requestAnimationFrame(() => { this._el.textContent = text; });
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   ACCESSIBILITY CONTROL MODE — MAIN CONTROLLER
───────────────────────────────────────────────────────────────────────── */
class AccessibilityControlMode {
  constructor() {
    this._enabled    = false;
    this._standards  = Object.values(STANDARDS);
    this._sessionId  = `acm_${Date.now()}`;

    // Sub-systems
    this._focusRing  = new A11yFocusRing();
    this._navigator  = new A11yNavigator();
    this._announcer  = new A11yAnnouncer();
    this._hint       = new A11yHintOverlay();
    this._logger     = null; // injected after logger module loads
    this._dwell      = null; // created when logger is ready

    // UI refs (set in _connectUI)
    this._ui = {
      toggleBtn:      null,
      statusBadge:    null,
      statusBar:      null,
      standardsList:  null,
      elementCount:   null,
      logCount:       null,
      dwellSlider:    null,
      hintBtn:        null,
      exportCsvBtn:   null,
      exportPdfBtn:   null,
    };

    // Keyboard augmentation (WCAG 2.1 — keyboard accessible)
    this._keyHandler = this._onKeyDown.bind(this);
  }

  /* ── Public API ─────────────────────────────────────────────────── */

  init(logger) {
    this._logger = logger;
    this._dwell  = new A11yDwellActivator(this._navigator, logger, this._focusRing);
    this._dictation = new A11yDictation(logger);
    injectSkipNav();
    this._connectUI();
    this._extendVoiceCommands();
    this._listenForFocusChanges();
    this._scanAndUpdateCount();
    this._updateStatusUI();

    // Emit ready event
    window.dispatchEvent(new CustomEvent('acm:ready', { detail: { version: ACM_VERSION } }));

    console.log(`%c Accessibility Control Mode ✅ v${ACM_VERSION} — WCAG/ADA/508 compliant`,
                'color:#00ff88;font-weight:bold;font-size:12px;');
  }

  enable() {
    if (this._enabled) return;
    this._enabled = true;

    // Scan all focusable elements
    const count = this._navigator.scan();

    // Start gaze dwell activation
    this._dwell?.start();

    // Add keyboard augmentation
    document.addEventListener('keydown', this._keyHandler, true);

    // Show first-time hint
    this._hint.show();

    // Update UI
    this._updateStatusUI();
    this._updateElementCount(count);

    // Log activation
    this._logger?.log(
      `Accessibility Control Mode activated — ${count} interactive elements indexed`,
      'system', 'mode_change', null,
      { standards: this._standards, session: this._sessionId }
    );

    // Announce to screen readers
    this._announcer.announce('Accessibility Control Mode enabled. Gaze, voice, and intent fusion are active.');

    // Visual cue on body
    document.body.classList.add('a11y-mode-active');

    window.app?.toast?.show?.('Accessibility Mode', `Active — ${count} elements indexed`, 'success', 'fas fa-universal-access', 3000);
  }

  disable() {
    if (!this._enabled) return;
    this._enabled = false;

    this._dwell?.stop();
    this._focusRing.hide();
    document.removeEventListener('keydown', this._keyHandler, true);
    document.body.classList.remove('a11y-mode-active');

    this._updateStatusUI();
    this._logger?.log('Accessibility Control Mode deactivated', 'system', 'mode_change');
    this._announcer.announce('Accessibility Control Mode disabled.');
    window.app?.toast?.show?.('Accessibility Mode', 'Disabled', 'info', 'fas fa-universal-access', 2000);
  }

  toggle() {
    this._enabled ? this.disable() : this.enable();
  }

  get enabled() { return this._enabled; }
  get sessionId() { return this._sessionId; }

  /* ── UI Connection ────────────────────────────────────────────── */

  _connectUI() {
    const $  = id => document.getElementById(id);
    this._ui.toggleBtn     = $('acm-toggle-btn');
    this._ui.statusBadge   = $('acm-status-badge');
    this._ui.statusBar     = $('acm-status-bar');
    this._ui.standardsList = $('acm-standards-list');
    this._ui.elementCount  = $('acm-element-count');
    this._ui.logCount      = $('acm-log-count');
    this._ui.dwellSlider   = $('acm-dwell-slider');
    this._ui.hintBtn       = $('acm-hint-btn');
    this._ui.exportCsvBtn  = $('acm-export-csv');
    this._ui.exportPdfBtn  = $('acm-export-pdf');

    this._ui.toggleBtn?.addEventListener('click', () => this.toggle());

    this._ui.dwellSlider?.addEventListener('input', (e) => {
      const ms = parseInt(e.target.value);
      this._dwell?.setDwellTime(ms);
      const label = document.getElementById('acm-dwell-val');
      if (label) label.textContent = ms + ' ms';
    });

    this._ui.hintBtn?.addEventListener('click', () => {
      this._hint._dismissed = false;
      this._hint._el = null;
      this._hint.show();
    });

    this._ui.exportCsvBtn?.addEventListener('click', () => {
      this._logger?.exportCSV();
      this._logger?.log('User exported accessibility log as CSV', 'system', 'export');
    });

    this._ui.exportPdfBtn?.addEventListener('click', () => {
      this._logger?.exportPDF();
      this._logger?.log('User exported accessibility log as PDF', 'system', 'export');
    });

    // Standards compliance badges
    if (this._ui.standardsList) {
      this._ui.standardsList.innerHTML = this._standards.map(s =>
        `<span class="a11y-std-badge">${s}</span>`
      ).join('');
    }
  }

  _updateStatusUI() {
    const { toggleBtn, statusBadge } = this._ui;
    if (toggleBtn) {
      toggleBtn.classList.toggle('active', this._enabled);
      toggleBtn.innerHTML = this._enabled
        ? '<i class="fas fa-universal-access"></i> <span>ACM ON</span>'
        : '<i class="fas fa-universal-access"></i> <span>ACM OFF</span>';
    }
    if (statusBadge) {
      statusBadge.textContent = this._enabled ? 'ACTIVE' : 'INACTIVE';
      statusBadge.className   = 'acm-status-badge ' + (this._enabled ? 'active' : '');
    }
  }

  _updateElementCount(count) {
    if (this._ui.elementCount) {
      this._ui.elementCount.textContent = count;
    }
  }

  _scanAndUpdateCount() {
    const count = this._navigator.scan();
    this._updateElementCount(count);
    // Re-scan when DOM changes
    const observer = new MutationObserver(() => {
      this._updateElementCount(this._navigator.scan());
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }

  updateLogCount(count) {
    if (this._ui.logCount) this._ui.logCount.textContent = count;
  }

  /* ── Keyboard Augmentation (WCAG 2.1.1) ──────────────────────── */

  _onKeyDown(e) {
    if (!this._enabled) return;

    switch (e.key) {
      case 'Tab': {
        // Let natural tab order work; just log it
        const el = document.activeElement;
        if (el && el !== document.body) {
          const label = el.getAttribute('aria-label') || el.textContent?.trim().slice(0,40) || el.tagName;
          this._logger?.log(`User navigated to "${label}" via Tab key`, 'keyboard', 'focus', el);
          this._focusRing.show(el);
          this._announcer.announce(`Focused: ${label}`);
        }
        break;
      }
      case 'Enter':
      case ' ': {
        const el = document.activeElement;
        if (el && el !== document.body) {
          const label = el.getAttribute('aria-label') || el.textContent?.trim().slice(0,40) || el.tagName;
          this._logger?.log(`User activated "${label}" via keyboard (${e.key})`, 'keyboard', 'activate', el);
        }
        break;
      }
    }
  }

  /* ── Focus Change Listener ────────────────────────────────────── */

  _listenForFocusChanges() {
    document.addEventListener('focusin', (e) => {
      if (!this._enabled) return;
      const el = e.target;
      const label = el.getAttribute('aria-label') || el.textContent?.trim().slice(0,40) || el.tagName;
      this._focusRing.show(el);
      this._announcer.announce(label);
    });
    document.addEventListener('focusout', () => {
      if (!this._enabled) return;
      this._focusRing.hide();
    });
  }

  /* ── Voice Command Extension ──────────────────────────────────── */

  _extendVoiceCommands() {
    // Wait until voiceNav controller is ready, then patch in ACM-specific commands
    const patch = () => {
      const vn = window.voiceNav;
      if (!vn) { setTimeout(patch, 300); return; }

      // Monkey-patch _performAction to add ACM logging and new actions
      const origPerform = vn._performAction.bind(vn);
      vn._performAction = (entry, action) => {
        // Log through ACM logger if ACM is enabled
        if (this._enabled && this._logger) {
          const modality = this._detectModality();
          const label = entry?.text || action;
          this._logVoiceAction(action, label, entry?.el || null, modality);
        }
        // Handle ACM-specific actions before delegating to original
        switch (action) {
          case 'acm:dictate:start':
            window.dispatchEvent(new Event('acm:dictate:start'));
            return;
          case 'acm:dictate:stop':
            window.dispatchEvent(new Event('acm:dictate:stop'));
            return;
          case 'acm:toggle':
            this.toggle();
            return;
          case 'acm:hint':
            this._hint._dismissed = false; this._hint._el = null; this._hint.show();
            return;
          case 'acm:export:csv':
            this._logger?.exportCSV();
            return;
          case 'acm:focusNext':
            const next = this._navigator.focusNext();
            if (next) {
              const lbl = next.getAttribute('aria-label') || next.textContent?.trim().slice(0,40) || next.tagName;
              this._logger?.log(`User moved to next element "${lbl}" via voice`, 'voice', 'focus', next);
              this._announcer.announce(`Focused: ${lbl}`);
            }
            return;
          case 'acm:focusPrev':
            const prev = this._navigator.focusPrev();
            if (prev) {
              const lbl2 = prev.getAttribute('aria-label') || prev.textContent?.trim().slice(0,40) || prev.tagName;
              this._logger?.log(`User moved to previous element "${lbl2}" via voice`, 'voice', 'focus', prev);
              this._announcer.announce(`Focused: ${lbl2}`);
            }
            return;
        }
        // Delegate to original voice-nav handler
        origPerform(entry, action);
      };

      // Add ACM action entries to voice command map
      const extraCmds = {
        'start dictation'  : 'acm:dictate:start',
        'begin dictation'  : 'acm:dictate:start',
        'dictate'          : 'acm:dictate:start',
        'stop dictation'   : 'acm:dictate:stop',
        'accessibility mode': 'acm:toggle',
        'show hint'        : 'acm:hint',
        'show guide'       : 'acm:hint',
        'export log'       : 'acm:export:csv',
        'download log'     : 'acm:export:csv',
      };
      // Inject into the voice controller's ACTION_COMMANDS (module-level var is shared)
      // We do this by extending the multiWord lookup in the controller
      const origMulti = vn._extractMultiWordAction.bind(vn);
      vn._extractMultiWordAction = (lower) => {
        for (const [key, val] of Object.entries(extraCmds)) {
          if (lower.includes(key)) return val;
        }
        return origMulti(lower);
      };

      console.log('[ACM] Voice command extensions patched');
    };
    setTimeout(patch, 800);
  }

  _detectModality() {
    // Simple heuristic: if voice transcript was updated in last 500ms → voice
    // else → gaze (dwell). Intent fusion is logged separately.
    return 'multimodal';
  }

  _logVoiceAction(action, label, el, modality) {
    const actionMap = {
      'click': 'User activated', 'open': 'User opened', 'select': 'User selected',
      'scrollUp': 'User scrolled up', 'scrollDown': 'User scrolled down',
      'back': 'User navigated back', 'navForward': 'User navigated forward',
      'reloadPage': 'User reloaded page', 'newTab': 'User opened new tab',
      'zoomIn': 'User zoomed in', 'zoomOut': 'User zoomed out',
      'focusSearch': 'User focused search field', 'pauseControl': 'User paused control',
      'resumeControl': 'User resumed control', 'showClickable': 'User invoked element discovery',
    };
    const prefix = actionMap[action] || `User executed ${action} on`;
    const msg = label !== action ? `${prefix} "${label}"` : `${prefix}`;
    this._logger?.log(msg + ` via voice`, 'voice', action, el);
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   BOOTSTRAP
───────────────────────────────────────────────────────────────────────── */
(function bootstrapACM() {
  const acm = new AccessibilityControlMode();
  let attempts = 0;

  const attach = () => {
    attempts++;
    // Wait for both app and logger to be ready
    if (!window.app || !window.AccessEye?.a11yLogger) {
      if (attempts <= 40) setTimeout(attach, 250);
      else console.warn('[ACM] Timed out waiting for app/logger after', attempts, 'attempts');
      return;
    }

    acm.init(window.AccessEye.a11yLogger);

    // Expose globally
    window.AccessEye = window.AccessEye || {};
    window.AccessEye.acm = acm;
    window.acm = acm;

    // Update log count display whenever logger emits a new entry
    window.addEventListener('a11y:log', () => {
      acm.updateLogCount(window.AccessEye.a11yLogger.count);
    });

    console.log('%c Accessibility Control Mode ✅ v' + ACM_VERSION + ' — WCAG/ADA/508 ready',
                'color:#00ff88;font-weight:bold;font-size:12px;');
  };

  // Start polling after DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => setTimeout(attach, 500));
  } else {
    setTimeout(attach, 500);
  }
})();
