/**
 * AccessEye — Snap-To Engine  (snap-engine.js)
 * ═══════════════════════════════════════════════════════════════════
 *  Three co-operating modules:
 *
 *  1. SnapToEngine        — Detects nearby interactive elements,
 *                           smooth-interpolates cursor to the best
 *                           candidate, and emits snap/release events.
 *
 *  2. TargetPredictor     — Scores every candidate using distance,
 *                           size, semantic type, and interaction
 *                           history; picks the highest-scoring
 *                           element within snapThresholdDistance.
 *
 *  3. AdaptiveGazeLearner — Builds a per-user profile from every
 *                           dwell/snap event and continuously tunes
 *                           snapThresholdDistance, dwellClickTime,
 *                           cursorSmoothing, and predictionWeight.
 *
 *  Integration (app.js):
 *    const snapEngine = new SnapToEngine(config);
 *    snapEngine.enable();
 *    // Each gaze frame:
 *    const {x, y, snapped} = snapEngine.update(rawPx, rawPy);
 *    cursor.style.left = x + 'px'; cursor.style.top = y + 'px';
 * ═══════════════════════════════════════════════════════════════════
 */

/* ─── Tiny utilities (duplicated here so the file is self-contained) ─── */
const _clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const _lerp  = (a, b, t)   => a + (b - a) * t;
const _dist  = (x1, y1, x2, y2) => Math.hypot(x2 - x1, y2 - y1);
const _now   = () => performance.now();

/* ─── Selector for interactive elements ─── */
const SNAP_SELECTOR = [
  'button:not([disabled])',
  'a[href]',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[role="button"]',
  '[role="link"]',
  '[role="menuitem"]',
  '[role="tab"]',
  '[role="checkbox"]',
  '[role="radio"]',
  '[tabindex]:not([tabindex="-1"])',
  '[data-accessible-target]',
  '.gaze-target',
].join(',');

/* Semantic priority weights (higher = more attractive snap target) */
const SEMANTIC_WEIGHT = {
  'BUTTON'   : 1.0,
  'A'        : 0.85,
  'INPUT'    : 0.80,
  'SELECT'   : 0.75,
  'TEXTAREA' : 0.70,
  'DEFAULT'  : 0.50,
};

/**
 * Local-storage key for the adaptive profile.
 */
const PROFILE_KEY = 'accesseye_snap_profile';

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1.  TargetPredictor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
class TargetPredictor {
  /**
   * @param {Object}  profile       Live AdaptiveGazeLearner profile ref
   * @param {number}  predictionWeight  0-1, scales history vs geometry
   */
  constructor(profile, predictionWeight = 0.35) {
    this._profile          = profile;
    this._predictionWeight = predictionWeight;

    // Cached element list: [{el, bbox, center, tag, role, clickable}]
    this._cache      = [];
    this._cacheTime  = 0;
    this.CACHE_TTL   = 300; // ms – refresh at ~3 Hz

    // Observers for DOM/layout changes
    this._resizeObs  = null;
    this._mutObs     = null;
    this._dirty      = true;   // force first scan

    this._initObservers();
  }

  /** Tear down observers */
  destroy() {
    this._resizeObs?.disconnect();
    this._mutObs?.disconnect();
  }

  /** Set prediction weight externally (AdaptiveGazeLearner calls this) */
  setPredictionWeight(w) {
    this._predictionWeight = _clamp(w, 0, 1);
  }

  /**
   * Return the best snap candidate for (px, py) within maxDist pixels.
   * Returns null when nothing qualifies.
   * @returns {{el, center, bbox, score, dist}|null}
   */
  predict(px, py, maxDist) {
    this._maybeRefreshCache();
    if (this._cache.length === 0) return null;

    let best = null;
    let bestScore = -Infinity;

    for (const entry of this._cache) {
      const cx = entry.center.x;
      const cy = entry.center.y;
      const d  = _dist(px, py, cx, cy);
      if (d > maxDist) continue;

      const score = this._score(entry, d, maxDist);
      if (score > bestScore) {
        bestScore = score;
        best = { ...entry, score, dist: d };
      }
    }
    return best;
  }

  /** Score one candidate */
  _score(entry, dist, maxDist) {
    // Distance score: 1 at 0 px, 0 at maxDist
    const dScore = 1 - dist / maxDist;

    // Size score: larger = slightly more attractive (logarithmic)
    const area = entry.bbox.w * entry.bbox.h;
    const sScore = _clamp(Math.log10(area + 1) / 5, 0, 1);

    // Semantic score
    const semScore = entry.semantic;

    // History score: interaction frequency 0-1
    const freq = this._profile.interactionFreq[entry._uid] || 0;
    const hScore = _clamp(freq / 10, 0, 1);   // saturates at 10 interactions

    const pw  = this._predictionWeight;
    const gw  = 1 - pw;   // geometry weight

    return (gw * (0.55 * dScore + 0.25 * sScore + 0.20 * semScore))
         + (pw * hScore);
  }

  /** Maybe rebuild the element cache */
  _maybeRefreshCache() {
    const t = _now();
    if (!this._dirty && t - this._cacheTime < this.CACHE_TTL) return;
    this._rebuildCache();
    this._dirty    = false;
    this._cacheTime = t;
  }

  /** Scan the DOM and cache bounding boxes */
  _rebuildCache() {
    const els = document.querySelectorAll(SNAP_SELECTOR);
    const viewport = { w: window.innerWidth, h: window.innerHeight };
    const list = [];

    els.forEach((el, idx) => {
      // Skip hidden, zero-size, or off-screen elements
      if (el.offsetParent === null && getComputedStyle(el).position !== 'fixed') return;
      const r = el.getBoundingClientRect();
      if (r.width < 2 || r.height < 2) return;
      if (r.right < 0 || r.bottom < 0 || r.left > viewport.w || r.top > viewport.h) return;

      // Unique ID for history tracking
      const uid = el.dataset.id || el.id || el.dataset.accessibleTarget ||
                  (el.textContent?.trim().slice(0, 20) || '') + '_' + idx;

      const tag = el.tagName;
      const role = (el.getAttribute('role') || '').toUpperCase();
      const semantic = SEMANTIC_WEIGHT[tag] || SEMANTIC_WEIGHT[role === 'BUTTON' ? 'BUTTON' : 'DEFAULT'] || SEMANTIC_WEIGHT.DEFAULT;

      list.push({
        el,
        _uid    : uid,
        bbox    : { x: r.left, y: r.top, w: r.width, h: r.height },
        center  : { x: r.left + r.width / 2, y: r.top + r.height / 2 },
        tag,
        semantic,
        clickable: tag === 'BUTTON' || tag === 'A' || tag === 'INPUT' ||
                   !!el.onclick || el.getAttribute('role') === 'button',
      });
    });

    this._cache = list;
  }

  /** Record an interaction with an element (boosts future score) */
  recordInteraction(el) {
    if (!el) return;
    const uid = el.dataset.id || el.id || el.dataset.accessibleTarget ||
                (el.textContent?.trim().slice(0, 20) || '');
    const p = this._profile;
    p.interactionFreq[uid] = (p.interactionFreq[uid] || 0) + 1;
  }

  _initObservers() {
    // ResizeObserver – mark dirty on any element resize
    if (window.ResizeObserver) {
      this._resizeObs = new ResizeObserver(() => { this._dirty = true; });
      this._resizeObs.observe(document.body);
    }
    // MutationObserver – mark dirty on DOM changes
    if (window.MutationObserver) {
      this._mutObs = new MutationObserver(() => { this._dirty = true; });
      this._mutObs.observe(document.body, {
        childList  : true,
        subtree    : true,
        attributes : true,
        attributeFilter: ['style', 'class', 'disabled', 'hidden'],
      });
    }
    // Scroll and resize events
    window.addEventListener('scroll', () => { this._dirty = true; }, { passive: true });
    window.addEventListener('resize', () => { this._dirty = true; }, { passive: true });
  }
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   2.  AdaptiveGazeLearner
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
class AdaptiveGazeLearner {
  constructor() {
    /** Live tunable config — SnapToEngine reads these */
    this.config = {
      snapThresholdDistance : 90,    // px
      dwellClickTime        : 900,   // ms
      cursorSmoothing       : 0.22,  // lerp alpha (0=slow, 1=instant)
      predictionWeight      : 0.35,  // history vs geometry
      zoneActivationTime    : 1200,  // ms for gaze-command zones
    };

    /** Per-user profile (persisted in localStorage) */
    this.profile = {
      version           : 1,
      interactionFreq   : {},   // uid → count
      dwellSamples      : [],   // last N dwell durations (ms)
      snapDistSamples   : [],   // last N snap-success distances (px)
      velocitySamples   : [],   // last N cursor velocities (px/ms)
      driftSamples      : [],   // last N horizontal drift values
      totalActivations  : 0,
      sessionStart      : _now(),
    };

    this._MAX_SAMPLES = 40;
    this._load();
  }

  /** Called by SnapToEngine after a successful dwell activation */
  recordDwell(durationMs, snapDist, velocityPxPerMs) {
    const p = this.profile;
    p.dwellSamples.push(durationMs);
    if (snapDist !== null) p.snapDistSamples.push(snapDist);
    if (velocityPxPerMs !== null) p.velocitySamples.push(velocityPxPerMs);
    p.totalActivations++;

    // Trim to window size
    if (p.dwellSamples.length   > this._MAX_SAMPLES) p.dwellSamples.shift();
    if (p.snapDistSamples.length > this._MAX_SAMPLES) p.snapDistSamples.shift();
    if (p.velocitySamples.length > this._MAX_SAMPLES) p.velocitySamples.shift();

    this._adapt();
    this._save();
  }

  /** Called by SnapToEngine on every gaze frame for drift tracking */
  recordDrift(dx) {
    const p = this.profile;
    p.driftSamples.push(Math.abs(dx));
    if (p.driftSamples.length > this._MAX_SAMPLES) p.driftSamples.shift();
    // Drift doesn't trigger full _adapt to avoid overhead
  }

  /** Update config based on accumulated samples */
  _adapt() {
    const p = this.profile;
    const c = this.config;

    // ── Dwell click time ──────────────────────────────────────────────
    if (p.dwellSamples.length >= 5) {
      const sorted = [...p.dwellSamples].sort((a, b) => a - b);
      // Use 25th-percentile dwell: fast users → shorter threshold
      const p25 = sorted[Math.floor(sorted.length * 0.25)];
      // Clamp to [400 ms, 1800 ms]
      c.dwellClickTime = _clamp(Math.round(p25 * 0.9), 400, 1800);
    }

    // ── Snap distance ─────────────────────────────────────────────────
    if (p.snapDistSamples.length >= 5) {
      const avg = p.snapDistSamples.reduce((s, v) => s + v, 0) / p.snapDistSamples.length;
      // Pad by 25 % with bounds [50, 200]
      c.snapThresholdDistance = _clamp(Math.round(avg * 1.25), 50, 200);
    }

    // ── Cursor smoothing ──────────────────────────────────────────────
    if (p.velocitySamples.length >= 5) {
      const avgV = p.velocitySamples.reduce((s, v) => s + v, 0) / p.velocitySamples.length;
      // Faster gaze → faster cursor (more responsive alpha)
      // velocity range ~0.01–0.5 px/ms → alpha range 0.12–0.35
      c.cursorSmoothing = _clamp(0.12 + avgV * 0.46, 0.12, 0.35);
    }

    // ── Prediction weight ─────────────────────────────────────────────
    if (p.totalActivations >= 10) {
      // Gradually trust history more as activations accumulate (cap at 0.6)
      c.predictionWeight = _clamp(0.35 + (p.totalActivations / 200), 0.35, 0.60);
    }
  }

  /** Persist profile to localStorage */
  _save() {
    try {
      localStorage.setItem(PROFILE_KEY, JSON.stringify(this.profile));
    } catch (_) { /* storage full – skip */ }
  }

  /** Load profile from localStorage */
  _load() {
    try {
      const raw = localStorage.getItem(PROFILE_KEY);
      if (!raw) return;
      const saved = JSON.parse(raw);
      if (saved?.version === this.profile.version) {
        // Merge arrays; keep session-specific fields fresh
        Object.assign(this.profile, saved, { sessionStart: _now() });
        this._adapt();
      }
    } catch (_) { /* corrupt – ignore */ }
  }

  /** Reset profile and return to defaults */
  reset() {
    this.profile = {
      version          : 1,
      interactionFreq  : {},
      dwellSamples     : [],
      snapDistSamples  : [],
      velocitySamples  : [],
      driftSamples     : [],
      totalActivations : 0,
      sessionStart     : _now(),
    };
    this.config = {
      snapThresholdDistance : 90,
      dwellClickTime        : 900,
      cursorSmoothing       : 0.22,
      predictionWeight      : 0.35,
      zoneActivationTime    : 1200,
    };
    try { localStorage.removeItem(PROFILE_KEY); } catch (_) {}
  }
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   3.  SnapToEngine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
class SnapToEngine {
  /**
   * @param {Object} [opts]
   * @param {boolean} [opts.enabled=false]        Start enabled?
   * @param {number}  [opts.snapThreshold=90]     px radius
   * @param {number}  [opts.smoothing=0.22]       lerp alpha
   * @param {number}  [opts.dwellClickTime=900]   ms to auto-click
   * @param {number}  [opts.predictionWeight=0.35]
   * @param {boolean} [opts.autoDwellClick=false] dwell-to-click toggle
   */
  constructor(opts = {}) {
    this.learner   = new AdaptiveGazeLearner();
    this.predictor = new TargetPredictor(
      this.learner.profile,
      this.learner.config.predictionWeight,
    );

    // Live config — always read from learner.config (which adapts over time)
    this._cfg = this.learner.config;

    // Override any initial values from opts
    if (opts.snapThreshold    !== undefined) this._cfg.snapThresholdDistance = opts.snapThreshold;
    if (opts.smoothing        !== undefined) this._cfg.cursorSmoothing       = opts.smoothing;
    if (opts.dwellClickTime   !== undefined) this._cfg.dwellClickTime        = opts.dwellClickTime;
    if (opts.predictionWeight !== undefined) this._cfg.predictionWeight      = opts.predictionWeight;

    this.enabled       = opts.enabled ?? false;
    this.autoDwellClick = opts.autoDwellClick ?? false;

    // Smooth cursor state
    this._curX   = 0;
    this._curY   = 0;
    this._prevX  = 0;
    this._prevY  = 0;
    this._prevT  = _now();

    // Current snap state
    this._snapTarget   = null;   // current highlighted element
    this._snapCenterX  = 0;
    this._snapCenterY  = 0;
    this._dwelling     = false;
    this._dwellStart   = 0;
    this._dwellProgress = 0;

    // Event callbacks
    this._callbacks = {};

    // Build highlight overlay element
    this._highlightEl = this._createHighlightEl();
  }

  /* ── Public API ─────────────────────────────────────────────── */

  enable()  { this.enabled = true;  this._clearHighlight(); }
  disable() { this.enabled = false; this._clearHighlight(); }
  toggle()  { this.enabled ? this.disable() : this.enable(); return this.enabled; }

  /**
   * SECTION 1 — Full cleanup when exiting Lock-On mode.
   * Stops ALL snap-related processes deterministically:
   *  • clears highlight overlays on every tracked element
   *  • cancels dwell timer
   *  • resets interpolated cursor state
   *  • marks disabled so update() is a no-op
   * Does NOT disconnect observers (they're cheap and needed on re-enable).
   */
  fullCleanup() {
    // Clear any active snap highlight on the DOM element
    this._clearHighlight();

    // Force-remove snap classes from ALL elements that may have been touched
    document.querySelectorAll('.snap-highlight, .snap-dwell-active, .snap-activated').forEach(el => {
      el.classList.remove('snap-highlight', 'snap-dwell-active', 'snap-activated');
      const bar = el.querySelector('.dwell-progress');
      if (bar) bar.style.width = '0%';
    });

    // Reset internal state
    this._snapTarget    = null;
    this._dwelling      = false;
    this._dwellStart    = 0;
    this._dwellProgress = 0;

    // Reset smooth-cursor state so next enable() starts from current raw position
    this._curX  = 0;
    this._curY  = 0;
    this._prevX = 0;
    this._prevY = 0;

    // Disable so update() is skipped entirely in free-look
    this.enabled = false;

    // Mark predictor cache dirty so next enable() rescans DOM fresh
    if (this.predictor) this.predictor._dirty = true;
  }

  on(event, cb) {
    if (!this._callbacks[event]) this._callbacks[event] = [];
    this._callbacks[event].push(cb);
  }

  _emit(event, data) {
    (this._callbacks[event] || []).forEach(cb => cb(data));
  }

  /**
   * Process one gaze frame.
   * Call this instead of directly positioning the cursor.
   *
   * @param {number} rawPx  Raw gaze X in viewport pixels
   * @param {number} rawPy  Raw gaze Y in viewport pixels
   * @returns {{ x: number, y: number, snapped: boolean, target: Element|null, dwellProgress: number }}
   */
  update(rawPx, rawPy) {
    const t   = _now();
    const dt  = t - this._prevT;
    this._prevT = t;

    // Velocity for adaptive learning (px/ms, clamped)
    const vel = dt > 0 ? _clamp(_dist(rawPx, rawPy, this._prevX, this._prevY) / dt, 0, 2) : 0;
    this._prevX = rawPx;
    this._prevY = rawPy;

    const alpha = _clamp(this._cfg.cursorSmoothing, 0.05, 1.0);

    let targetX = rawPx;
    let targetY = rawPy;
    let snapped  = false;
    let snapEl   = null;

    if (this.enabled) {
      // Update predictor's prediction weight from learner
      this.predictor.setPredictionWeight(this._cfg.predictionWeight);

      const candidate = this.predictor.predict(rawPx, rawPy, this._cfg.snapThresholdDistance);

      if (candidate) {
        targetX = candidate.center.x;
        targetY = candidate.center.y;
        snapped = true;
        snapEl  = candidate.el;

        // Drift tracking
        this.learner.recordDrift(rawPx - targetX);

        // Highlight
        if (this._snapTarget !== candidate.el) {
          this._clearHighlight();
          this._snapTarget   = candidate.el;
          this._snapCenterX  = targetX;
          this._snapCenterY  = targetY;
          this._applyHighlight(candidate.el);
          this._dwelling     = false;
          this._dwellStart   = 0;
          this._dwellProgress = 0;
          this._emit('snap', { el: candidate.el, score: candidate.score, dist: candidate.dist });
        }

        // Dwell timer
        if (!this._dwelling) {
          this._dwelling   = true;
          this._dwellStart = t;
        }
        const elapsed = t - this._dwellStart;
        this._dwellProgress = _clamp(elapsed / this._cfg.dwellClickTime, 0, 1);
        this._updateDwellRing(candidate.el, this._dwellProgress);

        // Auto-click on dwell completion
        if (this.autoDwellClick && this._dwellProgress >= 1) {
          this._activateTarget(candidate.el, candidate.dist, vel);
        }
      } else {
        // No nearby target — release snap
        if (this._snapTarget) {
          this._emit('release', { el: this._snapTarget });
          this._clearHighlight();
          this._snapTarget    = null;
          this._dwelling      = false;
          this._dwellStart    = 0;
          this._dwellProgress = 0;
        }
      }
    } else {
      // Snap disabled — clear any leftover highlight
      if (this._snapTarget) {
        this._clearHighlight();
        this._snapTarget = null;
      }
    }

    // Smooth interpolation: cursorX += (target - cursor) * alpha
    this._curX += (targetX - this._curX) * alpha;
    this._curY += (targetY - this._curY) * alpha;

    return {
      x            : this._curX,
      y            : this._curY,
      snapped,
      target       : snapEl,
      dwellProgress: this._dwellProgress,
    };
  }

  /**
   * Call this when a blink or gesture fires to activate the current snap target.
   * @param {'blink'|'pinch'|'airTap'|string} method
   */
  activateSnapped(method = 'gesture') {
    if (this._snapTarget) {
      this._activateTarget(this._snapTarget, 0, 0, method);
      return true;
    }
    return false;
  }

  /** Get the current snapped element (or null) */
  getSnappedTarget() { return this._snapTarget; }

  /** Expose live config for external UI (settings panel) */
  getConfig() { return { ...this._cfg }; }

  /** Update individual config keys from settings panel */
  setConfig(overrides) {
    Object.assign(this._cfg, overrides);
  }

  /** Reset adaptive profile */
  resetProfile() {
    this.learner.reset();
    this._emit('profileReset', {});
  }

  /* ── Private helpers ────────────────────────────────────────── */

  _activateTarget(el, dist, vel, method = 'dwell') {
    const duration = this._dwelling ? _now() - this._dwellStart : 0;

    // Record learning sample
    this.learner.recordDwell(duration, dist || null, vel || null);
    this.predictor.recordInteraction(el);

    // Visual flash
    el.classList.add('snap-activated');
    setTimeout(() => el.classList.remove('snap-activated'), 600);

    // Reset dwell state so it doesn't fire again immediately
    this._dwelling      = false;
    this._dwellStart    = 0;
    this._dwellProgress = 0;

    this._emit('activate', { el, method, duration });
  }

  _applyHighlight(el) {
    el.classList.add('snap-highlight');
    // Position the overlay ring
    const r = el.getBoundingClientRect();
    const h = this._highlightEl;
    h.style.left   = `${r.left   - 4}px`;
    h.style.top    = `${r.top    - 4}px`;
    h.style.width  = `${r.width  + 8}px`;
    h.style.height = `${r.height + 8}px`;
    h.style.opacity = '1';
    h.style.display = 'block';
  }

  _clearHighlight() {
    if (this._snapTarget) {
      this._snapTarget.classList.remove('snap-highlight', 'snap-dwell-active');
      this._clearDwellRing(this._snapTarget);
    }
    this._highlightEl.style.opacity = '0';
    this._highlightEl.style.display = 'none';
  }

  _updateDwellRing(el, progress) {
    // Drive the element's own dwell-progress bar if present
    const bar = el.querySelector('.dwell-progress');
    if (bar) bar.style.width = `${progress * 100}%`;
    // Mark as actively dwelling
    if (progress > 0) el.classList.add('snap-dwell-active');
    // Update highlight ring arc via CSS custom property
    this._highlightEl.style.setProperty('--dwell-pct', progress.toString());
  }

  _clearDwellRing(el) {
    const bar = el?.querySelector('.dwell-progress');
    if (bar) bar.style.width = '0%';
    el?.classList.remove('snap-dwell-active');
  }

  /** Create a fixed-position overlay ring (lives outside the snap target) */
  _createHighlightEl() {
    const el = document.createElement('div');
    el.id        = 'snap-highlight-ring';
    el.className = 'snap-highlight-ring';
    el.style.cssText = [
      'position:fixed',
      'pointer-events:none',
      'z-index:99998',
      'display:none',
      'opacity:0',
      'border-radius:8px',
      'transition:opacity 0.15s ease, left 0.08s ease, top 0.08s ease, width 0.08s ease, height 0.08s ease',
    ].join(';');
    document.body.appendChild(el);
    return el;
  }

  /** Clean up DOM elements and observers */
  destroy() {
    this.fullCleanup();
    this._highlightEl.remove();
    this.predictor.destroy();
  }
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Export to global scope (no bundler required)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
window.SnapToEngine        = SnapToEngine;
window.TargetPredictor     = TargetPredictor;
window.AdaptiveGazeLearner = AdaptiveGazeLearner;
