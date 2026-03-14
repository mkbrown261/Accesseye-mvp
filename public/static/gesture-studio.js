/**
 * AccessEye — Gesture Studio  (gesture-studio.js)
 * ═══════════════════════════════════════════════════════════════════
 *
 *  Four co-operating modules:
 *
 *  1. FacialGestureEngine   — Built-in lip-tap (scroll-up) and bite-bottom-lip
 *                             (scroll-down, held) detectors with confidence scoring,
 *                             minimum-duration gates, and per-gesture cooldowns.
 *
 *  2. CustomGestureRecorder — Captures 2–4 s of facial landmark data and
 *                             stores it as a reference profile (mouth shape,
 *                             brows, eye closure, cheek expansion, head dir).
 *
 *  3. CustomGestureRecognizer — Continuously compares live landmarks to saved
 *                               profiles; fires when similarity ≥ threshold.
 *
 *  4. GestureStudio         — Manages the full lifecycle: create, record,
 *                             assign action, save, edit, delete, and fire
 *                             gestures, with localStorage persistence.
 *
 *  Integration (app.js):
 *    const studio = new GestureStudio();
 *    studio.onAction((actionId, gestureName) => { … });
 *    // Each face frame:
 *    studio.processFaceLandmarks(lm);
 * ═══════════════════════════════════════════════════════════════════
 */

/* ─── Configuration constants ─── */
const SCROLL_AMOUNT               = 180;   // px per scroll event
const SCROLL_INTERVAL_MS          = 80;    // ms between repeated scroll ticks (bite-lip hold)
const LIP_TAP_TIME_WINDOW         = 750;   // ms between two closures
const LIP_TAP_CONFIDENCE_THRESHOLD = 0.70; // 0-1
const BITE_LIP_THRESHOLD          = 0.55;  // lower-lip-Y / mouth-height ratio for bite detection
const GESTURE_COOLDOWN            = 1200;  // ms between any built-in fire
const CUSTOM_GESTURE_CONFIDENCE   = 0.72;  // 0-1

const BLOW_DETECTION_THRESHOLD    = BITE_LIP_THRESHOLD;  // alias for config compat
const STUDIO_STORAGE_KEY = 'accesseye_gesture_studio';

/* MediaPipe FaceMesh landmark indices */
const LM = {
  // Mouth outer
  MOUTH_TOP    : 13,   // upper lip centre
  MOUTH_BOT    : 14,   // lower lip centre
  MOUTH_LEFT   : 61,
  MOUTH_RIGHT  : 291,
  // Mouth inner
  MOUTH_INNER_TOP : 82,
  MOUTH_INNER_BOT : 87,
  // Upper lip
  UPPER_LIP_L : 40,
  UPPER_LIP_R : 270,
  // Lower lip outer bottom edge (used for bite-lip detection)
  LOWER_LIP_L : 178,
  LOWER_LIP_R : 402,
  // Lower lip inner (rises when bottom lip is bitten inward)
  LOWER_LIP_INNER_L : 95,
  LOWER_LIP_INNER_R : 325,
  // Upper lip inner top
  UPPER_LIP_INNER_TOP : 13,
  // Nose tip (for head dir)
  NOSE_TIP    : 1,
  // Brow landmarks
  LEFT_BROW_INNER  : 107,
  LEFT_BROW_MID    : 105,
  LEFT_BROW_OUTER  : 70,
  RIGHT_BROW_INNER : 336,
  RIGHT_BROW_MID   : 334,
  RIGHT_BROW_OUTER : 300,
  // Eye closure
  LEFT_EYE_TOP    : 159,
  LEFT_EYE_BOT    : 145,
  RIGHT_EYE_TOP   : 386,
  RIGHT_EYE_BOT   : 374,
  LEFT_EYE_SPAN_L : 33,
  LEFT_EYE_SPAN_R : 133,
  RIGHT_EYE_SPAN_L: 362,
  RIGHT_EYE_SPAN_R: 263,
  // Cheek reference (distance between cheek bones to detect puffing)
  CHEEK_L : 234,
  CHEEK_R : 454,
};

/* ─── Tiny distance helper ─── */
function _d2(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Helper: extract a compact feature vector from face landmarks
   Returns 20-element Float32Array covering:
   - mouth open ratio (vertical / horizontal)
   - cheek expansion ratio
   - left/right brow heights (normalised)
   - left/right eye closure ratios
   - nose tip X/Y (head direction)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
function extractFeatures(lm) {
  const f = new Float32Array(20);

  // 0: mouth open ratio (inner vertical / outer horizontal)
  const mTop  = lm[LM.MOUTH_TOP];
  const mBot  = lm[LM.MOUTH_BOT];
  const mL    = lm[LM.MOUTH_LEFT];
  const mR    = lm[LM.MOUTH_RIGHT];
  const mITop = lm[LM.MOUTH_INNER_TOP];
  const mIBot = lm[LM.MOUTH_INNER_BOT];
  const mW    = _d2(mL, mR) || 0.001;
  const mH    = _d2(mTop, mBot);
  const mIH   = _d2(mITop, mIBot);
  f[0]  = mH  / mW;      // outer open ratio
  f[1]  = mIH / mW;      // inner open ratio  ← key for lip-tap
  f[2]  = mW;             // absolute width (for blow/cheek)

  // 3: cheek expansion (cheek-to-cheek / eye-to-eye baseline)
  const ckW  = _d2(lm[LM.CHEEK_L], lm[LM.CHEEK_R]) || 0.001;
  // use left eye span as head-width baseline
  const eyeBase = _d2(lm[LM.LEFT_EYE_SPAN_L], lm[LM.LEFT_EYE_SPAN_R]) || 0.001;
  f[3]  = ckW;            // raw cheek span (normalise later with eyeBase)
  f[4]  = eyeBase;

  // 5-7: left brow height relative to eye top
  const lEyeTop  = lm[LM.LEFT_EYE_TOP];
  const lBrowMid = lm[LM.LEFT_BROW_MID];
  f[5] = (lBrowMid.y - lEyeTop.y) / (mW || 0.001); // neg = raised

  const rEyeTop  = lm[LM.RIGHT_EYE_TOP];
  const rBrowMid = lm[LM.RIGHT_BROW_MID];
  f[6] = (rBrowMid.y - rEyeTop.y) / (mW || 0.001);

  // 7-8: eye closure ratios
  const lEyeH = _d2(lm[LM.LEFT_EYE_TOP], lm[LM.LEFT_EYE_BOT]);
  const lEyeW = _d2(lm[LM.LEFT_EYE_SPAN_L], lm[LM.LEFT_EYE_SPAN_R]) || 0.001;
  f[7]  = lEyeH / lEyeW;

  const rEyeH = _d2(lm[LM.RIGHT_EYE_TOP], lm[LM.RIGHT_EYE_BOT]);
  const rEyeW = _d2(lm[LM.RIGHT_EYE_SPAN_L], lm[LM.RIGHT_EYE_SPAN_R]) || 0.001;
  f[8]  = rEyeH / rEyeW;

  // 9-10: nose tip position (head direction proxy)
  f[9]  = lm[LM.NOSE_TIP].x;
  f[10] = lm[LM.NOSE_TIP].y;

  // 11: upper-lip L/R distance (lip pursing / puffing)
  f[11] = _d2(lm[LM.UPPER_LIP_L], lm[LM.UPPER_LIP_R]) / mW;
  // 12: lower-lip L/R distance
  f[12] = _d2(lm[LM.LOWER_LIP_L], lm[LM.LOWER_LIP_R]) / mW;

  // 13-14: reserved (zeros)
  // 15-19: reserved (zeros)

  return f;
}

/* Cosine similarity between two Float32Arrays */
function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

/* Mean of a Float32Array list */
function meanFeatures(frames) {
  if (frames.length === 0) return new Float32Array(20);
  const out = new Float32Array(20);
  for (const f of frames) for (let i = 0; i < 20; i++) out[i] += f[i];
  for (let i = 0; i < 20; i++) out[i] /= frames.length;
  return out;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1.  FacialGestureEngine — built-in lip-tap & blow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
class FacialGestureEngine {
  constructor(config = {}) {
    this.config = {
      scrollAmount            : config.scrollAmount            ?? SCROLL_AMOUNT,
      lipTapTimeWindow        : config.lipTapTimeWindow        ?? LIP_TAP_TIME_WINDOW,
      lipTapConfidence        : config.lipTapConfidence        ?? LIP_TAP_CONFIDENCE_THRESHOLD,
      blowThreshold           : config.blowThreshold           ?? BLOW_DETECTION_THRESHOLD,
      biteLipThreshold        : config.biteLipThreshold        ?? BITE_LIP_THRESHOLD,
      scrollIntervalMs        : config.scrollIntervalMs        ?? SCROLL_INTERVAL_MS,
      gestureCooldown         : config.gestureCooldown         ?? GESTURE_COOLDOWN,
      customGestureConfidence : config.customGestureConfidence ?? CUSTOM_GESTURE_CONFIDENCE,
    };

    this._callbacks = {};

    // ── Lip-tap state ──────────────────────────────────────────────
    // Two full lip closures within LIP_TAP_TIME_WINDOW
    this._lipOpen         = false;   // was mouth open last frame?
    this._lipClosureCount = 0;
    this._firstClosureT   = 0;
    this._minDwellFrames  = 2;       // must hold closed/open for ≥2 frames (false-trigger gate)
    this._closedFrames    = 0;
    this._openFrames      = 0;
    this._cycleComplete   = false;   // one open→closed→open complete

    // ── Bite-lip state ────────────────────────────────────────────────
    this._biteLipActive   = false;
    this._biteLipFrames   = 0;
    this._BITE_MIN_FRAMES = 3;
    this._biteLipInterval = null;
    this._lastBiteConf    = 0;

    // ── Cooldown ───────────────────────────────────────────────────
    this._lastFire = {};   // gestureName → timestamp

    // ── Baseline window for auto-calibration ──────────────────────
    this._baselineFrames = [];
    this._baselineSize   = 30;       // 1 s @ 30fps
    this._mouthBaseline  = null;     // average resting mouth-open ratio
  }

  on(event, cb) {
    if (!this._callbacks[event]) this._callbacks[event] = [];
    this._callbacks[event].push(cb);
    return this;
  }

  _emit(event, data) {
    (this._callbacks[event] || []).forEach(cb => cb(data));
  }

  _canFire(name) {
    const cd = this.config.gestureCooldown;
    const last = this._lastFire[name] || 0;
    return (performance.now() - last) >= cd;
  }

  _fired(name) {
    this._lastFire[name] = performance.now();
  }

  /**
   * Call every face frame with the 468+ FaceMesh landmarks array.
   * @param {Array} lm  MediaPipe face landmarks (normalised 0-1)
   */
  process(lm) {
    if (!lm || lm.length < 468) return;

    const f = extractFeatures(lm);

    // ── Auto-calibrate resting mouth ──────────────────────────────
    if (!this._mouthBaseline) {
      this._baselineFrames.push(f[1]);
      if (this._baselineFrames.length >= this._baselineSize) {
        this._mouthBaseline = this._baselineFrames.reduce((s, v) => s + v, 0) / this._baselineFrames.length;
        this._baselineFrames = [];
      }
      return; // don't detect until we have baseline
    }

    this._detectLipTap(f);
    this._detectBiteLip(f);
  }

  // ── Lip-tap: two full open→close→open cycles within LIP_TAP_TIME_WINDOW ──
  _detectLipTap(f) {
    const innerRatio = f[1];  // inner open ratio
    const baseline   = this._mouthBaseline;
    const CLOSE_THRESH = baseline + 0.03;  // closing = ratio near baseline
    const OPEN_THRESH  = baseline + 0.06;  // open = significantly above baseline

    const isClosed = innerRatio <= CLOSE_THRESH;
    const isOpen   = innerRatio >= OPEN_THRESH;

    // Confidence = how clearly closed the mouth is (0→1)
    const closeConf = isClosed
      ? Math.min(1, (CLOSE_THRESH - innerRatio + 0.05) / 0.05)
      : 0;

    if (!this._lipOpen && isOpen) {
      // Mouth opened
      this._openFrames++;
      if (this._openFrames >= this._minDwellFrames) {
        this._lipOpen    = true;
        this._closedFrames = 0;
      }
    } else if (this._lipOpen && isClosed && closeConf >= this.config.lipTapConfidence) {
      // Mouth closed while it was open
      this._closedFrames++;
      if (this._closedFrames >= this._minDwellFrames) {
        const now = performance.now();
        if (this._lipClosureCount === 0) {
          // First closure
          this._lipClosureCount = 1;
          this._firstClosureT   = now;
        } else if (this._lipClosureCount === 1 && now - this._firstClosureT <= this.config.lipTapTimeWindow) {
          // Second closure within window — double lip-tap!
          this._lipClosureCount = 0;
          if (this._canFire('lipTap')) {
            this._fired('lipTap');
            this._emit('lipTap', { confidence: closeConf, action: 'scrollUp' });
            this._emit('gesture', { name: 'lipTap', confidence: closeConf, action: 'scrollUp' });
          }
        } else {
          // Too slow — restart
          this._lipClosureCount = 1;
          this._firstClosureT   = now;
        }
        this._lipOpen     = false;
        this._openFrames  = 0;
      }
    } else if (!isOpen && !isClosed) {
      // Neutral — do nothing, let frames accumulate
    } else {
      this._openFrames   = 0;
      this._closedFrames = 0;
    }

    // Reset if window expired
    if (this._lipClosureCount > 0 && performance.now() - this._firstClosureT > this.config.lipTapTimeWindow + 200) {
      this._lipClosureCount = 0;
      this._firstClosureT   = 0;
    }
  }

  // ── Bite bottom lip: lower lip bitten inward → scroll down (held) ──
  // Detection: when the bottom lip is bitten, the inner mouth opening
  // (f[1] = innerRatio) collapses relative to the outer mouth opening (f[0]).
  // biteRatio = innerRatio / outerRatio → low when bitten, high when neutral.
  // Continuous scroll fires every scrollIntervalMs while the gesture is held.
  _detectBiteLip(f) {
    const outerRatio = f[0];
    const innerRatio = f[1];
    const baseline   = this._mouthBaseline || 0.05;

    const outerOpen = Math.max(outerRatio, baseline);
    const biteRatio = outerOpen > 0.01 ? innerRatio / outerOpen : 1.0;

    const threshold  = this.config.biteLipThreshold;
    const confidence = Math.min(1, Math.max(0, (threshold - biteRatio) / threshold));
    // Must have some mouth separation (not just closed) and low inner ratio
    const isBiting   = biteRatio < threshold && outerRatio > baseline * 0.8;

    if (isBiting) {
      this._biteLipFrames = Math.min(this._biteLipFrames + 1, this._BITE_MIN_FRAMES + 5);

      if (this._biteLipFrames >= this._BITE_MIN_FRAMES && !this._biteLipActive) {
        this._biteLipActive = true;
        this._lastBiteConf  = confidence;
        // Fire immediately
        this._emit('biteLip', { confidence, active: true, action: 'scrollDown' });
        this._emit('gesture', { name: 'biteLip', confidence, action: 'scrollDown' });
        // Continuous scroll while held
        this._biteLipInterval = setInterval(() => {
          this._emit('biteLip', { confidence: this._lastBiteConf, active: true, action: 'scrollDown' });
          this._emit('gesture', { name: 'biteLip', confidence: this._lastBiteConf, action: 'scrollDown' });
        }, this.config.scrollIntervalMs);
      }
      if (this._biteLipActive) this._lastBiteConf = confidence;
    } else {
      this._biteLipFrames = Math.max(0, this._biteLipFrames - 1);
      if (this._biteLipActive && this._biteLipFrames === 0) {
        this._biteLipActive = false;
        if (this._biteLipInterval) {
          clearInterval(this._biteLipInterval);
          this._biteLipInterval = null;
        }
        this._emit('biteLipRelease', { action: 'scrollDown' });
      }
    }
  }

  /** Reset all state (e.g. on camera restart) */
  reset() {
    this._lipOpen = false;
    this._lipClosureCount = 0;
    this._firstClosureT   = 0;
    this._closedFrames    = 0;
    this._openFrames      = 0;
    this._biteLipActive   = false;
    this._biteLipFrames   = 0;
    if (this._biteLipInterval) {
      clearInterval(this._biteLipInterval);
      this._biteLipInterval = null;
    }
    this._mouthBaseline   = null;
    this._baselineFrames  = [];
    this._lastFire        = {};
  }
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   2.  CustomGestureRecorder — captures a reference profile
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
class CustomGestureRecorder {
  /**
   * @param {Object} opts
   * @param {number} opts.minDuration  ms (default 2000)
   * @param {number} opts.maxDuration  ms (default 4000)
   * @param {Function} opts.onProgress called(progress 0-1, framesCollected)
   * @param {Function} opts.onComplete called(gesturePatternData)
   * @param {Function} opts.onCancel   called()
   */
  constructor(opts = {}) {
    this._minDuration  = opts.minDuration  ?? 2000;
    this._maxDuration  = opts.maxDuration  ?? 4000;
    this._onProgress   = opts.onProgress   || (() => {});
    this._onComplete   = opts.onComplete   || (() => {});
    this._onCancel     = opts.onCancel     || (() => {});

    this._recording    = false;
    this._frames       = [];
    this._startT       = 0;
    this._autoComplete = null;
  }

  get isRecording() { return this._recording; }

  /** Start capturing landmark frames */
  start() {
    this._recording = true;
    this._frames    = [];
    this._startT    = performance.now();
    // Auto-complete at maxDuration
    this._autoComplete = setTimeout(() => this.stop(), this._maxDuration);
  }

  /** Feed one frame of landmarks */
  feed(lm) {
    if (!this._recording || !lm || lm.length < 468) return;
    const f = extractFeatures(lm);
    this._frames.push(f);
    const elapsed  = performance.now() - this._startT;
    const progress = Math.min(1, elapsed / this._maxDuration);
    this._onProgress(progress, this._frames.length);
  }

  /** Stop recording and return the pattern data */
  stop() {
    if (!this._recording) return null;
    clearTimeout(this._autoComplete);
    this._recording = false;

    const elapsed = performance.now() - this._startT;
    if (elapsed < this._minDuration || this._frames.length < 20) {
      this._onCancel();
      return null;
    }

    const pattern = {
      version   : 1,
      frameCount: this._frames.length,
      durationMs: Math.round(elapsed),
      meanVector: Array.from(meanFeatures(this._frames)),
      // Store 5 percentile-sampled frames for richer matching
      samples   : this._sampleFrames(this._frames, 5).map(f => Array.from(f)),
      recordedAt: Date.now(),
    };

    this._onComplete(pattern);
    return pattern;
  }

  cancel() {
    clearTimeout(this._autoComplete);
    this._recording = false;
    this._frames    = [];
    this._onCancel();
  }

  _sampleFrames(frames, n) {
    const step = Math.max(1, Math.floor(frames.length / n));
    const out  = [];
    for (let i = 0; i < n && i * step < frames.length; i++) {
      out.push(frames[i * step]);
    }
    return out;
  }
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   3.  CustomGestureRecognizer — live matching against saved profiles
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
class CustomGestureRecognizer {
  /**
   * @param {number} windowMs   Rolling window for frame averaging (ms)
   * @param {number} confidence Minimum similarity to trigger (0-1)
   */
  constructor(windowMs = 600, confidence = CUSTOM_GESTURE_CONFIDENCE) {
    this._windowMs    = windowMs;
    this._confidence  = confidence;
    this._profiles    = [];       // [{id, name, pattern, action, threshold}]
    this._liveWindow  = [];       // {t, features}[]
    this._lastFire    = {};       // id → timestamp
    this._callbacks   = {};
    this._cooldown    = GESTURE_COOLDOWN;
  }

  on(event, cb) {
    if (!this._callbacks[event]) this._callbacks[event] = [];
    this._callbacks[event].push(cb);
    return this;
  }

  _emit(event, data) {
    (this._callbacks[event] || []).forEach(cb => cb(data));
  }

  /** Load profiles array (called by GestureStudio) */
  setProfiles(profiles) {
    this._profiles = profiles;
  }

  /** Process one frame — call every face frame */
  process(lm) {
    if (!lm || lm.length < 468 || this._profiles.length === 0) return;

    const t  = performance.now();
    const f  = extractFeatures(lm);

    // Maintain rolling window
    this._liveWindow.push({ t, f });
    while (this._liveWindow.length > 0 && t - this._liveWindow[0].t > this._windowMs) {
      this._liveWindow.shift();
    }
    if (this._liveWindow.length < 5) return; // not enough frames yet

    // Average live window
    const liveFeatures = meanFeatures(this._liveWindow.map(e => e.f));

    for (const profile of this._profiles) {
      const mean     = new Float32Array(profile.pattern.meanVector);
      const thresh   = profile.threshold ?? this._confidence;
      const sim      = cosineSim(liveFeatures, mean);

      if (sim >= thresh) {
        const last = this._lastFire[profile.id] || 0;
        if (t - last >= this._cooldown) {
          this._lastFire[profile.id] = t;
          this._emit('gesture', {
            id        : profile.id,
            name      : profile.gestureName,
            action    : profile.assignedAction,
            confidence: sim,
          });
        }
      }
    }
  }

  reset() {
    this._liveWindow = [];
    this._lastFire   = {};
  }
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   4.  GestureStudio — full lifecycle management
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

/**
 * Scroll the best available scrollable container.
 * Priority: focused element's scrollable ancestor → .demo-main → window
 * @param {number} delta  positive = down, negative = up
 */
function _scrollPage(delta) {
  // Walk up from active element to find a scrollable ancestor
  let el = document.activeElement;
  while (el && el !== document.body) {
    const st = getComputedStyle(el);
    if ((st.overflowY === 'auto' || st.overflowY === 'scroll') && el.scrollHeight > el.clientHeight) {
      el.scrollBy({ top: delta, behavior: 'smooth' });
      return;
    }
    el = el.parentElement;
  }
  // Fall back to the demo-main container (the primary scrollable pane in the demo)
  const demoMain = document.querySelector('.demo-main');
  if (demoMain && demoMain.scrollHeight > demoMain.clientHeight) {
    demoMain.scrollBy({ top: delta, behavior: 'smooth' });
    return;
  }
  // Last resort: window
  window.scrollBy({ top: delta, behavior: 'smooth' });
}

/** Available assignable actions */
const GESTURE_ACTIONS = {
  scrollUp       : { label: 'Scroll Up',           fn: () => _scrollPage(-SCROLL_AMOUNT) },
  scrollDown     : { label: 'Scroll Down',          fn: () => _scrollPage(SCROLL_AMOUNT) },
  click          : { label: 'Click / Activate',     fn: () => document.activeElement?.click() },
  doubleClick    : { label: 'Double Click',          fn: (el) => { el = el || document.activeElement; el?.click(); setTimeout(() => el?.click(), 80); } },
  navBack        : { label: 'Navigate Back',         fn: () => history.back() },
  navForward     : { label: 'Navigate Forward',      fn: () => history.forward() },
  openMenu       : { label: 'Open Menu',             fn: () => document.dispatchEvent(new CustomEvent('accesseye:openMenu')) },
  playPause      : { label: 'Play / Pause',          fn: () => { const v = document.querySelector('video,audio'); if(v) v.paused ? v.play() : v.pause(); } },
  customScript   : { label: 'Custom Script',         fn: (_, script) => { try { new Function(script)(); } catch(e) { console.warn('[GestureStudio] custom script error', e); } } },
};

class GestureStudio {
  constructor() {
    this._gestures         = [];     // array of gesture records
    this._recorder         = null;   // active CustomGestureRecorder
    this._recognizer       = new CustomGestureRecognizer();
    this._faceEngine       = new FacialGestureEngine();
    this._callbacks        = {};
    this._recordingId      = null;   // id being recorded/re-recorded
    this._biteLipActionFired = false; // prevents toast spam on continuous bite-lip

    this._load();
    this._syncRecognizer();
    this._wireBuiltins();

    console.log('[GestureStudio] Initialised with', this._gestures.length, 'saved gestures');
  }

  /* ── Public API ──────────────────────────────────────────── */

  /** Register callback for actions */
  onAction(cb) { return this.on('action', cb); }

  on(event, cb) {
    if (!this._callbacks[event]) this._callbacks[event] = [];
    this._callbacks[event].push(cb);
    return this;
  }

  _emit(event, data) {
    (this._callbacks[event] || []).forEach(cb => cb(data));
  }

  /** Feed face landmarks every frame (call from app.js) */
  processFaceLandmarks(lm) {
    this._faceEngine.process(lm);
    if (!this._recorder?.isRecording) {
      this._recognizer.process(lm);
    } else {
      this._recorder.feed(lm);
    }
  }

  /** Reset engine state (e.g. camera restart) */
  reset() {
    this._faceEngine.reset();
    this._recognizer.reset();
    if (this._recorder?.isRecording) this._recorder.cancel();
  }

  /* ── Gesture CRUD ────────────────────────────────────────── */

  /** Return a copy of all saved gestures */
  getGestures() {
    return this._gestures.map(g => ({ ...g }));
  }

  /**
   * Create a new empty gesture entry.
   * @returns {string} id
   */
  create(gestureName, assignedAction = 'scrollDown', confidenceThreshold = CUSTOM_GESTURE_CONFIDENCE) {
    const id = 'gs_' + Date.now() + '_' + Math.random().toString(36).slice(2, 7);
    this._gestures.push({
      id,
      gestureName,
      assignedAction,
      confidenceThreshold,
      gesturePatternData : null,   // filled after recording
      createdAt          : Date.now(),
    });
    this._save();
    this._emit('gesturesChanged', this.getGestures());
    return id;
  }

  /** Rename a gesture */
  rename(id, newName) {
    const g = this._byId(id);
    if (!g) return false;
    g.gestureName = newName;
    this._save();
    this._emit('gesturesChanged', this.getGestures());
    return true;
  }

  /** Change assigned action */
  setAction(id, actionId) {
    const g = this._byId(id);
    if (!g || !GESTURE_ACTIONS[actionId]) return false;
    g.assignedAction = actionId;
    this._save();
    this._emit('gesturesChanged', this.getGestures());
    return true;
  }

  /** Change confidence threshold */
  setThreshold(id, threshold) {
    const g = this._byId(id);
    if (!g) return false;
    g.confidenceThreshold = Math.max(0.4, Math.min(0.99, threshold));
    this._syncRecognizer();
    this._save();
    return true;
  }

  /** Delete a gesture */
  delete(id) {
    const idx = this._gestures.findIndex(g => g.id === id);
    if (idx === -1) return false;
    this._gestures.splice(idx, 1);
    this._syncRecognizer();
    this._save();
    this._emit('gesturesChanged', this.getGestures());
    return true;
  }

  /* ── Recording ───────────────────────────────────────────── */

  /**
   * Start recording for gesture `id`.
   * @param {string} id
   * @param {Function} onProgress (progress 0-1, frameCount)
   * @param {Function} onDone     (success: boolean)
   */
  startRecording(id, onProgress, onDone) {
    const g = this._byId(id);
    if (!g) { onDone?.(false); return; }

    this._recordingId = id;
    this._recorder    = new CustomGestureRecorder({
      onProgress: (p, n) => onProgress?.(p, n),
      onComplete: (pattern) => {
        g.gesturePatternData = pattern;
        this._recordingId = null;
        this._recorder    = null;
        this._syncRecognizer();
        this._save();
        this._emit('gesturesChanged', this.getGestures());
        onDone?.(true, pattern);
      },
      onCancel: () => {
        this._recordingId = null;
        this._recorder    = null;
        onDone?.(false);
      },
    });
    this._recorder.start();
    this._emit('recordingStarted', { id });
  }

  /** Stop an active recording early */
  stopRecording() {
    if (this._recorder?.isRecording) {
      this._recorder.stop();
    }
  }

  cancelRecording() {
    if (this._recorder?.isRecording) {
      this._recorder.cancel();
    }
  }

  get isRecording() { return !!(this._recorder?.isRecording); }
  get recordingId() { return this._recordingId; }

  /* ── Action execution ────────────────────────────────────── */

  _executeAction(actionId, gestureName, customScript) {
    const action = GESTURE_ACTIONS[actionId];
    if (!action) return;
    try {
      action.fn(document.activeElement, customScript);
    } catch (e) {
      console.warn('[GestureStudio] action error', e);
    }
    this._emit('action', { actionId, gestureName, label: action.label });
  }

  /* ── Internal helpers ────────────────────────────────────── */

  _byId(id) { return this._gestures.find(g => g.id === id); }

  _syncRecognizer() {
    const withPattern = this._gestures
      .filter(g => g.gesturePatternData)
      .map(g => ({
        id             : g.id,
        gestureName    : g.gestureName,
        assignedAction : g.assignedAction,
        pattern        : g.gesturePatternData,
        threshold      : g.confidenceThreshold,
      }));
    this._recognizer.setProfiles(withPattern);
  }

  _wireBuiltins() {
    // ── Lip-tap (double) → scroll up ────────────────────────────────
    this._faceEngine.on('lipTap', ({ confidence }) => {
      _scrollPage(-SCROLL_AMOUNT);
      this._emit('action', { actionId: 'scrollUp', gestureName: 'Double Lip-Tap', label: 'Scroll Up', confidence });
      this._emit('builtinGesture', { name: 'lipTap', confidence, action: 'scrollUp' });
    });

    // ── Bite bottom lip (held) → scroll down continuously ───────────
    this._faceEngine.on('biteLip', ({ confidence }) => {
      _scrollPage(SCROLL_AMOUNT);
      // Emit action only once per gesture onset to avoid toast spam
      if (!this._biteLipActionFired) {
        this._biteLipActionFired = true;
        this._emit('action', { actionId: 'scrollDown', gestureName: 'Bite Bottom Lip', label: 'Scroll Down', confidence });
        this._emit('builtinGesture', { name: 'biteLip', confidence, action: 'scrollDown' });
      }
    });
    this._faceEngine.on('biteLipRelease', () => {
      this._biteLipActionFired = false;
    });

    // ── Custom gestures ──────────────────────────────────────────────
    // Recognizer emits { id, name, action, confidence }
    // where action = the assignedAction string key into GESTURE_ACTIONS
    this._recognizer.on('gesture', ({ id, name, action, confidence }) => {
      const actionId = action || 'scrollDown';  // fallback
      this._executeAction(actionId, name);
      this._emit('builtinGesture', { name, confidence, action: actionId });
    });
  }

  _save() {
    try {
      localStorage.setItem(STUDIO_STORAGE_KEY, JSON.stringify(this._gestures));
    } catch (_) {}
  }

  _load() {
    try {
      const raw = localStorage.getItem(STUDIO_STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) this._gestures = parsed;
      }
    } catch (_) {}
  }

  /** Expose available action list for UI dropdowns */
  static getAvailableActions() {
    return Object.entries(GESTURE_ACTIONS).map(([id, v]) => ({ id, label: v.label }));
  }
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   5.  GestureStudioUI — minimal controller (wires the HTML panel)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
class GestureStudioUI {
  /**
   * @param {GestureStudio} studio
   * @param {string}        rootId   id of the root panel element in the DOM
   */
  constructor(studio, rootId = 'gesture-studio-panel') {
    this._studio     = studio;
    this._root       = null;
    this._rootId     = rootId;
    this._editingId  = null;
    this._recInterval = null;

    // Re-render whenever gestures change
    studio.on('gesturesChanged', () => this._render());

    // Recording progress
    studio.on('recordingStarted', () => this._renderRecordingState(true));

    // Builtin gesture feedback
    studio.on('builtinGesture', ({ name, confidence, action }) => {
      let gLabel = name === 'lipTap' ? '👄 Double Lip-Tap' : name === 'biteLip' ? '😬 Bite Lip' : name;
      this._showFeedback(`${gLabel}: ${action} (conf ${(confidence * 100).toFixed(0)}%)`, 'success');
    });

    // Custom gesture feedback
    studio.on('customGesture', ({ name, action, confidence }) => {
      this._showFeedback(`🎯 ${name} → ${action} (${(confidence * 100).toFixed(0)}%)`, 'success');
    });
  }

  /** Call once after DOM is ready */
  init() {
    this._root = document.getElementById(this._rootId);
    if (!this._root) return;
    this._render();
  }

  _render() {
    if (!this._root) return;
    const gestures = this._studio.getGestures();
    const actions  = GestureStudio.getAvailableActions();

    this._root.innerHTML = `
<div class="gs-panel">
  <!-- Built-in gestures info -->
  <div class="gs-section">
    <div class="gs-section-title"><i class="fas fa-bolt"></i> Built-In Gestures</div>
    <div class="gs-builtin-row">
      <div class="gs-builtin-card">
        <div class="gs-builtin-icon">👄</div>
        <div class="gs-builtin-info">
          <div class="gs-builtin-name">Double Lip-Tap</div>
          <div class="gs-builtin-desc">Two lip closures within 750 ms</div>
          <div class="gs-builtin-action"><i class="fas fa-arrow-up"></i> Scroll Up</div>
        </div>
      </div>
      <div class="gs-builtin-card">
        <div class="gs-builtin-icon">😬</div>
        <div class="gs-builtin-info">
          <div class="gs-builtin-name">Bite Bottom Lip</div>
          <div class="gs-builtin-desc">Bite lower lip — hold to scroll, release to stop</div>
          <div class="gs-builtin-action"><i class="fas fa-arrow-down"></i> Scroll Down (held)</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Custom gestures list -->
  <div class="gs-section">
    <div class="gs-section-header">
      <div class="gs-section-title"><i class="fas fa-magic"></i> Custom Gestures <span class="gs-count">${gestures.length}</span></div>
      <button class="gs-btn gs-btn-primary" id="gs-create-btn"><i class="fas fa-plus"></i> New Gesture</button>
    </div>

    ${gestures.length === 0 ? `
      <div class="gs-empty">
        <i class="fas fa-hand-paper"></i>
        <p>No custom gestures yet.<br/>Click <strong>New Gesture</strong> to create one.</p>
      </div>
    ` : `
      <div class="gs-list">
        ${gestures.map(g => this._renderGestureCard(g, actions)).join('')}
      </div>
    `}
  </div>

  <!-- Feedback area -->
  <div id="gs-feedback" class="gs-feedback" style="display:none"></div>
</div>`;

    // Create button
    this._root.querySelector('#gs-create-btn')?.addEventListener('click', () => {
      this._showCreateDialog(actions);
    });

    // Wire gesture card buttons
    gestures.forEach(g => {
      this._wireCard(g, actions);
    });
  }

  _renderGestureCard(g, actions) {
    const hasPattern = !!g.gesturePatternData;
    const actionLabel = actions.find(a => a.id === g.assignedAction)?.label || g.assignedAction;
    const statusClass = hasPattern ? 'gs-status-ready' : 'gs-status-unrecorded';
    const statusText  = hasPattern ? 'Ready' : 'Needs Recording';

    return `
<div class="gs-card" data-id="${g.id}">
  <div class="gs-card-header">
    <div class="gs-card-title">
      <span class="gs-card-name">${this._esc(g.gestureName)}</span>
      <span class="gs-status ${statusClass}">${statusText}</span>
    </div>
    <div class="gs-card-actions">
      <button class="gs-btn gs-btn-sm gs-btn-record ${g.id === this._studio.recordingId ? 'active' : ''}"
              data-action="record" data-id="${g.id}" title="Record gesture">
        <i class="fas fa-circle"></i> ${g.id === this._studio.recordingId ? 'Stop' : 'Record'}
      </button>
      <button class="gs-btn gs-btn-sm" data-action="rename" data-id="${g.id}" title="Rename">
        <i class="fas fa-pen"></i>
      </button>
      <button class="gs-btn gs-btn-sm gs-btn-danger" data-action="delete" data-id="${g.id}" title="Delete">
        <i class="fas fa-trash"></i>
      </button>
    </div>
  </div>
  <div class="gs-card-body">
    <label class="gs-label">Action:</label>
    <select class="gs-select" data-action="setAction" data-id="${g.id}">
      ${actions.map(a => `<option value="${a.id}" ${a.id === g.assignedAction ? 'selected' : ''}>${a.label}</option>`).join('')}
    </select>
    <label class="gs-label">Confidence: <span class="gs-thresh-val">${Math.round(g.confidenceThreshold * 100)}%</span></label>
    <input type="range" class="gs-slider" data-action="setThreshold" data-id="${g.id}"
           min="40" max="99" value="${Math.round(g.confidenceThreshold * 100)}">
  </div>
  ${hasPattern ? `<div class="gs-card-meta">
    <i class="fas fa-database"></i> ${g.gesturePatternData.frameCount} frames
    · ${g.gesturePatternData.durationMs}ms
    · recorded ${this._relTime(g.gesturePatternData.recordedAt)}
  </div>` : ''}
  <!-- Recording progress bar (hidden by default) -->
  <div class="gs-rec-progress" id="gs-rec-progress-${g.id}" style="display:none">
    <div class="gs-rec-bar"><div class="gs-rec-fill" id="gs-rec-fill-${g.id}"></div></div>
    <div class="gs-rec-label" id="gs-rec-label-${g.id}">Hold your gesture…</div>
  </div>
</div>`;
  }

  _wireCard(g, actions) {
    const root = this._root;

    root.querySelector(`[data-action="record"][data-id="${g.id}"]`)?.addEventListener('click', () => {
      if (this._studio.isRecording) {
        this._studio.stopRecording();
        return;
      }
      this._startRecording(g.id);
    });

    root.querySelector(`[data-action="rename"][data-id="${g.id}"]`)?.addEventListener('click', () => {
      const newName = prompt('Rename gesture:', g.gestureName);
      if (newName?.trim()) {
        this._studio.rename(g.id, newName.trim());
      }
    });

    root.querySelector(`[data-action="delete"][data-id="${g.id}"]`)?.addEventListener('click', () => {
      if (confirm(`Delete "${g.gestureName}"?`)) {
        this._studio.delete(g.id);
      }
    });

    root.querySelector(`[data-action="setAction"][data-id="${g.id}"]`)?.addEventListener('change', (e) => {
      this._studio.setAction(g.id, e.target.value);
    });

    root.querySelector(`[data-action="setThreshold"][data-id="${g.id}"]`)?.addEventListener('input', (e) => {
      const val = parseInt(e.target.value) / 100;
      this._studio.setThreshold(g.id, val);
      const valEl = e.target.closest('.gs-card-body')?.querySelector('.gs-thresh-val');
      if (valEl) valEl.textContent = `${Math.round(val * 100)}%`;
    });
  }

  _startRecording(id) {
    const progressEl = this._root?.querySelector(`#gs-rec-progress-${id}`);
    const fillEl     = this._root?.querySelector(`#gs-rec-fill-${id}`);
    const labelEl    = this._root?.querySelector(`#gs-rec-label-${id}`);
    if (progressEl) progressEl.style.display = 'block';

    this._studio.startRecording(
      id,
      (progress, frames) => {
        if (fillEl) fillEl.style.width = `${progress * 100}%`;
        if (labelEl) labelEl.textContent = `Recording… ${Math.round(progress * 100)}% (${frames} frames)`;
      },
      (success, pattern) => {
        if (progressEl) progressEl.style.display = 'none';
        if (success) {
          this._showFeedback(`✅ Gesture recorded (${pattern.frameCount} frames, ${pattern.durationMs}ms)`, 'success');
        } else {
          this._showFeedback('❌ Recording too short — hold the gesture longer (min 2 s)', 'error');
        }
        this._render();
      }
    );

    // Re-render to flip button to "Stop"
    this._render();
  }

  _showCreateDialog(actions) {
    const name = prompt('Name your new gesture (e.g. "Brow Raise"):');
    if (!name?.trim()) return;
    const actionList = actions.map((a, i) => `${i + 1}. ${a.label}`).join('\n');
    const choice = prompt(`Assign an action:\n${actionList}\n\nEnter number (default 2 = Scroll Down):`);
    const idx    = parseInt(choice) - 1;
    const action = actions[isNaN(idx) || idx < 0 || idx >= actions.length ? 1 : idx];
    const id     = this._studio.create(name.trim(), action.id);
    this._showFeedback(`Gesture "${name.trim()}" created. Click Record to capture it.`, 'info');
  }

  _showFeedback(msg, type = 'info') {
    const el = this._root?.querySelector('#gs-feedback');
    if (!el) return;
    el.textContent = msg;
    el.className   = `gs-feedback gs-feedback-${type}`;
    el.style.display = 'block';
    clearTimeout(this._fbTimer);
    this._fbTimer = setTimeout(() => { el.style.display = 'none'; }, 3500);
  }

  _renderRecordingState(active) {
    // handled via _render()
  }

  _esc(s) { return s.replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

  _relTime(ts) {
    const diff = Date.now() - ts;
    if (diff < 60000)  return 'just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    return `${Math.floor(diff / 3600000)}h ago`;
  }
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Export to global scope
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
window.GestureStudio          = GestureStudio;
window.GestureStudioUI        = GestureStudioUI;
window.FacialGestureEngine    = FacialGestureEngine;
window.CustomGestureRecorder  = CustomGestureRecorder;
window.CustomGestureRecognizer = CustomGestureRecognizer;
window.GESTURE_ACTIONS        = GESTURE_ACTIONS;

console.log('[AccessEye] Gesture Studio loaded ✅');
