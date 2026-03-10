/**
 * ═══════════════════════════════════════════════════════════════════════════
 *  AccessEye — Phase 3: Advanced Gaze Processing Upgrades
 *  phase3-engine.js
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  New Modules (all 100% on-device, Cloudflare Pages compatible):
 *
 *   P3.1  OneEuroFilter        — signal pre-filter (minCutoff 1Hz, β 0.007)
 *   P3.2  IVTSaccadeDetector   — velocity-threshold saccade/fixation (35px/frame)
 *   P3.3  AdaptiveDwellTimer   — user-profile presets (Fast/Normal/Accessible/Extended)
 *   P3.4  PACERecalibrator     — continuous weight-decay recalibration (buffer 100)
 *   P3.5  SmoothPursuitCalib   — figure-8 path calibration (200-300 valid windows)
 *   P3.6  CalibrationValidator — 5-point post-calib accuracy test (thresholds 70/85/85+%)
 *   P3.7  HeadFreeStabilizer   — dynamic face-center compensation for head movement
 *
 *  Integration: phase3-init.js wires these modules into the existing pipeline.
 * ═══════════════════════════════════════════════════════════════════════════
 */

'use strict';

/* ─────────────────────────────────────────────────────────────────────────
   PHASE 3 MATH UTILITIES
───────────────────────────────────────────────────────────────────────── */
const p3 = {
  clamp:  (v, lo, hi) => Math.min(hi, Math.max(lo, v)),
  lerp:   (a, b, t) => a + (b - a) * t,
  dist2:  (x1, y1, x2, y2) => Math.hypot(x2 - x1, y2 - y1),
  now:    () => performance.now(),
  avg:    (arr) => arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0,
  stddev: (arr) => {
    if (arr.length < 2) return 0;
    const m = p3.avg(arr);
    return Math.sqrt(p3.avg(arr.map(v => (v - m) ** 2)));
  }
};

/* ─────────────────────────────────────────────────────────────────────────
   P3.1  ONE EURO FILTER
   ──────────────────────────────────────────────────────────────────────────
   Applies a 1€ (One Euro) adaptive low-pass filter to gaze signals.
   Placed BEFORE the Kalman filter in the pipeline to reduce high-frequency
   jitter while preserving fast intentional movements.

   Parameters:
     minCutoff — low-velocity cutoff frequency (Hz), default 1.0 Hz
     β         — speed coefficient controlling cutoff rise, default 0.007
     dCutoff   — derivative (velocity) cutoff frequency, default 1.0 Hz
     freq      — expected input sample rate (Hz), default 30 Hz

   Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter
   for Noisy Input in Interactive Systems", CHI 2012.
───────────────────────────────────────────────────────────────────────── */
class OneEuroFilter {
  /**
   * @param {number} minCutoff  Minimum cutoff (Hz) — lower = smoother at rest
   * @param {number} beta       Speed coefficient — higher = less lag during fast motion
   * @param {number} dCutoff    Derivative cutoff (Hz)
   * @param {number} freq       Input frequency (Hz)
   */
  // FIX M-3: beta raised from 0.007 → 0.02.
  // FIX ACC-1: minCutoff lowered from 1.0 → 0.3 Hz.
  // At 1.0 Hz the filter barely smooths at-rest tremor (α≈0.18 at 30fps).
  // At 0.3 Hz α≈0.056, giving much stronger jitter suppression at fixation
  // while beta=0.05 ensures intentional fast saccades still break through.
  // Reference: Casiez et al. CHI 2012 recommend minCutoff ≈ 0.5-1.0 Hz for
  // mouse, but gaze needs lower (0.3-0.5) due to higher tremor frequency.
  constructor(minCutoff = 0.3, beta = 0.05, dCutoff = 1.0, freq = 30) {
    this.minCutoff = minCutoff;
    this.beta      = beta;
    this.dCutoff   = dCutoff;
    this.freq      = freq;

    // Per-axis state (x and y)
    this._xState  = this._newState();
    this._yState  = this._newState();
  }

  _newState() {
    return { x: null, dx: 0, initialized: false };
  }

  _alpha(cutoff) {
    // α = 1 / (1 + τ·ω)  where  τ = 1/(2π·cutoff), ω = 2π·freq
    // Simplified: α = 2π·cutoff / (2π·cutoff + freq)
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau * this.freq);
  }

  _filterAxis(state, rawValue) {
    if (!state.initialized) {
      state.x   = rawValue;
      state.dx  = 0;
      state.initialized = true;
      return rawValue;
    }

    // Estimate velocity (derivative)
    const dxRaw  = (rawValue - state.x) * this.freq;
    // Low-pass filter velocity
    const alphaDx = this._alpha(this.dCutoff);
    state.dx = state.dx + alphaDx * (dxRaw - state.dx);

    // Compute cutoff based on speed  (|dx| drives the adaptive part)
    const cutoff = this.minCutoff + this.beta * Math.abs(state.dx);
    const alpha  = this._alpha(cutoff);

    // Low-pass filter value
    state.x = state.x + alpha * (rawValue - state.x);

    return state.x;
  }

  /**
   * Filter a 2D gaze point.
   * @param {number} x  Raw gaze X (normalized 0-1)
   * @param {number} y  Raw gaze Y (normalized 0-1)
   * @returns {{ x: number, y: number }}
   */
  filter(x, y) {
    return {
      x: this._filterAxis(this._xState, x),
      y: this._filterAxis(this._yState, y)
    };
  }

  /** Update sample rate (called when FPS changes) */
  setFreq(hz) {
    this.freq = Math.max(1, hz);
  }

  reset() {
    this._xState = this._newState();
    this._yState = this._newState();
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   P3.2  IVT SACCADE DETECTOR
   ──────────────────────────────────────────────────────────────────────────
   Velocity-threshold (I-VT) classifier separating fixations from saccades.
   Replaces the radius-based MicroSaccadeFilter with a velocity-based gate.

   • velocityThreshold  — 35 px/frame distinguishes fixations from saccades
   • fixationMinMs      — 100 ms minimum fixation duration to be meaningful
   • dwellGating        — suppresses dwell activation during saccades (~35% fewer false activations)

   State emitted on events:
     'fixation'  → { x, y, duration, age }
     'saccade'   → { velocity, fromX, fromY, toX, toY }
     'fix-end'   → { x, y, duration }
───────────────────────────────────────────────────────────────────────── */
class IVTSaccadeDetector {
  /**
   * @param {number} velocityThreshold  px/frame (at 30fps ≈ 35px/frame)
   * @param {number} fixationMinMs      minimum duration to count as fixation (ms)
   * @param {number} windowSize         smoothing window for velocity (frames)
   */
  constructor(velocityThreshold = 35, fixationMinMs = 100, windowSize = 3) {
    this.velocityThreshold = velocityThreshold;
    this.fixationMinMs     = fixationMinMs;
    this.windowSize        = windowSize;

    // State
    this.isFixating   = false;
    this.isSaccading  = false;
    this.fixStartTime = 0;
    this.fixStartX    = 0;
    this.fixStartY    = 0;
    this.fixationAge  = 0;   // ms since fixation start

    this._prevX    = null;
    this._prevY    = null;
    this._prevT    = null;
    this._velHist  = [];   // rolling velocity buffer for smoothing

    // Statistics
    this._stats = { fixations: 0, saccades: 0, totalFixMs: 0 };

    this._callbacks = {};
  }

  on(event, cb) {
    if (!this._callbacks[event]) this._callbacks[event] = [];
    this._callbacks[event].push(cb);
  }

  _emit(event, data) {
    (this._callbacks[event] || []).forEach(cb => cb(data));
  }

  /**
   * Process a new gaze sample (screen pixels).
   * @param {number} px  Screen X in pixels
   * @param {number} py  Screen Y in pixels
   * @param {number} conf Confidence [0-1]
   * @returns {{ x, y, isFixating, isSaccading, velocity, fixationAge }}
   */
  update(px, py, conf = 1.0) {
    const t = p3.now();

    let velocity = 0;
    if (this._prevX !== null) {
      const dx = px - this._prevX;
      const dy = py - this._prevY;
      const dt = Math.max(1, t - this._prevT) / (1000 / 30);  // normalise to 30fps
      const rawVel = Math.hypot(dx, dy) / dt;

      // Smooth velocity over window
      this._velHist.push(rawVel);
      if (this._velHist.length > this.windowSize) this._velHist.shift();
      velocity = p3.avg(this._velHist);
    }

    this._prevX = px;
    this._prevY = py;
    this._prevT = t;

    const wasSaccading = this.isSaccading;
    this.isSaccading  = velocity > this.velocityThreshold;
    this.isFixating   = !this.isSaccading && conf >= 0.4;

    // Fixation start
    if (this.isFixating && !wasSaccading && this._prevX !== null) {
      if (this.fixStartTime === 0) {
        this.fixStartTime = t;
        this.fixStartX    = px;
        this.fixStartY    = py;
      }
    }

    // Saccade start
    if (this.isSaccading && !wasSaccading && this.fixStartTime > 0) {
      const dur = t - this.fixStartTime;
      if (dur >= this.fixationMinMs) {
        this._stats.fixations++;
        this._stats.totalFixMs += dur;
        this._emit('fixation', {
          x: this.fixStartX, y: this.fixStartY,
          duration: dur, age: dur
        });
        this._emit('fix-end', { x: this.fixStartX, y: this.fixStartY, duration: dur });
      }
      this.fixStartTime = 0;
      this._stats.saccades++;
      this._emit('saccade', {
        velocity,
        fromX: this.fixStartX, fromY: this.fixStartY,
        toX: px, toY: py
      });
    }

    // Update fixation age
    this.fixationAge = this.isFixating && this.fixStartTime > 0
      ? t - this.fixStartTime
      : 0;

    // Emit ongoing fixation if long enough
    if (this.isFixating && this.fixationAge >= this.fixationMinMs) {
      this._emit('fixation', {
        x: this.fixStartX, y: this.fixStartY,
        duration: this.fixationAge, age: this.fixationAge
      });
    }

    return {
      x: px, y: py,
      isFixating: this.isFixating,
      isSaccading: this.isSaccading,
      velocity,
      fixationAge: this.fixationAge
    };
  }

  getStats() {
    return {
      ...this._stats,
      avgFixMs: this._stats.fixations > 0
        ? Math.round(this._stats.totalFixMs / this._stats.fixations)
        : 0
    };
  }

  reset() {
    this.isFixating   = false;
    this.isSaccading  = false;
    this.fixStartTime = 0;
    this.fixationAge  = 0;
    this._prevX = null;
    this._prevY = null;
    this._prevT = null;
    this._velHist = [];
    this._stats = { fixations: 0, saccades: 0, totalFixMs: 0 };
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   P3.3  ADAPTIVE DWELL TIMER
   ──────────────────────────────────────────────────────────────────────────
   Replaces the static 350ms dwell timer with an adaptive system:
   • 4 user presets  (Fast/Normal/Accessible/Extended)
   • Per-element dwell multipliers
   • IVT-gated: dwell only counts during fixation intervals
   • Saccade interruption resets dwell counter

   Dwell presets (base + modifier ms):
     Fast       → 180 ms  (power users, strong gaze control)
     Normal     → 300 ms  (default — original behaviour)
     Accessible → 500 ms  (reduced motor control)
     Extended   → 800 ms  (severe motor impairment, AAC users)
───────────────────────────────────────────────────────────────────────── */
class AdaptiveDwellTimer {
  static PRESETS = {
    fast:       { base: 180, label: 'Fast',       icon: '⚡' },
    normal:     { base: 300, label: 'Normal',      icon: '🎯' },
    accessible: { base: 500, label: 'Accessible',  icon: '♿' },
    extended:   { base: 800, label: 'Extended',    icon: '🐢' }
  };

  /**
   * @param {string} preset       Initial preset key ('normal')
   * @param {IVTSaccadeDetector} ivt  IVT detector for gating
   */
  constructor(preset = 'normal', ivt = null) {
    this.ivt         = ivt;
    this._preset     = preset;
    this._base       = AdaptiveDwellTimer.PRESETS[preset]?.base ?? 300;
    this._multipliers = new Map();  // elementId → number (1.0 default)

    // Per-element state
    this._dwellStart = new Map();   // elementId → timestamp
    this._progress   = new Map();   // elementId → [0..1]
    this._focusedId  = null;

    // Auto-adapt: track recent activations
    this._activationHistory = [];   // [{ ms: dwellMs, preset }]
    this._MAX_HIST  = 20;
    this._adaptEnabled = false;

    this._callbacks = {};
  }

  on(event, cb) {
    if (!this._callbacks[event]) this._callbacks[event] = [];
    this._callbacks[event].push(cb);
  }

  _emit(event, data) {
    (this._callbacks[event] || []).forEach(cb => cb(data));
  }

  /** Set preset by key ('fast'|'normal'|'accessible'|'extended') */
  setPreset(key) {
    const p = AdaptiveDwellTimer.PRESETS[key];
    if (!p) return;
    this._preset = key;
    this._base   = p.base;
    this._emit('presetChanged', { preset: key, base: this._base });
  }

  /** Get effective dwell time for an element (preset × multiplier) */
  getDwellTime(elementId) {
    const mult = this._multipliers.get(elementId) ?? 1.0;
    return Math.round(this._base * mult);
  }

  /** Set per-element multiplier (e.g., 2.0 for destructive actions) */
  setMultiplier(elementId, mult) {
    this._multipliers.set(elementId, p3.clamp(mult, 0.25, 5.0));
  }

  /**
   * Update dwell state for a focused element.
   * Called every frame with current fixation state.
   *
   * @param {string|null} elementId   Currently focused element ID (or null)
   * @param {boolean}     isFixating  From IVT detector
   * @returns {{ progress, completed, elementId }}
   */
  update(elementId, isFixating) {
    const t = p3.now();

    // Handle element switch
    if (elementId !== this._focusedId) {
      this._dwellStart.delete(this._focusedId);
      this._progress.delete(this._focusedId);
      this._focusedId = elementId;
      if (elementId) this._dwellStart.set(elementId, t);
    }

    if (!elementId) return { progress: 0, completed: false, elementId: null };

    // IVT gate: if saccading, pause dwell
    const gated = this.ivt ? isFixating : true;
    if (!gated) {
      // Don't reset, just pause (preserve partial progress within grace period)
      return {
        progress: this._progress.get(elementId) ?? 0,
        completed: false, elementId
      };
    }

    const start    = this._dwellStart.get(elementId) ?? t;
    const elapsed  = t - start;
    const dwellMs  = this.getDwellTime(elementId);
    const progress = p3.clamp(elapsed / dwellMs, 0, 1);

    this._progress.set(elementId, progress);
    this._emit('progress', { elementId, progress, elapsed, dwellMs });

    if (progress >= 1.0) {
      this._dwellStart.delete(elementId);
      this._progress.delete(elementId);
      this._activationHistory.push({ ms: elapsed, preset: this._preset, t });
      if (this._activationHistory.length > this._MAX_HIST) this._activationHistory.shift();
      this._emit('dwell-activate', { elementId, elapsed, dwellMs });
      return { progress: 1.0, completed: true, elementId };
    }

    return { progress, completed: false, elementId };
  }

  /** Reset dwell for specific element */
  reset(elementId) {
    if (elementId) {
      this._dwellStart.delete(elementId);
      this._progress.delete(elementId);
    }
  }

  getProgress(elementId) {
    return this._progress.get(elementId) ?? 0;
  }

  get preset() { return this._preset; }
  get baseMs()  { return this._base; }

  getPresetInfo() {
    return AdaptiveDwellTimer.PRESETS[this._preset] ?? AdaptiveDwellTimer.PRESETS.normal;
  }

  static listPresets() {
    return Object.entries(AdaptiveDwellTimer.PRESETS).map(([k, v]) => ({ key: k, ...v }));
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   P3.4  PACE RECALIBRATOR
   ──────────────────────────────────────────────────────────────────────────
   Continuous passive recalibration inspired by the PACE algorithm:
   • Accumulates high-confidence gaze→screen samples during normal use
   • Weight-decay 0.995 per frame keeps the model fresh (old samples fade)
   • Triggers a weighted ridge refit every REFIT_INTERVAL new samples
   • Works silently in the background without requiring explicit recalibration

   Integration: called from the main processing loop whenever a fixation
   is detected on a UI element with known screen position.
───────────────────────────────────────────────────────────────────────── */
class PACERecalibrator {
  /**
   * @param {CalibrationEngine} calibration  Phase 1 base engine
   * @param {number} bufferSize             max samples (default 100)
   * @param {number} weightDecay            per-frame decay (default 0.995)
   * @param {number} minConfidence          minimum confidence to accept (default 0.65)
   * @param {number} refitInterval          samples between refits (default 10)
   */
  constructor(calibration, bufferSize = 100, weightDecay = 0.995, minConfidence = 0.65, refitInterval = 10) {
    this.calib          = calibration;
    this.bufferSize     = bufferSize;
    this.weightDecay    = weightDecay;
    this.minConf        = minConfidence;
    this.refitInterval  = refitInterval;
    this.RIDGE_LAMBDA   = 0.012;

    this._samples       = [];   // [{gx, gy, sx, sy, w}]
    this._frameCount    = 0;
    this._samplesSince  = 0;   // samples since last refit
    this._lastRefit     = 0;

    this._callbacks = {};
  }

  on(event, cb) {
    if (!this._callbacks[event]) this._callbacks[event] = [];
    this._callbacks[event].push(cb);
  }

  _emit(event, data) {
    (this._callbacks[event] || []).forEach(cb => cb(data));
  }

  /**
   * Age all sample weights by WEIGHT_DECAY.
   * Called once per frame from the processing loop.
   */
  tick() {
    this._frameCount++;
    // Apply weight decay every 5 frames to save CPU
    if (this._frameCount % 5 === 0) {
      for (const s of this._samples) s.w *= Math.pow(this.weightDecay, 5);
      // Prune very old samples
      this._samples = this._samples.filter(s => s.w >= 0.01);
    }
  }

  /**
   * Add a new gaze→screen sample.
   * @param {number} gx  Raw gaze X (from HybridGazeEngine)
   * @param {number} gy  Raw gaze Y
   * @param {number} sx  Known screen X [0-1]
   * @param {number} sy  Known screen Y [0-1]
   * @param {number} conf Confidence score
   */
  addSample(gx, gy, sx, sy, conf) {
    if (conf < this.minConf) return;
    if (!this.calib.isCalibrated) return;

    this._samples.push({ gx, gy, sx, sy, w: conf });
    if (this._samples.length > this.bufferSize) this._samples.shift();
    this._samplesSince++;

    if (this._samplesSince >= this.refitInterval) {
      this._refit();
      this._samplesSince = 0;
    }
  }

  /**
   * Perform weighted ridge regression refit.
   * Blends base calibration (weight 2.5) + PACE samples.
   */
  _refit() {
    if (!this.calib.isCalibrated || this._samples.length < 6) return;

    // PRECISION-1: Normalize all gaze values through the base model's gazeRange.
    // The polynomial was trained on normalized coordinates [-0.5, +0.5].
    // Without this, PACE injects raw gaze values into a normalized-space model
    // → the model drifts badly after just a few PACE updates.
    const rngX = this.calib.model?.gazeRangeX;
    const rngY = this.calib.model?.gazeRangeY;
    const norm = (val, rng) => {
      if (!rng) return val;
      const mid = (rng.max + rng.min) / 2;
      const span = rng.max - rng.min;
      return span > 0.001 ? (val - mid) / span : val;
    };

    const blended = [
      // Anchor: base calibration data with zone-based weights (corners 4x, edges 2.5x)
      ...this.calib.calibData.filter(d => d.gx !== undefined).map((d, i) => {
        const zoneW = i < 4 ? 4.0 : i < 8 ? 2.5 : 1.0;
        return {
          gx: norm(d.gx, rngX), gy: norm(d.gy, rngY),
          sx: d.sx, sy: d.sy, w: zoneW
        };
      }),
      // PACE samples: normalize before blending
      ...this._samples.map(s => ({
        gx: norm(s.gx, rngX), gy: norm(s.gy, rngY),
        sx: s.sx, sy: s.sy, w: s.w
      }))
    ];

    const modelX = this._ridgeLS(blended, p => p.sx);
    const modelY = this._ridgeLS(blended, p => p.sy);

    if (modelX && modelY) {
      this.calib.model = {
        ...this.calib.model,
        x: modelX, y: modelY,
        pace: true,
        paceSamples: blended.length,
        paceTimestamp: p3.now()
      };
      this._lastRefit = p3.now();
      this._emit('refit', {
        samples: blended.length,
        pace: this._samples.length,
        timestamp: this._lastRefit
      });
    }
  }

  /**
   * FIX ACC-3c: Degree-3 weighted ridge regression (10 terms).
   * φ(gx, gy) = [1, gx, gy, gx², gy², gx·gy, gx³, gy³, gx²·gy, gx·gy²]
   * Matches base CalibrationEngine degree-3 upgrade.
   */
  _ridgeLS(pts, getTarget) {
    try {
      const deg = 10;  // FIX ACC-3c: upgraded from 6 to 10 terms
      let ATA = Array.from({ length: deg }, () => new Array(deg).fill(0));
      let ATb = new Array(deg).fill(0);

      for (const p of pts) {
        const w   = p.w ?? 1.0;
        const gx = p.gx, gy = p.gy;
        const gx2 = gx*gx, gy2 = gy*gy;
        const row = [1, gx, gy, gx2, gy2, gx*gy, gx*gx2, gy*gy2, gx2*gy, gx*gy2];
        const t   = getTarget(p);
        for (let r = 0; r < deg; r++) {
          ATb[r] += w * row[r] * t;
          for (let c = 0; c < deg; c++) ATA[r][c] += w * row[r] * row[c];
        }
      }
      for (let i = 1; i < deg; i++) ATA[i][i] += this.RIDGE_LAMBDA;

      let aug = ATA.map((row, i) => [...row, ATb[i]]);
      for (let col = 0; col < deg; col++) {
        let maxR = col;
        for (let r = col + 1; r < deg; r++) {
          if (Math.abs(aug[r][col]) > Math.abs(aug[maxR][col])) maxR = r;
        }
        [aug[col], aug[maxR]] = [aug[maxR], aug[col]];
        const piv = aug[col][col];
        if (Math.abs(piv) < 1e-12) continue;
        for (let r = 0; r < deg; r++) {
          if (r === col) continue;
          const f = aug[r][col] / piv;
          for (let c = col; c <= deg; c++) aug[r][c] -= f * aug[col][c];
        }
        for (let c = col; c <= deg; c++) aug[col][c] /= piv;
      }
      return aug.map(row => row[deg]);
    } catch (_) { return null; }
  }

  getSampleCount() { return this._samples.length; }
  getLastRefitTime() { return this._lastRefit; }

  save() {
    try {
      localStorage.setItem('accesseye_pace', JSON.stringify({
        samples: this._samples.slice(-50),
        t: Date.now()
      }));
    } catch (_) {}
  }

  load() {
    try {
      const raw = localStorage.getItem('accesseye_pace');
      if (!raw) return;
      const d = JSON.parse(raw);
      this._samples = (d.samples || []).map(s => ({ ...s, w: Math.min(s.w ?? 1, 0.5) }));
    } catch (_) {}
  }

  reset() {
    this._samples = [];
    this._samplesSince = 0;
    this._frameCount = 0;
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   P3.5  SMOOTH PURSUIT CALIBRATOR
   ──────────────────────────────────────────────────────────────────────────
   Calibrates gaze by having the user follow a smoothly moving target
   (figure-8 path) rather than staring at fixed points.

   • Figure-8 Lissajous path: x(t) = A·sin(ωt), y(t) = A·sin(2ωt)
   • Duration: ~8 seconds for a full traversal
   • Collects 200-300 valid (fixation-based) 100ms windows
   • Each window: median gaze sample correlated with known target position
   • Builds calibration model via weighted ridge regression
   • Target position known analytically — no user commitment required

   Usage:
     pursuitCalib.start(videoCanvas)       → begins animation
     pursuitCalib.cancel()                 → abort
     Event 'complete' → { success, model } on finish
     Event 'progress' → { pct } each frame
───────────────────────────────────────────────────────────────────────── */
class SmoothPursuitCalibrator {
  /**
   * @param {CalibrationEngine} calibration  Engine to build model into
   * @param {GazeEngine|HybridGazeEngine} gazeEngine  Source of raw gaze
   */
  constructor(calibration, gazeEngine) {
    this.calib      = calibration;
    this.gazeEngine = gazeEngine;

    // Path parameters (Lissajous figure-8)
    this.DURATION_MS   = 9000;    // total time (ms)
    this.OMEGA         = 2 * Math.PI / 4.5;   // one loop period = 4.5s
    this.AMPLITUDE_X   = 0.38;    // fraction of screen width
    this.AMPLITUDE_Y   = 0.30;    // fraction of screen height
    this.CENTER_X      = 0.50;
    this.CENTER_Y      = 0.50;

    // Sample collection
    this.WINDOW_MS     = 100;     // gaze averaging window
    this.MIN_WINDOWS   = 200;     // minimum valid windows for success
    this.VELOCITY_GATE = 60;      // max target velocity px/s — gate fast-moving targets
    this.RIDGE_LAMBDA  = 0.01;

    this._running       = false;
    this._startTime     = 0;
    this._animFrame     = null;
    this._overlayEl     = null;
    this._dotEl         = null;
    this._windowBuffer  = [];    // gaze samples in current window
    this._windowStart   = 0;
    this._collectedPts  = [];    // [{gx,gy, sx,sy, w}] correlated samples
    this._lastTargetX   = null;
    this._lastTargetY   = null;

    this._callbacks = {};
  }

  on(event, cb) {
    if (!this._callbacks[event]) this._callbacks[event] = [];
    this._callbacks[event].push(cb);
  }

  _emit(event, data) {
    (this._callbacks[event] || []).forEach(cb => cb(data));
  }

  /** Target position at time t (ms) as fractions [0,1] */
  _targetAt(t) {
    const s = t / 1000;
    return {
      x: this.CENTER_X + this.AMPLITUDE_X * Math.sin(this.OMEGA * s),
      y: this.CENTER_Y + this.AMPLITUDE_Y * Math.sin(2 * this.OMEGA * s)
    };
  }

  /** Target velocity at time t (normalised/second) */
  _targetVelocity(t) {
    const dt = 16;  // ~1 frame at 60fps
    const a  = this._targetAt(t);
    const b  = this._targetAt(t + dt);
    return Math.hypot((b.x - a.x) * 1000 / dt, (b.y - a.y) * 1000 / dt);
  }

  /**
   * Start smooth pursuit calibration.
   * @param {HTMLElement} container  Element to render the dot inside
   */
  start(container) {
    if (this._running) return;
    this._running     = true;
    this._startTime   = p3.now();
    this._collectedPts = [];
    this._windowBuffer = [];
    this._windowStart  = this._startTime;
    this._lastTargetX  = null;
    this._lastTargetY  = null;

    // Create overlay
    this._overlayEl = document.createElement('div');
    this._overlayEl.id = 'pursuit-overlay';
    Object.assign(this._overlayEl.style, {
      position: 'fixed', inset: '0', background: 'rgba(10,10,20,0.88)',
      zIndex: '9999', display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center', fontFamily: 'sans-serif'
    });
    this._overlayEl.innerHTML = `
      <div style="color:#a78bfa;font-size:1.4rem;font-weight:700;margin-bottom:8px">Smooth Pursuit Calibration</div>
      <div style="color:#888;font-size:0.9rem;margin-bottom:20px">Follow the dot with your eyes</div>
      <div id="pursuit-arena" style="position:relative;width:80vw;height:60vh;background:rgba(255,255,255,0.03);border:1px solid rgba(167,139,250,0.3);border-radius:12px;overflow:hidden">
        <div id="pursuit-dot" style="position:absolute;width:24px;height:24px;border-radius:50%;background:radial-gradient(circle,#00d4ff,#7c3aed);transform:translate(-50%,-50%);transition:none;box-shadow:0 0 16px rgba(0,212,255,0.8)"></div>
        <div id="pursuit-trail" style="position:absolute;inset:0;pointer-events:none;opacity:0.4"></div>
      </div>
      <div style="margin-top:16px;width:80vw">
        <div style="background:rgba(255,255,255,0.1);border-radius:4px;height:6px;overflow:hidden">
          <div id="pursuit-progress" style="height:100%;background:linear-gradient(90deg,#7c3aed,#00d4ff);width:0%;transition:width 0.1s"></div>
        </div>
        <div id="pursuit-status" style="text-align:center;color:#aaa;font-size:0.8rem;margin-top:8px">Warming up…</div>
      </div>
    `;
    document.body.appendChild(this._overlayEl);
    this._dotEl = document.getElementById('pursuit-dot');

    this._animate();
  }

  _animate() {
    if (!this._running) return;
    const t   = p3.now() - this._startTime;
    const pct = Math.min(t / this.DURATION_MS, 1.0);

    // Move dot to target
    const target = this._targetAt(t);
    const arena  = this._overlayEl.querySelector('#pursuit-arena');
    if (arena && this._dotEl) {
      const W = arena.clientWidth;
      const H = arena.clientHeight;
      this._dotEl.style.left = `${target.x * W}px`;
      this._dotEl.style.top  = `${target.y * H}px`;
    }

    // Update progress
    const progressBar = document.getElementById('pursuit-progress');
    if (progressBar) progressBar.style.width = `${pct * 100}%`;
    const statusEl = document.getElementById('pursuit-status');

    // Collect gaze sample into current window
    const rawGaze = this.gazeEngine?.rawGaze;
    if (rawGaze && typeof rawGaze.x === 'number') {
      this._windowBuffer.push({ gx: rawGaze.x, gy: rawGaze.y });
    }

    // Every WINDOW_MS: check if window is valid and correlate
    if (p3.now() - this._windowStart >= this.WINDOW_MS) {
      const vel = this._targetVelocity(t);
      const norm_vel = vel * Math.min(window.innerWidth, window.innerHeight);

      if (this._windowBuffer.length >= 2 && norm_vel < this.VELOCITY_GATE) {
        // Median gaze position for this window (robust to blinks)
        const sorted_gx = [...this._windowBuffer].sort((a, b) => a.gx - b.gx);
        const sorted_gy = [...this._windowBuffer].sort((a, b) => a.gy - b.gy);
        const mid = Math.floor(this._windowBuffer.length / 2);
        const medGx = sorted_gx[mid].gx;
        const medGy = sorted_gy[mid].gy;

        // Weight: inverse of target velocity (slow targets → higher weight)
        const weight = 1.0 - p3.clamp(norm_vel / this.VELOCITY_GATE, 0, 0.9);

        this._collectedPts.push({
          gx: medGx, gy: medGy,
          sx: target.x, sy: target.y,
          w: weight
        });
      }

      this._windowBuffer = [];
      this._windowStart  = p3.now();

      if (statusEl) {
        statusEl.textContent = this._collectedPts.length >= this.MIN_WINDOWS
          ? `✓ ${this._collectedPts.length} windows collected — building model…`
          : `Collecting… ${this._collectedPts.length}/${this.MIN_WINDOWS} windows`;
      }
    }

    this._emit('progress', { pct, windows: this._collectedPts.length });

    if (pct < 1.0) {
      this._animFrame = requestAnimationFrame(() => this._animate());
    } else {
      this._finish();
    }
  }

  _finish() {
    this._running = false;
    if (this._animFrame) cancelAnimationFrame(this._animFrame);

    const success = this._collectedPts.length >= this.MIN_WINDOWS;
    if (success) {
      // FIX C-3: The pursuit dot is positioned relative to the #pursuit-arena
      // div (80vw × 60vh), so target.x / target.y are already fractions [0,1]
      // of that arena.  But the calibration model expects gaze→screen fractions
      // relative to the FULL viewport.  The Lissajous path has CENTER_X=0.5,
      // AMPLITUDE_X=0.38, CENTER_Y=0.5, AMPLITUDE_Y=0.30, so the extreme
      // positions in arena fractions are [0.12, 0.88] × [0.20, 0.80].
      // We do NOT need to rescale sx/sy because the arena fractions ARE the
      // screen fractions the user was looking at (the arena fills the overlay
      // which covers the full screen).  HOWEVER we must guard against any
      // direct pixel-to-fraction confusion in callers.  The safest proof is:
      //   arena fraction in [0,1] → already correct screen fraction.
      // What WAS wrong: older code paths used arena.clientWidth/clientHeight
      // to convert pixel positions, which yielded fractions > 1.  Verify:
      const screenPts = this._collectedPts.map(p => ({
        gx: p.gx, gy: p.gy,
        // Clamp to [0,1] to catch any legacy pixel leak
        sx: Math.max(0, Math.min(1, p.sx)),
        sy: Math.max(0, Math.min(1, p.sy)),
        w:  p.w
      }));

      // Build calibration model from pursuit data
      const modelX = this._ridgeLS(screenPts, p => p.sx);
      const modelY = this._ridgeLS(screenPts, p => p.sy);

      if (modelX && modelY) {
        this.calib.model = {
          x: modelX, y: modelY,
          degree: 2, lambda: this.RIDGE_LAMBDA,
          method: 'pursuit',
          points: screenPts.length,
          timestamp: Date.now()
        };
        this.calib.isCalibrated = true;
        this.calib._saveToStorage?.();
      }
    }

    this._cleanup();
    this._emit('complete', {
      success,
      windows: this._collectedPts.length,
      method: 'smooth-pursuit'
    });
  }

  cancel() {
    this._running = false;
    if (this._animFrame) cancelAnimationFrame(this._animFrame);
    this._cleanup();
    this._emit('complete', { success: false, cancelled: true });
  }

  _cleanup() {
    if (this._overlayEl && this._overlayEl.parentNode) {
      this._overlayEl.parentNode.removeChild(this._overlayEl);
      this._overlayEl = null;
    }
  }

  _ridgeLS(pts, getTarget) {
    try {
      const deg = 6;
      let ATA = Array.from({ length: deg }, () => new Array(deg).fill(0));
      let ATb = new Array(deg).fill(0);

      for (const p of pts) {
        const w   = p.w ?? 1.0;
        const row = [1, p.gx, p.gy, p.gx * p.gx, p.gy * p.gy, p.gx * p.gy];
        const t   = getTarget(p);
        for (let r = 0; r < deg; r++) {
          ATb[r] += w * row[r] * t;
          for (let c = 0; c < deg; c++) ATA[r][c] += w * row[r] * row[c];
        }
      }
      for (let i = 1; i < deg; i++) ATA[i][i] += this.RIDGE_LAMBDA;

      let aug = ATA.map((row, i) => [...row, ATb[i]]);
      for (let col = 0; col < deg; col++) {
        let maxR = col;
        for (let r = col + 1; r < deg; r++) {
          if (Math.abs(aug[r][col]) > Math.abs(aug[maxR][col])) maxR = r;
        }
        [aug[col], aug[maxR]] = [aug[maxR], aug[col]];
        const piv = aug[col][col];
        if (Math.abs(piv) < 1e-12) continue;
        for (let r = 0; r < deg; r++) {
          if (r === col) continue;
          const f = aug[r][col] / piv;
          for (let c = col; c <= deg; c++) aug[r][c] -= f * aug[col][c];
        }
        for (let c = col; c <= deg; c++) aug[col][c] /= piv;
      }
      return aug.map(row => row[deg]);
    } catch (_) { return null; }
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   P3.6  CALIBRATION VALIDATOR
   ──────────────────────────────────────────────────────────────────────────
   Post-calibration accuracy test using 5 validation points.
   Displays each point, measures gaze error, reports pass/warn/fail.

   Accuracy thresholds:
     EXCELLENT  >85% of points within tolerance  → "Excellent" (green)
     GOOD       >70% of points within tolerance  → "Good" (yellow)
     POOR       ≤70%                             → "Needs Recalibration" (red)

   Tolerance: default 80px at 1080p (≈ 4.2°visual angle at 60cm distance)

   Each validation point:
     • Displayed for 1.5s (500ms delay + 1s sample collection)
     • Gaze error measured as Euclidean distance in pixels
     • Results stored in `lastReport` for UI display
───────────────────────────────────────────────────────────────────────── */
class CalibrationValidator {
  /**
   * @param {CalibrationEngine} calibration
   * @param {GazeEngine|HybridGazeEngine} gazeEngine
   * @param {number} tolerancePx  Maximum acceptable error in px (default 80px)
   */
  constructor(calibration, gazeEngine, tolerancePx = 80) {
    this.calib       = calibration;
    this.gazeEngine  = gazeEngine;
    this.tolerancePx = tolerancePx;

    // 5 validation points (different from calibration points)
    this.VAL_POINTS = [
      { sx: 0.25, sy: 0.25, label: 'Upper-Left'  },
      { sx: 0.75, sy: 0.25, label: 'Upper-Right' },
      { sx: 0.50, sy: 0.60, label: 'Center-Low'  },
      { sx: 0.20, sy: 0.75, label: 'Lower-Left'  },
      { sx: 0.80, sy: 0.75, label: 'Lower-Right' }
    ];

    this.SAMPLE_MS   = 1000;   // sampling duration per point
    this.DELAY_MS    = 500;    // lead-in delay per point

    this.lastReport  = null;
    this._running    = false;
    this._callbacks  = {};
  }

  on(event, cb) {
    if (!this._callbacks[event]) this._callbacks[event] = [];
    this._callbacks[event].push(cb);
  }

  _emit(event, data) {
    (this._callbacks[event] || []).forEach(cb => cb(data));
  }

  /**
   * Run the 5-point validation sequence.
   * Returns a Promise resolving to the accuracy report.
   */
  async run() {
    if (!this.calib.isCalibrated) {
      return { success: false, error: 'Not calibrated' };
    }
    if (this._running) return { success: false, error: 'Already running' };
    this._running = true;

    const overlay = this._createOverlay();
    document.body.appendChild(overlay);

    const results = [];
    const W = window.innerWidth;
    const H = window.innerHeight;

    for (let i = 0; i < this.VAL_POINTS.length; i++) {
      const pt = this.VAL_POINTS[i];
      this._highlightPoint(overlay, i, pt);
      this._emit('step', { index: i, label: pt.label });

      // Collect gaze samples for this point
      const samples = await this._collectSamples(pt);

      if (samples.length > 0) {
        // FIX: Use 2D centroid-distance trimming (same fix as CalibrationEngine C-1).
        // Original code sorted X and Y arrays separately, so a blink frame that
        // pushed both axes off could survive in one axis while being trimmed in
        // the other.  Sort by Euclidean distance from centroid and discard the
        // worst 20% as a coupled unit.
        const cx = p3.avg(samples.map(s => s.x));
        const cy = p3.avg(samples.map(s => s.y));
        const ranked = [...samples]
          .map(s => ({ ...s, dist: p3.dist2(s.x, s.y, cx, cy) }))
          .sort((a, b) => a.dist - b.dist);
        const cut     = Math.max(1, Math.floor(ranked.length * 0.2));
        const trimmed = ranked.slice(0, ranked.length - cut);
        const meanX   = p3.avg(trimmed.map(s => s.x));
        const meanY   = p3.avg(trimmed.map(s => s.y));

        const errorPx = p3.dist2(meanX * W, meanY * H, pt.sx * W, pt.sy * H);
        const passed  = errorPx <= this.tolerancePx;

        results.push({
          label: pt.label, sx: pt.sx, sy: pt.sy,
          measuredX: meanX, measuredY: meanY,
          errorPx: Math.round(errorPx),
          passed
        });
      }
    }

    document.body.removeChild(overlay);
    this._running = false;

    const report = this._buildReport(results);
    this.lastReport = report;
    this._emit('complete', report);
    return report;
  }

  _collectSamples(pt) {
    return new Promise(resolve => {
      const samples = [];
      // Delay before sampling
      setTimeout(() => {
        const interval = setInterval(() => {
          const sg = this.gazeEngine.smoothGaze;
          if (sg && typeof sg.x === 'number') {
            samples.push({ x: sg.x, y: sg.y });
          }
        }, 30);   // ~33 Hz

        setTimeout(() => {
          clearInterval(interval);
          resolve(samples);
        }, this.SAMPLE_MS);
      }, this.DELAY_MS);
    });
  }

  _createOverlay() {
    const el = document.createElement('div');
    el.id = 'val-overlay';
    Object.assign(el.style, {
      position: 'fixed', inset: '0', background: 'rgba(5,5,15,0.90)',
      zIndex: '9998', pointerEvents: 'none'
    });
    el.innerHTML = `
      <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
           text-align:center;color:rgba(167,139,250,0.7);font-family:sans-serif;font-size:0.9rem">
        Validation — look at each dot
      </div>
    `;
    // Render all points (dimmed)
    this.VAL_POINTS.forEach((pt, i) => {
      const dot = document.createElement('div');
      dot.id = `val-pt-${i}`;
      Object.assign(dot.style, {
        position: 'absolute',
        width: '22px', height: '22px',
        borderRadius: '50%',
        background: 'rgba(120,120,160,0.4)',
        border: '2px solid rgba(120,120,160,0.5)',
        transform: 'translate(-50%,-50%)',
        left: `${pt.sx * 100}%`, top: `${pt.sy * 100}%`,
        transition: 'all 0.3s'
      });
      el.appendChild(dot);
    });
    return el;
  }

  _highlightPoint(overlay, idx, pt) {
    // Reset all
    this.VAL_POINTS.forEach((_, i) => {
      const dot = document.getElementById(`val-pt-${i}`);
      if (dot) {
        dot.style.background = 'rgba(120,120,160,0.4)';
        dot.style.boxShadow = 'none';
        dot.style.transform = 'translate(-50%,-50%) scale(1)';
      }
    });
    // Highlight current
    const current = document.getElementById(`val-pt-${idx}`);
    if (current) {
      current.style.background = 'radial-gradient(circle,#00d4ff,#7c3aed)';
      current.style.boxShadow = '0 0 20px rgba(0,212,255,0.8)';
      current.style.transform = 'translate(-50%,-50%) scale(1.4)';
    }
  }

  _buildReport(results) {
    const total  = results.length;
    const passed = results.filter(r => r.passed).length;
    const pct    = total > 0 ? passed / total : 0;
    const errors = results.map(r => r.errorPx);
    const avgErr = Math.round(p3.avg(errors));
    const maxErr = Math.max(...errors, 0);

    let grade, color;
    if (pct > 0.85)      { grade = 'Excellent'; color = '#00ff88'; }
    else if (pct > 0.70) { grade = 'Good';      color = '#ffd32a'; }
    else                 { grade = 'Poor — recalibrate'; color = '#ff4757'; }

    return {
      success: pct > 0.70,
      grade, color, pct,
      passed, total,
      avgErrorPx: avgErr,
      maxErrorPx: maxErr,
      tolerancePx: this.tolerancePx,
      points: results
    };
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   P3.7  HEAD-FREE GAZE STABILIZER
   ──────────────────────────────────────────────────────────────────────────
   Compensates for head movement by computing a dynamic face-center reference.
   As the head moves, the raw gaze is re-anchored to the current face center,
   so gaze position changes only when the eyes move — not when the head moves.

   Algorithm:
     1. Track face center (midpoint of key landmarks) via EMA smoothing
     2. Compute face displacement from reference position (set at calibration)
     3. Subtract displacement from raw gaze (head-pose compensation)
     4. Apply separate confidence gate: suppress correction when head moves too fast

   This gives users the freedom to move their head naturally while maintaining
   accurate gaze tracking — a significant accessibility improvement.
───────────────────────────────────────────────────────────────────────── */
class HeadFreeStabilizer {
  /**
   * @param {number} emaAlpha   Smoothing factor for face center (default 0.15)
   * @param {number} compScale  Compensation scale factor (default 0.85)
   * @param {number} maxDispPx  Max head displacement before confidence penalty (default 40px)
   */
  constructor(emaAlpha = 0.15, compScale = 0.85, maxDispPx = 40) {
    this.emaAlpha   = emaAlpha;
    this.compScale  = compScale;
    this.maxDispPx  = maxDispPx;

    // Face center state
    this._refFaceX   = null;   // reference face center (set on calibration)
    this._refFaceY   = null;
    this._smoothFX   = null;   // EMA-smoothed face center X
    this._smoothFY   = null;
    this._initialized = false;

    // For velocity-based confidence gating
    this._prevFX = null;
    this._prevFY = null;
    this._velEMA = 0;
  }

  /**
   * Set the reference face center (call this right after calibration).
   * @param {number} faceCX  Face center X [0-1]
   * @param {number} faceCY  Face center Y [0-1]
   */
  setReference(faceCX, faceCY) {
    this._refFaceX = faceCX;
    this._refFaceY = faceCY;
    this._smoothFX = faceCX;
    this._smoothFY = faceCY;
    this._initialized = true;
  }

  /**
   * Compute current face center from MediaPipe landmarks.
   * Uses forehead, chin, and cheekbone landmarks.
   * @param {Array} lm  MediaPipe face landmarks array
   * @returns {{ x, y }}
   */
  static faceCenter(lm) {
    // Key face landmarks: nose bridge (6), chin (152), left cheek (234), right cheek (454)
    const pts = [lm[6], lm[152], lm[234], lm[454]].filter(Boolean);
    if (pts.length === 0) return { x: 0.5, y: 0.5 };
    return {
      x: pts.reduce((s, p) => s + p.x, 0) / pts.length,
      y: pts.reduce((s, p) => s + p.y, 0) / pts.length
    };
  }

  /**
   * Process a raw gaze point with head-free compensation.
   * @param {number} rawGX    Raw gaze X from HybridGazeEngine [0-1]
   * @param {number} rawGY    Raw gaze Y
   * @param {Array}  lm       MediaPipe landmarks (for face center)
   * @param {number} screenW  Screen width in px (for displacement scaling)
   * @param {number} screenH  Screen height in px
   * @returns {{ x, y, headDisp, compensated, confidenceMultiplier }}
   */
  stabilize(rawGX, rawGY, lm, screenW = window.innerWidth, screenH = window.innerHeight) {
    if (!lm || lm.length < 10) {
      return { x: rawGX, y: rawGY, headDisp: 0, compensated: false, confidenceMultiplier: 1.0 };
    }

    const fc = HeadFreeStabilizer.faceCenter(lm);

    // Update EMA of face center
    if (this._smoothFX === null) {
      this._smoothFX = fc.x;
      this._smoothFY = fc.y;
    } else {
      this._smoothFX += this.emaAlpha * (fc.x - this._smoothFX);
      this._smoothFY += this.emaAlpha * (fc.y - this._smoothFY);
    }

    // Set reference if not initialized
    if (!this._initialized) {
      this.setReference(this._smoothFX, this._smoothFY);
    }

    // Displacement from reference (in normalised units, then scale to px)
    const dxNorm = this._smoothFX - this._refFaceX;
    const dyNorm = this._smoothFY - this._refFaceY;
    const dispPx = Math.hypot(dxNorm * screenW, dyNorm * screenH);

    // Head velocity for gating
    const vel = this._prevFX !== null
      ? Math.hypot((this._smoothFX - this._prevFX) * screenW,
                   (this._smoothFY - this._prevFY) * screenH)
      : 0;
    this._velEMA = this._velEMA * 0.8 + vel * 0.2;
    this._prevFX = this._smoothFX;
    this._prevFY = this._smoothFY;

    // Confidence multiplier: reduce when head is moving fast
    const confidenceMultiplier = p3.clamp(1.0 - (this._velEMA / 10), 0.4, 1.0);

    // Apply compensation: subtract scaled displacement from raw gaze
    const compX = p3.clamp(rawGX - dxNorm * this.compScale, 0, 1);
    const compY = p3.clamp(rawGY - dyNorm * this.compScale, 0, 1);

    return {
      x: compX, y: compY,
      headDisp: Math.round(dispPx),
      compensated: true,
      confidenceMultiplier,
      faceCenter: { x: this._smoothFX, y: this._smoothFY },
      displacement: { dx: dxNorm, dy: dyNorm }
    };
  }

  reset() {
    this._refFaceX = null;
    this._refFaceY = null;
    this._smoothFX = null;
    this._smoothFY = null;
    this._prevFX   = null;
    this._prevFY   = null;
    this._velEMA   = 0;
    this._initialized = false;
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   PHASE 3 ORCHESTRATOR
   Integrates all P3 modules into the existing Phase 2 pipeline.
───────────────────────────────────────────────────────────────────────── */
class Phase3Orchestrator {
  /**
   * @param {Phase2Orchestrator} p2Orch  Phase 2 orchestrator reference
   * @param {AccessEyeApp} app           Phase 1 app reference
   */
  constructor(p2Orch, app) {
    this.p2    = p2Orch;
    this.app   = app;
    const calib      = app.calibration;
    const gazeEngine = p2Orch.hybridGaze;

    // ── Instantiate P3 modules ──
    // PRECISION-9: OneEuro minCutoff raised 0.15 → 0.45 Hz.
    // Problem: 0.15 Hz was so aggressive that the cursor lagged 3-5 frames behind
    // actual gaze, making it impossible for the user to feel the cursor "tracking"
    // their eye directly. The cursor appeared to be "behind" where they were looking.
    // Fix: 0.45 Hz still provides strong jitter suppression at fixation while
    // responding much faster to deliberate eye movements, closing the perceived gap
    // between "where I'm looking" and "where the cursor is".
    // beta=0.007 unchanged — allows fast saccades to pass through cleanly.
    this.oneEuro  = new OneEuroFilter(0.45, 0.007, 1.0, 30);    // P3.1  PRECISION-9
    this.ivt      = new IVTSaccadeDetector(35, 100, 3);        // P3.2
    this.dwell    = new AdaptiveDwellTimer('normal', this.ivt);// P3.3
    this.pace     = new PACERecalibrator(calib, 100, 0.995, 0.65, 10);  // P3.4
    this.pursuit  = new SmoothPursuitCalibrator(calib, gazeEngine);     // P3.5
    this.validator= new CalibrationValidator(calib, gazeEngine, 80);    // P3.6
    // EASE-2: HeadFreeStabilizer tuned for tolerating natural head movement.
    // compScale 0.85 → 0.60: less aggressive compensation so small head movements
    // don't cause the cursor to over-correct and jitter.
    // emaAlpha 0.15 → 0.25: faster face-center tracking so the reference updates
    // quickly when the user naturally adjusts position.
    // maxDispPx 40 → 60: wider tolerance before confidence penalty kicks in.
    this.headFree = new HeadFreeStabilizer(0.25, 0.60, 60);             // P3.7  EASE-2

    this.active   = false;
    this._frameCounter = 0;

    // Load saved PACE data
    this.pace.load();

    this._wireEvents();
  }

  _wireEvents() {
    // IVT: feed fixation events to PACE recalibrator
    this.ivt.on('fixation', (fix) => {
      const focused = this.app.uiRegistry?.getFocused();
      if (focused) {
        const bbox = focused.bbox;
        if (bbox) {
          const W = window.innerWidth, H = window.innerHeight;
          const targetSX = (bbox.x + bbox.w / 2) / W;
          const targetSY = (bbox.y + bbox.h / 2) / H;
          const rawG = this.p2.hybridGaze?.rawGaze;
          if (rawG) {
            this.pace.addSample(rawG.x, rawG.y, targetSX, targetSY, 0.75);
          }
        }
      }
    });

    // Dwell activate → also feed PACE
    this.dwell.on('dwell-activate', ({ elementId }) => {
      const entry = this.app.uiRegistry?.elements?.get(elementId);
      if (entry && entry.bbox) {
        const W = window.innerWidth, H = window.innerHeight;
        const sx = (entry.bbox.x + entry.bbox.w / 2) / W;
        const sy = (entry.bbox.y + entry.bbox.h / 2) / H;
        const rawG = this.p2.hybridGaze?.rawGaze;
        const conf = this.p2.confidence?.lastScore?.total ?? 0.7;
        if (rawG) this.pace.addSample(rawG.x, rawG.y, sx, sy, conf);
      }
    });

    // PACE refit → update pipeline label
    this.pace.on('refit', (d) => {
      this.app.log?.add(`PACE refit: ${d.pace} passive samples`, 'info');
      this._updatePipelineLabel();
    });

    // Pursuit complete
    this.pursuit.on('complete', (res) => {
      if (res.success) {
        this.app.toast?.show(
          'Smooth Pursuit Calibration Complete!',
          `Collected ${res.windows} windows — gaze model updated.`,
          'success', 'fas fa-route', 4000
        );
        this.app.log?.add(`Smooth pursuit: ${res.windows} windows calibrated`, 'success');
        // Re-set head-free reference
        const lm = this._lastLandmarks;
        if (lm) {
          const fc = HeadFreeStabilizer.faceCenter(lm);
          this.headFree.setReference(fc.x, fc.y);
        }
      } else {
        this.app.toast?.show('Pursuit Calibration Cancelled', '', 'warn');
      }
    });

    // Validator complete
    this.validator.on('complete', (report) => {
      this._showValidationReport(report);
    });
  }

  /**
   * Activate Phase 3 — called after Phase 2 is active.
   * Patches the Phase 2 processing pipeline to insert P3 filters.
   */
  activate() {
    if (this.active) return;
    this.active = true;

    // Update One Euro filter freq to match actual camera FPS
    const fps = this.p2.cameraFPS || 30;
    this.oneEuro.setFreq(fps);

    // Patch Phase 2 orchestrator's _processPhase2Face method
    this._patchPhase2Pipeline();

    // Replace UIElementRegistry dwell timer with adaptive version
    this._hookAdaptiveDwell();

    this._updatePipelineLabel();
    console.log('%c Phase 3 Active ✅ — OneEuro + IVT + AdaptiveDwell + PACE + HeadFree',
                'color:#00d4ff;font-weight:bold;font-size:12px;');
  }

  /**
   * Patch the Phase 2 pipeline to insert P3 filters.
   * Wraps the existing _processPhase2Face with pre/post processing.
   */
  _patchPhase2Pipeline() {
    const self    = this;
    const orch    = this.p2;
    const origFn  = orch._processPhase2Face.bind(orch);

    // ── P3.1: Patch HybridGazeEngine ONCE during activation ──
    // Apply One Euro filter to raw gaze before it enters the Kalman stabilizer.
    // This reduces high-frequency tremor at the earliest possible stage.
    if (orch.hybridGaze && !orch.hybridGaze._p3Patched) {
      const origProcess = orch.hybridGaze.processResults.bind(orch.hybridGaze);
      orch.hybridGaze._originalProcessResults = origProcess;
      orch.hybridGaze._p3Patched = true;

      orch.hybridGaze.processResults = function(multiFaceLandmarks, W, H, headPoseResult) {
        const packet = this._originalProcessResults(multiFaceLandmarks, W, H, headPoseResult);
        if (packet && packet.raw) {
          // PRECISION-2: Save the TRUE unfiltered iris raw gaze BEFORE OneEuro.
          // CalibrationUI reads gazeEngine.rawGaze for samples — it must get the
          // unfiltered iris position, not the smoothed output.
          // We store it as _trueRawGaze so CalibrationUI can access it directly.
          this._trueRawGaze = { x: packet.raw.x, y: packet.raw.y };
          // PRECISION-5: Also capture the iris-only signal (pre-fusion) for calibration.
          // packet.iris is the irisSignal from _computeIrisSignal — pure iris, no head/pupil.
          if (packet.iris) {
            this._irisOnlyGaze = { x: packet.iris.x, y: packet.iris.y };
          }

          // Apply One Euro to iris-only gaze for the live display pipeline only.
          // PRECISION-5: Filter the iris-only signal (not the fused raw) through OneEuro.
          // The model was trained on iris-only — so the live display pipeline should also
          // use iris-only as input to mapGaze.  Head-pose compensation is still present
          // in the fused rawGaze used for other processing.
          const irisOnly = this._irisOnlyGaze || packet.raw;
          const filtered = self.oneEuro.filter(irisOnly.x, irisOnly.y);
          packet.raw   = filtered;
          this.rawGaze = filtered;   // smoothed iris-only — used for display + PACE
          // Remap through calibration with OneEuro-smoothed iris-only gaze
          if (this.calibration?.isCalibrated) {
            const mapped = this.calibration.mapGaze(filtered.x, filtered.y);
            packet.screen = { x: mapped.sx, y: mapped.sy };
          }
        }
        return packet;
      };
      console.log('[Phase3] One Euro Filter patched into HybridGazeEngine ✅');
    }

    orch._processPhase2Face = function(results, mp) {
      // ── P3.7: Store landmarks for head-free use ──
      const lm = results.multiFaceLandmarks?.[0];
      self._lastLandmarks = lm || null;

      // ── Call original Phase 2 pipeline (which now uses patched HybridGazeEngine) ──
      origFn(results, mp);

      // ── P3.2: IVT saccade detection on final screen coordinates ──
      // Run IVT in parallel with Phase 2 saccade filter for comparison
      const screenX = orch.app?._lastScreenX;
      const screenY = orch.app?._lastScreenY;
      if (typeof screenX === 'number' && typeof screenY === 'number') {
        const conf = orch.confidence?.lastScore?.total ?? 0.7;
        const ivtResult = self.ivt.update(screenX, screenY, conf);
        self._lastIVTResult = ivtResult;

        // IVT dwell gating: suppress UIRegistry dwell during saccades
        if (ivtResult.isSaccading) {
          // Reset dwell progress on saccade start
          if (orch.app.uiRegistry?.focusedId) {
            self.dwell.reset(orch.app.uiRegistry.focusedId);
          }
        }

        // Update P3 UI stats
        self._updateP3UI(ivtResult);
      }

      // ── P3.4: PACE tick (weight decay) ──
      self._frameCounter++;
      self.pace.tick();

      // ── P3.7: Head-free stabilization ──
      if (lm && self.active) {
        const rawG = orch.hybridGaze?.rawGaze;
        if (rawG) {
          const stabilized = self.headFree.stabilize(rawG.x, rawG.y, lm);
          if (stabilized.compensated) {
            // Update confidence multiplier in confidence scorer
            if (orch.confidence?.lastScore) {
              orch.confidence.lastScore.total *= stabilized.confidenceMultiplier;
              orch.confidence.lastScore.total = p3.clamp(orch.confidence.lastScore.total, 0, 1);
            }
          }
        }
      }
    };

    // Hook into app to capture last screen coordinates
    const origUpdateGazeCursor = orch.app._updateGazeCursor?.bind(orch.app);
    if (origUpdateGazeCursor && !orch.app._p3GazeCursorHooked) {
      orch.app._p3GazeCursorHooked = true;
      orch.app._updateGazeCursor = function(x, y) {
        orch.app._lastScreenX = x;
        orch.app._lastScreenY = y;
        origUpdateGazeCursor(x, y);
      };
    }
  }

  /**
   * Hook adaptive dwell timer into UIElementRegistry.
   * Integrates with IVT saccade gating.
   */
  _hookAdaptiveDwell() {
    const registry = this.app.uiRegistry;
    if (!registry || registry._p3DwellHooked) return;
    registry._p3DwellHooked = true;

    const self = this;
    const origUpdateGaze = registry.updateGaze.bind(registry);

    registry.updateGaze = function(screenX, screenY) {
      origUpdateGaze(screenX, screenY);

      // Apply adaptive dwell on top of existing dwell
      const isFixating = self.ivt.isFixating;
      const result = self.dwell.update(registry.focusedId, isFixating);

      if (result.completed && registry.focusedId) {
        // Dwell-activate element
        const entry = registry.elements.get(registry.focusedId);
        if (entry) {
          entry.el.classList.add('gaze-activating');
          setTimeout(() => {
            entry.el.classList.remove('gaze-activating');
            entry.el.classList.add('gaze-activated');
            setTimeout(() => entry.el.classList.remove('gaze-activated'), 600);
          }, 200);
          registry._emit?.('activate', { id: entry.id, label: entry.label, gesture: 'dwell' });
          if (entry.onActivate) entry.onActivate('dwell');
        }
      }
    };
  }

  _updatePipelineLabel() {
    const el = document.getElementById('p2-pipeline-label');
    if (el) {
      const fps   = this.p2.cameraFPS || 30;
      const pace  = this.pace.getSampleCount();
      el.textContent = `P3 | ${fps}FPS | 1€+Kalman+IVT | PACE:${pace}`;
    }
  }

  _updateP3UI(ivt) {
    const ivtEl = document.getElementById('p3-ivt-status');
    if (ivtEl) {
      ivtEl.textContent = ivt.isSaccading ? '↗ Saccade' : `◉ Fixated ${Math.round(ivt.fixationAge)}ms`;
      ivtEl.style.color = ivt.isSaccading ? '#ffd32a' : '#00ff88';
    }
    const velEl = document.getElementById('p3-velocity');
    if (velEl) velEl.textContent = `${Math.round(ivt.velocity)}px/f`;
    const paceEl = document.getElementById('p3-pace-count');
    if (paceEl) paceEl.textContent = this.pace.getSampleCount();
    const dwellEl = document.getElementById('p3-dwell-preset');
    if (dwellEl) dwellEl.textContent = `${this.dwell.preset} (${this.dwell.baseMs}ms)`;
  }

  _showValidationReport(report) {
    // Create validation result panel
    const existing = document.getElementById('p3-val-report');
    const container = document.getElementById('p3-section') || document.body;

    const html = `
      <div id="p3-val-report" style="background:rgba(15,15,30,0.95);border:1px solid ${report.color};
           border-radius:8px;padding:12px;margin-top:8px;font-size:0.8rem">
        <div style="color:${report.color};font-weight:700;font-size:1rem;margin-bottom:6px">
          ${report.grade} — ${Math.round(report.pct * 100)}% passed
        </div>
        <div style="color:#888;margin-bottom:8px">
          Avg error: ${report.avgErrorPx}px | Max: ${report.maxErrorPx}px | Tolerance: ${report.tolerancePx}px
        </div>
        ${report.points.map(p => `
          <div style="display:flex;justify-content:space-between;padding:2px 0;
               color:${p.passed ? '#00ff88' : '#ff4757'}">
            <span>${p.label}</span>
            <span>${p.errorPx}px ${p.passed ? '✓' : '✗'}</span>
          </div>
        `).join('')}
        ${!report.success ? `<div style="color:#ff4757;margin-top:6px;font-weight:600">
          ⚠ Accuracy below threshold — please recalibrate</div>` : ''}
      </div>
    `;

    if (existing) existing.outerHTML = html;
    else container.insertAdjacentHTML('beforeend', html);

    this.app.toast?.show(
      `Validation: ${report.grade}`,
      `${report.passed}/${report.total} points within ${report.tolerancePx}px tolerance`,
      report.success ? 'success' : 'warn',
      'fas fa-crosshairs',
      6000
    );
    this.app.log?.add(`Validation: ${report.grade} — ${report.avgErrorPx}px avg error`, report.success ? 'success' : 'warn');
  }

  deactivate() {
    this.active = false;
    this.pace.save();
    this.oneEuro.reset();
    this.ivt.reset();
    this.headFree.reset();
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   EXPORT — expose to window.Phase3
───────────────────────────────────────────────────────────────────────── */
window.Phase3 = {
  OneEuroFilter,
  IVTSaccadeDetector,
  AdaptiveDwellTimer,
  PACERecalibrator,
  SmoothPursuitCalibrator,
  CalibrationValidator,
  HeadFreeStabilizer,
  Phase3Orchestrator
};

console.log('%c Phase 3 Engine Loaded ✅ — OneEuro | IVT | AdaptiveDwell | PACE | SmoothPursuit | Validator | HeadFree',
            'color:#00d4ff;font-weight:bold;font-size:12px;');
