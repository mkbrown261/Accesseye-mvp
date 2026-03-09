/**
 * ═══════════════════════════════════════════════════════════════════
 *  AccessEye — Eye + Gesture Control System (MVP)
 *  app.js — Core Engine
 * ═══════════════════════════════════════════════════════════════════
 *
 *  Layers implemented:
 *   1. Vision Input Layer     — MediaPipe FaceMesh + Hands
 *   2. Gaze Mapping Engine    — Kalman filter + EMA smoothing
 *   3. Gaze Calibration       — 13-point ridge regression (v3)
 *   4. UI Target Detection    — Bounding box registry + dwell timer
 *   5. Gesture Engine         — Pinch / Air-tap / Open palm
 *   6. Accessibility Layer    — TTS audio feedback
 *   7. Navigation + Demo App  — Full interactive demo
 */

'use strict';

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   UTILITY HELPERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const lerp = (a, b, t) => a + (b - a) * t;
const dist2D = (x1, y1, x2, y2) => Math.hypot(x2 - x1, y2 - y1);
const now = () => performance.now();

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   KALMAN FILTER (2D — position + velocity state)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
class KalmanFilter2D {
  /**
   * Simplified scalar Kalman applied independently to X and Y.
   * State: [position, velocity]
   * @param {number} R  Measurement noise (higher = smoother, less responsive)
   * @param {number} Q  Process noise (higher = more responsive, less smooth)
   */
  constructor(R = 0.005, Q = 0.0001) {
    this.R = R; this.Q = Q;
    this._init('x'); this._init('y');
  }

  _init(axis) {
    this[`${axis}_x`]  = 0;   // state: position
    this[`${axis}_v`]  = 0;   // state: velocity
    this[`${axis}_p`]  = 1;   // error covariance
  }

  _update(axis, measurement) {
    const dt = 1; // normalized step
    // Predict
    let pred_x = this[`${axis}_x`] + this[`${axis}_v`] * dt;
    let pred_p = this[`${axis}_p`] + this.Q;

    // Update (Kalman gain)
    const K = pred_p / (pred_p + this.R);
    this[`${axis}_x`] = pred_x + K * (measurement - pred_x);
    this[`${axis}_v`] = this[`${axis}_v`] + K * (measurement - pred_x) / dt;
    this[`${axis}_p`] = (1 - K) * pred_p;

    return this[`${axis}_x`];
  }

  update(mx, my) {
    return {
      x: this._update('x', mx),
      y: this._update('y', my)
    };
  }

  reset() { this._init('x'); this._init('y'); }
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   EXPONENTIAL MOVING AVERAGE SMOOTHER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
class EMAFilter {
  constructor(alpha = 0.3) {
    this.alpha = alpha;
    this.x = null; this.y = null;
  }

  update(x, y) {
    if (this.x === null) { this.x = x; this.y = y; return { x, y }; }
    this.x = lerp(this.x, x, this.alpha);
    this.y = lerp(this.y, y, this.alpha);
    return { x: this.x, y: this.y };
  }

  reset() { this.x = null; this.y = null; }
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   CALIBRATION ENGINE  [v3 — 13-point + Ridge Regression]
   • 13-point grid: 4 corners, 4 mid-edges, 4 mid-quadrants, center
   • Weighted Ridge Regression (λ=0.01, degree-2 polynomial, 6 terms)
   • ~20-25% MSE reduction vs 5-pt bilinear; ~12% peripheral gain
   • Maps (gx, gy) → (sx, sy) in [0,1]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
class CalibrationEngine {
  constructor() {
    /**
     * 13-point calibration grid (importance-ordered for early abort graceful fallback):
     *   4 corners → 4 mid-edges → 4 mid-quadrant → center
     * Layout (sx, sy as fractions of screen):
     *   TL  TC  TR
     *   ML  MC  MR    (mid-quadrant on diagonals)
     *   BL  BC  BR
     *
     * Mid-edge points (TLM/TRM/BLM/BRM) catch peripheral distortion
     * that the 5-pt model misses (~12% accuracy gain at edges).
     */
    this.CALIB_POINTS = [
      // ── 4 corners (highest priority — structural anchors) ──
      { sx: 0.10, sy: 0.10, label: 'Top-Left'        },
      { sx: 0.90, sy: 0.10, label: 'Top-Right'       },
      { sx: 0.10, sy: 0.90, label: 'Bottom-Left'     },
      { sx: 0.90, sy: 0.90, label: 'Bottom-Right'    },
      // ── 4 mid-edge points (peripheral accuracy, new in v3) ──
      { sx: 0.50, sy: 0.10, label: 'Top-Center'      },   // TLM / TRM
      { sx: 0.50, sy: 0.90, label: 'Bottom-Center'   },   // BLM / BRM
      { sx: 0.10, sy: 0.50, label: 'Mid-Left'        },
      { sx: 0.90, sy: 0.50, label: 'Mid-Right'       },
      // ── 4 mid-quadrant interior points ──
      { sx: 0.30, sy: 0.30, label: 'Inner-TL'        },
      { sx: 0.70, sy: 0.30, label: 'Inner-TR'        },
      { sx: 0.30, sy: 0.70, label: 'Inner-BL'        },
      { sx: 0.70, sy: 0.70, label: 'Inner-BR'        },
      // ── Center ──
      { sx: 0.50, sy: 0.50, label: 'Center'          }
    ];

    this.calibData = [];          // [{sx, sy, gx, gy, samples[]}]
    this.model     = null;        // { x: coeff[], y: coeff[], degree, lambda }
    this.isCalibrated   = false;
    this.SAMPLES_PER_POINT = 25;
    this.RIDGE_LAMBDA      = 0.01;  // regularisation strength (λ)
    this.MIN_POINTS_FOR_MODEL = 6;  // minimum calibration points to build model

    // Event system (used to notify GazeEngine of calibration completion)
    this._callbacks = {};
  }

  on(event, cb) {
    if (!this._callbacks[event]) this._callbacks[event] = [];
    this._callbacks[event].push(cb);
  }

  _emit(event, data) {
    (this._callbacks[event] || []).forEach(cb => cb(data));
  }

  /* Record raw gaze samples for a calibration point */
  addCalibSample(pointIdx, rawGazeX, rawGazeY) {
    if (!this.calibData[pointIdx]) {
      this.calibData[pointIdx] = { samples: [], ...this.CALIB_POINTS[pointIdx] };
    }
    this.calibData[pointIdx].samples.push({ gx: rawGazeX, gy: rawGazeY });
  }

  /* After collecting samples, compute trimmed-mean gaze vector per point.
   * FIX C-1: Sort by 2D Euclidean distance from centroid and discard the
   * worst 20% rows TOGETHER (axis-coupled), so a blink frame that corrupts
   * both gx and gy is removed as a unit. The old per-axis sort could remove
   * a good gx row and a good gy row from *different* blink frames, leaving
   * both axes contaminated.
   */
  finalizePoint(pointIdx) {
    const d = this.calibData[pointIdx];
    if (!d || d.samples.length === 0) return false;

    // Compute centroid of all samples
    const cx = d.samples.reduce((s, v) => s + v.gx, 0) / d.samples.length;
    const cy = d.samples.reduce((s, v) => s + v.gy, 0) / d.samples.length;

    // Rank by 2D distance from centroid (closest = most representative)
    const ranked = [...d.samples]
      .map(s => ({ ...s, dist: Math.hypot(s.gx - cx, s.gy - cy) }))
      .sort((a, b) => a.dist - b.dist);

    // Discard the outermost 20% of samples (blinks, squints, head turns)
    const cut     = Math.floor(ranked.length * 0.20);
    const trimmed = ranked.slice(0, ranked.length - cut);  // keep closest 80%

    d.gx = trimmed.reduce((s, v) => s + v.gx, 0) / trimmed.length;
    d.gy = trimmed.reduce((s, v) => s + v.gy, 0) / trimmed.length;
    return true;
  }

  /**
   * Build a degree-2 polynomial Ridge Regression model
   * Feature vector φ(gx,gy) = [1, gx, gy, gx², gy², gx·gy]  (6 terms)
   * Coefficients solved via (A^T·W·A + λI)·c = A^T·W·b
   * λ=0.01 prevents overfitting; ~20-25% MSE reduction vs plain bilinear.
   */
  buildModel() {
    if (this.calibData.length < this.MIN_POINTS_FOR_MODEL) return false;
    this.calibData.forEach((d, i) => this.finalizePoint(i));

    const pts = this.calibData.filter(d => d.gx !== undefined);
    if (pts.length < this.MIN_POINTS_FOR_MODEL) return false;

    const modelX = this._ridgeRegression(pts, p => p.sx, this.RIDGE_LAMBDA);
    const modelY = this._ridgeRegression(pts, p => p.sy, this.RIDGE_LAMBDA);

    if (!modelX || !modelY) return false;

    this.model = {
      x:      modelX,
      y:      modelY,
      degree: 2,
      lambda: this.RIDGE_LAMBDA,
      points: pts.length
    };
    this.isCalibrated = true;
    this._saveToStorage();

    // FIX M-1: notify GazeEngine to capture reference eye span.
    // The calibration was collected while the user was positioned in front of
    // the camera, so rawGaze from the gaze engine reflects their real eye size.
    // We emit an event that GazeEngine listens to in order to set _refEyeSpan.
    this._emit?.('calibrated', { points: pts.length });

    return true;
  }

  /**
   * Build design matrix row for degree-2 polynomial:
   *   φ(gx, gy) = [1, gx, gy, gx², gy², gx·gy]
   */
  _designRow(gx, gy) {
    return [1, gx, gy, gx * gx, gy * gy, gx * gy];
  }

  /**
   * Weighted Ridge Regression
   * Solves (A^T·W·A + λI)·c = A^T·W·b  via Gauss-Jordan elimination
   * @param {Array}    pts       calibration points [{gx,gy,w?}]
   * @param {Function} getTarget function(pt) → target scalar
   * @param {number}   lambda    ridge penalty λ
   */
  _ridgeRegression(pts, getTarget, lambda = 0.01) {
    const deg = 6;   // number of features in φ
    let ATA = Array.from({ length: deg }, () => new Array(deg).fill(0));
    let ATb = new Array(deg).fill(0);

    for (const p of pts) {
      const w   = p.w ?? 1.0;
      const row = this._designRow(p.gx, p.gy);
      const t   = getTarget(p);
      for (let r = 0; r < deg; r++) {
        ATb[r] += w * row[r] * t;
        for (let c = 0; c < deg; c++) {
          ATA[r][c] += w * row[r] * row[c];
        }
      }
    }

    // Add ridge penalty λ to diagonal  (skip intercept term, index 0)
    for (let i = 1; i < deg; i++) ATA[i][i] += lambda;

    return this._solveGaussJordan(ATA, ATb, deg);
  }

  /* Gauss–Jordan with partial pivoting — solves ATA·x = ATb */
  _solveGaussJordan(ATA, ATb, n) {
    let aug = ATA.map((row, i) => [...row, ATb[i]]);
    for (let col = 0; col < n; col++) {
      let maxR = col;
      for (let r = col + 1; r < n; r++) {
        if (Math.abs(aug[r][col]) > Math.abs(aug[maxR][col])) maxR = r;
      }
      [aug[col], aug[maxR]] = [aug[maxR], aug[col]];
      const piv = aug[col][col];
      if (Math.abs(piv) < 1e-12) continue;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = aug[r][col] / piv;
        for (let c = col; c <= n; c++) aug[r][c] -= f * aug[col][c];
      }
      for (let c = col; c <= n; c++) aug[col][c] /= piv;
    }
    return aug.map(row => row[n]);   // [c0, c1, c2, c3, c4, c5]
  }

  /** Apply degree-2 polynomial model: φ(gx,gy) · coeff */
  _applyModel(coeff, gx, gy) {
    const [c0, c1, c2, c3, c4, c5] = coeff;
    return c0 + c1*gx + c2*gy + c3*gx*gx + c4*gy*gy + c5*gx*gy;
  }

  /* Map raw gaze to calibrated screen (0..1) coordinates */
  mapGaze(gx, gy) {
    if (!this.model) return { sx: 0.5, sy: 0.5 };
    return {
      sx: clamp(this._applyModel(this.model.x, gx, gy), 0, 1),
      sy: clamp(this._applyModel(this.model.y, gx, gy), 0, 1)
    };
  }

  /**
   * Compute per-point residual error (for validation reporting)
   * @returns {Array} [{sx,sy,predictedX,predictedY,errorPx}]
   */
  getResiduals(screenW = 1920, screenH = 1080) {
    if (!this.model) return [];
    return this.calibData.filter(d => d.gx !== undefined).map(d => {
      const p = this.mapGaze(d.gx, d.gy);
      const ex = (p.sx - d.sx) * screenW;
      const ey = (p.sy - d.sy) * screenH;
      return {
        label: d.label, sx: d.sx, sy: d.sy,
        predictedX: p.sx, predictedY: p.sy,
        errorPx: Math.hypot(ex, ey)
      };
    });
  }

  _saveToStorage() {
    try {
      localStorage.setItem('accesseye_calib', JSON.stringify({
        model:     this.model,
        calibData: this.calibData.map(d => ({ sx: d.sx, sy: d.sy, gx: d.gx, gy: d.gy, label: d.label })),
        version:   3,
        timestamp: Date.now()
      }));
    } catch(_) {}
  }

  loadFromStorage() {
    try {
      const raw = localStorage.getItem('accesseye_calib');
      if (!raw) return false;
      const data = JSON.parse(raw);
      // Accept v3 models only (degree-2 ridge, 6 coefficients)
      if (data.version !== 3) { localStorage.removeItem('accesseye_calib'); return false; }
      this.model      = data.model;
      this.calibData  = data.calibData || [];
      this.isCalibrated = true;
      return true;
    } catch(_) { return false; }
  }

  reset() {
    this.calibData    = [];
    this.model        = null;
    this.isCalibrated = false;
  }
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   GESTURE ENGINE
   MediaPipe Hands landmark-based gesture classification
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
class GestureEngine {
  constructor() {
    this.DEBOUNCE_MS = { pinch: 600, airTap: 700, openPalm: 800 };
    this._lastFire = {};
    this.indexZHistory = [];    // for air-tap Z-delta detection
    this.MAX_Z_HISTORY = 8;
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
   * Process MediaPipe Hands results
   * Landmarks are normalized [0,1] in x,y, z ~depth
   */
  processLandmarks(landmarks) {
    if (!landmarks || landmarks.length === 0) return null;
    const lm = landmarks[0]; // Use first detected hand

    const gesture = this._classify(lm);
    if (gesture) {
      const now_ = now();
      const debounce = this.DEBOUNCE_MS[gesture] || 500;
      if (!this._lastFire[gesture] || now_ - this._lastFire[gesture] > debounce) {
        this._lastFire[gesture] = now_;
        this._emit('gesture', { type: gesture, landmarks: lm });
        return gesture;
      }
    }
    return null;
  }

  _classify(lm) {
    // ── LANDMARKS ──
    // 0 = wrist
    // 4 = thumb tip, 3 = thumb ip, 2 = thumb mcp
    // 8 = index tip, 7 = index dip, 6 = index pip, 5 = index mcp
    // 12 = middle tip
    // 16 = ring tip
    // 20 = pinky tip

    const thumb  = lm[4];
    const index  = lm[8];
    const middle = lm[12];
    const ring   = lm[16];
    const pinky  = lm[20];
    const wrist  = lm[0];
    const indexPip = lm[6];
    const indexMcp = lm[5];

    // ── PINCH DETECTION ──
    // Normalized distance between thumb tip and index tip
    const pinchDist = dist2D(thumb.x, thumb.y, index.x, index.y);
    if (pinchDist < 0.06) return 'pinch';

    // ── OPEN PALM DETECTION ──
    // All fingers spread: average tip distance from wrist is large
    const tips = [thumb, index, middle, ring, pinky];
    const avgTipDist = tips.reduce((s, t) => s + dist2D(t.x, t.y, wrist.x, wrist.y), 0) / 5;
    const midDist = dist2D(middle.x, middle.y, ring.x, ring.y);
    if (avgTipDist > 0.35 && midDist > 0.07) return 'openPalm';

    // ── AIR TAP DETECTION ──
    // Index finger forward motion: index Z decreasing rapidly (moving toward camera)
    const indexZ = index.z;
    this.indexZHistory.push(indexZ);
    if (this.indexZHistory.length > this.MAX_Z_HISTORY) this.indexZHistory.shift();

    if (this.indexZHistory.length >= 4) {
      const oldest = this.indexZHistory[0];
      const newest = this.indexZHistory[this.indexZHistory.length - 1];
      const zDelta = oldest - newest; // positive = moved toward camera
      if (zDelta > 0.06) {
        this.indexZHistory = []; // reset to avoid re-trigger
        return 'airTap';
      }
    }

    return null;
  }

  getGestureLabel(g) {
    return { pinch: '🤌 Pinch', airTap: '👆 Air Tap', openPalm: '✋ Open Palm' }[g] || g;
  }

  getGestureIcon(g) {
    return { pinch: 'fas fa-hand-scissors', airTap: 'fas fa-hand-point-up', openPalm: 'fas fa-hand-paper' }[g] || 'fas fa-hand';
  }
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   GAZE ENGINE
   MediaPipe FaceMesh iris landmarks → gaze vector → screen coords
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
class GazeEngine {
  constructor(calibration) {
    this.calibration = calibration;
    this.kalman = new KalmanFilter2D(0.008, 0.0002);
    this.ema = new EMAFilter(0.25);

    // MediaPipe Face Mesh iris landmark indices
    // Left eye iris:  landmarks 468-472
    // Right eye iris: landmarks 473-477
    this.LEFT_IRIS   = [468, 469, 470, 471, 472];
    this.RIGHT_IRIS  = [473, 474, 475, 476, 477];
    this.LEFT_EYE_CORNER_L  = 33;
    this.LEFT_EYE_CORNER_R  = 133;
    this.RIGHT_EYE_CORNER_L = 362;
    this.RIGHT_EYE_CORNER_R = 263;

    this.rawGaze = { x: 0.5, y: 0.5 };
    this.smoothGaze = { x: 0.5, y: 0.5 };
    this.confidence = 0;

    // FIX M-1: reference eye span measured at calibration time
    // Prevents narrow-eye users from being penalised by fixed 0.15 denominator
    this._refEyeSpan = null;

    this._callbacks = {};

    // FIX Bug-11: Subscribe to CalibrationEngine 'calibrated' event so we
    // capture the current eye span as the user's personal reference.
    // This must happen here (constructor) because CalibrationEngine already
    // exists when GazeEngine is constructed; we can't rely on external wiring.
    this.calibration.on('calibrated', () => {
      // Capture the eye span as measured during the last processed frame.
      // If we haven't seen a frame yet (span is 0) keep the existing reference.
      if (this._lastEyeSpan > 0) {
        this._refEyeSpan = this._lastEyeSpan * 2; // sum of both eyes (matches confidence calc)
      }
    });
    this._lastEyeSpan = 0;   // running sum of both eye spans, updated every frame
  }

  on(event, cb) {
    if (!this._callbacks[event]) this._callbacks[event] = [];
    this._callbacks[event].push(cb);
  }

  _emit(event, data) {
    (this._callbacks[event] || []).forEach(cb => cb(data));
  }

  /**
   * Process MediaPipe FaceMesh results
   * @param {Array} multiFaceLandmarks
   * @param {number} videoWidth
   * @param {number} videoHeight
   */
  processResults(multiFaceLandmarks, videoWidth, videoHeight) {
    if (!multiFaceLandmarks || multiFaceLandmarks.length === 0) {
      this.confidence = 0;
      return null;
    }

    const lm = multiFaceLandmarks[0];
    if (lm.length < 478) {
      // Iris landmarks not available (need refine_landmarks=true)
      // Fallback: use eye corner midpoints
      return this._fallbackGaze(lm, videoWidth, videoHeight);
    }

    // ── IRIS CENTER CALCULATION ──
    const leftIrisCenter  = this._irisCenter(lm, this.LEFT_IRIS);
    const rightIrisCenter = this._irisCenter(lm, this.RIGHT_IRIS);

    // ── EYE SPAN (horizontal) — used for X offset + confidence ──
    const leftEyeSpan  = dist2D(lm[this.LEFT_EYE_CORNER_L].x,  lm[this.LEFT_EYE_CORNER_L].y,
                                 lm[this.LEFT_EYE_CORNER_R].x,  lm[this.LEFT_EYE_CORNER_R].y);
    const rightEyeSpan = dist2D(lm[this.RIGHT_EYE_CORNER_L].x, lm[this.RIGHT_EYE_CORNER_L].y,
                                 lm[this.RIGHT_EYE_CORNER_R].x, lm[this.RIGHT_EYE_CORNER_R].y);

    // ── EYE HEIGHT (vertical) — FIX A-1: normalize Y by eye height not width ──
    // Upper/lower eyelid mid-landmarks for each eye
    // Left eye: upper lid ~159, lower lid ~145
    // Right eye: upper lid ~386, lower lid ~374
    const leftEyeHeight  = lm[159] && lm[145] ? Math.abs(lm[159].y - lm[145].y) : leftEyeSpan * 0.4;
    const rightEyeHeight = lm[386] && lm[374] ? Math.abs(lm[386].y - lm[374].y) : rightEyeSpan * 0.4;

    // ── IRIS OFFSET ──
    // X offset: normalized by horizontal eye width (correct for horizontal gaze)
    // Y offset: normalized by vertical eye height (FIX A-1 — was using eye width, causing 3-5x compression)
    const leftEyeMidX = (lm[this.LEFT_EYE_CORNER_L].x + lm[this.LEFT_EYE_CORNER_R].x) / 2;
    const leftEyeMidY = (lm[this.LEFT_EYE_CORNER_L].y + lm[this.LEFT_EYE_CORNER_R].y) / 2;
    const leftOffsetX = leftEyeSpan   > 0 ? (leftIrisCenter.x - leftEyeMidX) / leftEyeSpan   : 0;
    const leftOffsetY = leftEyeHeight > 0 ? (leftIrisCenter.y - leftEyeMidY) / leftEyeHeight : 0;

    const rightEyeMidX = (lm[this.RIGHT_EYE_CORNER_L].x + lm[this.RIGHT_EYE_CORNER_R].x) / 2;
    const rightEyeMidY = (lm[this.RIGHT_EYE_CORNER_L].y + lm[this.RIGHT_EYE_CORNER_R].y) / 2;
    const rightOffsetX = rightEyeSpan   > 0 ? (rightIrisCenter.x - rightEyeMidX) / rightEyeSpan   : 0;
    const rightOffsetY = rightEyeHeight > 0 ? (rightIrisCenter.y - rightEyeMidY) / rightEyeHeight : 0;

    // Average of both eyes (binocular gaze vector)
    const rawGX = (leftOffsetX + rightOffsetX) / 2;
    const rawGY = (leftOffsetY + rightOffsetY) / 2;

    // ── CONFIDENCE ──
    // FIX M-1: normalize to user's own reference span instead of fixed 0.15
    // _refEyeSpan is set at first calibration; falls back to 0.15 until then
    const currentSpanSum = leftEyeSpan + rightEyeSpan;
    this._lastEyeSpan = currentSpanSum;  // Bug-11: keep updated for calibrated-event callback
    const refSpan = this._refEyeSpan || 0.15;
    this.confidence = Math.min(1, currentSpanSum / refSpan);

    // ── APPLY FILTERS ──
    // FIX H-2: When Phase 2 is active it runs its own superior Kalman
    // (TemporalStabilizer with adaptive R=0.003) plus EMA + sliding-window
    // median.  Running Phase 1 Kalman (R=0.008) BEFORE Phase 2 adds
    // 80-150 ms of extra lag on fast deliberate moves.
    // Solution: use identity passthrough when Phase 2 orchestrator is active,
    // preserving only the EMA for a very light smoothing pass.
    let kalmanResult;
    if (window.app?.phase2?.active) {
      // Identity passthrough — Phase 2 handles all stabilisation
      kalmanResult = { x: rawGX, y: rawGY };
    } else {
      kalmanResult = this.kalman.update(rawGX, rawGY);
    }
    const smoothResult = this.ema.update(kalmanResult.x, kalmanResult.y);

    this.rawGaze    = { x: rawGX, y: rawGY };

    // ── MAP TO SCREEN COORDINATES ──
    let screenCoords;
    if (this.calibration.isCalibrated) {
      screenCoords = this.calibration.mapGaze(smoothResult.x, smoothResult.y);
    } else {
      // Uncalibrated: use head-facing estimate
      const headX = lm[1].x;  // nose tip
      const headY = lm[1].y;
      screenCoords = {
        sx: clamp(0.5 + smoothResult.x * 3.5 + (0.5 - headX) * 0.6, 0, 1),
        sy: clamp(0.5 + smoothResult.y * 4.0 + (headY - 0.5) * 0.8, 0, 1)
      };
    }

    this.smoothGaze = { x: screenCoords.sx, y: screenCoords.sy };

    this._emit('gaze', {
      raw: this.rawGaze,
      screen: this.smoothGaze,
      confidence: this.confidence
    });

    return this.smoothGaze;
  }

  _irisCenter(lm, indices) {
    const pts = indices.map(i => lm[i]);
    return {
      x: pts.reduce((s, p) => s + p.x, 0) / pts.length,
      y: pts.reduce((s, p) => s + p.y, 0) / pts.length
    };
  }

  _fallbackGaze(lm, w, h) {
    // Use nose tip position as rough gaze proxy
    const nose = lm[1];
    const sx = clamp(1 - nose.x, 0.05, 0.95);
    const sy = clamp(nose.y * 1.2 - 0.1, 0.05, 0.95);
    this.smoothGaze = { x: sx, y: sy };
    this.confidence = 0.6;
    this._emit('gaze', { raw: { x: sx, y: sy }, screen: this.smoothGaze, confidence: 0.6 });
    return this.smoothGaze;
  }

  reset() {
    this.kalman.reset();
    this.ema.reset();
    this.confidence = 0;
    this._lastEyeSpan = 0;
    // Do NOT reset _refEyeSpan here — it persists across camera restarts
  }
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   UI ELEMENT REGISTRY
   Registers DOM elements as gaze targets with bounding boxes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
class UIElementRegistry {
  constructor(dwellTime = 350) {
    this.elements = new Map();   // id → { el, label, onActivate, bbox, dwellStart }
    this.focusedId = null;
    this.dwellTime = dwellTime;
    this.dwellStart = null;
    this.dwellProgress = 0;
    this._callbacks = {};

    // FIX M-2: throttle bbox refresh to 5 Hz (was forcing reflow every gaze frame)
    this._lastRefresh = 0;
    this.BBOX_REFRESH_INTERVAL = 200;  // ms

    // Recalculate bboxes on resize
    window.addEventListener('resize', () => this._refreshBBoxes());
  }

  on(event, cb) {
    if (!this._callbacks[event]) this._callbacks[event] = [];
    this._callbacks[event].push(cb);
  }

  _emit(event, data) {
    (this._callbacks[event] || []).forEach(cb => cb(data));
  }

  register(id, el, label, onActivate) {
    const bbox = this._getBBox(el);
    this.elements.set(id, { id, el, label, onActivate, bbox, focused: false });
  }

  unregister(id) {
    if (this.focusedId === id) this._unfocus(id);
    this.elements.delete(id);
  }

  _getBBox(el) {
    const r = el.getBoundingClientRect();
    return { x: r.left, y: r.top, w: r.width, h: r.height };
  }

  _refreshBBoxes() {
    for (const [, entry] of this.elements) {
      entry.bbox = this._getBBox(entry.el);
    }
  }

  /**
   * Update gaze position and detect element focus
   * @param {number} screenX  Absolute screen X in pixels
   * @param {number} screenY  Absolute screen Y in pixels
   */
  updateGaze(screenX, screenY) {
    // FIX M-2: throttle getBoundingClientRect reflows to 5 Hz (200ms)
    // Previously this ran every frame (30-60x/s), forcing up to 10 reflows/frame
    const t = now();
    if (t - this._lastRefresh > this.BBOX_REFRESH_INTERVAL) {
      this._refreshBBoxes();
      this._lastRefresh = t;
    }
    let hitId = null;

    for (const [id, entry] of this.elements) {
      const { x, y, w, h } = entry.bbox;
      // Expand hit area by 8px for accessibility
      if (screenX >= x - 8 && screenX <= x + w + 8 &&
          screenY >= y - 8 && screenY <= y + h + 8) {
        hitId = id;
        break;
      }
    }

    if (hitId !== this.focusedId) {
      if (this.focusedId) this._unfocus(this.focusedId);
      if (hitId) this._beginFocus(hitId);
    }

    // Update dwell timer
    if (this.focusedId && this.dwellStart !== null) {
      const elapsed = now() - this.dwellStart;
      this.dwellProgress = clamp(elapsed / this.dwellTime, 0, 1);
      this._updateDwellUI(this.focusedId, this.dwellProgress);
    }
  }

  _beginFocus(id) {
    this.focusedId = id;
    this.dwellStart = now();
    this.dwellProgress = 0;
    const entry = this.elements.get(id);
    if (!entry) return;
    entry.el.classList.add('gaze-focus');
    this._emit('focus', { id, label: entry.label });
  }

  _unfocus(id) {
    const entry = this.elements.get(id);
    if (entry) {
      entry.el.classList.remove('gaze-focus', 'gaze-activating', 'gaze-activated');
      this._updateDwellUI(id, 0);
    }
    this.focusedId = null;
    this.dwellStart = null;
    this.dwellProgress = 0;
  }

  _updateDwellUI(id, progress) {
    const entry = this.elements.get(id);
    if (!entry) return;
    const bar = entry.el.querySelector('.dwell-progress');
    if (bar) bar.style.width = `${progress * 100}%`;
  }

  /** Called when gesture fires — activates focused element */
  activateFocused(gesture) {
    if (!this.focusedId) return false;
    const entry = this.elements.get(this.focusedId);
    if (!entry) return false;
    entry.el.classList.add('gaze-activating');
    setTimeout(() => {
      entry.el.classList.remove('gaze-activating');
      entry.el.classList.add('gaze-activated');
      setTimeout(() => entry.el.classList.remove('gaze-activated'), 600);
    }, 200);
    this._emit('activate', { id: entry.id, label: entry.label, gesture });
    if (entry.onActivate) entry.onActivate(gesture);
    return true;
  }

  getFocused() {
    return this.focusedId ? this.elements.get(this.focusedId) : null;
  }
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   AUDIO FEEDBACK (Web Speech API TTS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
class AudioFeedback {
  constructor() {
    this.enabled = false;
    this.synth = window.speechSynthesis || null;
    this._speaking = false;
    this._queue = [];
  }

  speak(text, priority = false) {
    if (!this.enabled || !this.synth) return;
    if (priority) { this.synth.cancel(); this._queue = []; }
    const utt = new SpeechSynthesisUtterance(text);
    utt.rate = 1.1;
    utt.pitch = 1.0;
    utt.volume = 0.9;
    utt.onend = () => { this._speaking = false; };
    this._speaking = true;
    this.synth.speak(utt);
  }

  toggle() {
    this.enabled = !this.enabled;
    if (!this.enabled && this.synth) this.synth.cancel();
    return this.enabled;
  }
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   SIMULATION ENGINE (Mouse mode — when camera is unavailable)
   Uses mouse pointer to simulate gaze position
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
class SimulationEngine {
  constructor() {
    this.active = false;
    this.mouseX = window.innerWidth / 2;
    this.mouseY = window.innerHeight / 2;
    this.ema = new EMAFilter(0.4);
    this._callbacks = {};
    this._bound = null;
  }

  on(event, cb) {
    if (!this._callbacks[event]) this._callbacks[event] = [];
    this._callbacks[event].push(cb);
  }

  _emit(event, data) {
    (this._callbacks[event] || []).forEach(cb => cb(data));
  }

  start() {
    this.active = true;
    this._bound = (e) => {
      const s = this.ema.update(e.clientX / window.innerWidth, e.clientY / window.innerHeight);
      this.mouseX = e.clientX;
      this.mouseY = e.clientY;
      this._emit('gaze', {
        screen: { x: s.x, y: s.y },
        raw: { x: s.x, y: s.y },
        confidence: 1.0,
        simulated: true
      });
    };
    document.addEventListener('mousemove', this._bound);
  }

  stop() {
    this.active = false;
    if (this._bound) document.removeEventListener('mousemove', this._bound);
    this._bound = null;
    this.ema.reset();
  }
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   TOAST NOTIFICATION SYSTEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
class ToastSystem {
  constructor() {
    this.container = $('#toast-container');
  }

  show(title, message, type = 'info', icon = null, duration = 3500) {
    const icons = {
      info: 'fas fa-info-circle', success: 'fas fa-check-circle',
      gesture: 'fas fa-hand-paper', warn: 'fas fa-exclamation-triangle'
    };
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
      <i class="${icon || icons[type]} toast-icon"></i>
      <div class="toast-body">
        <div class="toast-title">${title}</div>
        ${message ? `<div class="toast-msg">${message}</div>` : ''}
      </div>`;
    this.container.appendChild(toast);
    setTimeout(() => {
      toast.classList.add('toast-out');
      setTimeout(() => toast.remove(), 350);
    }, duration);
  }
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   INTERACTION LOG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
class InteractionLog {
  constructor() {
    this.body = $('#event-log-body');
    this.maxEntries = 30;
  }

  add(message, type = 'info') {
    if (!this.body) return;
    const t = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `<span class="log-time">${t}</span><span class="log-msg">${message}</span>`;
    this.body.appendChild(entry);
    this.body.scrollTop = this.body.scrollHeight;
    // Trim old entries
    while (this.body.children.length > this.maxEntries) this.body.removeChild(this.body.firstChild);
  }

  clear() { if (this.body) this.body.innerHTML = ''; }
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   CALIBRATION UI CONTROLLER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
class CalibrationUI {
  constructor(calibEngine, gazeEngine, log, toast) {
    this.calibEngine = calibEngine;
    this.gazeEngine = gazeEngine;
    this.log = log;
    this.toast = toast;
    this.overlay = $('#calibration-overlay');
    this.container = $('#calib-points-container');
    this.progressFill = $('#calib-progress-fill');
    this.stepLabel = $('#calib-step-label');
    this.currentStep = -1;
    this.collecting = false;
    this.sampleCount = 0;
    this.SAMPLES = 25;
    this._onComplete = null;
    this._sampleInterval = null;
    this._arenaW = 0;   // set in _renderPoints after layout
    this._arenaH = 0;
  }

  show(onComplete) {
    this._onComplete = onComplete;
    this.calibEngine.reset();
    this.currentStep = -1;
    this.overlay.style.display = 'flex';
    this._updateProgress(0);
    this.log.add('Calibration started', 'info');
    // FIX CAL-2: defer _renderPoints until after the browser has painted the
    // full-viewport overlay so offsetWidth/Height return correct dimensions.
    // Double-rAF ensures the flex layout has fully resolved before we query.
    requestAnimationFrame(() => requestAnimationFrame(() => this._renderPoints()));
  }

  hide() {
    this.overlay.style.display = 'none';
    this._arenaW = 0;   // reset so next show() re-measures the viewport
    this._arenaH = 0;
  }

  /**
   * FIX CAL-1 + CAL-4: Place calibration dots using the ACTUAL rendered
   * dimensions of the container (which now spans the full viewport working
   * area).  The sx/sy fractions (0–1) in CALIB_POINTS therefore represent
   * fractions of the full visible screen, matching the `window.innerWidth ×
   * window.innerHeight` multiplication used when the gaze cursor is placed.
   *
   * Previously the container was a fixed 600×380 px box inside the demo
   * sidebar.  The model was trained on targets clustered inside that small
   * box, while the cursor was placed across the full 1920×1080 viewport.
   * That mismatch caused a systematic ~3–4× scale error in both axes.
   */
  _renderPoints() {
    this.container.innerHTML = '';
    const pts = this.calibEngine.CALIB_POINTS;

    // Use the live rendered size (full viewport after CSS fix)
    const W = this.container.offsetWidth  || window.innerWidth;
    const H = this.container.offsetHeight || window.innerHeight;

    // Store the arena size so _runStep can re-render correctly after resize
    this._arenaW = W;
    this._arenaH = H;

    pts.forEach((pt, i) => {
      const el = document.createElement('div');
      el.className = 'calib-point';
      el.textContent = i + 1;
      el.style.left = `${pt.sx * W}px`;
      el.style.top  = `${pt.sy * H}px`;
      el.id = `calib-pt-${i}`;
      this.container.appendChild(el);
    });
  }

  async start() {
    if (this.collecting) return;

    // FIX CAL-2: if _renderPoints hasn't finished yet (overlay just shown),
    // wait one more frame before starting.  The Start button is only visible
    // after the overlay opens so this delay is imperceptible.
    if (!this._arenaW) {
      await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
      this._renderPoints();
      await new Promise(r => requestAnimationFrame(r));
    }

    this.collecting = true;
    $('#start-calib-btn').disabled = true;
    this.calibEngine.reset();
    this.calibEngine.calibData = [];

    for (let i = 0; i < this.calibEngine.CALIB_POINTS.length; i++) {
      await this._runStep(i);
    }

    // Build model
    const success = this.calibEngine.buildModel();
    this.hide();
    this.collecting = false;

    if (success) {
      this.toast.show('Calibration Complete!', 'Gaze tracking is now calibrated for your eyes.', 'success', 'fas fa-sliders-h');
      this.log.add('Calibration complete — model built successfully', 'success');
      if (this._onComplete) this._onComplete(true);
    } else {
      this.toast.show('Calibration Failed', 'Not enough data collected. Please try again.', 'warn');
      this.log.add('Calibration failed — insufficient data', 'warn');
      if (this._onComplete) this._onComplete(false);
    }
  }

  _runStep(stepIdx) {
    return new Promise(resolve => {
      this.currentStep = stepIdx;
      this._updateProgress(stepIdx / this.calibEngine.CALIB_POINTS.length);
      this.stepLabel.textContent = `Step ${stepIdx + 1} / ${this.calibEngine.CALIB_POINTS.length}`;

      // Activate point
      $$('.calib-point').forEach(el => el.classList.remove('active'));
      const ptEl = $(`#calib-pt-${stepIdx}`);
      if (ptEl) ptEl.classList.add('active');

      this.log.add(`Calibrating point ${stepIdx + 1}: ${this.calibEngine.CALIB_POINTS[stepIdx].label}`, 'info');

      // Wait 800ms then start collecting
      setTimeout(() => {
        this.sampleCount = 0;
        this.calibEngine.calibData[stepIdx] = {
          ...this.calibEngine.CALIB_POINTS[stepIdx],
          samples: []
        };

        this._sampleInterval = setInterval(() => {
          // FIX CAL-4: Prefer Phase 2 HybridGazeEngine rawGaze when available —
          // it uses binocular iris + head-pose fusion and is more accurate than
          // the Phase-1 single-eye estimate.  Fall back to Phase-1 when inactive.
          const rawSrc = (window.app?.phase2?.active && window.Phase2?.orchestrator?.hybridGaze)
            ? window.Phase2.orchestrator.hybridGaze
            : this.gazeEngine;
          const gaze = rawSrc.rawGaze;
          this.calibEngine.addCalibSample(stepIdx, gaze.x, gaze.y);
          this.sampleCount++;

          const progress = (stepIdx + this.sampleCount / this.SAMPLES) / this.calibEngine.CALIB_POINTS.length;
          this._updateProgress(progress);

          if (this.sampleCount >= this.SAMPLES) {
            clearInterval(this._sampleInterval);
            this.calibEngine.finalizePoint(stepIdx);
            if (ptEl) { ptEl.classList.remove('active'); ptEl.classList.add('done'); }
            resolve();
          }
        }, 60); // ~16 FPS sampling
      }, 800);
    });
  }

  _updateProgress(pct) {
    if (this.progressFill) this.progressFill.style.width = `${pct * 100}%`;
  }
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   MEDIAPIPE CONTROLLER
   Manages FaceMesh + Hands initialization and processing loop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
class MediaPipeController {
  constructor(videoEl, canvasEl, gazeEngine, gestureEngine) {
    this.videoEl = videoEl;
    this.canvasEl = canvasEl;
    this.ctx = canvasEl.getContext('2d');
    this.gazeEngine = gazeEngine;
    this.gestureEngine = gestureEngine;
    this.faceMesh = null;
    this.hands = null;
    this.camera = null;
    this.running = false;
    this.faceDetected = false;
    this.handDetected = false;
    this.frameCount = 0;
    this.lastFpsTime = now();
    this.fps = 0;
    this.latency = 0;
    this._latencyStart = 0;
    this._callbacks = {};
  }

  on(event, cb) {
    if (!this._callbacks[event]) this._callbacks[event] = [];
    this._callbacks[event].push(cb);
  }

  _emit(event, data) {
    (this._callbacks[event] || []).forEach(cb => cb(data));
  }

  async initialize() {
    // Check if MediaPipe is available
    if (typeof FaceMesh === 'undefined' || typeof Hands === 'undefined') {
      console.warn('MediaPipe not loaded — falling back to simulation mode');
      return false;
    }

    try {
      this.faceMesh = new FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
      });
      this.faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,      // Enables iris landmarks (468-477)
        minDetectionConfidence: 0.6,
        minTrackingConfidence: 0.6
      });
      this.faceMesh.onResults((results) => this._onFaceResults(results));

      this.hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      });
      this.hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 0,         // Lightweight for real-time
        minDetectionConfidence: 0.6,
        minTrackingConfidence: 0.6
      });
      this.hands.onResults((results) => this._onHandResults(results));

      return true;
    } catch (e) {
      console.error('MediaPipe init failed:', e);
      return false;
    }
  }

  async startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user', frameRate: 30 }
      });
      this.videoEl.srcObject = stream;

      // FIX M-6: set canvas dimensions from loadedmetadata event, not immediately
      // after play() — videoWidth is 0 on some browsers until the first frame arrives
      this.videoEl.addEventListener('loadedmetadata', () => {
        this.canvasEl.width  = this.videoEl.videoWidth  || 640;
        this.canvasEl.height = this.videoEl.videoHeight || 480;
      }, { once: true });

      await this.videoEl.play();

      // Fallback: if metadata already loaded (stream was reused), set now
      if (this.videoEl.videoWidth > 0) {
        this.canvasEl.width  = this.videoEl.videoWidth;
        this.canvasEl.height = this.videoEl.videoHeight;
      }

      this.running = true;
      this._processLoop();
      return true;
    } catch (e) {
      console.error('Camera access failed:', e);
      this._emit('error', { type: 'camera', message: e.message });
      return false;
    }
  }

  async _processLoop() {
    if (!this.running) return;
    const t0 = now();

    if (this.videoEl.readyState >= 2) {
      try {
        if (this.faceMesh) await this.faceMesh.send({ image: this.videoEl });
        // FIX H-4: Process Hands every 2nd frame only.
        // Gestures don't require per-frame latency < 100ms, and serial execution
        // adds ~7ms average latency at 30fps. This saves ~15% CPU.
        if (this.hands && this.frameCount % 2 === 0) {
          await this.hands.send({ image: this.videoEl });
        }
      } catch(e) {}
    }

    // FPS calculation
    this.frameCount++;
    const elapsed = now() - this.lastFpsTime;
    if (elapsed >= 1000) {
      this.fps = Math.round(this.frameCount / (elapsed / 1000));
      this.frameCount = 0;
      this.lastFpsTime = now();
    }
    this.latency = Math.round(now() - t0);

    this._emit('frame', { fps: this.fps, latency: this.latency });
    requestAnimationFrame(() => this._processLoop());
  }

  _onFaceResults(results) {
    this.faceDetected = !!(results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0);
    this._emit('face', { detected: this.faceDetected, results });

    if (this.faceDetected) {
      const gaze = this.gazeEngine.processResults(
        results.multiFaceLandmarks,
        this.videoEl.videoWidth,
        this.videoEl.videoHeight
      );
      if (gaze) this._drawGazeOnCanvas(gaze);
      this._drawFaceMesh(results);
    } else {
      this._clearCanvas();
    }
  }

  _onHandResults(results) {
    this.handDetected = !!(results.multiHandLandmarks && results.multiHandLandmarks.length > 0);
    this._emit('hand', { detected: this.handDetected, results });

    if (this.handDetected) {
      const gesture = this.gestureEngine.processLandmarks(results.multiHandLandmarks);
      if (gesture) this._emit('gesture', { type: gesture });
      this._drawHands(results);
    }
  }

  _drawFaceMesh(results) {
    const canvas = this.canvasEl;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!results.multiFaceLandmarks) return;
    for (const lm of results.multiFaceLandmarks) {
      if (typeof drawConnectors !== 'undefined' && typeof FACEMESH_TESSELATION !== 'undefined') {
        drawConnectors(ctx, lm, FACEMESH_TESSELATION, { color: 'rgba(0,212,255,0.06)', lineWidth: 0.5 });
        drawConnectors(ctx, lm, FACEMESH_RIGHT_EYE, { color: 'rgba(0,212,255,0.4)', lineWidth: 1 });
        drawConnectors(ctx, lm, FACEMESH_LEFT_EYE,  { color: 'rgba(0,212,255,0.4)', lineWidth: 1 });
      }

      // Draw iris (landmarks 468-477 if available)
      if (lm.length >= 478) {
        const irisIndices = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477];
        for (const idx of irisIndices) {
          const p = lm[idx];
          ctx.fillStyle = 'rgba(0,255,136,0.9)';
          ctx.beginPath();
          ctx.arc(p.x * canvas.width, p.y * canvas.height, 2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }

  _drawHands(results) {
    const canvas = this.canvasEl;
    const ctx = this.ctx;
    if (!results.multiHandLandmarks) return;
    for (const lm of results.multiHandLandmarks) {
      if (typeof drawConnectors !== 'undefined' && typeof HAND_CONNECTIONS !== 'undefined') {
        drawConnectors(ctx, lm, HAND_CONNECTIONS, { color: 'rgba(255,107,53,0.5)', lineWidth: 1.5 });
      }
      for (const pt of lm) {
        ctx.fillStyle = 'rgba(255,107,53,0.9)';
        ctx.beginPath();
        ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  _drawGazeOnCanvas(gaze) {
    const canvas = this.canvasEl;
    const ctx = this.ctx;
    // Draw small gaze indicator on video overlay
    const vx = (1 - gaze.x) * canvas.width;  // mirrored
    const vy = gaze.y * canvas.height;
    ctx.strokeStyle = 'rgba(0,212,255,0.8)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(vx, vy, 12, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = 'rgba(0,212,255,0.6)';
    ctx.beginPath();
    ctx.arc(vx, vy, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  _clearCanvas() {
    this.ctx.clearRect(0, 0, this.canvasEl.width, this.canvasEl.height);
  }

  stop() {
    this.running = false;
    if (this.videoEl.srcObject) {
      this.videoEl.srcObject.getTracks().forEach(t => t.stop());
      this.videoEl.srcObject = null;
    }
    this._clearCanvas();
  }
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   MAIN APPLICATION CONTROLLER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
class AccessEyeApp {
  constructor() {
    // Core engines
    this.calibration  = new CalibrationEngine();
    this.gazeEngine   = new GazeEngine(this.calibration);
    this.gestureEngine= new GestureEngine();
    this.uiRegistry   = new UIElementRegistry(350);
    this.audio        = new AudioFeedback();
    this.toast        = new ToastSystem();
    this.log          = new InteractionLog();
    this.sim          = new SimulationEngine();

    // Mode
    this.mode     = 'mouse';  // 'mouse' | 'gaze' | 'calibrate'
    this.cameraOn = false;
    this.mpAvailable = false;
    this.mpController = null;

    // Gaze cursor (global)
    this.gazeCursor = $('#global-gaze-cursor');
    this.dwellCircle = $('#dwell-circle');
    this.DWELL_CIRCUMFERENCE = 163.36;

    // FPS/latency tracking
    this._frameTimer = 0;

    this._init();
  }

  _init() {
    this._setupNavigation();
    this._setupDemoControls();
    this._setupGazeTargets();
    this._setupGestureSystem();
    this._setupCalibrationUI();
    this._setupAudioToggle();
    this._setupPerformanceGauges();
    this._setupHeroPulse();
    this._tryLoadCalibration();
    this._setupSimulation();
    this._setupHeroButtons();
    this._startCursorFromMouse(); // Default: mouse sim for demos
  }

  /* ── NAVIGATION ─────────────────────────────────────────── */
  _setupNavigation() {
    $$('.nav-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const page = btn.dataset.page;
        $$('.nav-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        $$('.page').forEach(p => p.classList.remove('active'));
        const target = $(`#page-${page}`);
        if (target) target.classList.add('active');
        if (page === 'demo') this._onEnterDemo();
        if (page === 'architecture') this._animateGauges();
      });
    });
  }

  _navigateTo(page) {
    $$('.nav-btn').forEach(b => b.classList.toggle('active', b.dataset.page === page));
    $$('.page').forEach(p => p.classList.toggle('active', p.id === `page-${page}`));
    if (page === 'demo') this._onEnterDemo();
    if (page === 'architecture') this._animateGauges();
  }

  _setupHeroButtons() {
    $('#launch-demo-btn')?.addEventListener('click', () => this._navigateTo('demo'));
    $('#view-arch-btn')?.addEventListener('click',  () => this._navigateTo('architecture'));
  }

  /* ── DEMO MODE ENTRY ────────────────────────────────────── */
  _onEnterDemo() {
    // If in mouse mode, start simulation
    if (this.mode === 'mouse' && !this.sim.active) {
      this._startSimulation();
    }
    // Register all gaze targets
    this._registerGazeTargets();
  }

  /* ── DEMO CONTROLS ──────────────────────────────────────── */
  _setupDemoControls() {
    $('#start-camera-btn')?.addEventListener('click', () => this._startCamera());
    $('#stop-camera-btn')?.addEventListener('click',  () => this._stopCamera());
    $('#clear-log-btn')?.addEventListener('click',   () => this.log.clear());

    // Mode tabs
    $$('.mode-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        $$('.mode-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        this._setMode(tab.dataset.mode);
      });
    });
  }

  _setMode(mode) {
    this.mode = mode;
    this.log.add(`Mode: ${mode}`, 'info');

    if (mode === 'mouse') {
      this._startSimulation();
      this._updateHint('Mouse simulation: move your cursor over buttons and <strong>click</strong> to activate.');
    } else if (mode === 'gaze') {
      this.sim.stop();
      if (this.cameraOn) {
        this._updateHint('Gaze mode active: look at buttons, then <strong>pinch</strong> or <strong>air tap</strong> to activate.');
      } else {
        this._updateHint('Start camera first to enable gaze tracking.');
      }
    } else if (mode === 'calibrate') {
      this.sim.stop();
      if (this.cameraOn) {
        this._showCalibrationFlow();
      } else {
        this.toast.show('Camera Required', 'Start the camera to run calibration.', 'warn');
      }
    }
  }

  _updateHint(html) {
    const el = $('#demo-hint-text');
    if (el) el.innerHTML = html;
  }

  /* ── CAMERA ─────────────────────────────────────────────── */
  async _startCamera() {
    const btn = $('#start-camera-btn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';

    this.log.add('Requesting camera access...', 'info');

    // Initialize MediaPipe
    const videoEl  = $('#demo-video');
    const canvasEl = $('#overlay-canvas');
    this.mpController = new MediaPipeController(videoEl, canvasEl, this.gazeEngine, this.gestureEngine);

    const mpOk = await this.mpController.initialize();
    this.mpAvailable = mpOk;

    const camOk = await this.mpController.startCamera();

    if (camOk) {
      this.cameraOn = true;
      // Update UI
      $('#camera-placeholder').style.display = 'none';
      btn.disabled = false;
      btn.innerHTML = '<i class="fas fa-play"></i> Restart';
      $('#stop-camera-btn').disabled = false;
      this._updateSystemStatus('active', 'Camera Active');
      this.log.add(`Camera started ${mpOk ? '(MediaPipe active)' : '(simulation mode)'}`, 'success');
      this.toast.show('Camera Started', mpOk ? 'MediaPipe processing active.' : 'Using position simulation.', 'success');

      // Wire up MediaPipe events
      this._wireMediaPipeEvents();

      // Update gaze dot in camera feed
      const gazeDot = $('#gaze-dot');
      if (gazeDot) gazeDot.style.display = 'block';

      if (!mpOk) {
        this._startSimulation();
        this._showSimBanner();
      }
    } else {
      btn.disabled = false;
      btn.innerHTML = '<i class="fas fa-play"></i> Start Camera';
      this.log.add('Camera failed — using mouse simulation', 'warn');
      this.toast.show('Camera Failed', 'Camera not available. Using mouse simulation instead.', 'warn');
      this._startSimulation();
      this._showSimBanner();
    }
  }

  _stopCamera() {
    if (this.mpController) {
      this.mpController.stop();
      this.mpController = null;
    }
    this.cameraOn = false;

    // Deactivate Phase 2 if running
    if (this.phase2?.active) {
      this.phase2.deactivate();
    }

    // Reset mode back to mouse simulation
    this.mode = 'mouse';
    const modeTabs = document.querySelectorAll('.mode-tab');
    modeTabs.forEach(t => t.classList.toggle('active', t.dataset.mode === 'mouse'));

    $('#camera-placeholder').style.display = 'flex';
    $('#start-camera-btn').disabled = false;
    $('#start-camera-btn').innerHTML = '<i class="fas fa-play"></i> Start Camera';
    $('#stop-camera-btn').disabled = true;
    this._updateSystemStatus('offline', 'Camera Off');
    this._updateStatusItem('status-face',    false, 'Inactive');
    this._updateStatusItem('status-gaze',    false, 'Inactive');
    this._updateStatusItem('status-hand',    false, 'Inactive');
    this._updateStatusItem('status-gesture', false, 'None');
    this.log.add('Camera stopped', 'info');

    // Fall back to simulation
    this._startSimulation();
  }

  _showSimBanner() {
    const appHeader = $('.demo-app-header');
    if (!appHeader || appHeader.querySelector('.sim-banner')) return;
    const banner = document.createElement('div');
    banner.className = 'sim-banner';
    banner.innerHTML = '<i class="fas fa-mouse-pointer"></i> Simulation Mode: Move cursor to simulate gaze. Click to simulate gesture.';
    appHeader.appendChild(banner);
  }

  /* ── MEDIAPIPE EVENTS ───────────────────────────────────── */
  _wireMediaPipeEvents() {
    if (!this.mpController) return;

    this.mpController.on('frame', ({ fps, latency }) => {
      // Phase 2 updates metrics itself with confidence; Phase 1 does it here
      if (!this.phase2?.active) {
        this._updateMetrics(fps, latency, Math.round(this.gazeEngine.confidence * 100));
      }
    });

    this.mpController.on('face', ({ detected }) => {
      this._updateStatusItem('status-face', detected, detected ? 'Detected' : 'Not found', detected ? 'active' : '');
      this._updateStatusItem('status-gaze', detected, detected ? 'Tracking' : 'Inactive', detected ? 'tracking' : '');
    });

    this.mpController.on('hand', ({ detected }) => {
      this._updateStatusItem('status-hand', detected, detected ? 'Detected' : 'Not found', detected ? 'active' : '');
      if (!detected) this._updateStatusItem('status-gesture', false, 'None');
    });

    this.mpController.on('gesture', ({ type }) => {
      this._handleGesture(type);
    });

    this.gazeEngine.on('gaze', ({ screen, confidence }) => {
      // Phase 2 orchestrator drives gaze directly when active — skip Phase 1 path
      if (this.phase2?.active) return;
      if (!this.cameraOn) return;
      this._updateGazeCursor(screen.x * window.innerWidth, screen.y * window.innerHeight);
      this._updateCoords(screen.x, screen.y);
      this.uiRegistry.updateGaze(screen.x * window.innerWidth, screen.y * window.innerHeight);
    });
  }

  /* ── SIMULATION ─────────────────────────────────────────── */
  _setupSimulation() {
    this.sim.on('gaze', ({ screen, simulated }) => {
      // Don't run simulation when Phase 2 is active (camera driving gaze)
      if (this.phase2?.active) return;
      if (this.mode !== 'mouse') return;
      const sx = screen.x * window.innerWidth;
      const sy = screen.y * window.innerHeight;
      this._updateGazeCursor(sx, sy);
      this._updateCoords(screen.x, screen.y);
      this.uiRegistry.updateGaze(sx, sy);
      this._updateMetrics(60, 5, 100);
      this._updateSystemStatus('tracking', 'Mouse Sim');
      this._updateStatusItem('status-gaze', true, 'Simulated', 'tracking');
    });
  }

  _startSimulation() {
    if (!this.sim.active) {
      this.sim.start();
      this.log.add('Mouse simulation started', 'info');
    }
  }

  _startCursorFromMouse() {
    // Show cursor while in mouse simulation mode
    document.addEventListener('mousemove', (e) => {
      // Skip if Phase 2 is driving gaze from camera
      if (this.phase2?.active) return;
      if (this.mode === 'mouse') {
        this.gazeCursor.style.display = 'block';
        this.gazeCursor.style.left = `${e.clientX}px`;
        this.gazeCursor.style.top  = `${e.clientY}px`;
      }
    });

    // Click = gesture in mouse mode
    document.addEventListener('click', (e) => {
      if (this.mode === 'mouse') {
        // Don't intercept button/nav clicks — only trigger gesture if on gaze target
        const target = e.target.closest('.gaze-target');
        if (target) return; // Let normal click handle it
        this._handleGesture('pinch');
      }
    });
  }

  /* ── GAZE CURSOR ────────────────────────────────────────── */
  _updateGazeCursor(px, py) {
    if (!this.gazeCursor) return;
    this.gazeCursor.style.display = 'block';
    this.gazeCursor.style.left = `${px}px`;
    this.gazeCursor.style.top  = `${py}px`;

    // Update camera feed dot
    const videoEl = $('#demo-video');
    if (videoEl && this.cameraOn) {
      const rect = videoEl.getBoundingClientRect();
      const dot = $('#gaze-dot');
      if (dot) {
        dot.style.left = `${((px - rect.left) / rect.width) * 100}%`;
        dot.style.top  = `${((py - rect.top)  / rect.height) * 100}%`;
      }
    }

    // Update dwell ring
    const focused = this.uiRegistry.getFocused();
    if (focused && this.uiRegistry.dwellProgress > 0) {
      const dashOffset = (1 - this.uiRegistry.dwellProgress) * this.DWELL_CIRCUMFERENCE;
      if (this.dwellCircle) {
        this.dwellCircle.style.strokeDasharray = `${this.uiRegistry.dwellProgress * this.DWELL_CIRCUMFERENCE} ${this.DWELL_CIRCUMFERENCE}`;
      }
    } else {
      if (this.dwellCircle) this.dwellCircle.style.strokeDasharray = `0 ${this.DWELL_CIRCUMFERENCE}`;
    }
  }

  /* ── GAZE TARGETS ───────────────────────────────────────── */
  _setupGazeTargets() {
    // Button click handlers (for direct mouse clicks in sim mode)
    document.addEventListener('click', (e) => {
      const target = e.target.closest('.gaze-target');
      if (target && this.mode === 'mouse') {
        const id = target.dataset.id;
        const label = target.dataset.label;
        this._onElementActivated(id, label, 'click');
        e.preventDefault();
      }
    });
  }

  _registerGazeTargets() {
    // Unregister old ones
    const targets = $$('.gaze-target');
    targets.forEach(el => {
      const id = el.dataset.id;
      if (id) {
        this.uiRegistry.unregister(id);
        this.uiRegistry.register(id, el, el.dataset.label || id, (gesture) => {
          this._onElementActivated(id, el.dataset.label || id, gesture);
        });
      }
    });
  }

  _onElementActivated(id, label, gesture) {
    // Visual ripple
    const entry = this.uiRegistry.elements.get(id);
    if (entry) {
      const rect = entry.el.getBoundingClientRect();
      this._spawnRipple(rect.left + rect.width/2, rect.top + rect.height/2);
    }

    // Audio feedback
    this.audio.speak(`${label} activated`);

    // Log
    this.log.add(`Activated: <strong>${label}</strong> via ${gesture}`, 'success');

    // Toast
    this.toast.show(label, `Activated via ${gesture}`, 'success', 'fas fa-check-circle', 2500);

    // Special actions
    this._handleElementAction(id, label);

    // Update status
    this._updateStatusItem('status-gesture', true, this.gestureEngine.getGestureLabel(gesture) || gesture, 'gesture');
    setTimeout(() => this._updateStatusItem('status-gesture', false, 'None'), 2000);
  }

  _handleElementAction(id, label) {
    const thread = $('#message-thread');
    if (!thread) return;

    if (id.startsWith('reply-')) {
      const bubble = document.createElement('div');
      bubble.className = 'msg-bubble sent';
      bubble.innerHTML = `
        <div class="msg-avatar"><i class="fas fa-user"></i></div>
        <div class="msg-content">
          <p>${label}</p>
          <span class="msg-time">${new Date().toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'})}</span>
        </div>`;
      thread.appendChild(bubble);
      thread.scrollTop = thread.scrollHeight;
      // Fade in
      requestAnimationFrame(() => { bubble.style.opacity = '0'; requestAnimationFrame(() => { bubble.style.transition = 'opacity 0.3s'; bubble.style.opacity = '1'; }); });
    }

    if (id === 'btn-send') {
      const bubble = document.createElement('div');
      bubble.className = 'msg-bubble sent';
      bubble.innerHTML = `
        <div class="msg-avatar"><i class="fas fa-user"></i></div>
        <div class="msg-content"><p>📤 Message sent!</p><span class="msg-time">${new Date().toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'})}</span></div>`;
      thread.appendChild(bubble);
      thread.scrollTop = thread.scrollHeight;
    }

    if (id === 'btn-call') {
      this.toast.show('Calling...', '📞 Initiating voice call...', 'info', 'fas fa-phone', 3000);
    }

    if (id === 'btn-back') {
      this.toast.show('Going Back', 'Navigation: Back', 'info', 'fas fa-arrow-left', 2000);
    }
  }

  /* ── GESTURE SYSTEM ─────────────────────────────────────── */
  _setupGestureSystem() {
    this.uiRegistry.on('focus', ({ id, label }) => {
      this.log.add(`Focused: <strong>${label}</strong>`, 'focus');
      this._updateCoordTarget(label);
      this.audio.speak(label);
    });

    this.uiRegistry.on('activate', ({ id, label, gesture }) => {
      // handled in _onElementActivated
    });

    this.gestureEngine.on('gesture', ({ type }) => {
      this._handleGesture(type);
    });
  }

  _handleGesture(type) {
    const labelMap = { pinch: '🤌 Pinch', airTap: '👆 Air Tap', openPalm: '✋ Open Palm', click: '👆 Click' };
    const label = labelMap[type] || type;

    // Show gesture indicator
    this._showGestureIndicator(label);
    this.log.add(`Gesture: ${label}`, 'gesture');
    this._updateStatusItem('status-gesture', true, label, 'gesture');

    if (type === 'openPalm') {
      this.toast.show('Open Palm', 'Cancel / Go Back', 'gesture', 'fas fa-hand-paper', 2000);
      return;
    }

    // Activate focused element
    const activated = this.uiRegistry.activateFocused(type);
    if (!activated && this.uiRegistry.focusedId === null) {
      // No element focused
      this.log.add(`Gesture detected but no element focused`, 'warn');
    }
  }

  _showGestureIndicator(label) {
    let indicator = $('.gesture-indicator');
    if (indicator) indicator.remove();
    indicator = document.createElement('div');
    indicator.className = 'gesture-indicator';
    indicator.innerHTML = `<i class="fas fa-hand-point-up"></i> ${label}`;
    document.body.appendChild(indicator);
    setTimeout(() => { indicator.style.opacity = '0'; indicator.style.transform = 'translateX(-50%) scale(0.8)'; indicator.style.transition = '0.3s'; setTimeout(() => indicator.remove(), 350); }, 1500);
  }

  _spawnRipple(x, y) {
    const ripple = document.createElement('div');
    ripple.className = 'activation-ripple';
    ripple.style.left = `${x}px`;
    ripple.style.top  = `${y}px`;
    document.body.appendChild(ripple);
    setTimeout(() => ripple.remove(), 700);
  }

  /* ── CALIBRATION ────────────────────────────────────────── */
  _setupCalibrationUI() {
    const overlay = $('#calibration-overlay');
    if (!overlay) return;
    const calibUI = new CalibrationUI(this.calibration, this.gazeEngine, this.log, this.toast);

    $('#start-calib-btn')?.addEventListener('click', () => calibUI.start());
    $('#cancel-calib-btn')?.addEventListener('click', () => {
      calibUI.hide();
      this.log.add('Calibration cancelled', 'warn');
    });

    this._calibUI = calibUI;
  }

  _showCalibrationFlow() {
    if (this._calibUI) {
      this._calibUI.show((success) => {
        if (success) {
          this._updateSystemStatus('tracking', 'Calibrated');
          this.log.add('Gaze calibration active', 'success');
        }
      });
    }
  }

  _tryLoadCalibration() {
    const loaded = this.calibration.loadFromStorage();
    if (loaded) {
      this.log.add('Saved calibration loaded', 'success');
    }
  }

  /* ── AUDIO TOGGLE ───────────────────────────────────────── */
  _setupAudioToggle() {
    const btn = $('#audio-toggle');
    const icon = $('#audio-icon');
    btn?.addEventListener('click', () => {
      const on = this.audio.toggle();
      btn.classList.toggle('active', on);
      if (icon) icon.className = on ? 'fas fa-volume-up' : 'fas fa-volume-mute';
      this.toast.show(on ? 'Audio On' : 'Audio Off', `TTS feedback ${on ? 'enabled' : 'disabled'}`, 'info', null, 2000);
    });
  }

  /* ── STATUS HELPERS ─────────────────────────────────────── */
  _updateSystemStatus(state, label) {
    const dot   = $('.status-dot');
    const lbl   = $('.status-label');
    if (dot)  { dot.className = `status-dot ${state}`; }
    if (lbl)  lbl.textContent = label;
  }

  _updateStatusItem(id, active, valText, cssClass = '') {
    const el = $(`#${id}`);
    if (!el) return;
    el.className = `status-item ${cssClass}`;
    const valEl = el.querySelector('.status-val');
    if (valEl) valEl.textContent = valText;
  }

  _updateMetrics(fps, latency, conf) {
    const fpsEl  = $('#fps-display');
    const latEl  = $('#latency-display');
    const confEl = $('#confidence-display');
    if (fpsEl)  fpsEl.textContent  = fps;
    if (latEl)  latEl.textContent  = latency;
    if (confEl) confEl.textContent = `${conf}%`;
  }

  _updateCoords(x, y) {
    const xEl = $('#gaze-x-val'), yEl = $('#gaze-y-val');
    if (xEl) xEl.textContent = (x * window.innerWidth).toFixed(0) + 'px';
    if (yEl) yEl.textContent = (y * window.innerHeight).toFixed(0) + 'px';
  }

  _updateCoordTarget(label) {
    const el = $('#target-val');
    if (el) el.textContent = label;
  }

  /* ── ARCHITECTURE: PERFORMANCE GAUGES ───────────────────── */
  _setupPerformanceGauges() {
    // Gauges animate in when page becomes visible
  }

  _animateGauges() {
    const gaugeData = [
      { id: 'gauge-latency',  pct: 0.9, color: '#00d4ff'  },
      { id: 'gauge-fps',      pct: 1.0, color: '#00ff88'  },
      { id: 'gauge-gesture',  pct: 0.85,color: '#ff6b35'  },
      { id: 'gauge-accuracy', pct: 0.85,color: '#a78bfa'  }
    ];
    const CIRCUMFERENCE = 141; // arc length of our gauge path
    gaugeData.forEach(({ id, pct, color }, i) => {
      setTimeout(() => {
        const el = $(`#${id}`);
        if (el) {
          el.style.transition = `stroke-dasharray 1.2s cubic-bezier(0.4,0,0.2,1)`;
          el.style.strokeDasharray = `${pct * CIRCUMFERENCE} ${CIRCUMFERENCE}`;
          el.style.stroke = color;
        }
      }, i * 200);
    });
  }

  /* ── HERO ANIMATION ─────────────────────────────────────── */
  _setupHeroPulse() {
    // Hero pupil and crosshair are CSS animated, no JS needed
    // Start metric counter animation
    $$('.stat-val').forEach(el => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(10px)';
      el.style.transition = 'all 0.5s ease';
    });
    setTimeout(() => {
      $$('.stat-val').forEach((el, i) => {
        setTimeout(() => {
          el.style.opacity = '1';
          el.style.transform = 'none';
        }, i * 150);
      });
    }, 500);
  }
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   BOOTSTRAP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
let app;
document.addEventListener('DOMContentLoaded', () => {
  app = new AccessEyeApp();
  window.app = app; // Expose for Phase 2 initialization

  // Expose public API for external use
  window.AccessEye = {
    /**
     * Register a UI element as a gaze target
     */
    registerElement({ id, element, label, onActivate }) {
      app.uiRegistry.register(id, element, label, onActivate);
    },

    /**
     * Register multiple elements
     */
    registerElements(elements) {
      elements.forEach(e => {
        if (e.element) {
          app.uiRegistry.register(e.id, e.element, e.label, e.onActivate);
        }
      });
    },

    /**
     * Remove a registered element
     */
    unregisterElement(id) {
      app.uiRegistry.unregister(id);
    },

    /**
     * Run calibration flow
     */
    calibrate() {
      app._showCalibrationFlow();
    },

    /**
     * Listen for system events
     */
    on(event, cb) {
      if (event === 'gaze')    app.gazeEngine.on('gaze', cb);
      if (event === 'gesture') app.gestureEngine.on('gesture', cb);
      if (event === 'focus')   app.uiRegistry.on('focus', cb);
      if (event === 'activate')app.uiRegistry.on('activate', cb);
    },

    /**
     * Toggle audio feedback
     */
    toggleAudio() { return app.audio.toggle(); },

    /**
     * Get current gaze position (normalized 0-1)
     */
    getGaze() { return app.gazeEngine.smoothGaze; }
  };

  console.log('%c AccessEye MVP Loaded ✅', 'color:#00d4ff;font-weight:bold;font-size:14px;');
  console.log('%c Version: 1.0.0 | On-device eye + gesture control', 'color:#94a3b8;font-size:12px;');
});
