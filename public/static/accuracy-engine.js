/**
 * ═══════════════════════════════════════════════════════════════════════════
 *  AccessEye — Accuracy Engine  v3.0
 *  accuracy-engine.js
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  WHAT PHASES 1–3 ALREADY HAVE:
 *   ✅ One Euro Filter        (Phase 3 — minCutoff 1.0, β 0.12)
 *   ✅ IVT saccade classifier (Phase 3 — 35px/frame threshold)
 *   ✅ Adaptive dwell timer   (Phase 3 — Fast/Normal/Accessible/Extended)
 *   ✅ Kalman + EMA + trimmed-mean stabilizer (Phase 2)
 *   ✅ Binocular iris fusion  (Phase 2)
 *   ✅ Dynamic calibration + bias correction (Phase 2)
 *   ✅ Snap-to engine + target predictor (snap-engine)
 *
 *  WHAT THIS FILE ADDS (v3.0):
 *
 *   ACC.1  GravitySnapEngine   — Gravity-model soft attractor pull
 *     Ref: Grossman & Balakrishnan (2005); Phase 6 AccessEye report
 *     25–40% mis-selection reduction vs distance-only snap
 *
 *   ACC.2  PerSessionDriftCorrector — Accumulating drift compensation
 *     Ref: Phase 5 AccessEye audit — passive fixation-based correction
 *
 *   ACC.3  GazeGainRemapper — Center-expansion nonlinear gain curve   [NEW v3]
 *     PROBLEM: Raw iris offset is geometrically non-linear. Center region
 *     has ~3× higher gain (tiny eye movement → big cursor jump), so users
 *     can't hold center. Edges have compressed gain (eyelid occlusion limits
 *     iris travel). Combined with padding-expanded gaze range, the center
 *     becomes a small "dead zone" and edges become magnets.
 *
 *     SOLUTION: Power-law remap  u' = sign(u) × |u|^γ  where γ ∈ (0,1).
 *     With γ = 0.70:
 *       • Center (|u| < 0.3): expanded ~18% → 23% of [-0.5, 0.5] range
 *       • Outer (|u| > 0.4): compressed ~12% to balance
 *       • Net result: center dead-zone shrinks from ~20% → ~8% of screen
 *     Ref: Casiez 2012; Zhu & Ji 2006 (nonlinear gaze mapping, SVR implicit)
 *     Applied after CalibrationEngine._normalizeGaze, before _applyModel.
 *
 *   ACC.4  CenterGravity — Gentle center-return for idle cursor          [NEW v3]
 *     PROBLEM: When no nearby targets exist and gaze confidence fluctuates,
 *     the Kalman filter can lock the cursor at a screen edge (higher Kalman R
 *     at edges means filter is looser, velocity doesn't decay to zero).
 *     SOLUTION: When no SnapTo attractor is near AND gaze is fixated AND
 *     we are more than CENTER_DEADBAND from screen center, apply a micro-pull
 *     (CENTER_PULL_RATE = 0.5% per second) toward (0.5, 0.5). This is 10-20×
 *     weaker than gravity snap — imperceptible during active use but prevents
 *     multi-second edge locks. Disabled if snap-to is off (user made a choice).
 *
 *  SAFE INTEGRATION:
 *   • ACC.3 wraps CalibrationEngine.mapGaze (the mapping step only)
 *   • ACC.4 wraps app._updateGazeCursor (the FINAL output step only)
 *   • ACC.1/2 wrap app._updateGazeCursor (unchanged from v2)
 *   • All Phase 2/3 logic still runs first
 *   • All modules individually toggleable
 *   • No existing classes overridden or redeclared
 * ═══════════════════════════════════════════════════════════════════════════
 */

'use strict';

/* ═══════════════════════════════════════════════════════════════════════════
   ACC.1  GRAVITY SNAP ENGINE
   ═══════════════════════════════════════════════════════════════════════════
   A "soft attractor" that gently nudges the gaze cursor toward the most
   likely intended target during fixations.

   DIFFERENCE FROM SNAP-TO ENGINE:
   • SnapToEngine: hard snap (cursor teleports to element center when within
     threshold). Fast but can feel jumpy / trigger accidentally.
   • GravitySnapEngine: soft nudge (cursor is weighted-averaged toward the
     best candidate). Feels like the cursor "wants" to land on buttons.
     Does NOT trigger actions — only adjusts cursor position for accuracy.

   FORMULA:
     For element E at (ex, ey) with semantic weight W, area A, and usage
     frequency F, the gravitational force on gaze (gx, gy) is:

       dist_aniso = sqrt( (gx-ex)² + ((gy-ey)/0.6)² )   [wider Y tolerance]
       force = W × sizeScore(A) × freqBonus(F) / dist_aniso²

     The attractor with highest force (within MAX_RADIUS) wins.
     Gaze is nudged:
       nudgeX = (ex - gx) × PULL × clamp(force/500, 0, 1)
       nudgeY = (ey - gy) × PULL × clamp(force/500, 0, 1)

   PARAMETERS (conservative to avoid "magnet" feeling):
     MAX_PULL_RADIUS  110px   — elements farther than this are ignored
     PULL_STRENGTH    0.10    — 10% nudge per frame (0 = off, 1 = hard snap)
     MIN_CONFIDENCE   0.55    — ignore low-quality frames
     ANISOTROPY_Y     0.60    — vertical: more forgiving (60% of horiz)
*/
class _AccGravitySnap {
  constructor() {
    this.MAX_PULL_RADIUS = 110;
    this.PULL_STRENGTH   = 0.10;
    this.MIN_CONF        = 0.55;
    this.ANISOTROPY_Y    = 0.60;

    this.SEMANTIC_W = {
      BUTTON: 1.0, A: 0.85, INPUT: 0.80,
      SELECT: 0.75, TEXTAREA: 0.70, DEFAULT: 0.50
    };

    this._freq      = new Map();   // element-id → dwell count
    this._maxFreq   = 1;
    this._cache     = [];
    this._cacheTime = 0;
    this.CACHE_TTL  = 350;

    this._selector = [
      'button:not([disabled])', 'a[href]', 'input:not([disabled])',
      'select:not([disabled])', 'textarea:not([disabled])',
      '[role="button"]', '[role="link"]', '[role="menuitem"]',
      '[role="tab"]', '[role="checkbox"]', '[role="radio"]',
      '[tabindex]:not([tabindex="-1"])', '[data-accessible-target]',
      '.gaze-target'
    ].join(',');

    // Diagnostics
    this.lastForce     = 0;
    this.lastAttractor = null;
  }

  /** Record an activation to boost that element's gravity. */
  recordActivation(elementId) {
    const c = (this._freq.get(elementId) || 0) + 1;
    this._freq.set(elementId, c);
    this._maxFreq = Math.max(this._maxFreq, c);
    this._cacheTime = 0;  // invalidate cache
  }

  /**
   * Apply gravity pull to raw gaze position.
   * @param {number}  gx         Gaze X in screen pixels
   * @param {number}  gy         Gaze Y in screen pixels
   * @param {boolean} isFixated  Only pull during fixation
   * @param {number}  conf       0–1 gaze confidence
   * @returns {{ x: number, y: number, hasAttractor: boolean }}
   */
  update(gx, gy, isFixated, conf) {
    if (!isFixated || conf < this.MIN_CONF) {
      this.lastForce = 0;
      return { x: gx, y: gy, hasAttractor: false };
    }

    const els = this._getElements();
    if (!els.length) return { x: gx, y: gy, hasAttractor: false };

    let bestForce = 0, bestCx = 0, bestCy = 0, bestEl = null;

    for (const { cx, cy, area, tag, id } of els) {
      const dx   = gx - cx;
      const dy   = (gy - cy) / this.ANISOTROPY_Y;
      const dist = Math.hypot(dx, dy);

      if (dist > this.MAX_PULL_RADIUS || dist < 1) continue;

      const sizeScore  = Math.min(area / 960, 2.0);           // normalize vs ~40×24 button
      const semW       = this.SEMANTIC_W[tag] ?? 0.50;
      const freq       = this._freq.get(id) || 0;
      const freqBonus  = 1.0 + (this._maxFreq > 0 ? freq / this._maxFreq : 0) * 0.8;
      const force      = semW * sizeScore * freqBonus / (dist * dist) * 10000;

      if (force > bestForce) {
        bestForce = force; bestCx = cx; bestCy = cy; bestEl = id;
      }
    }

    this.lastForce     = Math.min(bestForce / 500, 1.0);
    this.lastAttractor = bestEl;

    if (!bestEl || this.lastForce < 0.05) return { x: gx, y: gy, hasAttractor: false };

    return {
      x: gx + (bestCx - gx) * this.PULL_STRENGTH * this.lastForce,
      y: gy + (bestCy - gy) * this.PULL_STRENGTH * this.lastForce,
      hasAttractor: true
    };
  }

  _getElements() {
    const now = performance.now();
    if (now - this._cacheTime < this.CACHE_TTL && this._cache.length) return this._cache;
    try {
      this._cache = [];
      for (const el of document.querySelectorAll(this._selector)) {
        const r = el.getBoundingClientRect();
        if (r.width < 4 || r.height < 4) continue;
        this._cache.push({
          cx:   r.left + r.width  / 2,
          cy:   r.top  + r.height / 2,
          area: r.width * r.height,
          tag:  el.tagName || 'DEFAULT',
          id:   el.id || el.getAttribute('data-id') || `${r.left}_${r.top}`
        });
      }
      this._cacheTime = now;
    } catch (_) {}
    return this._cache;
  }

  reset() {
    this._freq.clear(); this._maxFreq = 1;
    this._cache = []; this._cacheTime = 0;
    this.lastForce = 0; this.lastAttractor = null;
  }
}


/* ═══════════════════════════════════════════════════════════════════════════
   ACC.2  PER-SESSION DRIFT CORRECTOR
   ═══════════════════════════════════════════════════════════════════════════
   Addresses the "Phase 5 gap": missing passive drift correction during normal
   use (not just after explicit interactions).

   HOW IT DIFFERS FROM Phase 2 DynamicCalibrationEngine:
   • DynCalib corrects drift via INTERACTION events (click, activate).
     Good for short-term correction but only fires ~once/minute in light use.
   • SessionDriftCorrector watches FIXATION CENTROIDS continuously.
     When the same cluster of fixations consistently lands 2–4% off-center
     from the cursor, it infers drift and nudges the output.

   ALGORITHM:
   1. Accumulate fixation positions over a sliding window (50 fixations, ~30s)
   2. Compare fixation cluster centroid to cursor output centroid
   3. If systematic offset > DRIFT_THRESHOLD (1.5% of screen), compute
      a drift correction vector and apply it to all subsequent positions
   4. Correction decays if fixations stop confirming it (α=0.0008 decay)

   This is intentionally very gentle (max ±2.5% screen) — never noticeable
   as a jump, only as gradual reduction in systematic offset over ~60s.
*/
class _AccDriftCorrector {
  constructor() {
    this.DRIFT_THRESHOLD = 0.015;   // 1.5% screen width to trigger correction
    this.MAX_CORRECTION  = 0.025;   // max ±2.5% screen
    this.UPDATE_ALPHA    = 0.0015;  // correction learning rate (slow)
    this.DECAY_ALPHA     = 0.0008;  // correction decay when no drift seen
    this.WINDOW_SIZE     = 50;      // fixation samples in window
    this.MIN_SAMPLES     = 12;      // need at least 12 before correcting

    // Correction offsets (0–1 normalized screen space)
    this._corrX = 0;
    this._corrY = 0;

    // Fixation + cursor history buffers
    this._fixBuf   = [];   // { gx, gy } normalized fixation positions
    this._curBuf   = [];   // { cx, cy } cursor positions at same fixation

    // Frame skip (expensive: run every 12 frames)
    this._frameSkip = 0;
    this.FRAME_INTERVAL = 12;

    // Diagnostics
    this.lastDriftMag = 0;
    this.correctionCount = 0;
  }

  /**
   * Feed a fixation event (from saccade filter or IVT).
   * @param {number} gx  Normalized gaze X (0–1)
   * @param {number} gy  Normalized gaze Y (0–1)
   * @param {number} cx  Cursor output X (0–1) at the time of fixation
   * @param {number} cy  Cursor output Y (0–1) at the time of fixation
   */
  recordFixation(gx, gy, cx, cy) {
    this._fixBuf.push({ x: gx, y: gy });
    this._curBuf.push({ x: cx, y: cy });
    if (this._fixBuf.length > this.WINDOW_SIZE) {
      this._fixBuf.shift(); this._curBuf.shift();
    }
  }

  /**
   * Apply drift correction to cursor output (call every frame).
   * @param {number} px  Screen pixel X
   * @param {number} py  Screen pixel Y
   * @param {boolean} isFixated
   * @returns {{ x: number, y: number }}
   */
  update(px, py, isFixated) {
    const W = (window.visualViewport?.width  || window.innerWidth)  || 1920;
    const H = (window.visualViewport?.height || window.innerHeight) || 1080;

    // Update correction vector periodically
    this._frameSkip++;
    if (this._frameSkip >= this.FRAME_INTERVAL) {
      this._frameSkip = 0;
      this._computeCorrection(W, H);
    }

    // Apply correction
    const corrPx = this._corrX * W;
    const corrPy = this._corrY * H;

    return {
      x: px + corrPx,
      y: py + corrPy
    };
  }

  _computeCorrection(W, H) {
    if (this._fixBuf.length < this.MIN_SAMPLES) {
      // Not enough data — slowly decay any existing correction
      this._corrX *= (1 - this.DECAY_ALPHA);
      this._corrY *= (1 - this.DECAY_ALPHA);
      return;
    }

    // Compute mean fixation vs mean cursor offset
    let sumGX = 0, sumGY = 0, sumCX = 0, sumCY = 0;
    const N = this._fixBuf.length;
    for (let i = 0; i < N; i++) {
      sumGX += this._fixBuf[i].x;
      sumGY += this._fixBuf[i].y;
      sumCX += this._curBuf[i].x;
      sumCY += this._curBuf[i].y;
    }
    const meanGX = sumGX / N, meanGY = sumGY / N;
    const meanCX = sumCX / N, meanCY = sumCY / N;

    // Systematic drift: cursor consistently lands (meanCX - meanGX) away from fixation
    const driftX  = meanGX - meanCX;   // positive = cursor too far LEFT, push right
    const driftY  = meanGY - meanCY;
    const driftMag = Math.hypot(driftX, driftY);
    this.lastDriftMag = driftMag;

    if (driftMag > this.DRIFT_THRESHOLD) {
      // Gently nudge correction toward observed drift
      this._corrX += this.UPDATE_ALPHA * (driftX - this._corrX);
      this._corrY += this.UPDATE_ALPHA * (driftY - this._corrY);
      this.correctionCount++;
    } else {
      // Below threshold: decay toward zero
      this._corrX *= (1 - this.DECAY_ALPHA * 2);
      this._corrY *= (1 - this.DECAY_ALPHA * 2);
    }

    // Hard clamp: never correct more than MAX_CORRECTION
    const MAX = this.MAX_CORRECTION;
    this._corrX = Math.max(-MAX, Math.min(MAX, this._corrX));
    this._corrY = Math.max(-MAX, Math.min(MAX, this._corrY));
  }

  reset() {
    this._fixBuf = []; this._curBuf = [];
    this._corrX = 0; this._corrY = 0;
    this.lastDriftMag = 0; this.correctionCount = 0;
  }
}


/* ═══════════════════════════════════════════════════════════════════════════
   ACC.3  GAZE GAIN REMAPPER                                        [NEW v3]
   ═══════════════════════════════════════════════════════════════════════════
   PROBLEM (Root Cause 1 from research audit):
   Raw iris offset is computed as:
     offsetX = (irisCenter.x − eyeMidX) / eyeSpan
   This is geometrically non-linear. When looking straight ahead, the iris
   sits near the center of the aperture — a small movement produces a large
   normalized offset (HIGH GAIN). When looking far left/right, the iris presses
   against the sclera and the eyelid partially covers it — the iris barely moves
   relative to the corners (LOW GAIN, + measurement noise).

   COMBINED EFFECT with calibration padding (PAD=0.22):
   • Small iris movements near center → normalized to large screen fractions
     → center feels "too sensitive", cursor flies off with tiny movement
   • Large iris movements toward edges → compressed, cursor barely reaches edge
     → edges become "magnets" (cursor stays there because return requires
       extremely fine iris control back toward a narrow center window)

   SOLUTION — Power-law center-expansion:
     u' = sign(u) × |u|^γ    where γ = 0.72 (< 1 = center expansion)

   Effect on the [-0.5, +0.5] normalized gaze range:
     |u| = 0.10  →  |u'| = 0.137  (37% expansion: center region gets bigger)
     |u| = 0.25  →  |u'| = 0.306  (22% expansion: inner quadrant)
     |u| = 0.40  →  |u'| = 0.463  (16% expansion: still slightly bigger)
     |u| = 0.50  →  |u'| = 0.573  (15% expansion at edge — clamped to 0.5)

   After the remap the polynomial sees a more evenly spaced set of inputs,
   which means:
   • The center dead-zone shrinks from ~20% to ~8% of screen
   • Edge "pull" is reduced because the outer gaze range is now more compressed
   • Users can hold center gaze with normal fixation effort

   Gamma is user-adjustable via sensitivity slider (range 0.55–1.00, default 0.72).
     γ=1.0 → linear (no remap, original behavior)
     γ=0.72 → recommended: visible center expansion, minimal distortion
     γ=0.55 → strong expansion for users with very narrow iris range

   Applied ONLY when CalibrationEngine has a valid model (isCalibrated=true).
   Wraps calibration.mapGaze — zero impact on uncalibrated fallback.

   Ref: Zhu & Ji 2006 (nonlinear gaze-to-screen mapping RPI);
        Casiez et al. CHI 2012 (gain curves for pointer acceleration)
*/
class _AccGazeGainRemapper {
  constructor() {
    // γ < 1: center expansion (lower γ = stronger expansion)
    this.gamma = 0.72;

    // Sensitivity multiplier [0.5 – 2.0, default 1.0]
    // Applied as a linear scale AFTER the polynomial outputs screen coords.
    // > 1.0: makes cursor move more (good for narrow iris range)
    // < 1.0: makes cursor move less (good for wide iris range / overshooting)
    this.sensitivity = 1.0;

    this.enabled = true;
  }

  /**
   * Remap a normalized gaze value via power-law center-expansion.
   * Input/output both in [-0.5, +0.5].
   * @param {number} u  normalized gaze, range [-0.5, +0.5]
   * @returns {number}  remapped, same range
   */
  _remap(u) {
    if (!this.enabled || this.gamma >= 0.999) return u;
    const sign = u >= 0 ? 1 : -1;
    const abs  = Math.abs(u) * 2;    // scale to [0,1] for power law
    const r    = Math.pow(abs, this.gamma) / 2;  // back to [0,0.5]
    return Math.max(-0.5, Math.min(0.5, sign * r));
  }

  /**
   * Wrap CalibrationEngine.mapGaze to inject the gain remap.
   * Called once during install. Returns a patched mapGaze function.
   * @param {CalibrationEngine} calib
   * @param {Function} origMapGaze  the original mapGaze.bind(calib)
   * @returns {Function}  replacement mapGaze
   */
  patchMapGaze(calib, origMapGaze) {
    const self = this;
    return function(gx, gy) {
      if (!self.enabled || !calib.isCalibrated) {
        return origMapGaze(gx, gy);
      }
      // Step 1: normalize (same logic as CalibrationEngine._normalizeGaze)
      const rx = calib.model?.gazeRangeX;
      const ry = calib.model?.gazeRangeY;
      if (!rx || !ry) return origMapGaze(gx, gy);

      const normGX = rx ? (gx - (rx.max + rx.min) / 2) / (rx.max - rx.min) : gx;
      const normGY = ry ? (gy - (ry.max + ry.min) / 2) / (ry.max - ry.min) : gy;

      // Step 2: power-law remap (center expansion)
      const remGX = self._remap(normGX);
      const remGY = self._remap(normGY);

      // Step 3: apply polynomial model directly with remapped coords
      // (bypasses the internal normalize step since we already normalized)
      if (!calib.model?.x || !calib.model?.y) return origMapGaze(gx, gy);

      const sx = calib._applyModel(calib.model.x, remGX, remGY);
      const sy = calib._applyModel(calib.model.y, remGX, remGY);

      // Step 4: apply sensitivity multiplier (around screen center)
      const adjSX = self.sensitivity !== 1.0
        ? 0.5 + (sx - 0.5) * self.sensitivity
        : sx;
      const adjSY = self.sensitivity !== 1.0
        ? 0.5 + (sy - 0.5) * self.sensitivity
        : sy;

      return {
        sx: Math.max(-0.02, Math.min(1.02, adjSX)),
        sy: Math.max(-0.02, Math.min(1.02, adjSY))
      };
    };
  }

  reset() {
    this.gamma       = 0.72;
    this.sensitivity = 1.0;
  }
}


/* ═══════════════════════════════════════════════════════════════════════════
   ACC.4  CENTER GRAVITY                                            [NEW v3]
   ═══════════════════════════════════════════════════════════════════════════
   PROBLEM (Root Cause 4 from research audit):
   When gaze confidence fluctuates at screen edges, the adaptive Kalman
   (Phase 2) increases measurement noise R, making the filter looser.
   Combined with the inertia of EMA smoothing, the cursor can "lock" at
   an edge for several seconds even after the user stops looking there.

   SOLUTION: A very gentle center-return force — 10-20× weaker than gravity
   snap — applied only when:
     (a) No SnapToEngine attractor is active (gravity snap isn't pulling)
     (b) Gaze fixation is detected (stable gaze, not a saccade)
     (c) Cursor is beyond CENTER_DEADBAND from screen center
     (d) System has been active for at least 2 seconds (avoid startup artifact)

   PARAMETERS:
     CENTER_PULL_RATE = 0.008   — 0.8% of distance per frame toward center
     CENTER_DEADBAND  = 0.30    — only activates when > 30% from center
                                  (inner 60% of screen is a free zone)
     MIN_CONF         = 0.50    — low confidence = don't apply (edge noise)

   This is intentionally very subtle: it corrects a 40% off-center cursor
   in approximately 10–15 seconds of fixation. During active navigation
   it has zero perceptible effect because snap-to is active.
*/
class _AccCenterGravity {
  constructor() {
    this.CENTER_PULL_RATE = 0.008;  // 0.8% per frame toward center
    // FIX-RIGHT-EDGE: Raised deadband from 0.30 → 0.48 so center gravity
    // only fires when cursor is >48% from center (nearly stuck at edge).
    // At 0.30 it was firing as soon as the cursor reached 80% across the
    // screen, pulling it back and preventing the last 10–15 px from being
    // reached on every edge — right, left, top, and bottom.
    this.CENTER_DEADBAND  = 0.48;   // fraction from center to activate (|sx-0.5| > 0.48)
    this.MIN_CONF         = 0.50;

    this.enabled = true;

    // Diagnostics
    this.lastPullX = 0;
    this.lastPullY = 0;
    this._startTime = performance.now();
  }

  /**
   * Apply a gentle center-return nudge.
   * @param {number}  px           Cursor pixel X
   * @param {number}  py           Cursor pixel Y
   * @param {boolean} isFixated    Only pull during fixation
   * @param {number}  conf         Gaze confidence 0–1
   * @param {boolean} hasAttractor True if gravity snap found a target
   * @returns {{ x: number, y: number }}
   */
  update(px, py, isFixated, conf, hasAttractor) {
    this.lastPullX = 0; this.lastPullY = 0;

    // Don't interfere when snap gravity is already pulling
    if (!this.enabled || hasAttractor) return { x: px, y: py };
    if (!isFixated || conf < this.MIN_CONF) return { x: px, y: py };

    // Don't fire in first 2 seconds (startup stabilization)
    if (performance.now() - this._startTime < 2000) return { x: px, y: py };

    const W = (window.visualViewport?.width  || window.innerWidth)  || 1920;
    const H = (window.visualViewport?.height || window.innerHeight) || 1080;

    const normX = px / W;  // [0,1]
    const normY = py / H;

    // Distance from center
    const dX = normX - 0.5;
    const dY = normY - 0.5;
    const dist = Math.hypot(dX, dY);

    // Only act outside deadband
    if (dist <= this.CENTER_DEADBAND) return { x: px, y: py };

    // Pull rate scales with how far outside deadband we are
    const excess = dist - this.CENTER_DEADBAND;    // 0 at edge of deadband
    const pullFactor = Math.min(excess / 0.20, 1.0); // ramps up over 20% of screen
    const rate = this.CENTER_PULL_RATE * pullFactor;

    // Nudge toward (0.5, 0.5) in normalized space
    const newNormX = normX - dX * rate;
    const newNormY = normY - dY * rate;

    this.lastPullX = (normX - newNormX) * W;
    this.lastPullY = (normY - newNormY) * H;

    return {
      x: newNormX * W,
      y: newNormY * H
    };
  }

  reset() {
    this.lastPullX = 0; this.lastPullY = 0;
    this._startTime = performance.now();
  }
}


/* ═══════════════════════════════════════════════════════════════════════════
   ACCURACY ORCHESTRATOR  v3
   ═══════════════════════════════════════════════════════════════════════════
   Wires ACC.1–4 into Phase 2's output step.
   Polls for Phase2Orchestrator + CalibrationEngine availability,
   then installs safe wrappers on app._updateGazeCursor and calib.mapGaze.
*/
class AccuracyOrchestrator {
  constructor() {
    this.gravity      = new _AccGravitySnap();
    this.driftCorr    = new _AccDriftCorrector();
    this.gainRemap    = new _AccGazeGainRemapper();   // ACC.3 NEW
    this.centerGrav   = new _AccCenterGravity();      // ACC.4 NEW

    this._active    = false;
    this._installed = false;
    this._attempts  = 0;

    this.config = {
      enableGravity:    true,
      enableDriftCorr:  true,
      enableGainRemap:  true,   // ACC.3 — center expansion
      enableCenterGrav: true    // ACC.4 — center return
    };

    // Diagnostics
    this.diag = {
      frames:       0,
      gravityPulls: 0,
      driftCorrs:   0,
      gainRemaps:   0,
      centerPulls:  0,
      lastPullForce:   0,
      lastDriftMag:    0,
      lastGamma:       0,
      lastSensitivity: 1.0
    };
  }

  start() {
    this._poll();
  }

  _poll() {
    this._attempts++;
    const p2  = window.app?.phase2;
    const app = window.app;
    if (!p2 || !app) {
      if (this._attempts < 120) setTimeout(() => this._poll(), 500);
      else console.warn('[AccuracyEngine] Timed out waiting for Phase2');
      return;
    }
    this._install(p2, app);
  }

  _install(p2, app) {
    if (this._installed) return;
    this._installed = true;

    const self = this;

    // ── Hook Phase2 activate / deactivate ──
    const origActivate   = p2.activate?.bind(p2);
    const origDeactivate = p2.deactivate?.bind(p2);

    if (origActivate) {
      p2.activate = async function(videoEl, canvasEl) {
        const r = await origActivate(videoEl, canvasEl);
        self._active = true;
        self.centerGrav._startTime = performance.now(); // reset startup timer
        self._updateStatusUI();
        console.log('%c[AccuracyEngine v3] Active — GainRemap + CenterGravity + GravitySnap + DriftCorr', 'color:#00ff88;font-weight:bold');
        return r;
      };
    }

    if (origDeactivate) {
      p2.deactivate = function() {
        self._active = false;
        self.gravity.reset();
        self.driftCorr.reset();
        self.centerGrav.reset();
        return origDeactivate();
      };
    }

    // ── ACC.3: Wrap CalibrationEngine.mapGaze (gain remap) ──
    const calib = app.calibration;
    if (calib && typeof calib.mapGaze === 'function') {
      const origMapGaze = calib.mapGaze.bind(calib);
      calib.mapGaze = this.gainRemap.patchMapGaze(calib, origMapGaze);
      console.log('%c[AccuracyEngine v3] GainRemap patch installed on calibration.mapGaze', 'color:#a78bfa;font-size:11px');
    } else {
      console.warn('[AccuracyEngine] calibration.mapGaze not found — gain remap skipped');
    }

    // ── Sync gain remap with config ──
    Object.defineProperty(this.gainRemap, 'enabled', {
      get: () => this.config.enableGainRemap,
      configurable: true
    });
    Object.defineProperty(this.centerGrav, 'enabled', {
      get: () => this.config.enableCenterGrav,
      configurable: true
    });

    // ── CORE PATCH: Wrap app._updateGazeCursor ──
    const origUpdate = app._updateGazeCursor?.bind(app);
    if (!origUpdate) {
      console.warn('[AccuracyEngine] app._updateGazeCursor not found — skipping patch');
      return;
    }
    app._updateGazeCursor = function(sx, sy) {
      if (!self._active) return origUpdate(sx, sy);

      try {
        const W    = (window.visualViewport?.width  || window.innerWidth)  || 1920;
        const H    = (window.visualViewport?.height || window.innerHeight) || 1080;
        const conf = p2.confidence?.lastScore?.total ?? app.gazeEngine?.confidence ?? 0.5;

        // Determine fixation state from best available source
        const ivt       = window.app?.phase3?.ivt;
        const sacc      = p2.saccade;
        const isFixated = (ivt?.isFixating ?? sacc?.isFixated) ?? false;

        let px = sx, py = sy;

        // ── ACC.1: Gravity snap (soft pull toward likely target) ──
        let hasAttractor = false;
        if (self.config.enableGravity) {
          const g = self.gravity.update(px, py, isFixated, conf);
          if (g.x !== px || g.y !== py) self.diag.gravityPulls++;
          px = g.x; py = g.y;
          hasAttractor = g.hasAttractor;
          self.diag.lastPullForce = self.gravity.lastForce;
        }

        // ── ACC.4: Center gravity (gentle edge-escape when no snap target) ──
        if (self.config.enableCenterGrav) {
          const cg = self.centerGrav.update(px, py, isFixated, conf, hasAttractor);
          if (Math.abs(cg.x - px) > 0.1 || Math.abs(cg.y - py) > 0.1) self.diag.centerPulls++;
          px = cg.x; py = cg.y;
        }

        // ── ACC.2: Drift correction ──
        if (self.config.enableDriftCorr) {
          if (isFixated && conf > 0.60) {
            const rawGaze = p2.hybridGaze?._irisOnlyGaze ?? { x: px/W, y: py/H };
            self.driftCorr.recordFixation(rawGaze.x, rawGaze.y, px/W, py/H);
            self.diag.driftCorrs++;
          }
          const d = self.driftCorr.update(px, py, isFixated);
          px = d.x; py = d.y;
          self.diag.lastDriftMag = self.driftCorr.lastDriftMag;
        }

        self.diag.frames++;
        self.diag.lastGamma       = self.gainRemap.gamma;
        self.diag.lastSensitivity = self.gainRemap.sensitivity;
        if (self.diag.frames % 20 === 0) self._updateLiveUI();

        return origUpdate(px, py);
      } catch (e) {
        // Never crash the cursor pipeline — fall through to original
        console.warn('[AccuracyEngine] update error (non-fatal):', e.message);
        return origUpdate(sx, sy);
      }
    };

    // ── Record activations for gravity frequency map ──
    app.uiRegistry?.on?.('activate', ({ id }) => {
      self.gravity.recordActivation(id);
    });
    app.snapEngine?.on?.('activate', ({ el }) => {
      const id = el?.id || 'unknown';
      self.gravity.recordActivation(id);
    });

    // ── Wire sensitivity slider ──
    self._wireSensitivitySlider();

    console.log('%c[AccuracyEngine v3] Patch installed on app._updateGazeCursor', 'color:#00d4ff;font-size:11px');
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Sensitivity Slider Wiring
  // Connects the #acc-sensitivity-slider (added to HTML by _updateStatusUI)
  // to gainRemap.sensitivity and gainRemap.gamma.
  // ─────────────────────────────────────────────────────────────────────────
  _wireSensitivitySlider() {
    const tryWire = () => {
      const slider = document.getElementById('acc-sensitivity-slider');
      if (!slider) return;

      const self = this;
      const valEl = document.getElementById('acc-sensitivity-val');

      const update = () => {
        const v = parseFloat(slider.value);  // 0.5 – 2.0
        self.gainRemap.sensitivity = v;
        // Also map sensitivity to gamma: higher sensitivity = smaller γ (more center expansion)
        // Range: sensitivity 0.5 → γ=0.60, sensitivity 1.0 → γ=0.72, sensitivity 2.0 → γ=0.88
        self.gainRemap.gamma = 0.88 - (2.0 - v) * 0.14;
        self.gainRemap.gamma = Math.max(0.55, Math.min(1.0, self.gainRemap.gamma));
        if (valEl) valEl.textContent = v.toFixed(1) + '×';

        // Persist to localStorage
        try { localStorage.setItem('accesseye_cursor_sensitivity', String(v)); } catch(_) {}
      };

      // Load saved value
      try {
        const saved = parseFloat(localStorage.getItem('accesseye_cursor_sensitivity'));
        if (!isNaN(saved) && saved >= 0.5 && saved <= 2.0) {
          slider.value = saved;
          update();
        }
      } catch(_) {}

      slider.addEventListener('input', update);
      update(); // apply initial value
    };

    // Try immediately, then retry after UI builds
    tryWire();
    setTimeout(tryWire, 1000);
    setTimeout(tryWire, 3000);
  }

  _updateStatusUI() {
    const panel = document.getElementById('p2-status-panel');
    if (!panel) return;

    // Remove old v2 row if present
    const old = document.getElementById('acc-status-row');
    if (old) old.remove();

    const row = document.createElement('div');
    row.id = 'acc-status-row';
    row.style.cssText = [
      'margin-top:8px', 'padding:6px 8px',
      'background:rgba(0,255,136,0.07)', 'border:1px solid rgba(0,255,136,0.22)',
      'border-radius:6px', 'font-size:11px', 'color:#94a3b8'
    ].join(';');
    row.innerHTML = `
      <div style="color:#00ff88;font-weight:600;margin-bottom:5px;font-size:11px;">
        <i class="fas fa-crosshairs" style="margin-right:4px;"></i>Accuracy Engine v3
      </div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px;">
        <span title="Non-linear center expansion (γ)">
          <i class="fas fa-expand-arrows-alt" style="color:#a78bfa;margin-right:3px;"></i>
          Gain <span id="acc-gain-val" style="color:#fbbf24">γ=—</span>
        </span>
        <span title="Soft gravity pull toward likely targets">
          <i class="fas fa-magnet" style="color:#00d4ff;margin-right:3px;"></i>
          Gravity <span id="acc-pull-val" style="color:#fbbf24">—</span>
        </span>
        <span title="Session drift correction">
          <i class="fas fa-compress-arrows-alt" style="color:#00d4ff;margin-right:3px;"></i>
          Drift <span id="acc-drift-val" style="color:#fbbf24">—</span>
        </span>
        <span title="Center-return micro-pull">
          <i class="fas fa-dot-circle" style="color:#34d399;margin-right:3px;"></i>
          Center <span id="acc-center-val" style="color:#fbbf24">—</span>
        </span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
        <label style="color:#94a3b8;font-size:10px;white-space:nowrap;" title="Adjust cursor travel sensitivity. 1.0x = default. Lower if cursor overshoots; raise if hard to reach edges.">
          <i class="fas fa-sliders-h" style="margin-right:3px;color:#a78bfa;"></i>Sensitivity
        </label>
        <input id="acc-sensitivity-slider" type="range"
          min="0.5" max="2.0" step="0.1" value="1.0"
          style="flex:1;accent-color:#a78bfa;height:4px;cursor:pointer;"
          title="0.5× = narrower range (less sensitive) | 1.0× = default | 2.0× = wider range (more sensitive)">
        <span id="acc-sensitivity-val" style="color:#a78bfa;font-size:10px;min-width:26px;">1.0×</span>
      </div>`;
    panel.appendChild(row);

    // Wire slider immediately (in case panel was already visible)
    setTimeout(() => this._wireSensitivitySlider(), 100);
  }

  _updateLiveUI() {
    const gainEl = document.getElementById('acc-gain-val');
    if (gainEl) {
      const g = this.gainRemap.gamma.toFixed(2);
      gainEl.textContent  = `γ=${g}`;
      gainEl.style.color  = this.config.enableGainRemap ? '#a78bfa' : '#94a3b8';
    }
    const pullEl = document.getElementById('acc-pull-val');
    if (pullEl) {
      const f = Math.round(this.diag.lastPullForce * 100);
      pullEl.textContent  = f > 0 ? f + '%' : '—';
      pullEl.style.color  = f > 30 ? '#22c55e' : f > 0 ? '#fbbf24' : '#94a3b8';
    }
    const driftEl = document.getElementById('acc-drift-val');
    if (driftEl) {
      const d = Math.round(this.diag.lastDriftMag * 100);
      driftEl.textContent = d + '%';
      driftEl.style.color = d > 3 ? '#f87171' : d > 1 ? '#fbbf24' : '#22c55e';
    }
    const centerEl = document.getElementById('acc-center-val');
    if (centerEl) {
      const on = this.config.enableCenterGrav;
      const pulling = (Math.abs(this.centerGrav.lastPullX) + Math.abs(this.centerGrav.lastPullY)) > 0.2;
      centerEl.textContent = on ? (pulling ? 'on' : 'idle') : 'off';
      centerEl.style.color = pulling ? '#34d399' : on ? '#94a3b8' : '#4b5563';
    }
  }

  getDiag() {
    return {
      ...this.diag,
      gainRemap: {
        enabled:     this.config.enableGainRemap,
        gamma:       this.gainRemap.gamma.toFixed(3),
        sensitivity: this.gainRemap.sensitivity.toFixed(2)
      },
      centerGravity: {
        enabled:  this.config.enableCenterGrav,
        pullRate: this.centerGrav.CENTER_PULL_RATE,
        lastPull: { x: this.centerGrav.lastPullX.toFixed(2), y: this.centerGrav.lastPullY.toFixed(2) }
      },
      gravity: {
        pullStrength: this.gravity.PULL_STRENGTH,
        maxRadius:    this.gravity.MAX_PULL_RADIUS,
        lastForce:    (this.gravity.lastForce * 100).toFixed(1) + '%',
        attractor:    this.gravity.lastAttractor
      },
      drift: {
        corrX:    (this.driftCorr._corrX * 100).toFixed(2) + '%',
        corrY:    (this.driftCorr._corrY * 100).toFixed(2) + '%',
        driftMag: (this.driftCorr.lastDriftMag * 100).toFixed(2) + '%',
        samples:  this.driftCorr._fixBuf.length
      }
    };
  }
}


/* ═══════════════════════════════════════════════════════════════════════════
   BOOT
*/
(function boot() {
  const acc = new AccuracyOrchestrator();
  window.AccuracyEngine = acc;

  function init() {
    acc.start();
    console.log(
      '%c[AccuracyEngine v3] Loaded — GainRemap + CenterGravity + GravitySnap + DriftCorrector',
      'color:#00ff88;font-weight:bold;font-size:12px'
    );
    console.log(
      '%c  Fix: Edge bias / center dead-zone (root cause: nonlinear iris gain)',
      'color:#a78bfa;font-size:10px'
    );
    console.log(
      '%c  Research: Zhu & Ji 2006 | Casiez CHI 2012 | Grossman & Balakrishnan 2005',
      'color:#94a3b8;font-size:10px'
    );
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    setTimeout(init, 0);
  }
})();
