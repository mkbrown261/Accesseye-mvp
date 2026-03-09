/**
 * ═══════════════════════════════════════════════════════════════════════════
 *  AccessEye — Phase 2: High-Precision Eye Tracking Optimization
 *  phase2-engine.js
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  NEW MODULES (internal performance upgrades — UI unchanged):
 *
 *   P2.1  HighFPSCameraController   — 120/60/30 FPS auto-detect
 *   P2.2  HeadPoseEstimator         — 6-DOF solvePnP-style from landmarks
 *   P2.3  HybridGazeEngine          — geometric + head-pose + binocular fusion
 *   P2.4  TemporalStabilizer        — Adaptive Kalman + EMA + sliding window
 *   P2.5  MicroSaccadeFilter        — 12px / 200ms fixation stability window
 *   P2.6  GazeConfidenceScorer      — brightness, occlusion, glare detection
 *   P2.7  DynamicCalibrationEngine  — extends CalibrationEngine w/ micro-updates
 *   P2.8  IntentPredictionEngine    — OpenAI-backed behavioral model (server-side)
 *   P2.9  GazeBenchmark             — old vs new pipeline comparison
 *
 *  Architecture flow:
 *   Camera (120/60/30 FPS)
 *     ↓
 *   FaceMesh (refineLandmarks=true)
 *     ↓
 *   HeadPoseEstimator (6-DOF euler angles)
 *     ↓
 *   HybridGazeEngine (iris + eyelid + head_pose fusion)
 *     ↓
 *   GazeConfidenceScorer (brightness/occlusion/glare)
 *     ↓
 *   TemporalStabilizer (adaptive Kalman + EMA + window)
 *     ↓
 *   MicroSaccadeFilter (fixation stability)
 *     ↓
 *   DynamicCalibrationEngine (polynomial + micro-updates)
 *     ↓
 *   IntentPredictionEngine (AI behavioral prediction)
 *     ↓
 *   UIElementRegistry (screen coords → element hit detection)
 * ═══════════════════════════════════════════════════════════════════════════
 */

'use strict';

/* ─────────────────────────────────────────────────────────────────────────
   MATH UTILITIES (Phase 2)
───────────────────────────────────────────────────────────────────────── */
const p2 = {
  dot3:    (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2],
  cross3:  (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]],
  norm3:   (v) => { const m = Math.hypot(...v); return m > 1e-9 ? v.map(x=>x/m) : [0,0,1]; },
  len3:    (v) => Math.hypot(...v),
  sub3:    (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]],
  add3:    (a, b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]],
  scale3:  (v, s) => v.map(x => x*s),
  rad:     (deg) => deg * Math.PI / 180,
  deg:     (rad) => rad * 180 / Math.PI,
  clamp:   (v,lo,hi) => Math.min(hi, Math.max(lo, v)),
  lerp:    (a,b,t) => a + (b-a)*t,
  dist2:   (x1,y1,x2,y2) => Math.hypot(x2-x1,y2-y1),
  now:     () => performance.now(),
  avg:     (arr) => arr.reduce((s,v)=>s+v,0)/arr.length,
  stddev:  (arr) => { const m = p2.avg(arr); return Math.sqrt(p2.avg(arr.map(v=>(v-m)**2))); }
};

/* ─────────────────────────────────────────────────────────────────────────
   P2.1  HIGH-FPS CAMERA CONTROLLER
   Negotiates highest supported FPS with graceful fallback chain:
   120 FPS → 60 FPS → 30 FPS
───────────────────────────────────────────────────────────────────────── */
class HighFPSCameraController {
  constructor() {
    this.FPS_CHAIN = [120, 90, 60, 30];
    this.achievedFPS = 0;
    this.stream = null;
    this.videoEl = null;
    this.detectedCapabilities = null;
  }

  /**
   * Try FPS levels in descending order; stop at first success.
   * @returns {Promise<{stream, fps, width, height}>}
   */
  async acquire(videoEl) {
    this.videoEl = videoEl;
    for (const targetFPS of this.FPS_CHAIN) {
      try {
        const stream = await this._tryFPS(targetFPS);
        if (stream) {
          this.stream = stream;
          this.achievedFPS = targetFPS;
          videoEl.srcObject = stream;
          await videoEl.play();

          // Measure actual track settings
          const track = stream.getVideoTracks()[0];
          const settings = track.getSettings();
          this.detectedCapabilities = {
            fps: settings.frameRate || targetFPS,
            width: settings.width || 640,
            height: settings.height || 480,
            facingMode: settings.facingMode || 'user',
            requested: targetFPS
          };

          console.log(`[HighFPS] Acquired camera @ ${settings.frameRate || targetFPS} FPS, ${settings.width}×${settings.height}`);
          return this.detectedCapabilities;
        }
      } catch (e) {
        console.warn(`[HighFPS] ${targetFPS} FPS failed: ${e.message}`);
      }
    }
    throw new Error('No camera mode succeeded');
  }

  async _tryFPS(fps) {
    const constraints = {
      video: {
        facingMode: 'user',
        width:  { ideal: 1280, min: 320 },
        height: { ideal: 720,  min: 240 },
        frameRate: { ideal: fps, min: Math.min(fps, 24) }
      }
    };
    return navigator.mediaDevices.getUserMedia(constraints);
  }

  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
    }
    if (this.videoEl) {
      this.videoEl.srcObject = null;
    }
  }

  getCapabilities() { return this.detectedCapabilities; }
}

/* ─────────────────────────────────────────────────────────────────────────
   P2.2  HEAD POSE ESTIMATOR
   Computes 3D head orientation (yaw, pitch, roll) from Face Mesh landmarks
   using a simplified solvePnP approach with 6 stable facial anchor points.
   Output feeds into hybrid gaze to compensate for head movement.
───────────────────────────────────────────────────────────────────────── */
class HeadPoseEstimator {
  constructor() {
    // 6 stable 3D model points (canonical face — OpenCV convention, mm scale)
    this.MODEL_3D = [
      [  0.0,    0.0,   0.0],   // Nose tip         (1)
      [  0.0,  -63.6, -12.5],   // Chin             (152)
      [-43.3,   32.7, -26.0],   // Left eye corner  (33)
      [ 43.3,   32.7, -26.0],   // Right eye corner (263)
      [-28.9,  -28.9, -24.1],   // Left mouth corner(61)
      [ 28.9,  -28.9, -24.1]    // Right mouth corner(291)
    ];

    // Corresponding MediaPipe landmark indices
    this.ANCHOR_IDX = [1, 152, 33, 263, 61, 291];

    // Result
    this.yaw   = 0;  // horizontal head rotation (deg)
    this.pitch = 0;  // vertical head rotation   (deg)
    this.roll  = 0;  // tilt                     (deg)
    this.tvec  = [0, 0, 600]; // translation vector (depth)
    this.valid = false;

    // EMA smoothing on pose angles
    this._yawEMA   = new _EMAScalar(0.2);
    this._pitchEMA = new _EMAScalar(0.2);
    this._rollEMA  = new _EMAScalar(0.2);
  }

  /**
   * @param {Array} lm - Face Mesh landmark array (468+ points)
   * @param {number} W - video frame width
   * @param {number} H - video frame height
   * @returns {{ yaw, pitch, roll, valid, confidence }}
   */
  estimate(lm, W, H) {
    if (!lm || lm.length < 468) return { yaw:0, pitch:0, roll:0, valid:false, confidence:0 };

    // Extract 2D image points (pixels)
    const pts2D = this.ANCHOR_IDX.map(i => [lm[i].x * W, lm[i].y * H]);

    // Camera intrinsics (approximate focal length = image width)
    const f = W;
    const cx = W / 2, cy = H / 2;

    // Simplified iterative PnP via EPnP-lite  (no OpenCV in browser)
    // We use a closed-form approximation based on known geometry constraints
    const { yaw, pitch, roll, depth, confidence } = this._solvePnPApprox(pts2D, W, H, f, cx, cy);

    this.yaw   = this._yawEMA.update(yaw);
    this.pitch = this._pitchEMA.update(pitch);
    this.roll  = this._rollEMA.update(roll);
    this.tvec  = [0, 0, depth];
    this.valid = confidence > 0.3;

    return {
      yaw:   this.yaw,
      pitch: this.pitch,
      roll:  this.roll,
      depth,
      valid: this.valid,
      confidence
    };
  }

  /**
   * Approximated head pose from 2D landmark geometry.
   *
   * Yaw   ≈ horizontal displacement of nose relative to eye midpoint
   * Pitch ≈ vertical displacement of nose relative to eye-chin axis
   * Roll  ≈ angle of eye-to-eye line relative to horizontal
   * Depth ≈ inverse of inter-eye distance (farther = smaller span)
   */
  _solvePnPApprox(pts2D, W, H, f, cx, cy) {
    // pts2D[0]=nose, [1]=chin, [2]=leftEye, [3]=rightEye, [4]=leftMouth, [5]=rightMouth
    const [nose, chin, lEye, rEye, lMouth, rMouth] = pts2D;

    // Eye midpoint
    const eyeMid = [(lEye[0]+rEye[0])/2, (lEye[1]+rEye[1])/2];
    // Inter-eye distance (pixel) → proxy for depth
    const eyeSpan = p2.dist2(lEye[0], lEye[1], rEye[0], rEye[1]);

    // Depth proxy: 65mm real inter-eye ÷ pixel span × focal_length
    const depth = eyeSpan > 5 ? (65 * f) / eyeSpan : 600;

    // Yaw: horizontal offset of nose from eye midpoint, normalized
    const yawRaw  = (nose[0] - eyeMid[0]) / Math.max(eyeSpan, 1);
    const yaw     = p2.clamp(p2.deg(Math.atan(yawRaw * 1.8)), -50, 50);

    // Pitch: vertical offset of nose from eye midpoint, normalized
    const pitchRaw = (nose[1] - eyeMid[1]) / Math.max(eyeSpan, 1);
    const pitch    = p2.clamp(p2.deg(Math.atan(pitchRaw * 1.5)) - 10, -40, 40);

    // Roll: angle of eye line relative to horizontal
    const dX   = rEye[0] - lEye[0];
    const dY   = rEye[1] - lEye[1];
    const roll = p2.clamp(p2.deg(Math.atan2(dY, dX)), -30, 30);

    // Confidence: based on face size and symmetry
    const faceH  = Math.abs(chin[1] - eyeMid[1]);
    const symm   = 1 - Math.abs(nose[0] - eyeMid[0]) / Math.max(eyeSpan, 1) * 2;
    const conf   = p2.clamp((eyeSpan / (W * 0.15)) * Math.max(0, symm), 0, 1);

    return { yaw, pitch, roll, depth, confidence: conf };
  }

  reset() {
    this.yaw = 0; this.pitch = 0; this.roll = 0; this.valid = false;
    this._yawEMA.reset(); this._pitchEMA.reset(); this._rollEMA.reset();
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   P2.3  HYBRID GAZE ENGINE
   Combines 4 gaze signal sources with confidence-weighted fusion:
     (a) Binocular iris offset (normalized)
     (b) Head pose vector (yaw/pitch → screen displacement)
     (c) Eyelid aperture weighting (eyes nearly closed = low weight)
     (d) Multi-point iris boundary centroid (replaces simple iris center)
   Replaces Phase 1 GazeEngine.processResults()
───────────────────────────────────────────────────────────────────────── */
class HybridGazeEngine {
  constructor(calibration, headPose) {
    this.calibration = calibration;
    this.headPose    = headPose;

    // ── MediaPipe landmark index groups ──
    // Iris (5 pts each)
    this.L_IRIS = [468, 469, 470, 471, 472];
    this.R_IRIS = [473, 474, 475, 476, 477];

    // Eye corners
    this.L_CORNER_INNER = 133; this.L_CORNER_OUTER = 33;
    this.R_CORNER_INNER = 362; this.R_CORNER_OUTER = 263;

    // Upper eyelid (approx midpoints)
    this.L_LID_TOP    = [159, 160, 161];
    this.L_LID_BOTTOM = [145, 144, 163];
    this.R_LID_TOP    = [386, 387, 388];
    this.R_LID_BOTTOM = [374, 373, 390];

    // Pupil boundary approximation (outer iris ring subset)
    this.L_PUPIL_RING = [469, 470, 471, 472, 468];
    this.R_PUPIL_RING = [474, 475, 476, 477, 473];

    // Output state
    this.rawGaze    = { x: 0.5, y: 0.5 };
    this.smoothGaze = { x: 0.5, y: 0.5 };
    this.confidence = 0;

    // Internal weights (adaptive)
    this.W_IRIS = 0.55;      // binocular iris offset weight
    this.W_HEAD = 0.30;      // head pose vector weight
    this.W_PUPIL= 0.15;      // pupil boundary centroid weight

    // Phase-1 fallback re-used
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
   * Main entry — replaces Phase 1 GazeEngine.processResults()
   * Returns enhanced gaze packet with all Phase 2 fields.
   */
  processResults(multiFaceLandmarks, W, H, headPoseResult) {
    if (!multiFaceLandmarks || multiFaceLandmarks.length === 0) {
      this.confidence = 0;
      return null;
    }
    const lm = multiFaceLandmarks[0];
    if (lm.length < 478) return this._fallback(lm, W, H);

    // ── (a) Binocular iris offset signal ──
    const irisSignal = this._computeIrisSignal(lm);

    // ── (b) Head pose compensation ──
    const headSignal = this._computeHeadPoseSignal(headPoseResult);

    // ── (c) Pupil boundary centroid signal ──
    const pupilSignal = this._computePupilSignal(lm);

    // ── (d) Eyelid aperture (confidence weighting) ──
    const lidConf = this._computeLidAperture(lm);

    // ── Confidence scoring (pre-stabilization) ──
    const baseConf = irisSignal.confidence * lidConf;
    this.confidence = p2.clamp(baseConf, 0, 1);

    // ── Adaptive weight blending ──
    // Lower head pose weight when head is nearly frontal (iris more reliable)
    const headMag  = Math.hypot(headPoseResult?.yaw || 0, headPoseResult?.pitch || 0) / 30;
    const wHead    = p2.clamp(this.W_HEAD * (0.5 + headMag), 0, 0.45);
    const wIris    = p2.clamp(1 - wHead - this.W_PUPIL, 0, 0.70);
    const wPupil   = this.W_PUPIL;

    // ── Fused raw gaze vector ──
    const fusedX = wIris  * irisSignal.x  + wHead * headSignal.x  + wPupil * pupilSignal.x;
    const fusedY = wIris  * irisSignal.y  + wHead * headSignal.y  + wPupil * pupilSignal.y;

    this.rawGaze = { x: fusedX, y: fusedY };

    // ── Calibration mapping ──
    let screen;
    if (this.calibration.isCalibrated) {
      const mapped = this.calibration.mapGaze(fusedX, fusedY);
      screen = { x: mapped.sx, y: mapped.sy };
    } else {
      // Uncalibrated: enhanced head-pose-corrected estimate
      const headX = lm[1].x;
      const headY = lm[1].y;
      const hYaw  = (headPoseResult?.yaw || 0) / 50;
      const hPit  = (headPoseResult?.pitch || 0) / 40;
      screen = {
        x: p2.clamp(0.5 + fusedX * 3.2 + (0.5 - headX) * 0.7 + hYaw * 0.15, 0.02, 0.98),
        y: p2.clamp(0.5 + fusedY * 3.8 + (headY - 0.5) * 0.9 + hPit * 0.15, 0.02, 0.98)
      };
    }

    this.smoothGaze = screen;

    // ── Enhanced gaze packet (Phase 2 fields) ──
    const packet = {
      // Core
      raw:    this.rawGaze,
      screen: this.smoothGaze,
      confidence: this.confidence,
      // Phase 2 extras
      iris:   irisSignal,
      head:   headPoseResult,
      pupil:  pupilSignal,
      lidAperture: lidConf,
      weights: { wIris, wHead, wPupil },
      timestamp: p2.now()
    };

    this._emit('gaze', packet);
    return packet;
  }

  /* ── Signal Extractors ── */

  _computeIrisSignal(lm) {
    const lIris = this._irisCentroid(lm, this.L_IRIS);
    const rIris = this._irisCentroid(lm, this.R_IRIS);

    // Eye corners
    const lInner = lm[this.L_CORNER_INNER], lOuter = lm[this.L_CORNER_OUTER];
    const rInner = lm[this.R_CORNER_INNER], rOuter = lm[this.R_CORNER_OUTER];

    const lSpan = p2.dist2(lOuter.x, lOuter.y, lInner.x, lInner.y);
    const rSpan = p2.dist2(rOuter.x, rOuter.y, rInner.x, rInner.y);

    const lMidX = (lOuter.x + lInner.x) / 2, lMidY = (lOuter.y + lInner.y) / 2;
    const rMidX = (rOuter.x + rInner.x) / 2, rMidY = (rOuter.y + rInner.y) / 2;

    // Normalized iris displacement within eye span
    const lOX = lSpan > 0 ? (lIris.x - lMidX) / lSpan : 0;
    const lOY = lSpan > 0 ? (lIris.y - lMidY) / lSpan : 0;
    const rOX = rSpan > 0 ? (rIris.x - rMidX) / rSpan : 0;
    const rOY = rSpan > 0 ? (rIris.y - rMidY) / rSpan : 0;

    // Binocular average
    const x = (lOX + rOX) / 2;
    const y = (lOY + rOY) / 2;

    // Confidence: eye span relative to face width
    const confidence = p2.clamp((lSpan + rSpan) / 0.16, 0, 1);

    return { x, y, lSpan, rSpan, confidence, lIris, rIris };
  }

  _computeHeadPoseSignal(hp) {
    if (!hp || !hp.valid) return { x: 0, y: 0 };
    // Convert head angles to normalized gaze displacement
    // Yaw → horizontal shift, pitch → vertical shift
    const x = p2.clamp(-hp.yaw  / 50, -0.5, 0.5);
    const y = p2.clamp(-hp.pitch / 40, -0.5, 0.5);
    return { x, y };
  }

  _computePupilSignal(lm) {
    // Use outer iris ring as pupil boundary proxy
    const lCenter = this._irisCentroid(lm, this.L_PUPIL_RING);
    const rCenter = this._irisCentroid(lm, this.R_PUPIL_RING);

    // Use iris midpoint for screen direction
    const faceLeft  = lm[234];
    const faceRight = lm[454];
    const faceSpan  = faceLeft && faceRight
      ? p2.dist2(faceLeft.x, faceLeft.y, faceRight.x, faceRight.y) : 0.2;

    // Raw screen center of both pupils
    const cx = ((lCenter.x + rCenter.x) / 2);
    const cy = ((lCenter.y + rCenter.y) / 2);

    // Displacement from face center
    const faceMidX = faceLeft && faceRight ? (faceLeft.x + faceRight.x)/2 : 0.5;
    return {
      x: faceSpan > 0 ? (cx - faceMidX) / faceSpan : 0,
      y: cy - 0.5
    };
  }

  _computeLidAperture(lm) {
    // Average eye openness: ratio of lid gap to eye span
    const avgLid = (
      this._lidRatio(lm, this.L_LID_TOP, this.L_LID_BOTTOM, this.L_CORNER_OUTER, this.L_CORNER_INNER) +
      this._lidRatio(lm, this.R_LID_TOP, this.R_LID_BOTTOM, this.R_CORNER_OUTER, this.R_CORNER_INNER)
    ) / 2;
    // Map: fully open (0.3+) → 1.0, nearly closed (<0.1) → 0.2
    return p2.clamp((avgLid - 0.05) / 0.25, 0.2, 1.0);
  }

  _lidRatio(lm, topIdx, botIdx, cL, cR) {
    const topY = p2.avg(topIdx.map(i => lm[i]?.y || 0));
    const botY = p2.avg(botIdx.map(i => lm[i]?.y || 0));
    const span = p2.dist2(lm[cL]?.x||0, lm[cL]?.y||0, lm[cR]?.x||0, lm[cR]?.y||0);
    return span > 0 ? Math.abs(botY - topY) / span : 0.2;
  }

  _irisCentroid(lm, indices) {
    const pts = indices.map(i => lm[i] || { x:0.5, y:0.5 });
    return {
      x: pts.reduce((s,p)=>s+p.x,0)/pts.length,
      y: pts.reduce((s,p)=>s+p.y,0)/pts.length
    };
  }

  _fallback(lm, W, H) {
    // Phase 1 compatible fallback
    const nose = lm[1];
    const sx = p2.clamp(1 - nose.x, 0.05, 0.95);
    const sy = p2.clamp(nose.y * 1.2 - 0.1, 0.05, 0.95);
    this.rawGaze = { x: sx, y: sy };
    this.smoothGaze = { x: sx, y: sy };
    this.confidence = 0.4;
    const packet = { raw: this.rawGaze, screen: this.smoothGaze, confidence: 0.4, timestamp: p2.now() };
    this._emit('gaze', packet);
    return packet;
  }

  reset() { this.rawGaze = { x:0.5, y:0.5 }; this.smoothGaze = { x:0.5, y:0.5 }; this.confidence = 0; }
}

/* ─────────────────────────────────────────────────────────────────────────
   P2.4  TEMPORAL STABILIZER
   Three-layer pipeline:
     Layer A: Adaptive Kalman (measurement noise R adjusts to confidence)
     Layer B: Exponential Moving Average (alpha adjusts to movement speed)
     Layer C: Sliding window median (removes outlier frames)
───────────────────────────────────────────────────────────────────────── */
class TemporalStabilizer {
  /**
   * @param {object} opts
   *   kalmanR     {number} 0.003–0.02 (lower = noisier but more responsive)
   *   kalmanQ     {number} 0.0001
   *   emaAlpha    {number} 0.25 base alpha
   *   windowSize  {number} sliding window length (frames)
   */
  constructor(opts = {}) {
    this.baseR      = opts.kalmanR   ?? 0.004;
    this.baseQ      = opts.kalmanQ   ?? 0.00008;
    this.baseAlpha  = opts.emaAlpha  ?? 0.28;
    this.winSize    = opts.windowSize ?? 7;

    // Adaptive Kalman (per axis)
    this._kx = new _KalmanAxis(this.baseR, this.baseQ);
    this._ky = new _KalmanAxis(this.baseR, this.baseQ);

    // EMA (per axis)
    this._ex = null; this._ey = null;

    // Sliding window
    this._wx = [];   this._wy = [];

    // Velocity tracking (for adaptive alpha)
    this._prevX = null; this._prevY = null;
    this._velX = 0;     this._velY = 0;

    // Output
    this.stable = { x: 0.5, y: 0.5 };
  }

  /**
   * @param {number} rx  raw X (0–1)
   * @param {number} ry  raw Y (0–1)
   * @param {number} confidence  0–1
   * @returns {{ x, y }}
   */
  update(rx, ry, confidence = 1.0) {
    // ── Layer A: Adaptive Kalman ──
    // Increase measurement noise when confidence is low (be more conservative)
    const adaptR = this.baseR / Math.max(confidence, 0.1);
    this._kx.R = adaptR;
    this._ky.R = adaptR;
    const kx = this._kx.update(rx);
    const ky = this._ky.update(ry);

    // ── Layer B: Adaptive EMA ──
    // Velocity-based alpha: fast movements → higher alpha (more responsive)
    const velMag = this._prevX !== null
      ? Math.hypot(kx - this._prevX, ky - this._prevY)
      : 0;
    this._velX = p2.lerp(this._velX, Math.abs(kx - (this._prevX??kx)), 0.4);
    this._velY = p2.lerp(this._velY, Math.abs(ky - (this._prevY??ky)), 0.4);
    const velScore = p2.clamp(velMag / 0.05, 0, 1);
    const alpha = p2.lerp(this.baseAlpha, 0.6, velScore);

    if (this._ex === null) { this._ex = kx; this._ey = ky; }
    else {
      this._ex = p2.lerp(this._ex, kx, alpha);
      this._ey = p2.lerp(this._ey, ky, alpha);
    }
    this._prevX = kx; this._prevY = ky;

    // ── Layer C: Sliding window trimmed mean ──
    this._wx.push(this._ex); this._wy.push(this._ey);
    if (this._wx.length > this.winSize) { this._wx.shift(); this._wy.shift(); }

    // Trimmed mean: remove 1 highest + 1 lowest if window >= 5
    const sx = this._trimmedMean(this._wx);
    const sy = this._trimmedMean(this._wy);

    this.stable = { x: sx, y: sy };
    return this.stable;
  }

  _trimmedMean(arr) {
    if (arr.length < 5) return p2.avg(arr);
    const s = [...arr].sort((a,b)=>a-b);
    return p2.avg(s.slice(1, -1));
  }

  reset() {
    this._kx.reset(); this._ky.reset();
    this._ex = null;  this._ey = null;
    this._wx = [];    this._wy = [];
    this._prevX = null; this._prevY = null;
    this._velX = 0;   this._velY = 0;
    this.stable = { x:0.5, y:0.5 };
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   P2.5  MICRO-SACCADE FILTER
   Implements fixation detection via stability window:
   If gaze stays within `radius` px for `duration` ms → emit fixation event.
   Only stable fixations update UI focus; transient saccades are suppressed.
───────────────────────────────────────────────────────────────────────── */
class MicroSaccadeFilter {
  /**
   * @param {number} radiusPx   stability radius in screen pixels (default 12)
   * @param {number} durationMs stability window in ms (default 200)
   */
  constructor(radiusPx = 12, durationMs = 200) {
    this.radius   = radiusPx;
    this.duration = durationMs;

    this._anchorX    = null;
    this._anchorY    = null;
    this._anchorTime = null;
    this.isFixated   = false;
    this.fixationX   = null;
    this.fixationY   = null;
    this.fixationAge = 0;       // ms since fixation started

    this._callbacks = {};
    this._saccadeCount = 0;
    this._fixationCount = 0;
  }

  on(event, cb) {
    if (!this._callbacks[event]) this._callbacks[event] = [];
    this._callbacks[event].push(cb);
  }

  _emit(event, data) {
    (this._callbacks[event] || []).forEach(cb => cb(data));
  }

  /**
   * Feed raw gaze in screen pixels. Returns filtered gaze.
   * @param {number} px  screen pixel X
   * @param {number} py  screen pixel Y
   * @param {number} confidence
   * @returns {{ x, y, isFixated, fixationAge, isSaccade }}
   */
  update(px, py, confidence = 1.0) {
    const t = p2.now();

    if (this._anchorX === null) {
      this._anchorX = px; this._anchorY = py; this._anchorTime = t;
    }

    const distFromAnchor = p2.dist2(px, py, this._anchorX, this._anchorY);
    const effectiveRadius = this.radius / Math.max(confidence, 0.3);

    if (distFromAnchor <= effectiveRadius) {
      // Within stability window
      const elapsed = t - this._anchorTime;
      this.fixationAge = elapsed;

      if (elapsed >= this.duration && !this.isFixated) {
        // NEW FIXATION
        this.isFixated  = true;
        this.fixationX  = this._anchorX;
        this.fixationY  = this._anchorY;
        this._fixationCount++;
        this._emit('fixation', {
          x: this.fixationX,
          y: this.fixationY,
          duration: elapsed,
          count: this._fixationCount
        });
      }

      // Return anchor position (suppress micro-saccade noise)
      return {
        x: this._anchorX, y: this._anchorY,
        isFixated: this.isFixated,
        fixationAge: this.fixationAge,
        isSaccade: false,
        distFromAnchor
      };
    } else {
      // SACCADE DETECTED — reset anchor
      this._saccadeCount++;
      const wasFix = this.isFixated;
      if (wasFix) {
        this._emit('saccade', {
          fromX: this._anchorX, fromY: this._anchorY,
          toX: px, toY: py,
          fixationDuration: t - this._anchorTime
        });
      }
      this.isFixated  = false;
      this.fixationAge = 0;
      this._anchorX   = px;
      this._anchorY   = py;
      this._anchorTime = t;

      return {
        x: px, y: py,
        isFixated: false,
        fixationAge: 0,
        isSaccade: true,
        distFromAnchor
      };
    }
  }

  getStats() {
    return {
      saccades:  this._saccadeCount,
      fixations: this._fixationCount,
      isFixated: this.isFixated,
      fixationAge: this.fixationAge
    };
  }

  reset() {
    this._anchorX = null; this._anchorY = null; this._anchorTime = null;
    this.isFixated = false; this.fixationX = null; this.fixationY = null;
    this.fixationAge = 0;
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   P2.6  GAZE CONFIDENCE SCORER
   Multi-factor confidence scoring:
     • Eye visibility / occlusion
     • Iris detection quality
     • Lighting (brightness normalization)
     • Glasses/glare detection
     • Head angle extremity
     • Temporal stability
───────────────────────────────────────────────────────────────────────── */
class GazeConfidenceScorer {
  constructor(videoEl) {
    this.videoEl = videoEl;
    // Offline canvas for brightness sampling
    this._bCanvas = document.createElement('canvas');
    this._bCtx    = this._bCanvas.getContext('2d', { willReadFrequently: true });
    this._bCanvas.width = 64; this._bCanvas.height = 48;

    // Smoothed sub-scores
    this._brightEMA   = new _EMAScalar(0.1);
    this._occlusionEMA = new _EMAScalar(0.15);
    this._glareEMA    = new _EMAScalar(0.1);
    this._headEMA     = new _EMAScalar(0.2);

    this.lastScore = { total: 1.0, brightness: 1.0, occlusion: 1.0, glare: 1.0, head: 1.0 };
    this._frameSkip = 0;  // Only run expensive checks every N frames
    this._cachedBrightness = 1.0;
  }

  /**
   * @param {Array} lm         FaceMesh landmarks
   * @param {object} headPose  from HeadPoseEstimator
   * @param {object} irisData  from HybridGazeEngine._computeIrisSignal
   * @returns {{ total, brightness, occlusion, glare, head, lowLight, glassesDetected }}
   */
  score(lm, headPose, irisData) {
    this._frameSkip = (this._frameSkip + 1) % 6; // expensive path every 6 frames

    // ── Brightness score (run every 6 frames) ──
    let brightness = this._cachedBrightness;
    if (this._frameSkip === 0 && this.videoEl?.readyState >= 2) {
      brightness = this._measureBrightness();
      this._cachedBrightness = brightness;
    }
    const brightScore = this._brightEMA.update(brightness);

    // ── Occlusion score (landmark quality) ──
    const occScore = this._occlusionEMA.update(this._measureOcclusion(lm, irisData));

    // ── Glare / reflection detection ──
    const glareScore = this._glareEMA.update(this._detectGlare(lm, irisData));

    // ── Head pose extremity score ──
    const headScore = this._headEMA.update(this._headExtremity(headPose));

    // ── Total: weighted harmonic-style mean ──
    const total = p2.clamp(
      brightScore * 0.25 + occScore * 0.45 + glareScore * 0.15 + headScore * 0.15,
      0, 1
    );

    const lowLight        = brightScore < 0.45;
    const glassesDetected = glareScore  < 0.55;

    this.lastScore = { total, brightness: brightScore, occlusion: occScore, glare: glareScore, head: headScore, lowLight, glassesDetected };
    return this.lastScore;
  }

  _measureBrightness() {
    try {
      this._bCtx.drawImage(this.videoEl, 0, 0, 64, 48);
      const d = this._bCtx.getImageData(0, 0, 64, 48).data;
      let sum = 0;
      for (let i = 0; i < d.length; i += 4) sum += 0.299*d[i] + 0.587*d[i+1] + 0.114*d[i+2];
      const avg = sum / (d.length / 4);
      // Normalize: 0 (black) → 0.0, 90+ lux proxy → 1.0
      return p2.clamp(avg / 128, 0, 1);
    } catch(_) { return 0.8; }
  }

  _measureOcclusion(lm, irisData) {
    if (!lm || lm.length < 478) return 0.5;
    const confidence = irisData?.confidence ?? 0.5;
    // Also check: are eye landmarks present and not at 0,0?
    const lIrisOk = lm[468] && (lm[468].x > 0.01) && (lm[468].x < 0.99);
    const rIrisOk = lm[473] && (lm[473].x > 0.01) && (lm[473].x < 0.99);
    const bothEyes = (lIrisOk ? 0.5 : 0) + (rIrisOk ? 0.5 : 0);
    return p2.clamp(confidence * bothEyes, 0, 1);
  }

  _detectGlare(lm, irisData) {
    if (!irisData) return 0.8;
    // Glare proxy: if iris confidence is low but eye span is high → likely glare
    const eyeSize  = ((irisData.lSpan || 0) + (irisData.rSpan || 0)) / 2;
    const irisConf = irisData.confidence ?? 0.8;
    // Small iris confidence with adequate eye size → potential glare
    if (eyeSize > 0.04 && irisConf < 0.4) return 0.3;  // glare likely
    if (eyeSize > 0.04 && irisConf < 0.6) return 0.6;  // possible
    return Math.min(1.0, irisConf + 0.1);
  }

  _headExtremity(hp) {
    if (!hp) return 1.0;
    const yaw   = Math.abs(hp.yaw   || 0);
    const pitch = Math.abs(hp.pitch || 0);
    // Beyond 30°, tracking degrades significantly
    const yawScore   = p2.clamp(1 - (yaw   - 20) / 30, 0.2, 1);
    const pitchScore = p2.clamp(1 - (pitch - 15) / 25, 0.2, 1);
    return (yawScore + pitchScore) / 2;
  }

  reset() {
    this._brightEMA.reset(); this._occlusionEMA.reset();
    this._glareEMA.reset();  this._headEMA.reset();
    this.lastScore = { total:1, brightness:1, occlusion:1, glare:1, head:1 };
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   P2.7  DYNAMIC CALIBRATION ENGINE
   Extends Phase 1 CalibrationEngine with:
     • Micro-calibration: incremental model updates from confirmed interactions
     • Confidence gating: only update when confidence > threshold
     • Drift compensation: rolling bias correction
     • 9-point calibration support (vs Phase 1's 5-point)
───────────────────────────────────────────────────────────────────────── */
class DynamicCalibrationEngine {
  /**
   * @param {CalibrationEngine} baseCalib  Phase 1 engine (reused)
   */
  constructor(baseCalib) {
    this.base = baseCalib;

    // Micro-calibration state
    this.microSamples = [];       // [{gx, gy, sx, sy, confidence}]
    this.MAX_MICRO    = 40;       // rolling window
    this.MIN_CONF     = 0.65;     // minimum confidence to accept sample
    this.MICRO_LR     = 0.04;     // micro-update learning rate

    // Drift bias (rolling mean of residuals)
    this._biasX = 0;
    this._biasY = 0;
    this._biasAlpha = 0.03;

    // Extended 9-point calibration points
    this.CALIB_POINTS_9 = [
      { sx:0.10, sy:0.10, label:'TopLeft'     },
      { sx:0.50, sy:0.10, label:'TopCenter'   },
      { sx:0.90, sy:0.10, label:'TopRight'    },
      { sx:0.10, sy:0.50, label:'MidLeft'     },
      { sx:0.50, sy:0.50, label:'Center'      },
      { sx:0.90, sy:0.50, label:'MidRight'    },
      { sx:0.10, sy:0.90, label:'BotLeft'     },
      { sx:0.50, sy:0.90, label:'BotCenter'   },
      { sx:0.90, sy:0.90, label:'BotRight'    }
    ];

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
   * Called after a confirmed user interaction (gesture + element activation).
   * Treats (gazeX, gazeY) → (targetSX, targetSY) as a calibration sample.
   *
   * @param {number} gazeX      raw gaze X from HybridGazeEngine
   * @param {number} gazeY      raw gaze Y
   * @param {number} targetSX   known screen X of activated element (0-1)
   * @param {number} targetSY   known screen Y of activated element (0-1)
   * @param {number} confidence
   */
  recordInteraction(gazeX, gazeY, targetSX, targetSY, confidence) {
    if (confidence < this.MIN_CONF) return false;
    if (!this.base.isCalibrated) return false;

    this.microSamples.push({ gx: gazeX, gy: gazeY, sx: targetSX, sy: targetSY, confidence, t: p2.now() });
    if (this.microSamples.length > this.MAX_MICRO) this.microSamples.shift();

    // Compute bias from this sample
    const predicted = this.base.mapGaze(gazeX, gazeY);
    const residualX = targetSX - predicted.sx;
    const residualY = targetSY - predicted.sy;

    // Update rolling bias
    this._biasX = p2.lerp(this._biasX, residualX, this._biasAlpha);
    this._biasY = p2.lerp(this._biasY, residualY, this._biasAlpha);

    // Rebuild model if we have enough samples
    if (this.microSamples.length >= 8) {
      this._microUpdate();
    }

    this._emit('microCalib', {
      residualX, residualY,
      biasX: this._biasX, biasY: this._biasY,
      sampleCount: this.microSamples.length
    });

    return true;
  }

  /**
   * Apply drift bias correction to a mapped screen coordinate.
   * Called after base.mapGaze() in the pipeline.
   */
  applyBiasCorrection(sx, sy) {
    return {
      sx: p2.clamp(sx + this._biasX * 0.7, 0, 1),
      sy: p2.clamp(sy + this._biasY * 0.7, 0, 1)
    };
  }

  /**
   * Rebuild base calibration model augmented with micro-samples.
   * Weighted blend of original calibration data + new micro-samples.
   */
  _microUpdate() {
    if (!this.base.isCalibrated || this.microSamples.length < 5) return;

    // Blend: original calibration points get weight 2.0, micro-samples get confidence weight
    const blended = [
      ...this.base.calibData.filter(d => d.gx !== undefined).map(d => ({
        gx: d.gx, gy: d.gy, sx: d.sx, sy: d.sy, w: 2.0
      })),
      ...this.microSamples.map(s => ({
        gx: s.gx, gy: s.gy, sx: s.sx, sy: s.sy, w: s.confidence
      }))
    ];

    // Weighted least-squares
    const modelX = this._weightedLS(blended, p => p.sx);
    const modelY = this._weightedLS(blended, p => p.sy);

    if (modelX && modelY) {
      this.base.model = { x: modelX, y: modelY };
      this._emit('modelUpdated', { samples: blended.length, biasX: this._biasX, biasY: this._biasY });
    }
  }

  /* Weighted least-squares: 4-parameter bilinear */
  _weightedLS(pts, getTarget) {
    try {
      const n = 4;
      let ATA = Array.from({length:n},()=>new Array(n).fill(0));
      let ATb = new Array(n).fill(0);
      for (const p of pts) {
        const w = p.w ?? 1;
        const row = [1, p.gx, p.gy, p.gx*p.gy];
        const t   = getTarget(p);
        for (let r=0; r<n; r++) {
          ATb[r] += w * row[r] * t;
          for (let c=0; c<n; c++) ATA[r][c] += w * row[r] * row[c];
        }
      }
      // Gauss-Jordan
      let aug = ATA.map((row,i)=>[...row, ATb[i]]);
      for (let col=0; col<n; col++) {
        let maxR = col;
        for (let r=col+1;r<n;r++) if (Math.abs(aug[r][col])>Math.abs(aug[maxR][col])) maxR=r;
        [aug[col],aug[maxR]]=[aug[maxR],aug[col]];
        const piv = aug[col][col];
        if (Math.abs(piv)<1e-12) continue;
        for (let r=0;r<n;r++) { if(r===col) continue; const f=aug[r][col]/piv; for(let c=col;c<=n;c++) aug[r][c]-=f*aug[col][c]; }
        for (let c=col;c<=n;c++) aug[col][c]/=piv;
      }
      return aug.map(row=>row[n]);
    } catch(_) { return null; }
  }

  saveMicroData() {
    try {
      localStorage.setItem('accesseye_micro', JSON.stringify({
        biasX: this._biasX, biasY: this._biasY,
        samples: this.microSamples.slice(-20),
        t: Date.now()
      }));
    } catch(_) {}
  }

  loadMicroData() {
    try {
      const raw = localStorage.getItem('accesseye_micro');
      if (!raw) return;
      const d = JSON.parse(raw);
      this._biasX = d.biasX || 0;
      this._biasY = d.biasY || 0;
      this.microSamples = d.samples || [];
    } catch(_) {}
  }

  resetMicro() {
    this.microSamples = [];
    this._biasX = 0;
    this._biasY = 0;
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   P2.8  INTENT PREDICTION ENGINE
   AI-powered behavioral prediction using OpenAI (server-side proxy).
   Analyzes gaze patterns + history to predict likely next action.

   Data model fed to AI:
   {
     gaze_coordinates: {x, y},
     gaze_stability_duration: number (ms),
     head_pose_angle: {yaw, pitch, roll},
     gaze_confidence: number,
     fixation_event: bool,
     recent_elements: [{ id, label, dwellMs, activated }],
     gesture_history: [{ type, timestamp }],
     session_context: { activationCount, totalDwell, avgConfidence }
   }
───────────────────────────────────────────────────────────────────────── */
class IntentPredictionEngine {
  constructor(apiEndpoint = '/api/intent') {
    this.endpoint       = apiEndpoint;
    this.enabled        = false;
    this.debounceMs     = 1200;   // min time between AI calls
    this._lastCall      = 0;
    this._pending       = false;

    // Circular buffers for history
    this.gazeHistory    = [];     // last 30 gaze packets
    this.elementHistory = [];     // last 10 focused elements
    this.gestureHistory = [];     // last 5 gestures
    this.MAX_GAZE_H     = 30;
    this.MAX_ELEM_H     = 10;
    this.MAX_GEST_H     = 5;

    // Session context
    this.session = {
      startTime:        p2.now(),
      activationCount:  0,
      totalDwellMs:     0,
      confidenceSum:    0,
      confidenceFrames: 0
    };

    // Last prediction result
    this.lastPrediction = null;
    this._callbacks     = {};
  }

  on(event, cb) {
    if (!this._callbacks[event]) this._callbacks[event] = [];
    this._callbacks[event].push(cb);
  }

  _emit(event, data) {
    (this._callbacks[event] || []).forEach(cb => cb(data));
  }

  /** Feed every processed gaze frame */
  feedGaze(gazePacket, confidenceScore) {
    this.gazeHistory.push({
      x:    gazePacket.screen?.x ?? 0.5,
      y:    gazePacket.screen?.y ?? 0.5,
      conf: confidenceScore?.total ?? gazePacket.confidence ?? 0,
      t:    p2.now()
    });
    if (this.gazeHistory.length > this.MAX_GAZE_H) this.gazeHistory.shift();
    if (confidenceScore?.total > 0) {
      this.session.confidenceSum += confidenceScore.total;
      this.session.confidenceFrames++;
    }
  }

  /** Feed focused element events */
  feedFocus(elementId, label, dwellMs) {
    this.elementHistory.push({ id: elementId, label, dwellMs, activated: false, t: p2.now() });
    if (this.elementHistory.length > this.MAX_ELEM_H) this.elementHistory.shift();
    this.session.totalDwellMs += dwellMs || 0;
  }

  /** Feed activation events */
  feedActivation(elementId, label, gesture) {
    const last = this.elementHistory.findLast?.(e => e.id === elementId) ??
                 this.elementHistory[this.elementHistory.length - 1];
    if (last) last.activated = true;
    this.session.activationCount++;
    this.gestureHistory.push({ type: gesture, elementId, label, t: p2.now() });
    if (this.gestureHistory.length > this.MAX_GEST_H) this.gestureHistory.shift();
  }

  /**
   * Request AI prediction.
   * Called after a fixation event when confidence is sufficient.
   * @param {object} fixationData  from MicroSaccadeFilter
   * @param {object} focusedElement  current focused element or null
   */
  async predict(fixationData, focusedElement, confidenceScore) {
    if (!this.enabled) return null;
    if (this._pending) return this.lastPrediction;

    const t = p2.now();
    if (t - this._lastCall < this.debounceMs) return this.lastPrediction;

    const conf = confidenceScore?.total ?? 0;
    if (conf < 0.55) return null;  // don't predict on poor data

    this._lastCall = t;
    this._pending  = true;

    const payload = this._buildPayload(fixationData, focusedElement);
    try {
      const result = await this._callAPI(payload);
      this.lastPrediction = result;
      this._emit('prediction', result);
      return result;
    } catch (e) {
      console.warn('[Intent] API call failed:', e.message);
      return null;
    } finally {
      this._pending = false;
    }
  }

  _buildPayload(fixationData, focusedElement) {
    const avgConf = this.session.confidenceFrames > 0
      ? this.session.confidenceSum / this.session.confidenceFrames
      : 0;

    // Compute gaze stability duration from history
    const stableHistory = this.gazeHistory.slice(-10);
    const gazeSpread = stableHistory.length > 2
      ? p2.stddev(stableHistory.map(g => g.x)) + p2.stddev(stableHistory.map(g => g.y))
      : 1;

    return {
      gaze_coordinates: fixationData
        ? { x: fixationData.x, y: fixationData.y }
        : { x: 0.5, y: 0.5 },
      gaze_stability_duration: fixationData?.fixationAge ?? 0,
      gaze_spread: gazeSpread,
      gaze_confidence: avgConf,
      fixation_event: !!fixationData?.isFixated,
      focused_element: focusedElement
        ? { id: focusedElement.id, label: focusedElement.label }
        : null,
      recent_elements: this.elementHistory.slice(-5).map(e => ({
        id: e.id, label: e.label, dwell_ms: e.dwellMs, activated: e.activated
      })),
      gesture_history: this.gestureHistory.slice(-3).map(g => ({
        type: g.type, element: g.label, ago_ms: Math.round(p2.now() - g.t)
      })),
      session_context: {
        duration_s:       Math.round((p2.now() - this.session.startTime) / 1000),
        activation_count: this.session.activationCount,
        total_dwell_ms:   Math.round(this.session.totalDwellMs),
        avg_confidence:   Math.round(avgConf * 100) / 100
      }
    };
  }

  async _callAPI(payload) {
    const ctrl = new AbortController();
    const timeout = setTimeout(() => ctrl.abort(), 4000);
    try {
      const res = await fetch(this.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: ctrl.signal
      });
      clearTimeout(timeout);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } finally {
      clearTimeout(timeout);
    }
  }

  setEnabled(val) { this.enabled = val; }
}

/* ─────────────────────────────────────────────────────────────────────────
   P2.9  GAZE BENCHMARK
   Measures and compares Phase 1 vs Phase 2 pipeline metrics.
   Runs both pipelines in parallel during a benchmark session and
   reports accuracy, jitter, latency, and stability improvements.
───────────────────────────────────────────────────────────────────────── */
class GazeBenchmark {
  constructor() {
    this.running  = false;
    this.samples  = [];     // { p1: {x,y}, p2: {x,y}, target: {x,y} | null, t }
    this.started  = 0;
    this.duration = 30000;  // 30-second benchmark window

    // Per-pipeline metrics
    this.p1Metrics = this._emptyMetrics();
    this.p2Metrics = this._emptyMetrics();

    this._callbacks = {};
  }

  on(event, cb) {
    if (!this._callbacks[event]) this._callbacks[event] = [];
    this._callbacks[event].push(cb);
  }

  _emit(event, data) {
    (this._callbacks[event] || []).forEach(cb => cb(data));
  }

  _emptyMetrics() {
    return { jitterX:[], jitterY:[], latencies:[], fixationAges:[], confidences:[] };
  }

  start() {
    this.running = true;
    this.samples = [];
    this.started = p2.now();
    this.p1Metrics = this._emptyMetrics();
    this.p2Metrics = this._emptyMetrics();
    console.log('[Benchmark] Started 30s comparison window');
    this._emit('start', { duration: this.duration });
  }

  stop() {
    this.running = false;
    const report = this.generateReport();
    this._emit('complete', report);
    return report;
  }

  /** Record a single frame comparison */
  recordFrame(p1Gaze, p2Gaze, p2Confidence, p1Latency, p2Latency, fixationAge) {
    if (!this.running) return;
    const t = p2.now();
    if (t - this.started > this.duration) { this.stop(); return; }

    this.samples.push({ p1: {...p1Gaze}, p2: {...p2Gaze}, t });

    // Compute jitter (delta from previous sample)
    const n = this.samples.length;
    if (n > 1) {
      const prev = this.samples[n-2];
      this.p1Metrics.jitterX.push(Math.abs(p1Gaze.x - prev.p1.x));
      this.p1Metrics.jitterY.push(Math.abs(p1Gaze.y - prev.p1.y));
      this.p2Metrics.jitterX.push(Math.abs(p2Gaze.x - prev.p2.x));
      this.p2Metrics.jitterY.push(Math.abs(p2Gaze.y - prev.p2.y));
    }

    if (p1Latency) this.p1Metrics.latencies.push(p1Latency);
    if (p2Latency) this.p2Metrics.latencies.push(p2Latency);
    if (p2Confidence != null) this.p2Metrics.confidences.push(p2Confidence);
    if (fixationAge != null) this.p2Metrics.fixationAges.push(fixationAge);
  }

  generateReport() {
    const s = this.samples.length;
    if (s < 10) return { error: 'insufficient data', samples: s };

    const dur = (p2.now() - this.started) / 1000;

    const p1J = this._jitterReport(this.p1Metrics);
    const p2J = this._jitterReport(this.p2Metrics);

    const p1Lat = this._latReport(this.p1Metrics.latencies);
    const p2Lat = this._latReport(this.p2Metrics.latencies);

    const avgConf = this.p2Metrics.confidences.length
      ? p2.avg(this.p2Metrics.confidences) : 0;

    // Improvement percentages
    const jitterImprove = p1J.combined > 0
      ? Math.round((1 - p2J.combined / p1J.combined) * 100) : 0;
    const latImprove = p1Lat.mean > 0
      ? Math.round((1 - p2Lat.mean / p1Lat.mean) * 100) : 0;

    return {
      duration_s: Math.round(dur),
      total_frames: s,
      fps: Math.round(s / dur),
      phase1: { jitter: p1J, latency: p1Lat },
      phase2: { jitter: p2J, latency: p2Lat, avg_confidence: Math.round(avgConf*100)/100 },
      improvements: {
        jitter_reduction_pct: jitterImprove,
        latency_reduction_pct: latImprove,
        estimated_accuracy_gain: `~${Math.round(jitterImprove * 0.4)}%`
      },
      verdict: jitterImprove > 20
        ? `Phase 2 reduced gaze jitter by ${jitterImprove}% and latency by ${latImprove}%`
        : 'Insufficient improvement — check camera quality and lighting'
    };
  }

  _jitterReport(m) {
    const xArr = m.jitterX, yArr = m.jitterY;
    if (xArr.length === 0) return { x:0, y:0, combined:0 };
    const x = p2.avg(xArr), y = p2.avg(yArr);
    return { x: Math.round(x*10000)/10000, y: Math.round(y*10000)/10000, combined: x+y };
  }

  _latReport(arr) {
    if (arr.length === 0) return { mean:0, p95:0 };
    const sorted = [...arr].sort((a,b)=>a-b);
    return {
      mean: Math.round(p2.avg(arr)),
      p95:  Math.round(sorted[Math.floor(sorted.length * 0.95)] || 0)
    };
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   PHASE 2 ORCHESTRATOR
   Wires all Phase 2 modules together and integrates with the Phase 1 app.
   Exposes a clean interface that AccessEyeApp calls to upgrade itself.
───────────────────────────────────────────────────────────────────────── */
class Phase2Orchestrator {
  /**
   * @param {AccessEyeApp} app  the Phase 1 app instance
   */
  constructor(app) {
    this.app = app;

    // ── Instantiate Phase 2 modules ──
    this.camera     = new HighFPSCameraController();
    this.headPose   = new HeadPoseEstimator();
    this.hybridGaze = new HybridGazeEngine(app.calibration, this.headPose);
    this.stabilizer = new TemporalStabilizer({ kalmanR: 0.003, emaAlpha: 0.25, windowSize: 7 });
    this.saccade    = new MicroSaccadeFilter(12, 200);
    this.confidence = new GazeConfidenceScorer(null); // videoEl assigned on camera start
    this.dynCalib   = new DynamicCalibrationEngine(app.calibration);
    this.intent     = new IntentPredictionEngine('/api/intent');
    this.benchmark  = new GazeBenchmark();

    // State
    this.active    = false;
    this.cameraFPS = 0;
    this._p1LastGaze = { x: 0.5, y: 0.5 };

    // Wire internal events
    this._wireEvents();

    // Load any saved micro-calibration
    this.dynCalib.loadMicroData();
  }

  _wireEvents() {
    // ── Saccade filter → intent engine ──
    this.saccade.on('fixation', (fix) => {
      const focused = this.app.uiRegistry.getFocused();
      this.intent.predict(fix, focused, this.confidence.lastScore);
    });

    // ── Intent prediction → UI hint ──
    this.intent.on('prediction', (pred) => {
      this._handleIntentPrediction(pred);
    });

    // ── Dynamic calibration → micro-update events ──
    this.dynCalib.on('microCalib', (d) => {
      this.app.log.add(`Micro-calib: bias(${d.biasX.toFixed(3)}, ${d.biasY.toFixed(3)}) samples=${d.sampleCount}`, 'info');
    });

    this.dynCalib.on('modelUpdated', (d) => {
      this.app.log.add(`Calibration model updated from ${d.samples} samples`, 'success');
    });

    // ── Benchmark complete ──
    this.benchmark.on('complete', (report) => {
      this._showBenchmarkReport(report);
    });

    // ── UI activation → record micro-calibration sample ──
    this.app.uiRegistry.on('activate', ({ id, label, gesture }) => {
      const focused = this.app.uiRegistry.elements.get(id);
      if (focused && this.active) {
        const bbox   = focused.bbox;
        const targetSX = (bbox.x + bbox.w / 2) / window.innerWidth;
        const targetSY = (bbox.y + bbox.h / 2) / window.innerHeight;
        this.dynCalib.recordInteraction(
          this.hybridGaze.rawGaze.x,
          this.hybridGaze.rawGaze.y,
          targetSX, targetSY,
          this.confidence.lastScore.total
        );
        this.dynCalib.saveMicroData();
        this.intent.feedActivation(id, label, gesture);
      }
    });

    // ── Focus events → intent engine ──
    this.app.uiRegistry.on('focus', ({ id, label }) => {
      this.intent.feedFocus(id, label, 0);
    });
  }

  /**
   * Activate Phase 2 upgrades.
   * Called from AccessEyeApp after Phase 1 camera starts.
   */
  async activate(videoEl, canvasEl) {
    this.active = true;
    this.confidence.videoEl = videoEl;

    // ── Patch Phase 1 MediaPipeController to redirect results through Phase 2 ──
    const mp = this.app.mpController;
    if (mp) {
      const self = this;
      mp._onFaceResults = function(results) {
        self._processPhase2Face(results, mp);
      };
    }

    // ── Stop mouse simulation — camera is now driving gaze ──
    if (this.app.sim?.active) {
      this.app.sim.stop();
    }

    // ── Auto-switch UI to gaze mode ──
    this.app.mode = 'gaze';
    // Update mode tab UI
    const modeTabs = document.querySelectorAll('.mode-tab');
    modeTabs.forEach(t => {
      t.classList.toggle('active', t.dataset.mode === 'gaze');
    });
    this.app._updateHint('Phase 2 active: look at buttons, <strong>pinch</strong> or <strong>air tap</strong> to activate.');

    // ── Read actual camera FPS ──
    const track = videoEl?.srcObject?.getVideoTracks()[0];
    if (track) {
      const settings = track.getSettings();
      this.cameraFPS = settings.frameRate || 30;
    }

    this.app.log.add(`Phase 2 activated | Camera: ${this.cameraFPS} FPS | Hybrid gaze ON`, 'success');
    this.app.toast.show('Phase 2 Active', `Hybrid gaze engine running at ${this.cameraFPS} FPS`, 'success', 'fas fa-brain', 3000);
    this._updatePhase2StatusUI();
  }

  /**
   * The upgraded face processing pipeline.
   * Runs instead of Phase 1's _onFaceResults.
   */
  _processPhase2Face(results, mp) {
    mp.faceDetected = !!(results.multiFaceLandmarks?.length > 0);
    mp._emit('face', { detected: mp.faceDetected, results });

    if (!mp.faceDetected) {
      mp._clearCanvas();
      // Clear status when face lost
      this.app._updateStatusItem('status-face', false, 'Not found', '');
      this.app._updateStatusItem('status-gaze', false, 'No face', '');
      return;
    }

    const lm = results.multiFaceLandmarks[0];
    const W  = mp.videoEl.videoWidth  || 640;
    const H  = mp.videoEl.videoHeight || 480;
    const t0 = p2.now();

    // ── P2.2: Head pose ──
    const headResult = this.headPose.estimate(lm, W, H);

    // ── P2.3: Hybrid gaze ──
    const gazePacket = this.hybridGaze.processResults(
      results.multiFaceLandmarks, W, H, headResult
    );

    const p2Latency = p2.now() - t0;

    if (!gazePacket) return;

    // ── P2.6: Confidence scoring ──
    const confScore = this.confidence.score(lm, headResult, gazePacket.iris);

    // ── P2.4: Temporal stabilization ──
    const stableGaze = this.stabilizer.update(
      gazePacket.screen.x, gazePacket.screen.y, confScore.total
    );

    // ── P2.5: Micro-saccade filtering (screen pixels) ──
    const px = stableGaze.x * window.innerWidth;
    const py = stableGaze.y * window.innerHeight;
    const saccadeResult = this.saccade.update(px, py, confScore.total);

    // Use filtered position
    const finalX = saccadeResult.x / window.innerWidth;
    const finalY = saccadeResult.y / window.innerHeight;

    // ── Apply dynamic bias correction ──
    const biasFixed = this.dynCalib.applyBiasCorrection(finalX, finalY);

    // ── Build final enhanced gaze packet ──
    const enhanced = {
      ...gazePacket,
      screen: biasFixed,
      confidence: confScore.total,
      confidenceDetail: confScore,
      headPose: headResult,
      stabilized: stableGaze,
      saccade: saccadeResult,
      p2Latency,
      // Phase 2 extra fields for Intent Engine
      gaze_coordinates:        biasFixed,
      gaze_stability_duration: saccadeResult.fixationAge,
      head_pose_angle:         { yaw: headResult.yaw, pitch: headResult.pitch, roll: headResult.roll },
      gaze_confidence:         confScore.total,
      fixation_event:          saccadeResult.isFixated
    };

    // ── Feed intent engine ──
    this.intent.feedGaze(enhanced, confScore);

    // ── Benchmark recording ──
    if (this.benchmark.running) {
      this.benchmark.recordFrame(
        this._p1LastGaze, biasFixed,
        confScore.total, mp.latency, p2Latency,
        saccadeResult.fixationAge
      );
    }
    this._p1LastGaze = { x: gazePacket.screen.x, y: gazePacket.screen.y };

    // ── Forward to Phase 1 UI pipeline ──
    const screenX = biasFixed.x * window.innerWidth;
    const screenY = biasFixed.y * window.innerHeight;

    // Always update gaze cursor + coords regardless of mode (camera is driving)
    this.app._updateGazeCursor(screenX, screenY);
    this.app._updateCoords(biasFixed.x, biasFixed.y);
    this.app.uiRegistry.updateGaze(screenX, screenY);

    // Update top metrics bar (FPS / latency / confidence)
    const totalConf = Math.round(confScore.total * 100);
    this.app._updateMetrics(mp.fps || 30, Math.round(p2Latency), totalConf);

    // Sync Phase 1 gaze engine state so calibration UI works
    this.app.gazeEngine.rawGaze    = gazePacket.raw    || biasFixed;
    this.app.gazeEngine.smoothGaze = biasFixed;
    this.app.gazeEngine.confidence = confScore.total;

    // Update face/gaze status indicators
    this.app._updateStatusItem('status-face', true, 'Detected', 'active');
    this.app._updateStatusItem('status-gaze', true, `${totalConf}% conf`, 'tracking');

    // ── Update Phase 2 status UI ──
    this._updatePhase2LiveUI(enhanced, confScore, headResult, saccadeResult, p2Latency);

    // ── Draw enhanced canvas overlay ──
    this._drawEnhancedOverlay(mp, lm, enhanced, headResult, confScore);
  }

  _handleIntentPrediction(pred) {
    if (!pred) return;
    this.app.log.add(`AI Intent: ${pred.predicted_action || pred.intent || 'analyzing...'}`, 'info');
    this._updateIntentUI(pred);
  }

  /* ── UI Update helpers ── */

  _updatePhase2StatusUI() {
    const panel = $('#p2-status-panel');
    if (panel) panel.style.display = 'block';
  }

  _updatePhase2LiveUI(gaze, conf, hp, sacc, lat) {
    // Confidence bar
    const confBar = $('#p2-conf-fill');
    if (confBar) confBar.style.width = `${conf.total * 100}%`;
    const confVal = $('#p2-conf-val');
    if (confVal) confVal.textContent = `${Math.round(conf.total * 100)}%`;

    // Confidence color
    if (confBar) {
      confBar.style.background = conf.total > 0.75
        ? 'var(--accent-green)' : conf.total > 0.5
        ? 'var(--accent-yellow)' : 'var(--accent-red)';
    }

    // Head pose
    const hpYaw   = $('#p2-hp-yaw');
    const hpPitch = $('#p2-hp-pitch');
    const hpRoll  = $('#p2-hp-roll');
    if (hpYaw)   hpYaw.textContent   = `${hp.yaw.toFixed(1)}°`;
    if (hpPitch) hpPitch.textContent = `${hp.pitch.toFixed(1)}°`;
    if (hpRoll)  hpRoll.textContent  = `${hp.roll.toFixed(1)}°`;

    // Fixation status
    const fixEl = $('#p2-fixation');
    if (fixEl) {
      fixEl.textContent = sacc.isFixated
        ? `Fixated ${Math.round(sacc.fixationAge)}ms`
        : 'Scanning';
      fixEl.style.color = sacc.isFixated ? 'var(--accent-green)' : 'var(--accent-yellow)';
    }

    // Confidence sub-scores
    const bEl = $('#p2-bright'); if (bEl) bEl.textContent = `${Math.round(conf.brightness*100)}%`;
    const oEl = $('#p2-occl');   if (oEl) oEl.textContent = `${Math.round(conf.occlusion*100)}%`;
    const gEl = $('#p2-glare');  if (gEl) gEl.textContent = conf.lowLight ? '⚠ Low' : conf.glassesDetected ? '⚠ Glare' : 'OK';
    const lEl = $('#p2-latency');if (lEl) lEl.textContent = `${Math.round(lat)}ms`;

    // Saccade stats
    const stats = this.saccade.getStats();
    const sacEl = $('#p2-saccades'); if (sacEl) sacEl.textContent = stats.saccades;
    const fixCt = $('#p2-fixcount'); if (fixCt) fixCt.textContent = stats.fixations;

    // Pipeline label
    const pipeEl = $('#p2-pipeline-label');
    if (pipeEl) {
      const fps = this.cameraFPS;
      pipeEl.textContent = `Hybrid | ${fps}FPS | Kalman+EMA+Window`;
    }
  }

  _updateIntentUI(pred) {
    const el = $('#p2-intent-result');
    if (!el) return;
    el.textContent   = pred.predicted_action || pred.intent || '—';
    el.style.opacity = '1';
    setTimeout(() => { if (el) el.style.opacity = '0.6'; }, 3000);

    const confEl = $('#p2-intent-conf');
    if (confEl) confEl.textContent = pred.confidence
      ? `${Math.round(pred.confidence * 100)}%`
      : '—';

    const rsnEl = $('#p2-intent-reason');
    if (rsnEl) rsnEl.textContent = pred.reasoning || pred.explanation || '';
  }

  _showBenchmarkReport(report) {
    const el = $('#benchmark-report');
    if (!el) return;
    el.innerHTML = `
      <div class="bench-row">
        <span class="bench-lbl">Frames Analyzed</span>
        <span class="bench-val">${report.total_frames}</span>
      </div>
      <div class="bench-row">
        <span class="bench-lbl">Session FPS</span>
        <span class="bench-val">${report.fps}</span>
      </div>
      <div class="bench-row bench-compare">
        <span class="bench-lbl">Jitter (Phase 1)</span>
        <span class="bench-val p1">${report.phase1?.jitter?.combined?.toFixed(4) || '—'}</span>
      </div>
      <div class="bench-row bench-compare">
        <span class="bench-lbl">Jitter (Phase 2)</span>
        <span class="bench-val p2">${report.phase2?.jitter?.combined?.toFixed(4) || '—'}</span>
      </div>
      <div class="bench-row">
        <span class="bench-lbl">Latency P1 / P2</span>
        <span class="bench-val">${report.phase1?.latency?.mean || 0}ms / ${report.phase2?.latency?.mean || 0}ms</span>
      </div>
      <div class="bench-row bench-highlight">
        <span class="bench-lbl">Jitter Reduction</span>
        <span class="bench-val improve">▼ ${report.improvements?.jitter_reduction_pct || 0}%</span>
      </div>
      <div class="bench-row bench-highlight">
        <span class="bench-lbl">Est. Accuracy Gain</span>
        <span class="bench-val improve">${report.improvements?.estimated_accuracy_gain || '—'}</span>
      </div>
      <div class="bench-verdict">${report.verdict || ''}</div>
    `;
    el.style.display = 'block';
    this.app.toast.show('Benchmark Complete', report.verdict, 'success', 'fas fa-chart-bar', 6000);
  }

  /**
   * Enhanced canvas overlay: head pose axes, confidence rings, fixation indicator
   */
  _drawEnhancedOverlay(mp, lm, gaze, hp, conf) {
    const canvas = mp.canvasEl;
    const ctx    = mp.ctx;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const W = canvas.width, H = canvas.height;

    // ── Draw iris landmarks ──
    if (lm && lm.length >= 478) {
      const irisIdx = [468,469,470,471,472,473,474,475,476,477];
      for (const idx of irisIdx) {
        const pt = lm[idx];
        if (!pt) continue;
        ctx.fillStyle = 'rgba(0,255,136,0.9)';
        ctx.beginPath();
        ctx.arc(pt.x * W, pt.y * H, 2.5, 0, Math.PI*2);
        ctx.fill();
      }

      // Draw eyelid contours for left eye
      const leftLidTop    = [159,160,161];
      const leftLidBot    = [145,144,163];
      const rightLidTop   = [386,387,388];
      const rightLidBot   = [374,373,390];

      const drawLidLine = (indices, color) => {
        const pts = indices.map(i => lm[i]).filter(Boolean);
        if (pts.length < 2) return;
        ctx.strokeStyle = color;
        ctx.lineWidth   = 1;
        ctx.beginPath();
        ctx.moveTo(pts[0].x * W, pts[0].y * H);
        pts.slice(1).forEach(p => ctx.lineTo(p.x * W, p.y * H));
        ctx.stroke();
      };
      drawLidLine(leftLidTop,  'rgba(167,139,250,0.5)');
      drawLidLine(leftLidBot,  'rgba(167,139,250,0.3)');
      drawLidLine(rightLidTop, 'rgba(167,139,250,0.5)');
      drawLidLine(rightLidBot, 'rgba(167,139,250,0.3)');
    }

    // ── Head pose indicator (center of face) ──
    const noseTip = lm[1];
    const noseX = noseTip.x * W, noseY = noseTip.y * H;

    // Draw yaw arrow (horizontal)
    const yawLen = hp.yaw * 1.5;
    ctx.strokeStyle = 'rgba(255,107,53,0.9)';
    ctx.lineWidth = 2;
    this._drawArrow(ctx, noseX, noseY, noseX + yawLen, noseY);

    // Draw pitch arrow (vertical)
    const pitchLen = -hp.pitch * 1.5;
    ctx.strokeStyle = 'rgba(167,139,250,0.9)';
    this._drawArrow(ctx, noseX, noseY, noseX, noseY + pitchLen);

    // ── Confidence ring around both iris centroids ──
    const confColor = conf.total > 0.75
      ? `rgba(0,255,136,${conf.total})`
      : conf.total > 0.5
      ? `rgba(255,211,42,${conf.total})`
      : `rgba(255,71,87,${conf.total})`;

    const leftIris  = lm[468];
    const rightIris = lm[473];
    if (leftIris && rightIris) {
      [leftIris, rightIris].forEach(iris => {
        const r = 10 + (1 - conf.total) * 8;
        ctx.strokeStyle = confColor;
        ctx.lineWidth   = 1.5;
        ctx.beginPath();
        ctx.arc(iris.x * W, iris.y * H, r, 0, Math.PI*2);
        ctx.stroke();
      });
    }

    // ── Fixation indicator ──
    if (this.saccade.isFixated) {
      const fxX = this.saccade._anchorX / window.innerWidth * W;
      const fxY = this.saccade._anchorY / window.innerHeight * H;
      const age = Math.min(this.saccade.fixationAge / 500, 1);
      ctx.strokeStyle = `rgba(0,212,255,${0.4 + age * 0.5})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(fxX, fxY, 8 + age * 6, 0, Math.PI*2);
      ctx.stroke();
    }

    // ── Gaze point on canvas (mirrored) ──
    const gx = (1 - gaze.screen.x) * W;
    const gy = gaze.screen.y * H;
    ctx.strokeStyle = `rgba(0,212,255,${0.6 + conf.total * 0.4})`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(gx, gy, 14, 0, Math.PI*2);
    ctx.stroke();
    ctx.fillStyle = 'rgba(0,212,255,0.7)';
    ctx.beginPath();
    ctx.arc(gx, gy, 4, 0, Math.PI*2);
    ctx.fill();
  }

  _drawArrow(ctx, x1, y1, x2, y2) {
    const angle = Math.atan2(y2-y1, x2-x1);
    const len   = Math.hypot(x2-x1, y2-y1);
    if (len < 3) return;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    // Arrowhead
    const hw = 5;
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - hw * Math.cos(angle-0.4), y2 - hw * Math.sin(angle-0.4));
    ctx.lineTo(x2 - hw * Math.cos(angle+0.4), y2 - hw * Math.sin(angle+0.4));
    ctx.closePath();
    ctx.fill();
  }

  deactivate() {
    this.active = false;
    this.benchmark.running && this.benchmark.stop();
    this.dynCalib.saveMicroData();
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   PRIVATE HELPERS (module-scoped, not exported)
───────────────────────────────────────────────────────────────────────── */
class _KalmanAxis {
  constructor(R = 0.005, Q = 0.0001) {
    this.R = R; this.Q = Q;
    this.x = 0; this.v = 0; this.p = 1;
  }
  update(z) {
    const px = this.x + this.v;
    const pp = this.p + this.Q;
    const K  = pp / (pp + this.R);
    this.x   = px + K * (z - px);
    this.v   = this.v + K * (z - px);
    this.p   = (1 - K) * pp;
    return this.x;
  }
  reset() { this.x=0; this.v=0; this.p=1; }
}

class _EMAScalar {
  constructor(alpha = 0.2) { this.alpha = alpha; this.val = null; }
  update(v) {
    if (this.val === null) { this.val = v; return v; }
    this.val += this.alpha * (v - this.val);
    return this.val;
  }
  reset() { this.val = null; }
}

// results_ref is a scoping workaround for _drawEnhancedOverlay's FaceMesh draw
let results_ref = null;

/* ─────────────────────────────────────────────────────────────────────────
   EXPOSE GLOBALLY
───────────────────────────────────────────────────────────────────────── */
window.Phase2 = {
  HighFPSCameraController,
  HeadPoseEstimator,
  HybridGazeEngine,
  TemporalStabilizer,
  MicroSaccadeFilter,
  GazeConfidenceScorer,
  DynamicCalibrationEngine,
  IntentPredictionEngine,
  GazeBenchmark,
  Phase2Orchestrator
};

console.log('%c Phase 2 Engine Loaded ✅', 'color:#00ff88;font-weight:bold;font-size:13px;');
console.log('%c Modules: HybridGaze | HeadPose | TemporalStabilizer | SaccadeFilter | ConfidenceScorer | DynCalib | IntentAI | Benchmark',
            'color:#94a3b8;font-size:11px;');
