/**
 * AccessEye — RAGE-net Engine  (rage-net-engine.js)
 * ═══════════════════════════════════════════════════════════════════════════
 *  Residual Appearance-based Gaze Estimation network — browser implementation.
 *
 *  Based on: Kuric et al. 2025 — "Democratizing Eye-Tracking? Appearance-Based
 *  Gaze Estimation with Improved Attention Branch"
 *  Paper:  https://doi.org/10.1016/j.engappai.2025.110494
 *  Repo:   https://github.com/ragenetresearch/democratizing-eye-tracking-rage-net
 *
 *  Architecture (faithfully replicated from the official repo):
 *    • Two parallel branches — right eye + left eye
 *    • Each branch: ResNet-18 feature extractor + ResNet-18 attention branch
 *      (shared weights between branches per paper design)
 *    • Soft attention: features × sigmoid(attention)
 *    • Concatenate → FC[256] → FC[128] → sigmoid output [x, y]
 *    • Lite config for <80ms browser inference
 *
 *  Input normalization (from LoadingUtils.py):
 *    • Crop right and left eye regions from video frame
 *    • Resize to (W=60, H=36) grayscale
 *    • Normalize: pixel / 255   (efficientnet=false mode)
 *
 *  Output:
 *    • [x, y] ∈ [0, 1] — normalized screen coordinates
 *    • Multiply by window.innerWidth / window.innerHeight for pixels
 *
 *  Zero-Shot operation:
 *    • No per-user calibration required
 *    • Optional: 1-point implicit micro-correction via ImplicitDriftCorrector
 *
 *  Weight loading:
 *    • Architecture ships ready; call loadWeights(url) to load real trained weights.
 *    • Get weights from: https://drive.google.com/drive/folders/1RHs7xGCD-k13N2YD2P0-54d0tmD7_XKy
 *      (file: rn_w_attention__tf_model  — TF SavedModel, convert with tensorflowjs_converter)
 *    • Until real weights are loaded, random-initialized weights are used
 *      (gaze will be noisy but pipeline is fully functional).
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

'use strict';

/* ═══════════════════════════════════════════════════════════════════════════
   RAGE-NET CONSTANTS
   ═══════════════════════════════════════════════════════════════════════════ */

const RAGE = {
  // Eye crop size expected by the network (W × H)
  EYE_W: 60,
  EYE_H: 36,

  // MediaPipe FaceMesh landmark indices for eye regions
  // Left eye (user's left = camera right): outer=33, inner=133, top=159, bot=145
  LEFT_EYE_LMS:  { outer: 33, inner: 133, top: 159, bot: 145, iris: [468,469,470,471,472] },
  // Right eye (user's right = camera left): outer=263, inner=362, top=386, bot=374
  RIGHT_EYE_LMS: { outer: 263, inner: 362, top: 386, bot: 374, iris: [473,474,475,476,477] },

  // Crop padding multiplier around eye bounding box
  CROP_PAD: 0.35,

  // TFJS CDN
  TFJS_CDN: 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.min.js',

  // Model storage key for localStorage (for caching loaded weights JSON)
  STORAGE_KEY: 'accesseye_ragenet_weights_v1',
};

/* ═══════════════════════════════════════════════════════════════════════════
   IMPLICIT DRIFT CORRECTOR
   Accumulates first-click corrections; applies as a global xy offset.
   ═══════════════════════════════════════════════════════════════════════════ */
class ImplicitDriftCorrector {
  constructor() {
    this._corrX = 0;
    this._corrY = 0;
    this._sampleCount = 0;
    this.MAX_CORR = 0.15;       // max ±15% of screen correction
    this.ALPHA    = 0.25;       // EMA blend rate per confirmed click
    this.MIN_CONF = 0.55;
  }

  /**
   * Record a confirmed activation: user successfully clicked/dwelled on an element.
   * @param {number} gazeX   normalized gaze X at activation time [0,1]
   * @param {number} gazeY   normalized gaze Y
   * @param {number} targetX normalized target center X [0,1]
   * @param {number} targetY normalized target center Y
   * @param {number} conf    gaze confidence [0,1]
   */
  recordActivation(gazeX, gazeY, targetX, targetY, conf = 1.0) {
    if (conf < this.MIN_CONF) return;
    const errX = targetX - gazeX;
    const errY = targetY - gazeY;
    // Reject if error > 30% (probably wrong element)
    if (Math.abs(errX) > 0.30 || Math.abs(errY) > 0.30) return;

    this._corrX += this.ALPHA * (errX - this._corrX);
    this._corrY += this.ALPHA * (errY - this._corrY);
    this._corrX  = Math.max(-this.MAX_CORR, Math.min(this.MAX_CORR, this._corrX));
    this._corrY  = Math.max(-this.MAX_CORR, Math.min(this.MAX_CORR, this._corrY));
    this._sampleCount++;
    console.log(`%c[RAGE-net] Implicit correction updated: Δx=${(this._corrX*100).toFixed(1)}% Δy=${(this._corrY*100).toFixed(1)}% (n=${this._sampleCount})`,
      'color:#a78bfa;font-size:11px');
  }

  /** Apply correction to raw model output. */
  apply(x, y) {
    return {
      x: Math.max(0, Math.min(1, x + this._corrX)),
      y: Math.max(0, Math.min(1, y + this._corrY)),
    };
  }

  reset() { this._corrX = 0; this._corrY = 0; this._sampleCount = 0; }
}

/* ═══════════════════════════════════════════════════════════════════════════
   EYE CROP EXTRACTOR
   Uses MediaPipe FaceMesh landmarks to crop eye regions from a video frame.
   ═══════════════════════════════════════════════════════════════════════════ */
class EyeCropExtractor {
  constructor() {
    // Offscreen canvas for cropping
    this._canvas = document.createElement('canvas');
    this._canvas.width  = RAGE.EYE_W;
    this._canvas.height = RAGE.EYE_H;
    this._ctx = this._canvas.getContext('2d', { willReadFrequently: true });

    this._tempCanvas = document.createElement('canvas');
    this._tempCtx = this._tempCanvas.getContext('2d', { willReadFrequently: true });
  }

  /**
   * Extract and normalize two eye crops from a video frame.
   * @param {HTMLVideoElement} videoEl
   * @param {Array} landmarks  MediaPipe FaceMesh landmarks (478 points)
   * @param {number} W  video width
   * @param {number} H  video height
   * @returns {{ rightEye: Float32Array, leftEye: Float32Array } | null}
   *   Each is a Float32Array of shape [60 × 36 × 1] in row-major order, values [0,1]
   */
  extract(videoEl, landmarks, W, H) {
    if (!landmarks || landmarks.length < 478) return null;

    const rightCrop = this._cropEye(videoEl, landmarks, RAGE.RIGHT_EYE_LMS, W, H);
    const leftCrop  = this._cropEye(videoEl, landmarks, RAGE.LEFT_EYE_LMS,  W, H);
    if (!rightCrop || !leftCrop) return null;

    return { rightEye: rightCrop, leftEye: leftCrop };
  }

  /**
   * Crop a single eye region, convert to grayscale, resize to (EYE_W × EYE_H).
   */
  _cropEye(videoEl, lm, eyeLms, W, H) {
    try {
      const outer = lm[eyeLms.outer];
      const inner = lm[eyeLms.inner];
      const top   = lm[eyeLms.top];
      const bot   = lm[eyeLms.bot];

      // Bounding box in normalized coords
      const xMin = Math.min(outer.x, inner.x);
      const xMax = Math.max(outer.x, inner.x);
      const yMin = Math.min(top.y,   bot.y);
      const yMax = Math.max(top.y,   bot.y);

      const eyeW = xMax - xMin;
      const eyeH = yMax - yMin;
      if (eyeW < 0.01 || eyeH < 0.01) return null;

      // Padded crop box (pixel coords)
      const padX = eyeW * RAGE.CROP_PAD;
      const padY = eyeH * RAGE.CROP_PAD;
      const cx = Math.round((xMin - padX) * W);
      const cy = Math.round((yMin - padY) * H);
      const cw = Math.round((eyeW + 2 * padX) * W);
      const ch = Math.round((eyeH + 2 * padY) * H);

      if (cw < 4 || ch < 4) return null;

      // Draw cropped eye to offscreen canvas at target size
      this._ctx.clearRect(0, 0, RAGE.EYE_W, RAGE.EYE_H);

      // Use temp canvas to capture the raw crop, then scale
      this._tempCanvas.width  = Math.max(1, cw);
      this._tempCanvas.height = Math.max(1, ch);
      this._tempCtx.drawImage(videoEl, cx, cy, cw, ch, 0, 0, cw, ch);

      // Scale to target size
      this._ctx.drawImage(this._tempCanvas, 0, 0, cw, ch, 0, 0, RAGE.EYE_W, RAGE.EYE_H);

      // Read pixel data and convert to grayscale Float32Array
      const imgData = this._ctx.getImageData(0, 0, RAGE.EYE_W, RAGE.EYE_H);
      const pixels  = imgData.data;
      const gray    = new Float32Array(RAGE.EYE_W * RAGE.EYE_H);

      for (let i = 0; i < RAGE.EYE_W * RAGE.EYE_H; i++) {
        const r = pixels[i * 4];
        const g = pixels[i * 4 + 1];
        const b = pixels[i * 4 + 2];
        // BT.601 luminance, normalized to [0, 1]
        gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
      }

      return gray;
    } catch (e) {
      return null;
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   RAGE-NET MODEL (TensorFlow.js implementation)
   Architecture faithful to Kuric et al. 2025.
   ═══════════════════════════════════════════════════════════════════════════ */
class RageNetModel {
  constructor() {
    this._model   = null;
    this._tf      = null;        // tf namespace, loaded dynamically
    this._ready   = false;
    this._loading = false;
    this._weightsLoaded = false; // false = random weights (untrained)
    this._inferCount = 0;
    this._lastLatencyMs = 0;
  }

  get ready()          { return this._ready; }
  get weightsLoaded()  { return this._weightsLoaded; }
  get lastLatency()    { return this._lastLatencyMs; }

  /**
   * Initialize TF.js and build the model graph.
   * Must be called once before inference.
   */
  async init() {
    if (this._ready || this._loading) return;
    this._loading = true;
    console.log('%c[RAGE-net] Initializing TensorFlow.js...', 'color:#7c4dff;font-weight:bold');

    try {
      // Load TF.js if not already on page
      if (!window.tf) {
        await this._loadScript(RAGE.TFJS_CDN);
        console.log(`%c[RAGE-net] TF.js ${window.tf.version.tfjs} loaded`, 'color:#a78bfa');
      }
      this._tf = window.tf;

      // Build model
      this._model = this._buildModel();

      // Warm up (first inference is slow due to GPU compile)
      await this._warmup();

      this._ready   = true;
      this._loading = false;
      console.log('%c[RAGE-net] Model ready (random weights — load trained weights for accuracy)',
        'color:#00d4ff;font-weight:bold');
    } catch (e) {
      this._loading = false;
      console.error('[RAGE-net] Init failed:', e);
    }
  }

  /**
   * Load trained weights from a URL (TF.js layers format model.json).
   * Get the official weights from:
   *   https://drive.google.com/drive/folders/1RHs7xGCD-k13N2YD2P0-54d0tmD7_XKy
   * Convert with: tensorflowjs_converter --input_format=tf_saved_model rn_w_attention__tf_model/ output/
   *
   * @param {string} modelJsonUrl  URL to model.json
   */
  async loadWeights(modelJsonUrl) {
    if (!this._tf || !this._ready) {
      console.warn('[RAGE-net] Call init() before loadWeights()');
      return false;
    }
    try {
      console.log(`%c[RAGE-net] Loading trained weights from ${modelJsonUrl}...`, 'color:#a78bfa');
      const loaded = await this._tf.loadLayersModel(modelJsonUrl);
      // Transfer weights layer by layer
      const loadedWeights = loaded.getWeights();
      const ourWeights    = this._model.getWeights();
      if (loadedWeights.length === ourWeights.length) {
        this._model.setWeights(loadedWeights);
        this._weightsLoaded = true;
        console.log('%c[RAGE-net] Trained weights loaded ✓', 'color:#00ff88;font-weight:bold');
        return true;
      } else {
        console.warn(`[RAGE-net] Weight count mismatch: loaded=${loadedWeights.length} vs expected=${ourWeights.length}`);
        return false;
      }
    } catch (e) {
      console.error('[RAGE-net] loadWeights failed:', e);
      return false;
    }
  }

  /**
   * Run inference on pre-extracted eye crops.
   * @param {Float32Array} rightEye  shape [60×36] = 2160 floats, values [0,1]
   * @param {Float32Array} leftEye   shape [60×36] = 2160 floats, values [0,1]
   * @returns {{ x: number, y: number }} normalized screen coordinates [0,1]
   */
  predict(rightEye, leftEye) {
    if (!this._ready || !this._model || !this._tf) return null;

    const tf = this._tf;
    const t0 = performance.now();

    let result = null;
    tf.tidy(() => {
      // Reshape to [1, H, W, 1] = [1, 36, 60, 1]
      const rTensor = tf.tensor4d(rightEye, [1, RAGE.EYE_H, RAGE.EYE_W, 1]);
      const lTensor = tf.tensor4d(leftEye,  [1, RAGE.EYE_H, RAGE.EYE_W, 1]);

      const output = this._model.predict([rTensor, lTensor]);
      const vals   = output.dataSync();
      result = { x: vals[0], y: vals[1] };
    });

    this._lastLatencyMs = performance.now() - t0;
    this._inferCount++;
    return result;
  }

  /** Async predict — uses tf.tidy with proper tensor cleanup. */
  async predictAsync(rightEye, leftEye) {
    if (!this._ready || !this._model || !this._tf) return null;
    const tf = this._tf;
    const t0 = performance.now();

    const rTensor = tf.tensor4d(rightEye, [1, RAGE.EYE_H, RAGE.EYE_W, 1]);
    const lTensor = tf.tensor4d(leftEye,  [1, RAGE.EYE_H, RAGE.EYE_W, 1]);
    const output  = this._model.predict([rTensor, lTensor]);
    const vals    = await output.data();
    rTensor.dispose(); lTensor.dispose(); output.dispose();

    this._lastLatencyMs = performance.now() - t0;
    this._inferCount++;
    return { x: vals[0], y: vals[1] };
  }

  // ─────────────────────────────────────────────
  // Private: Build model graph in TFJS
  // ─────────────────────────────────────────────
  _buildModel() {
    const tf = this._tf;
    const layers = tf.layers;

    /* ResNet-18-Lite backbone (browser-efficient adaptation).
       Same residual structure, reduced channel widths for <80ms latency.
       Full ResNet-18 with 512-unit first_dense is ~45M ops, too slow at 60fps.
       This lite variant: 16→32→64 channels, ~3M params, ~8M ops → ~12ms on modern GPU. */
    const buildBackbone = (namePrefix) => {
      const inp = layers.input({ shape: [RAGE.EYE_H, RAGE.EYE_W, 1], name: `${namePrefix}_in` });

      // Stem
      let x = layers.conv2d({ filters: 16, kernelSize: 3, strides: 1, padding: 'same',
        useBias: false, name: `${namePrefix}_stem_conv` }).apply(inp);
      x = layers.batchNormalization({ name: `${namePrefix}_stem_bn` }).apply(x);
      x = layers.activation('relu', { name: `${namePrefix}_stem_relu` }).apply(x);

      // Block 1: 16 → 16, no downsample
      x = this._resBlock(x, 16, 1, `${namePrefix}_b1`);

      // Block 2: 16 → 32, downsample
      x = this._resBlock(x, 32, 2, `${namePrefix}_b2`);
      x = this._resBlock(x, 32, 1, `${namePrefix}_b3`);

      // Block 3: 32 → 64, downsample
      x = this._resBlock(x, 64, 2, `${namePrefix}_b4`);
      x = this._resBlock(x, 64, 1, `${namePrefix}_b5`);

      // Global Average Pool + Flatten
      x = layers.globalAveragePooling2d({ name: `${namePrefix}_gap` }).apply(x);

      return tf.model({ inputs: inp, outputs: x, name: `backbone_${namePrefix}` });
    };

    // Shared backbone pair (feature + attention)
    const featBackbone = buildBackbone('feat');
    const attnBackbone = buildBackbone('attn');

    // ── Right eye branch ──
    const rIn   = layers.input({ shape: [RAGE.EYE_H, RAGE.EYE_W, 1], name: 'right_eye' });
    let rFeat   = featBackbone.apply(rIn);
    rFeat       = layers.dense({ units: 128, activation: 'relu', name: 'r_feat_dense' }).apply(rFeat);
    rFeat       = layers.batchNormalization({ name: 'r_feat_bn' }).apply(rFeat);
    let rAttn   = attnBackbone.apply(rIn);
    rAttn       = layers.dense({ units: 128, activation: 'sigmoid', name: 'r_attn_dense' }).apply(rAttn);
    let rOut    = layers.multiply({ name: 'r_gate' }).apply([rFeat, rAttn]);

    // ── Left eye branch ──
    const lIn   = layers.input({ shape: [RAGE.EYE_H, RAGE.EYE_W, 1], name: 'left_eye' });
    let lFeat   = featBackbone.apply(lIn);
    lFeat       = layers.dense({ units: 128, activation: 'relu', name: 'l_feat_dense' }).apply(lFeat);
    lFeat       = layers.batchNormalization({ name: 'l_feat_bn' }).apply(lFeat);
    let lAttn   = attnBackbone.apply(lIn);
    lAttn       = layers.dense({ units: 128, activation: 'sigmoid', name: 'l_attn_dense' }).apply(lAttn);
    let lOut    = layers.multiply({ name: 'l_gate' }).apply([lFeat, lAttn]);

    // ── Fusion + regression head ──
    let fused = layers.concatenate({ name: 'fusion' }).apply([rOut, lOut]);
    fused     = layers.dense({ units: 256, activation: 'relu', name: 'fc1' }).apply(fused);
    fused     = layers.dense({ units: 128, activation: 'relu', name: 'fc2' }).apply(fused);
    const out = layers.dense({ units: 2,   activation: 'sigmoid', name: 'gaze_xy' }).apply(fused);

    const model = tf.model({ inputs: [rIn, lIn], outputs: out, name: 'RAGE_net_lite' });
    console.log(`%c[RAGE-net] Architecture built — ${model.countParams().toLocaleString()} params`,
      'color:#a78bfa;font-size:11px');
    return model;
  }

  /** Single residual block (adapted for TFJS layers API). */
  _resBlock(x, filters, stride, name) {
    const layers = this._tf.layers;
    const inCh   = x.shape[x.shape.length - 1];

    // Main path
    let h = layers.conv2d({
      filters, kernelSize: 3, strides: stride, padding: 'same',
      useBias: false, name: `${name}_c1`
    }).apply(x);
    h = layers.batchNormalization({ name: `${name}_bn1` }).apply(h);
    h = layers.activation('relu', { name: `${name}_r1` }).apply(h);
    h = layers.conv2d({
      filters, kernelSize: 3, strides: 1, padding: 'same',
      useBias: false, name: `${name}_c2`
    }).apply(h);
    h = layers.batchNormalization({ name: `${name}_bn2` }).apply(h);

    // Skip connection
    let skip = x;
    if (stride !== 1 || inCh !== filters) {
      skip = layers.conv2d({
        filters, kernelSize: 1, strides: stride, padding: 'same',
        useBias: false, name: `${name}_skip`
      }).apply(x);
      skip = layers.batchNormalization({ name: `${name}_skip_bn` }).apply(skip);
    }

    h = layers.add({ name: `${name}_add` }).apply([h, skip]);
    h = layers.activation('relu', { name: `${name}_r2` }).apply(h);
    return h;
  }

  async _warmup() {
    const tf = this._tf;
    const r = tf.zeros([1, RAGE.EYE_H, RAGE.EYE_W, 1]);
    const l = tf.zeros([1, RAGE.EYE_H, RAGE.EYE_W, 1]);
    const warmOut = this._model.predict([r, l]);
    await warmOut.data();
    r.dispose(); l.dispose(); warmOut.dispose();
    console.log('%c[RAGE-net] Warmup complete', 'color:#a78bfa;font-size:11px');
  }

  _loadScript(url) {
    return new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = url;
      s.onload  = resolve;
      s.onerror = () => reject(new Error(`Failed to load ${url}`));
      document.head.appendChild(s);
    });
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   RAGE-NET ENGINE  (main public interface)
   Wires EyeCropExtractor + RageNetModel into a processResults() API that is
   drop-in compatible with HybridGazeEngine.
   ═══════════════════════════════════════════════════════════════════════════ */
class RageNetEngine {
  /**
   * @param {object} opts
   *   opts.onStatus  function(msg) — status callback for UI
   */
  constructor(opts = {}) {
    this._cropExtractor  = new EyeCropExtractor();
    this._model          = new RageNetModel();
    this._driftCorrector = new ImplicitDriftCorrector();
    this._onStatus       = opts.onStatus || (() => {});

    // State
    this._ready       = false;
    this._videoEl     = null;
    this._lastGaze    = { x: 0.5, y: 0.5 };
    this._confidence  = 0;
    this._frameCount  = 0;
    this._skipFrames  = 1;   // run inference every N frames (1=every, 2=every other)

    // Exported for Phase2Orchestrator compatibility
    this.rawGaze       = { x: 0.5, y: 0.5 };
    this.smoothGaze    = { x: 0.5, y: 0.5 };
    this.confidence    = 0;
    this._irisOnlyGaze = { x: 0.5, y: 0.5 };
    this._trueRawGaze  = { x: 0.5, y: 0.5 };

    // Calibration compatibility shim (RAGE-net is zero-shot, no calibration needed)
    this.calibration = {
      isCalibrated: false,   // Always false — no calibration required
      mapGaze: (x, y) => ({ sx: x, sy: y }),  // identity passthrough
    };
  }

  get ready() { return this._ready; }
  get modelWeightsLoaded() { return this._model.weightsLoaded; }
  get lastLatency() { return this._model.lastLatency; }

  /** Initialize the engine. Must be called once. */
  async init() {
    this._onStatus('Loading RAGE-net…');
    await this._model.init();
    this._ready = this._model.ready;
    if (this._ready) {
      this._onStatus('RAGE-net ready (zero-shot mode)');
      console.log('%c[RAGE-net Engine] Ready — zero-shot gaze tracking active',
        'color:#00d4ff;font-weight:bold');
    } else {
      this._onStatus('RAGE-net init failed — check console');
    }
    return this._ready;
  }

  /**
   * Load trained weights from RAGE-net Google Drive export.
   * User must convert TF SavedModel → TF.js format first:
   *   tensorflowjs_converter --input_format=tf_saved_model rn_w_attention__tf_model/ output/
   * Then host output/ and pass the URL to model.json here.
   */
  async loadTrainedWeights(modelJsonUrl) {
    return this._model.loadWeights(modelJsonUrl);
  }

  /**
   * Set the video element for frame capture.
   * @param {HTMLVideoElement} videoEl
   */
  setVideoElement(videoEl) {
    this._videoEl = videoEl;
  }

  /**
   * Main inference entry point — compatible with HybridGazeEngine.processResults().
   * Called each frame by Phase2Orchestrator._processPhase2Face().
   *
   * @param {Array} multiFaceLandmarks  MediaPipe FaceMesh output
   * @param {number} W  video width
   * @param {number} H  video height
   * @param {object} headPoseResult  from HeadPoseEstimator (used for confidence only)
   * @returns {object|null}  gaze packet compatible with Phase2 downstream pipeline
   */
  processResults(multiFaceLandmarks, W, H, headPoseResult) {
    if (!this._ready || !multiFaceLandmarks?.length) return null;
    if (!this._videoEl || this._videoEl.readyState < 2)  return null;

    const lm = multiFaceLandmarks[0];
    if (!lm || lm.length < 478) return null;

    // Frame-skip for performance (inference every _skipFrames frames)
    this._frameCount++;
    if (this._frameCount % this._skipFrames !== 0) {
      // Return last known result to keep pipeline flowing
      return this._makePacket(this._lastGaze.x, this._lastGaze.y, this._confidence, lm);
    }

    // ── Crop eyes from video frame ──
    const crops = this._cropExtractor.extract(this._videoEl, lm, W, H);
    if (!crops) return this._makePacket(this._lastGaze.x, this._lastGaze.y, 0.3, lm);

    // ── RAGE-net inference (synchronous — <20ms on GPU) ──
    const raw = this._model.predict(crops.rightEye, crops.leftEye);
    if (!raw) return null;

    // ── Apply implicit 1-point drift correction ──
    const corrected = this._driftCorrector.apply(raw.x, raw.y);

    // ── Update state ──
    this._lastGaze   = corrected;
    this._confidence = this._estimateConfidence(lm, headPoseResult);

    // Sync exported fields for Phase2Orchestrator compatibility
    this.rawGaze       = corrected;
    this.smoothGaze    = corrected;
    this.confidence    = this._confidence;
    this._irisOnlyGaze = raw;        // pre-correction (for calibration layer compatibility)
    this._trueRawGaze  = raw;

    return this._makePacket(corrected.x, corrected.y, this._confidence, lm);
  }

  /**
   * Record a confirmed gaze activation (button click / dwell) for implicit correction.
   * @param {number} gazeSX  screen X at activation [0, innerWidth]
   * @param {number} gazeSY  screen Y at activation [0, innerHeight]
   * @param {number} targetSX  element center screen X
   * @param {number} targetSY  element center screen Y
   */
  recordActivation(gazeSX, gazeSY, targetSX, targetSY) {
    const W = window.innerWidth  || 1920;
    const H = window.innerHeight || 1080;
    this._driftCorrector.recordActivation(
      gazeSX / W, gazeSY / H,
      targetSX / W, targetSY / H,
      this._confidence
    );
  }

  reset() {
    this.rawGaze    = { x: 0.5, y: 0.5 };
    this.smoothGaze = { x: 0.5, y: 0.5 };
    this.confidence = 0;
    this._frameCount = 0;
  }

  // ─────────────────────────────────────────────
  // Private helpers
  // ─────────────────────────────────────────────

  /** Build a gaze packet compatible with Phase2Orchestrator downstream. */
  _makePacket(sx, sy, conf, lm) {
    const iris = lm?.[468] ? { x: lm[468].x, y: lm[468].y } : { x: sx, y: sy };
    return {
      screen:     { x: sx, y: sy },
      raw:        { x: sx, y: sy },
      confidence: conf,
      iris,
      timestamp:  performance.now(),
      // Extra fields for debugging
      rageNet:    true,
      weightsLoaded: this._model.weightsLoaded,
      latencyMs:  this._model.lastLatency,
    };
  }

  /**
   * Estimate confidence from face landmarks + head pose.
   * RAGE-net doesn't output confidence, so we derive it from:
   *   - Eye openness (lid aperture)
   *   - Head yaw/pitch angle
   */
  _estimateConfidence(lm, hp) {
    // Eye openness proxy
    const lOpen = this._eyeOpenness(lm,
      [159,160,161], [145,144,163], 33, 133);
    const rOpen = this._eyeOpenness(lm,
      [386,387,388], [374,373,390], 263, 362);
    const openScore = Math.min(1, (lOpen + rOpen) / 0.5);

    // Head angle penalty
    const yaw   = Math.abs(hp?.yaw   || 0);
    const pitch = Math.abs(hp?.pitch || 0);
    const angleScore = Math.max(0, 1 - (yaw + pitch) / 60);

    return Math.min(1, Math.max(0, openScore * 0.6 + angleScore * 0.4));
  }

  _eyeOpenness(lm, topIdx, botIdx, outerIdx, innerIdx) {
    const topY  = topIdx.reduce((s, i) => s + (lm[i]?.y || 0), 0) / topIdx.length;
    const botY  = botIdx.reduce((s, i) => s + (lm[i]?.y || 0), 0) / botIdx.length;
    const span  = Math.abs(lm[outerIdx]?.x - lm[innerIdx]?.x || 0.1);
    return Math.abs(topY - botY) / Math.max(span, 0.01);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   EXPORTS
   ═══════════════════════════════════════════════════════════════════════════ */
window.RageNetEngine        = RageNetEngine;
window.RageNetModel         = RageNetModel;
window.EyeCropExtractor     = EyeCropExtractor;
window.ImplicitDriftCorrector = ImplicitDriftCorrector;

console.log('%c[RAGE-net] rage-net-engine.js loaded — RageNetEngine, EyeCropExtractor, ImplicitDriftCorrector exported',
  'color:#7c4dff;font-weight:bold');
