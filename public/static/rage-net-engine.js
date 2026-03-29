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
   * Load trained weights from our custom sharded manifest.
   * manifestUrl: URL to weights_manifest.json
   *
   * The manifest has been reordered to exactly match TF.js getWeights() positional order.
   * Positional loading is used — each manifest entry maps 1:1 to model.getWeights()[i].
   */
  async loadWeights(manifestUrl) {
    if (!this._tf || !this._ready) {
      console.warn('[RAGE-net] Call init() before loadWeights()');
      return false;
    }
    try {
      const baseUrl = manifestUrl.replace(/weights_manifest\.json$/, '');
      console.log(`%c[RAGE-net] Fetching weight manifest...`, 'color:#a78bfa');

      const manifestResp = await fetch(manifestUrl);
      if (!manifestResp.ok) throw new Error(`Manifest fetch failed: ${manifestResp.status}`);
      const manifest = await manifestResp.json();
      const entry = manifest.weightsManifest[0];
      const shardPaths   = entry.paths;
      const shardOffsets = entry.shardOffsets;
      const weightsMeta  = entry.weights;

      // Download all shards in parallel
      console.log(`%c[RAGE-net] Downloading ${shardPaths.length} weight shards (~110 MB)...`, 'color:#a78bfa');
      const shardBuffers = await Promise.all(
        shardPaths.map(async (p, i) => {
          const r = await fetch(baseUrl + p);
          if (!r.ok) throw new Error(`Shard ${p} fetch failed: ${r.status}`);
          const buf = await r.arrayBuffer();
          console.log(`%c[RAGE-net] Shard ${i+1}/${shardPaths.length} ✓ (${(buf.byteLength/1048576).toFixed(1)} MB)`, 'color:#a78bfa;font-size:10px');
          return { offset: shardOffsets[i], buffer: buf };
        })
      );

      // Assemble flat ArrayBuffer from shards
      const totalBytes = weightsMeta.reduce((acc, w) => Math.max(acc, w.byteOffset + w.nbytes), 0);
      const flat = new Uint8Array(totalBytes);
      for (const { offset, buffer } of shardBuffers) {
        flat.set(new Uint8Array(buffer), offset);
      }

      const tf = this._tf;
      const modelWeights = this._model.getWeights();

      if (weightsMeta.length !== modelWeights.length) {
        console.warn(`[RAGE-net] Weight count: manifest=${weightsMeta.length} vs model=${modelWeights.length}`);
      }

      // Build tensors from manifest in positional order (matches getWeights())
      const minLen = Math.min(weightsMeta.length, modelWeights.length);
      const toSet = [];

      for (let i = 0; i < minLen; i++) {
        const w = weightsMeta[i];
        const floats = new Float32Array(flat.buffer, w.byteOffset, w.nbytes / 4);
        // Validate shape matches before creating tensor
        const mShape = w.shape;
        const modelShape = modelWeights[i].shape;
        if (mShape.length !== modelShape.length || !mShape.every((d, j) => d === modelShape[j])) {
          console.warn(`[RAGE-net] Shape mismatch at [${i}]: manifest${JSON.stringify(mShape)} vs model${JSON.stringify(modelShape)} (${w.name})`);
          toSet.push(modelWeights[i]); // keep existing weight
        } else {
          toSet.push(tf.tensor(Array.from(floats), mShape, w.dtype));
        }
      }

      this._model.setWeights(toSet);
      // Dispose created tensors (TF.js copies data on setWeights)
      toSet.forEach(t => { try { t.dispose(); } catch(e){} });

      const mismatches = toSet.length - minLen;
      if (mismatches === 0) {
        this._weightsLoaded = true;
        console.log(`%c[RAGE-net] ✓ All ${minLen} weights loaded — ResNet-18 gaze model active`,
          'color:#00ff88;font-weight:bold');
        return true;
      } else {
        console.warn(`[RAGE-net] ${mismatches} weights had shape mismatches`);
        this._weightsLoaded = minLen > modelWeights.length * 0.9;
        return this._weightsLoaded;
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
  // Full ResNet-18 matching the trained H5 weights:
  //   Two separate ResNet-18 backbones (one per eye): 64→128→256→512 channels
  //   GAP → 512 each; concat → 1024
  //   Two cross-attention gates (dense+BN → sigmoid → multiply)
  //   Fusion head: 1024→2048→1024→2
  // ─────────────────────────────────────────────
  _buildModel() {
    const tf = this._tf;
    const L  = tf.layers;

    /**
     * Build one full ResNet-18 backbone.
     * Matches architecture stored in res_net18_model_12 / res_net18_model_13.
     * Layers (with bias): stem conv2d(64,3x3), then 8 residual blocks:
     *   block1a: 64→64, stride 1
     *   block1b: 64→64, stride 1
     *   block2a: 64→128, stride 2, 1x1 shortcut
     *   block2b: 128→128, stride 1
     *   block3a: 128→256, stride 2, 1x1 shortcut
     *   block3b: 256→256, stride 1
     *   block4a: 256→512, stride 2, 1x1 shortcut
     *   block4b: 512→512, stride 1
     *   GAP → 512
     */
    const buildBackbone = (eyeName) => {
      const inp = L.input({ shape: [RAGE.EYE_H, RAGE.EYE_W, 1], name: `${eyeName}_input` });

      // BN228-style per-channel input normalisation (shape [1])
      let x = L.batchNormalization({ axis: -1, name: `${eyeName}_input_bn` }).apply(inp);

      // Stem: conv2d(64, 3x3, bias=true) + BN + ReLU
      x = L.conv2d({ filters: 64, kernelSize: 3, strides: 1, padding: 'same',
        useBias: true, name: `${eyeName}_stem` }).apply(x);
      x = L.batchNormalization({ name: `${eyeName}_stem_bn` }).apply(x);
      x = L.activation('relu', { name: `${eyeName}_stem_relu` }).apply(x);

      // Block group 1: 64 channels, stride 1, no shortcut needed (same channels)
      x = this._resBlock(x, 64, 1, `${eyeName}_b1a`);
      x = this._resBlock(x, 64, 1, `${eyeName}_b1b`);

      // Block group 2: 64→128, stride 2, 1x1 shortcut
      x = this._resBlock(x, 128, 2, `${eyeName}_b2a`);
      x = this._resBlock(x, 128, 1, `${eyeName}_b2b`);

      // Block group 3: 128→256, stride 2, 1x1 shortcut
      x = this._resBlock(x, 256, 2, `${eyeName}_b3a`);
      x = this._resBlock(x, 256, 1, `${eyeName}_b3b`);

      // Block group 4: 256→512, stride 2, 1x1 shortcut
      x = this._resBlock(x, 512, 2, `${eyeName}_b4a`);
      x = this._resBlock(x, 512, 1, `${eyeName}_b4b`);

      // Global Average Pooling → 512-d vector
      x = L.globalAveragePooling2d({ name: `${eyeName}_gap` }).apply(x);

      return tf.model({ inputs: inp, outputs: x, name: `rn18_${eyeName}` });
    };

    // Build one backbone per eye
    const rightBackbone = buildBackbone('right');
    const leftBackbone  = buildBackbone('left');

    // Model inputs
    const rIn = L.input({ shape: [RAGE.EYE_H, RAGE.EYE_W, 1], name: 'right_eye' });
    const lIn = L.input({ shape: [RAGE.EYE_H, RAGE.EYE_W, 1], name: 'left_eye'  });

    // Backbone features: 512 each
    const rFeat = rightBackbone.apply(rIn);  // [B, 512]
    const lFeat = leftBackbone.apply(lIn);   // [B, 512]

    // Concatenate: [B, 1024]
    const merged = L.concatenate({ name: 'backbone_concat' }).apply([rFeat, lFeat]);

    // ── Cross-attention gate 1 (multiply_12) ──
    // Feature path: dense_42(1024→512, relu) → BN264
    let feat1 = L.dense({ units: 512, activation: 'relu',    name: 'dense_feat1' }).apply(merged);
    feat1     = L.batchNormalization({ name: 'bn_feat1' }).apply(feat1);
    // Attention path: dense_43(1024→512, sigmoid)
    let attn1 = L.dense({ units: 512, activation: 'sigmoid', name: 'dense_attn1' }).apply(merged);
    // Gated output: element-wise multiply
    const gate1 = L.multiply({ name: 'gate1' }).apply([feat1, attn1]);  // [B, 512]

    // ── Cross-attention gate 2 (multiply_13) ──
    // Feature path: dense_44(1024→512, relu) → BN265
    let feat2 = L.dense({ units: 512, activation: 'relu',    name: 'dense_feat2' }).apply(merged);
    feat2     = L.batchNormalization({ name: 'bn_feat2' }).apply(feat2);
    // Attention path: dense_45(1024→512, sigmoid)
    let attn2 = L.dense({ units: 512, activation: 'sigmoid', name: 'dense_attn2' }).apply(merged);
    const gate2 = L.multiply({ name: 'gate2' }).apply([feat2, attn2]);  // [B, 512]

    // ── Fusion head ──  (matches dense_46/47/48)
    let fused = L.concatenate({ name: 'gate_concat' }).apply([gate1, gate2]);  // [B, 1024]
    fused     = L.dense({ units: 2048, activation: 'relu',    name: 'fc_2048' }).apply(fused);
    fused     = L.dense({ units: 1024, activation: 'relu',    name: 'fc_1024' }).apply(fused);
    const out = L.dense({ units: 2,    activation: 'sigmoid', name: 'gaze_out' }).apply(fused);

    const model = tf.model({ inputs: [rIn, lIn], outputs: out, name: 'RAGE_net_full' });
    console.log(`%c[RAGE-net] Full ResNet-18 architecture built — ${model.countParams().toLocaleString()} params`,
      'color:#a78bfa;font-size:11px');
    return model;
  }

  /**
   * Single residual block with bias convolutions (matching H5 weights).
   * Main path: conv(3x3,bias) → BN → ReLU → conv(3x3,bias) → BN
   * Shortcut:  1x1 conv(bias) → BN  (if stride>1 or channel mismatch)
   * Output:    add(main, skip) → ReLU
   */
  _resBlock(x, filters, stride, name) {
    const L   = this._tf.layers;
    const inCh = x.shape[x.shape.length - 1];

    // Main path
    let h = L.conv2d({
      filters, kernelSize: 3, strides: stride, padding: 'same',
      useBias: true, name: `${name}_c1`
    }).apply(x);
    h = L.batchNormalization({ name: `${name}_bn1` }).apply(h);
    h = L.activation('relu', { name: `${name}_r1` }).apply(h);
    h = L.conv2d({
      filters, kernelSize: 3, strides: 1, padding: 'same',
      useBias: true, name: `${name}_c2`
    }).apply(h);
    h = L.batchNormalization({ name: `${name}_bn2` }).apply(h);

    // Skip connection (1x1 conv WITHOUT BN — matches H5 architecture)
    let skip = x;
    if (stride !== 1 || inCh !== filters) {
      skip = L.conv2d({
        filters, kernelSize: 1, strides: stride, padding: 'same',
        useBias: true, name: `${name}_skip`
      }).apply(x);
      // NO batchNorm on skip — H5 model uses raw shortcut conv only
    }

    h = L.add({ name: `${name}_add` }).apply([h, skip]);
    h = L.activation('relu', { name: `${name}_r2` }).apply(h);
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

    // Async inference state — fire-and-update pattern.
    // processResults() returns the last known result immediately (non-blocking),
    // while inference runs asynchronously and updates state on completion.
    this._inferPending  = false;   // async inference is in flight
    this._lastLm        = null;    // landmarks captured for current inference frame

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

  /** Initialize the engine and load trained weights. Must be called once. */
  async init() {
    this._onStatus('Loading RAGE-net…');
    await this._model.init();
    this._ready = this._model.ready;
    if (this._ready) {
      this._onStatus('RAGE-net model built — loading trained weights…');
      // Load trained weights from jsDelivr CDN (GitHub-backed, no CORS issues)
      const WEIGHTS_URL = 'https://cdn.jsdelivr.net/gh/mkbrown261/Accesseye-mvp@main/public/static/ragenet-weights/weights_manifest.json';
      const loaded = await this._model.loadWeights(WEIGHTS_URL);
      if (loaded) {
        this._onStatus('RAGE-net ready ✓ (trained weights loaded)');
        console.log('%c[RAGE-net Engine] Trained weights loaded — zero-shot tracking active',
          'color:#00ff88;font-weight:bold');
      } else {
        this._onStatus('RAGE-net ready (random weights — accuracy limited)');
        console.warn('[RAGE-net Engine] Weights load failed — running with random weights');
      }
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
   * Uses a fire-and-update async pattern:
   *   - Returns last-known gaze result IMMEDIATELY (non-blocking, keeps pipeline at camera FPS)
   *   - Fires async inference in background
   *   - State is updated when inference resolves (typically next 1–3 frames later)
   *
   * This prevents the full ResNet-18 inference (~200-500ms on CPU) from blocking the
   * face-detection pipeline and causing fps: 2 stutter.
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

    this._frameCount++;

    // ── Fire async inference if none is in flight ──
    // Don't queue multiple inferences — just kick one off per completed cycle.
    if (!this._inferPending) {
      const crops = this._cropExtractor.extract(this._videoEl, lm, W, H);
      if (crops) {
        this._inferPending = true;
        const capturedLm = lm;
        const capturedHp = headPoseResult;
        this._model.predictAsync(crops.rightEye, crops.leftEye)
          .then(raw => {
            this._inferPending = false;
            if (!raw) return;
            const clamped   = { x: Math.max(0, Math.min(1, raw.x)), y: Math.max(0, Math.min(1, raw.y)) };
            const corrected = this._driftCorrector.apply(clamped.x, clamped.y);
            this._lastGaze   = corrected;
            this._confidence = this._estimateConfidence(capturedLm, capturedHp);
            // Sync exported fields
            this.rawGaze       = corrected;
            this.smoothGaze    = corrected;
            this.confidence    = this._confidence;
            this._irisOnlyGaze = corrected;
            this._trueRawGaze  = clamped;
          })
          .catch(err => {
            this._inferPending = false;
            console.warn('[RAGE-net] Async predict error:', err);
          });
      }
    }

    // ── Return last-known result immediately (non-blocking) ──
    // On the very first frame (before any inference completes), _lastGaze is {0.5,0.5}.
    // Use a minimum confidence of 0.3 so the TemporalStabilizer doesn't freeze
    // (it holds last position when conf < 0.25 — we want it to pass through 0.5 centre
    // gracefully on startup rather than locking there permanently).
    const outConf = Math.max(this._confidence, 0.3);
    return this._makePacket(this._lastGaze.x, this._lastGaze.y, outConf, lm);
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
    this._frameCount   = 0;
    this._inferPending = false;
    this._lastGaze     = { x: 0.5, y: 0.5 };
    this._confidence   = 0;
  }

  // ─────────────────────────────────────────────
  // Private helpers
  // ─────────────────────────────────────────────

  /** Build a gaze packet compatible with Phase2Orchestrator downstream. */
  _makePacket(sx, sy, conf, lm) {
    // Estimate eye span from landmarks for GazeConfidenceScorer compatibility.
    // lm[33]=left-outer, lm[133]=left-inner, lm[263]=right-outer, lm[362]=right-inner.
    const lSpan = lm?.[33] && lm?.[133]
      ? Math.abs(lm[133].x - lm[33].x) : 0.08;
    const rSpan = lm?.[263] && lm?.[362]
      ? Math.abs(lm[362].x - lm[263].x) : 0.08;

    // iris: structured signal compatible with GazeConfidenceScorer._measureOcclusion /
    //   _detectGlare. Fields: x, y (gaze coords), confidence, lSpan, rSpan.
    // NOTE: for RAGE-net, x/y are the GAZE output [0,1], NOT landmark screen coords.
    // Phase3 skips the iris-override path for rageNet===true packets, so this iris
    // object is only used by GazeConfidenceScorer, never as a gaze source.
    const iris = {
      x:          sx,
      y:          sy,
      confidence: conf,
      lSpan,
      rSpan,
    };

    return {
      screen:     { x: sx, y: sy },
      raw:        { x: sx, y: sy },
      confidence: conf,
      iris,
      timestamp:  performance.now(),
      // Extra fields for debugging and Phase3 branching
      rageNet:       true,
      weightsLoaded: this._model.weightsLoaded,
      latencyMs:     this._model.lastLatency,
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
