/**
 * ═══════════════════════════════════════════════════════════════════════════
 *  AccessEye — Eye Tracking Calibration Layer
 *  eye-calibration-layer.js   v1.2.0
 * ───────────────────────────────────────────────────────────────────────────
 *  ARCHITECTURE RULES (STRICT):
 *
 *  ✅ READS:   GazeEngine raw output via monkey-patch of GazeEngine._emit
 *              OR Phase2Orchestrator output — whichever is active
 *  ✅ PATCHES: The normalised {sx, sy} ∈ [0,1] coordinates BEFORE they are
 *              multiplied by window.innerWidth/Height to become pixels.
 *              This is the only correct intercept point.
 *  ✅ EMITS:   window.EyeCalibLayer public API only
 *
 *  ❌ NEVER touches: _updateGazeCursor (receives pixels — wrong space)
 *  ❌ NEVER touches: GazeEngine Kalman, EMA, iris offsets, landmark data
 *  ❌ NEVER touches: Phase2/3 engine internals
 *
 *  HOW IT WORKS:
 *  ─────────────
 *  Both call sites for _updateGazeCursor multiply by innerWidth/Height:
 *    app.js line ~2298:  _updateGazeCursor(screen.x * innerWidth,  screen.y * innerHeight)
 *    phase2-engine.js:   _updateGazeCursor(biasFixed.x * vpWidth,  biasFixed.y * vpHeight)
 *
 *  We intercept by patching app._updateGazeCursor to:
 *    1. Divide px back to [0,1] normalised
 *    2. Run the calibration pipeline (axis, center, normalise, sensitivity, EMA)
 *    3. Multiply back to pixels and call the original
 *
 *  This is safe because:
 *    - Division by current viewport size recovers the original normalised value
 *    - Phase 3's own _updateGazeCursor wrap runs AFTER ours (we store the
 *      Phase-3-wrapped function as origFn, so the chain is correct)
 *    - The dwell ring, snap engine, debug panel all still receive correct px
 *
 *  FEATURE FLAGS (all on by default, individually rollback-able):
 *    enableCalibrationLayer  — master switch (false = complete passthrough)
 *    enableAxisCorrection    — Phase 2: Y/X flip
 *    enableCenterCalib       — Phase 3: center shift
 *    enableNormalization     — Phase 3: min/max stretch
 *    enableSmoothing         — Phase 4: EMA smoothing
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

(function () {
  'use strict';

  /* ── guard double-load ─────────────────────────────────────────── */
  if (window.EyeCalibLayer) {
    console.warn('[EyeCalibLayer] Already loaded — skipping re-init.');
    return;
  }

  const ECL_VERSION = '1.2.0';

  /* ──────────────────────────────────────────────────────────────────
     CONFIGURATION
  ────────────────────────────────────────────────────────────────── */
  const DEFAULT_CONFIG = {
    /* Master feature flags */
    enableCalibrationLayer : true,
    enableAxisCorrection   : true,
    enableCenterCalib      : true,
    enableNormalization    : true,
    enableSmoothing        : true,

    /* Phase 2 — Axis correction */
    invertY : false,
    invertX : false,

    /* Phase 3 — Center calibration */
    observedCenterX : 0.50,
    observedCenterY : 0.50,

    /* Phase 3 — Symmetric normalisation */
    edgeBuffer       : 1.02,
    normWarmupFrames : 120,

    /* Phase 4 — Sensitivity scaling (1.0 = no change) */
    sensitivityX : 1.15,
    sensitivityY : 1.15,

    /* Phase 4 — EMA smoothing (0=max smooth, 1=raw) */
    smoothingAlpha : 0.20,

    /* Phase 5 — Edge clamp */
    clampMin : 0.0,
    clampMax : 1.0,
  };

  let cfg = Object.assign({}, DEFAULT_CONFIG);

  /* ──────────────────────────────────────────────────────────────────
     INTERNAL STATE
  ────────────────────────────────────────────────────────────────── */
  const state = {
    minX: 0.5, maxX: 0.5,
    minY: 0.5, maxY: 0.5,
    frameCount    : 0,
    normReady     : false,

    emaX : 0.5,
    emaY : 0.5,
    emaInitialized: false,

    centerSamples      : [],
    centerCalibActive  : false,

    patched            : false,
    _origFn            : null,

    lastRaw       : { x: 0.5, y: 0.5 },
    lastProcessed : { x: 0.5, y: 0.5 },
    log           : [],
  };

  /* ──────────────────────────────────────────────────────────────────
     HELPERS
  ────────────────────────────────────────────────────────────────── */
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  function _log(msg) {
    const entry = `[ECL ${new Date().toLocaleTimeString()}] ${msg}`;
    state.log.push(entry);
    if (state.log.length > 50) state.log.shift();
    console.log(entry);
  }

  /* ──────────────────────────────────────────────────────────────────
     CORE PIPELINE — input/output: normalised [0,1]
  ────────────────────────────────────────────────────────────────── */
  function process(sx, sy) {
    if (!cfg.enableCalibrationLayer) return { x: sx, y: sy };

    state.lastRaw = { x: sx, y: sy };
    state.frameCount++;

    let x = sx;
    let y = sy;

    /* Phase 2: Axis Correction */
    if (cfg.enableAxisCorrection) {
      if (cfg.invertY) y = 1.0 - y;
      if (cfg.invertX) x = 1.0 - x;
    }

    /* Phase 3a: Center Calibration */
    if (cfg.enableCenterCalib) {
      x += (0.5 - cfg.observedCenterX);
      y += (0.5 - cfg.observedCenterY);
    }

    /* Phase 3b: Symmetric Range Normalisation
       FIX RIGHT-EDGE-CLIP: raw gaze X range ~[0.05, 0.88] → map to [0,1]
       with 1.02 buffer so cursor reliably reaches both edges.             */
    if (cfg.enableNormalization) {
      if (x < state.minX) state.minX = x;
      if (x > state.maxX) state.maxX = x;
      if (y < state.minY) state.minY = y;
      if (y > state.maxY) state.maxY = y;

      if (state.frameCount >= cfg.normWarmupFrames) {
        state.normReady = true;
        const rangeX = state.maxX - state.minX;
        const rangeY = state.maxY - state.minY;
        if (rangeX > 0.05) {
          const buf = (cfg.edgeBuffer - 1.0) * rangeX / 2;
          x = (x - (state.minX - buf)) / (rangeX + buf * 2);
        }
        if (rangeY > 0.03) {
          const buf = (cfg.edgeBuffer - 1.0) * rangeY / 2;
          y = (y - (state.minY - buf)) / (rangeY + buf * 2);
        }
      }
    }

    /* Phase 4a: Sensitivity Scaling */
    x = 0.5 + (x - 0.5) * cfg.sensitivityX;
    y = 0.5 + (y - 0.5) * cfg.sensitivityY;

    /* Phase 4b: EMA Smoothing — bypassed when Snap-To is active */
    const snapActive = !!(window.app?.snapEngine?.enabled);
    if (cfg.enableSmoothing && !snapActive) {
      // FIX-CURSOR-Y: Skip EMA init/update when both coords are near 0
      // (camera warming up, before first real frame). Without this guard,
      // the first zero-valued call seeds emaY=0.0 which then decays the
      // cursor to the top-left corner over ~20 frames and stays there.
      const isZeroFrame = (x < 0.02 && y < 0.02);
      if (!isZeroFrame) {
        if (!state.emaInitialized) {
          state.emaX = x; state.emaY = y;
          state.emaInitialized = true;
        } else {
          const α = cfg.smoothingAlpha;
          state.emaX = α * x + (1 - α) * state.emaX;
          state.emaY = α * y + (1 - α) * state.emaY;
        }
        x = state.emaX;
        y = state.emaY;
      }
    }

    /* Phase 5: Edge Clamp */
    x = clamp(x, cfg.clampMin, cfg.clampMax);
    y = clamp(y, cfg.clampMin, cfg.clampMax);

    state.lastProcessed = { x, y };
    return { x, y };
  }

  /* ──────────────────────────────────────────────────────────────────
     CURSOR PATCH — correct coordinate space
     _updateGazeCursor(px, py) receives PIXEL values.
     We divide by viewport size → normalised → process → multiply back.
  ────────────────────────────────────────────────────────────────── */
  function _patchCursor() {
    const app = window.app;
    if (!app || typeof app._updateGazeCursor !== 'function') {
      setTimeout(_patchCursor, 300);
      return;
    }
    if (state.patched) return;

    // Store whatever is the current function (may already be Phase3-wrapped)
    state._origFn = app._updateGazeCursor.bind(app);
    state.patched = true;

    app._updateGazeCursor = function (px, py) {
      if (!cfg.enableCalibrationLayer) {
        state._origFn(px, py);
        return;
      }
      // Recover normalised coords from pixel values
      const W = window.visualViewport?.width  || window.innerWidth;
      const H = window.visualViewport?.height || window.innerHeight;
      if (!W || !H) { state._origFn(px, py); return; }

      // FIX-CURSOR-Y: If both px and py are near 0 (camera warming up or
      // no gaze detected yet), pass through unchanged — don't let the
      // calibration pipeline seed EMA with zeros and drag the cursor to
      // the top-left corner.
      if (px < 1 && py < 1) {
        state._origFn(px, py);
        return;
      }

      const nx = px / W;
      const ny = py / H;

      const result = process(nx, ny);

      // Convert back to pixels and pass to original
      state._origFn(result.x * W, result.y * H);
    };

    _log(`Cursor patch applied (pixel→normalised→process→pixel) v${ECL_VERSION}`);
  }

  function _unpatchCursor() {
    const app = window.app;
    if (!state.patched || !app || !state._origFn) return;
    app._updateGazeCursor = state._origFn;
    state._origFn  = null;
    state.patched  = false;
    _log('Cursor patch removed — full passthrough');
  }

  /* ──────────────────────────────────────────────────────────────────
     CENTER CALIBRATION PASS
  ────────────────────────────────────────────────────────────────── */
  function calibrateCenter(durationMs = 2000) {
    if (state.centerCalibActive) return;
    state.centerCalibActive = true;
    state.centerSamples = [];
    _log(`Center calibration started — look at screen center for ${durationMs / 1000}s`);
    if (window.app?.toast) {
      window.app.toast.show('Center Calibration',
        `Look at the screen center for ${durationMs / 1000}s`,
        'info', 'fas fa-crosshairs', durationMs + 200);
    }
    // Sample from our already-processed output (lastRaw before center shift)
    const _sample = () => {
      if (state.centerCalibActive) state.centerSamples.push({ x: state.lastRaw.x, y: state.lastRaw.y });
    };
    const iv = setInterval(_sample, 50);
    setTimeout(() => {
      clearInterval(iv);
      state.centerCalibActive = false;
      const n = state.centerSamples.length;
      if (n >= 10) {
        const meanX = state.centerSamples.reduce((s, p) => s + p.x, 0) / n;
        const meanY = state.centerSamples.reduce((s, p) => s + p.y, 0) / n;
        cfg.observedCenterX = meanX;
        cfg.observedCenterY = meanY;
        _log(`Center: (${meanX.toFixed(3)}, ${meanY.toFixed(3)}) from ${n} samples`);
        if (window.app?.toast) {
          window.app.toast.show('Center Calibration Complete',
            `Center locked at (${(meanX*100).toFixed(0)}%, ${(meanY*100).toFixed(0)}%)`,
            'success', 'fas fa-check-circle', 3000);
        }
      } else {
        _log('Center calibration: not enough samples.');
      }
    }, durationMs);
  }

  /* ──────────────────────────────────────────────────────────────────
     RANGE RESET
  ────────────────────────────────────────────────────────────────── */
  function resetRange() {
    state.minX = 0.5; state.maxX = 0.5;
    state.minY = 0.5; state.maxY = 0.5;
    state.frameCount = 0;
    state.normReady  = false;
    state.emaInitialized = false;
    state.emaX = 0.5; state.emaY = 0.5;
    _log('Range and EMA reset.');
  }

  /* ──────────────────────────────────────────────────────────────────
     REPORT
  ────────────────────────────────────────────────────────────────── */
  function getReport() {
    return {
      version          : ECL_VERSION,
      config           : Object.assign({}, cfg),
      normReady        : state.normReady,
      observedRangeX   : { min: state.minX, max: state.maxX },
      observedRangeY   : { min: state.minY, max: state.maxY },
      framesProcessed  : state.frameCount,
      lastRaw          : Object.assign({}, state.lastRaw),
      lastProcessed    : Object.assign({}, state.lastProcessed),
      snapBypassActive : !!(window.app?.snapEngine?.enabled),
      patched          : state.patched,
      rightEdgeClipFix : state.normReady
        ? `✅ Active — X range [${(state.minX*100).toFixed(0)}%–${(state.maxX*100).toFixed(0)}%] → [0%–100%]`
        : `⏳ Warmup ${state.frameCount}/${cfg.normWarmupFrames} frames`,
    };
  }

  /* ──────────────────────────────────────────────────────────────────
     INIT
  ────────────────────────────────────────────────────────────────── */
  function _init() {
    // Wait for Phase 3 to finish its own _updateGazeCursor wrap first,
    // so our patch sits on top of the full chain. Phase 3 init uses a
    // 400ms+ delay, so we wait 1.5s to be safe.
    const tryPatch = (attempts) => {
      if (window.app && typeof window.app._updateGazeCursor === 'function') {
        _patchCursor();
        _log(`EyeCalibLayer v${ECL_VERSION} ready. Flags: ` +
          `layer=${cfg.enableCalibrationLayer}, axis=${cfg.enableAxisCorrection}, ` +
          `center=${cfg.enableCenterCalib}, norm=${cfg.enableNormalization}, smooth=${cfg.enableSmoothing}`);
      } else if (attempts > 0) {
        setTimeout(() => tryPatch(attempts - 1), 300);
      } else {
        _log('EyeCalibLayer: window.app not ready after retries — not patched');
      }
    };
    // Give Phase 3 time to wrap first (Phase3 init waits 400ms + setup time)
    setTimeout(() => tryPatch(10), 1500);
  }

  /* ──────────────────────────────────────────────────────────────────
     PUBLIC API
  ────────────────────────────────────────────────────────────────── */
  window.EyeCalibLayer = {
    version : ECL_VERSION,

    setConfig(updates) {
      Object.assign(cfg, updates);
      _log(`Config: ${JSON.stringify(updates)}`);
    },
    getConfig : () => Object.assign({}, cfg),
    getReport,

    enable() {
      cfg.enableCalibrationLayer = true;
      if (!state.patched) _patchCursor();
      _log('EyeCalibLayer ENABLED');
    },
    disable() {
      cfg.enableCalibrationLayer = false;
      // Keep patch in place but pipeline is bypassed (fast passthrough path)
      _log('EyeCalibLayer DISABLED — passthrough');
    },
    toggle() {
      if (cfg.enableCalibrationLayer) this.disable(); else this.enable();
      return cfg.enableCalibrationLayer;
    },

    calibrateCenter,
    resetRange,
    processGaze : process,
    getLog      : () => [...state.log],

    phases: {
      axisCorrection : (on) => { cfg.enableAxisCorrection = on;  _log(`Axis correction: ${on}`); },
      centerCalib    : (on) => { cfg.enableCenterCalib    = on;  _log(`Center calib: ${on}`); },
      normalization  : (on) => { cfg.enableNormalization  = on;  _log(`Normalisation: ${on}`); },
      smoothing      : (on) => { cfg.enableSmoothing      = on;  _log(`Smoothing: ${on}`); },
    },
  };

  /* ── Kick off ───────────────────────────────────────────────────── */
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _init);
  } else {
    _init();
  }

  console.log(`[EyeCalibLayer] v${ECL_VERSION} loaded — patching after Phase 3 init...`);

})();
