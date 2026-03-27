/**
 * ═══════════════════════════════════════════════════════════════════════════
 *  AccessEye — Eye Tracking Calibration Layer
 *  eye-calibration-layer.js   v1.0.0
 * ───────────────────────────────────────────────────────────────────────────
 *  ARCHITECTURE RULES (STRICT — DO NOT VIOLATE):
 *
 *  ✅ READS:   window.AccessEye.on('gaze', …)  — Interaction Layer only
 *  ✅ EMITS:   window.AccessEye.emit('gaze:calibrated', …)  — new event
 *  ✅ WRITES:  window.EyeCalibLayer public API only
 *  ✅ MODIFIES: _updateGazeCursor via monkey-patch (restores on disable)
 *
 *  ❌ NEVER touches: GazeEngine, CalibrationEngine, Phase2/3 internals,
 *                    Kalman filter, EMA, raw iris offsets, landmark data.
 *
 *  PURPOSE:
 *  ─────────
 *  Adds a post-processing calibration layer that intercepts the normalised
 *  screen-space gaze coordinates (sx ∈ [0,1], sy ∈ [0,1]) AFTER the
 *  existing calibration model maps them, and applies:
 *
 *    Phase 2 — Axis correction       (flip Y if inverted)
 *    Phase 3 — Center calibration    (shift origin to observed screen center)
 *    Phase 3 — Symmetric normalisation (track min/max, normalize, 1.02 buffer)
 *    Phase 4 — Sensitivity scaling   (X 1.1–1.3, Y 1.1–1.3, adjustable)
 *    Phase 4 — Smoothing             (EMA α = 0.20, adjustable)
 *    Phase 5 — Edge clamping         (hard clamp [0,1])
 *    Phase 3 — Snap-mode protection  (bypass smoothing when snap-to active)
 *
 *  FEATURE FLAGS (all on by default, each independently rollback-able):
 *    enableCalibrationLayer  — master switch (false = complete passthrough)
 *    enableAxisCorrection    — Phase 2 Y-flip
 *    enableCenterCalib       — Phase 3 center shift
 *    enableNormalization     — Phase 3 min/max stretch
 *    enableSmoothing         — Phase 4 EMA smoothing
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

  const ECL_VERSION = '1.0.0';

  /* ──────────────────────────────────────────────────────────────────
     CONFIGURATION (all adjustable at runtime via EyeCalibLayer.setConfig)
  ────────────────────────────────────────────────────────────────── */
  const DEFAULT_CONFIG = {
    /* Master feature flags */
    enableCalibrationLayer : true,   // Phase 2: master on/off
    enableAxisCorrection   : true,   // Phase 2: flip Y axis if inverted
    enableCenterCalib      : true,   // Phase 3: center shift
    enableNormalization    : true,   // Phase 3: symmetric range stretch
    enableSmoothing        : true,   // Phase 4: EMA smoothing

    /* Phase 2 — Axis correction */
    invertY : false,   // set true if cursor moves UP when user looks DOWN
    invertX : false,   // set true if cursor moves RIGHT when user looks LEFT

    /* Phase 3 — Center calibration */
    // Observed screen-center coords (set during calibration walk or manually)
    observedCenterX : 0.50,
    observedCenterY : 0.50,

    /* Phase 3 — Symmetric normalisation */
    edgeBuffer      : 1.02,   // expand usable range by 2% past min/max
    normWarmupFrames: 120,    // frames to collect before normalisation kicks in

    /* Phase 4 — Sensitivity scaling */
    sensitivityX : 1.15,   // 1.1–1.3; horizontal expansion
    sensitivityY : 1.15,   // 1.1–1.3; vertical expansion

    /* Phase 4 — EMA smoothing */
    smoothingAlpha : 0.20,   // 0 = max smooth (laggy), 1 = no smooth (raw)

    /* Phase 5 — Edge clamp */
    clampMin : 0.0,
    clampMax : 1.0,
  };

  /* ──────────────────────────────────────────────────────────────────
     INTERNAL STATE
  ────────────────────────────────────────────────────────────────── */
  let cfg = Object.assign({}, DEFAULT_CONFIG);

  const state = {
    // Normalisation accumulators
    minX: 0.5, maxX: 0.5,   // initialised to center, expand on each frame
    minY: 0.5, maxY: 0.5,
    frameCount: 0,
    normReady : false,       // true once warmup frames collected

    // EMA state
    emaX: 0.5,
    emaY: 0.5,
    emaInitialized: false,

    // Center calibration — running mean of observed center during a
    // dedicated "look at center" pass (triggered by calibrateCenter())
    centerSamples: [],
    centerCalibActive: false,

    // Patch bookkeeping
    patched        : false,
    _origUpdateCursor: null,

    // Stats for report
    lastRaw       : { x: 0.5, y: 0.5 },
    lastProcessed : { x: 0.5, y: 0.5 },

    // Event log (circular, last 50 entries)
    log: [],
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
     CORE PIPELINE
     Input:  sx, sy ∈ [0,1] (after existing calibration model)
     Output: sx, sy ∈ [0,1] (post-processed)
  ────────────────────────────────────────────────────────────────── */
  function process(sx, sy) {
    if (!cfg.enableCalibrationLayer) return { x: sx, y: sy };

    state.lastRaw = { x: sx, y: sy };
    state.frameCount++;

    let x = sx;
    let y = sy;

    /* ── Phase 2: Axis Correction ──────────────────────────────────
       Diagnostic: if cursor moves DOWN when looking UP → set invertY=true
       Diagnostic: if cursor moves RIGHT when looking LEFT → set invertX=true
       Implementation: reflect around 0.5 (screen center) so range [0,1]
       is preserved.
    ────────────────────────────────────────────────────────────────── */
    if (cfg.enableAxisCorrection) {
      if (cfg.invertY) y = 1.0 - y;
      if (cfg.invertX) x = 1.0 - x;
    }

    /* ── Phase 3a: Center Calibration ─────────────────────────────
       Shift so that the user's natural forward-gaze center maps to
       screen (0.5, 0.5).  We measure the offset once (or per-session)
       by asking the user to look at the screen center for 2 seconds.
    ────────────────────────────────────────────────────────────────── */
    if (cfg.enableCenterCalib) {
      const dx = 0.5 - cfg.observedCenterX;
      const dy = 0.5 - cfg.observedCenterY;
      x += dx;
      y += dy;
    }

    /* ── Phase 3b: Symmetric Range Normalisation ───────────────────
       Track observed [min, max] across all frames. After warmup, linearly
       map the observed range → [0, 1], then apply 1.02 edge-buffer so the
       user can reach screen corners without physically over-rotating eyes.

       FIX RIGHT-EDGE-CLIP: The right edge was getting clipped because the
       raw gaze range for X doesn't reach 1.0 (approximately 0.02–0.15 gap
       on the right side depending on user). Normalisation removes this
       asymmetry by re-anchoring to the OBSERVED extremes, making left and
       right equally reachable.
    ────────────────────────────────────────────────────────────────── */
    if (cfg.enableNormalization) {
      // Expand tracked range
      if (x < state.minX) state.minX = x;
      if (x > state.maxX) state.maxX = x;
      if (y < state.minY) state.minY = y;
      if (y > state.maxY) state.maxY = y;

      if (state.frameCount >= cfg.normWarmupFrames) {
        state.normReady = true;

        const rangeX = state.maxX - state.minX;
        const rangeY = state.maxY - state.minY;

        if (rangeX > 0.05) {   // only normalize if meaningful range observed
          const buf = (cfg.edgeBuffer - 1.0) * rangeX / 2;
          x = (x - (state.minX - buf)) / (rangeX + buf * 2);
        }
        if (rangeY > 0.03) {
          const buf = (cfg.edgeBuffer - 1.0) * rangeY / 2;
          y = (y - (state.minY - buf)) / (rangeY + buf * 2);
        }
      }
    }

    /* ── Phase 4a: Sensitivity Scaling ────────────────────────────
       Expand from center outward.  sensitivityX=1.15 means a gaze 0.3
       away from center becomes 0.345 away → cursor reaches 93% of screen
       with the same physical eye movement that reached only 80% before.
    ────────────────────────────────────────────────────────────────── */
    x = 0.5 + (x - 0.5) * cfg.sensitivityX;
    y = 0.5 + (y - 0.5) * cfg.sensitivityY;

    /* ── Phase 4b: EMA Smoothing ───────────────────────────────────
       EMA α=0.20: output follows input with ~5-frame lag, removing
       high-frequency jitter while preserving intentional motion.

       Snap-mode protection: when Snap-To mode is active, we BYPASS
       smoothing entirely and use raw calibrated coordinates. This ensures
       snap-lock decisions are made on the freshest data, preventing
       the smoothing delay from causing the cursor to "slide past" a target.
    ────────────────────────────────────────────────────────────────── */
    const snapActive = !!(window.app?.snapEngine?.enabled);

    if (cfg.enableSmoothing && !snapActive) {
      if (!state.emaInitialized) {
        state.emaX = x;
        state.emaY = y;
        state.emaInitialized = true;
      } else {
        const α = cfg.smoothingAlpha;
        state.emaX = α * x + (1 - α) * state.emaX;
        state.emaY = α * y + (1 - α) * state.emaY;
      }
      x = state.emaX;
      y = state.emaY;
    }

    /* ── Phase 5: Edge Clamping ────────────────────────────────────
       Hard clamp after all transformations. Ensures cursor never
       escapes [0,1] regardless of upstream values.
    ────────────────────────────────────────────────────────────────── */
    x = clamp(x, cfg.clampMin, cfg.clampMax);
    y = clamp(y, cfg.clampMin, cfg.clampMax);

    state.lastProcessed = { x, y };
    return { x, y };
  }

  /* ──────────────────────────────────────────────────────────────────
     CENTER CALIBRATION PASS
     Call EyeCalibLayer.calibrateCenter() → asks user to look at center
     for 2s, then updates observedCenterX/Y automatically.
  ────────────────────────────────────────────────────────────────── */
  function calibrateCenter(durationMs = 2000) {
    if (state.centerCalibActive) return;
    state.centerCalibActive = true;
    state.centerSamples = [];

    _log(`Center calibration started — look at screen center for ${durationMs / 1000}s`);

    // Show non-blocking UI hint if toast is available
    if (window.app?.toast) {
      window.app.toast.show(
        'Center Calibration',
        `Look at the screen center for ${durationMs / 1000}s`,
        'info', 'fas fa-crosshairs', durationMs + 200
      );
    }

    // Temporarily collect raw (pre-processed) gaze samples
    const handler = ({ screen }) => {
      if (screen) state.centerSamples.push({ x: screen.x, y: screen.y });
    };

    if (window.AccessEye?.on) {
      window.AccessEye.on('gaze', handler);
    }

    setTimeout(() => {
      state.centerCalibActive = false;
      if (window.AccessEye?.off) window.AccessEye.off('gaze', handler);

      const n = state.centerSamples.length;
      if (n >= 10) {
        const meanX = state.centerSamples.reduce((s, p) => s + p.x, 0) / n;
        const meanY = state.centerSamples.reduce((s, p) => s + p.y, 0) / n;
        cfg.observedCenterX = meanX;
        cfg.observedCenterY = meanY;
        _log(`Center calibrated: observedCenter=(${meanX.toFixed(3)}, ${meanY.toFixed(3)}) from ${n} samples`);
        if (window.app?.toast) {
          window.app.toast.show(
            'Center Calibration Complete',
            `Center locked at (${(meanX * 100).toFixed(0)}%, ${(meanY * 100).toFixed(0)}%)`,
            'success', 'fas fa-check-circle', 3000
          );
        }
      } else {
        _log('Center calibration failed — not enough samples. Check camera.');
      }
    }, durationMs);
  }

  /* ──────────────────────────────────────────────────────────────────
     RANGE RESET
     Call when user switches posture / lighting to re-learn min/max.
  ────────────────────────────────────────────────────────────────── */
  function resetRange() {
    state.minX = 0.5; state.maxX = 0.5;
    state.minY = 0.5; state.maxY = 0.5;
    state.frameCount = 0;
    state.normReady  = false;
    state.emaInitialized = false;
    state.emaX = 0.5;
    state.emaY = 0.5;
    _log('Range and EMA state reset — re-learning from scratch.');
  }

  /* ──────────────────────────────────────────────────────────────────
     CURSOR PATCH
     Intercepts _updateGazeCursor in the main app singleton.
     Only patches once; restores cleanly on disable.
  ────────────────────────────────────────────────────────────────── */
  function _patchCursor() {
    const app = window.app;
    if (!app || typeof app._updateGazeCursor !== 'function') {
      _log('_patchCursor: window.app._updateGazeCursor not found — retry in 500ms');
      setTimeout(_patchCursor, 500);
      return;
    }
    if (state.patched) return;

    state._origUpdateCursor = app._updateGazeCursor.bind(app);
    state.patched = true;

    app._updateGazeCursor = function (sx, sy) {
      // Run calibration layer
      const result = process(sx, sy);
      // Call original with calibrated coords
      state._origUpdateCursor(result.x, result.y);
    };

    _log(`Cursor patch applied — EyeCalibLayer v${ECL_VERSION} active.`);
  }

  function _unpatchCursor() {
    const app = window.app;
    if (!state.patched || !app || !state._origUpdateCursor) return;
    app._updateGazeCursor = state._origUpdateCursor;
    state._origUpdateCursor = null;
    state.patched = false;
    _log('Cursor patch removed — passthrough mode.');
  }

  /* ──────────────────────────────────────────────────────────────────
     INIT — wait until window.app and AccessEye are ready
  ────────────────────────────────────────────────────────────────── */
  function _init() {
    if (window.app && typeof window.app._updateGazeCursor === 'function') {
      _patchCursor();
      _log(`EyeCalibLayer v${ECL_VERSION} initialised. Feature flags: ` +
        `layer=${cfg.enableCalibrationLayer}, axisCorr=${cfg.enableAxisCorrection}, ` +
        `center=${cfg.enableCenterCalib}, norm=${cfg.enableNormalization}, smooth=${cfg.enableSmoothing}`);
    } else {
      setTimeout(_init, 300);
    }
  }

  /* ──────────────────────────────────────────────────────────────────
     STATUS / REPORT
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
      deltaX           : (state.lastProcessed.x - state.lastRaw.x).toFixed(4),
      deltaY           : (state.lastProcessed.y - state.lastRaw.y).toFixed(4),
      snapBypassActive : !!(window.app?.snapEngine?.enabled),
      patched          : state.patched,
      rightEdgeClipFix : state.normReady
        ? `✅ Normalisation active — effective X range: [${(state.minX * 100).toFixed(0)}%–${(state.maxX * 100).toFixed(0)}%] mapped to [0%–100%]`
        : `⏳ Collecting warmup frames (${state.frameCount}/${cfg.normWarmupFrames})`,
    };
  }

  /* ──────────────────────────────────────────────────────────────────
     PUBLIC API
  ────────────────────────────────────────────────────────────────── */
  window.EyeCalibLayer = {
    version : ECL_VERSION,

    /** Runtime config update — any subset of DEFAULT_CONFIG keys */
    setConfig(updates) {
      Object.assign(cfg, updates);
      _log(`Config updated: ${JSON.stringify(updates)}`);
    },

    getConfig : () => Object.assign({}, cfg),
    getReport,

    /** Enable the full layer (re-patches cursor if needed) */
    enable() {
      cfg.enableCalibrationLayer = true;
      if (!state.patched) _patchCursor();
      _log('EyeCalibLayer ENABLED');
    },

    /** Disable entire layer — pure passthrough, cursor unpatch */
    disable() {
      cfg.enableCalibrationLayer = false;
      _unpatchCursor();
      _log('EyeCalibLayer DISABLED — full passthrough');
    },

    /** Toggle master switch */
    toggle() {
      if (cfg.enableCalibrationLayer) this.disable();
      else this.enable();
      return cfg.enableCalibrationLayer;
    },

    /** Start 2-second center calibration pass */
    calibrateCenter,

    /** Reset observed range so normalisation re-learns */
    resetRange,

    /** Expose processed gaze for external consumers */
    processGaze: process,

    /** Log access for debugging */
    getLog : () => [...state.log],

    /** Quick phase toggles for instant rollback */
    phases: {
      axisCorrection : (on) => { cfg.enableAxisCorrection = on;  _log(`Axis correction: ${on}`); },
      centerCalib    : (on) => { cfg.enableCenterCalib    = on;  _log(`Center calibration: ${on}`); },
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

  console.log(`[EyeCalibLayer] v${ECL_VERSION} loaded — waiting for window.app...`);

})();
