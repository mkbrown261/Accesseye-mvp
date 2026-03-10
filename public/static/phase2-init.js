/**
 * ═══════════════════════════════════════════════════════════════════════════
 *  AccessEye — Phase 2 Initialization & Integration Layer
 *  phase2-init.js
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  Responsibilities:
 *   1. Wait for Phase 1 app (AccessEyeApp) to bootstrap
 *   2. Instantiate Phase2Orchestrator and attach to app
 *   3. Wire Phase 2 UI controls (intent toggle, benchmark start/stop,
 *      micro-calibration reset, pipeline label)
 *   4. Upgrade camera start to activate Phase 2 pipeline automatically
 *   5. Expose window.AccessEye.phase2 public API
 *
 *  Load order (index.tsx script tags):
 *    1. app.js          — Phase 1 core engine + AccessEyeApp
 *    2. phase2-engine.js — Phase 2 modules + Phase2Orchestrator
 *    3. phase2-init.js   — THIS FILE — wires everything together
 * ═══════════════════════════════════════════════════════════════════════════
 */

'use strict';

/* ─────────────────────────────────────────────────────────────────────────
   PHASE 2 INIT CONTROLLER
   Runs after DOMContentLoaded (app.js bootstraps first via its own listener)
───────────────────────────────────────────────────────────────────────── */
class Phase2InitController {
  constructor() {
    this.orchestrator = null;
    this._ready       = false;
    this._activated   = false;
  }

  /**
   * Entry point — wait for Phase 1 app, then attach Phase 2.
   * Uses a short polling loop to ensure `window.app` and `window.Phase2`
   * are both available before wiring.
   */
  init() {
    const check = () => {
      // window.app is set in app.js bootstrap; window.Phase2 from phase2-engine.js
      if (window.app && window.Phase2 && window.Phase2.Phase2Orchestrator) {
        this._attach(window.app);
      } else {
        setTimeout(check, 80);
      }
    };
    // Start polling after a short delay to let DOMContentLoaded handlers run
    setTimeout(check, 200);
  }

  /* ── Attach to Phase 1 app ── */
  _attach(app) {
    if (this._ready) return;
    this._ready = true;

    try {
      this.orchestrator = new window.Phase2.Phase2Orchestrator(app);

      // Attach to app so _startCamera can find it
      app.phase2 = this.orchestrator;

      // ── Patch _startCamera to auto-activate Phase 2 ──
      this._patchCameraStart(app);

      // ── Wire Phase 2 UI controls ──
      this._wireUI(app);

      // ── Expose public API ──
      this._exposeAPI(app);

      console.log('%c Phase 2 Init ✅ — Orchestrator attached to AccessEyeApp',
                  'color:#00ff88;font-weight:bold;font-size:12px;');
    } catch (e) {
      console.error('[Phase2Init] Attach failed:', e);
    }
  }

  /* ── Patch camera start to trigger Phase 2 activation ── */
  _patchCameraStart(app) {
    const orch = this.orchestrator;
    const origStart = app._startCamera.bind(app);

    app._startCamera = async function() {
      // ── FIX H-1: Acquire the highest supported FPS BEFORE Phase 1 opens
      //    the camera.  Phase 1 _startCamera() calls getUserMedia with a
      //    hard-coded 640x480@30 constraint, overwriting any previous stream.
      //    By acquiring the high-FPS stream first and injecting it into the
      //    video element, Phase 1's getUserMedia call is replaced entirely.
      //
      //    Strategy:
      //      1. Try HighFPSCameraController (120→90→60→30 FPS, 1280×720 ideal)
      //      2. On success: inject stream into #demo-video so Phase 1's
      //         MediaPipeController finds it already playing.
      //      3. Monkey-patch navigator.mediaDevices.getUserMedia temporarily
      //         so Phase 1's call returns the same stream (avoids double acquire).
      //      4. On failure: fall through to Phase 1 original path.
      const videoEl = document.querySelector('#demo-video');
      let highFPSAcquired = false;
      let _origGetUserMedia = null;

      if (videoEl && window.Phase2?.HighFPSCameraController) {
        try {
          const hfps = new window.Phase2.HighFPSCameraController();
          const caps = await hfps.acquire(videoEl);

          // Store reference so Phase 2 orchestrator can read capabilities
          orch._highFPSController = hfps;
          orch.cameraFPS = caps.fps || caps.requested || 30;

          // Temporarily stub getUserMedia so Phase 1 reuses our stream
          const existingStream = hfps.stream;
          _origGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
          navigator.mediaDevices.getUserMedia = async () => existingStream;

          highFPSAcquired = true;
          console.log(`[Phase2Init] HighFPS camera: ${caps.fps || caps.requested} FPS @ ${caps.width}×${caps.height}`);
          app.log?.add(`Camera: ${caps.fps || caps.requested} FPS @ ${caps.width}×${caps.height} (HighFPS)`, 'success');
        } catch (err) {
          console.warn('[Phase2Init] HighFPS acquire failed, falling back:', err.message);
        }
      }

      // ── Run Phase 1 camera start (will reuse our stream if stub is active) ──
      await origStart();

      // Restore getUserMedia if we stubbed it
      if (_origGetUserMedia) {
        navigator.mediaDevices.getUserMedia = _origGetUserMedia;
      }

      // Auto-activate Phase 2 after camera + MediaPipe are ready
      const canvasEl = document.querySelector('#overlay-canvas');

      // FIX CAM-3: Always activate Phase 2 on camera start (not just first time).
      // The old `!orch.active` guard prevented re-activation after camera restart.
      // deactivate() now resets all state so re-activation is always safe.
      if (videoEl) {
        // Small delay to ensure MediaPipe controller is wired
        setTimeout(async () => {
          try {
            await orch.activate(videoEl, canvasEl);
            // Re-register gaze targets after mode switch
            app._registerGazeTargets();
            // Update system status bar
            const fpsBadge = orch.cameraFPS > 30 ? ` @ ${orch.cameraFPS}FPS` : '';
            app._updateSystemStatus('tracking', `Phase 2 Gaze${fpsBadge}`);
          } catch (err) {
            console.warn('[Phase2Init] Activation error:', err.message);
          }
        }, 600);
      }
    };
  }

  /* ── Wire all Phase 2 UI interactions ── */
  _wireUI(app) {
    const orch = this.orchestrator;

    /* ─ Intent Engine toggle ─ */
    const intentToggle = document.querySelector('#p2-intent-toggle');
    if (intentToggle) {
      intentToggle.addEventListener('click', () => {
        const newState = !orch.intent.enabled;
        orch.intent.setEnabled(newState);
        intentToggle.innerHTML = newState
          ? '<i class="fas fa-power-off"></i> Disable AI Intent'
          : '<i class="fas fa-power-off"></i> Enable AI Intent';
        intentToggle.classList.toggle('active', newState);

        const result = document.querySelector('#p2-intent-result');
        if (result) result.textContent = newState ? 'Listening for fixations...' : 'AI Intent disabled';

        app.toast.show(
          newState ? 'AI Intent Enabled' : 'AI Intent Disabled',
          newState ? 'Predicting user intent from gaze patterns.' : 'AI prediction paused.',
          newState ? 'success' : 'info',
          'fas fa-robot',
          2500
        );
        app.log.add(`AI Intent Engine: ${newState ? 'ENABLED' : 'DISABLED'}`, 'info');
      });
    }

    /* ─ Benchmark: Start ─ */
    const benchStart = document.querySelector('#p2-bench-start');
    const benchStop  = document.querySelector('#p2-bench-stop');
    const benchReport = document.querySelector('#benchmark-report');

    if (benchStart) {
      benchStart.addEventListener('click', () => {
        if (!orch.active) {
          app.toast.show('Camera Required', 'Start camera + Phase 2 to run benchmark.', 'warn');
          return;
        }
        orch.benchmark.start();
        benchStart.disabled = true;
        if (benchStop) benchStop.disabled = false;
        if (benchReport) benchReport.style.display = 'none';

        // Auto-stop after 30s
        const autoStop = setTimeout(() => {
          if (orch.benchmark.running) {
            orch.benchmark.stop();
            benchStart.disabled = false;
            if (benchStop) benchStop.disabled = true;
          }
        }, 31000);

        // Store ref for manual stop
        benchStart._autoStop = autoStop;
        app.log.add('Benchmark started — 30s window', 'info');
        app.toast.show('Benchmark Running', 'Comparing Phase 1 vs Phase 2 for 30 seconds...', 'info', 'fas fa-chart-bar', 3000);
      });
    }

    if (benchStop) {
      benchStop.addEventListener('click', () => {
        if (benchStart?._autoStop) clearTimeout(benchStart._autoStop);
        orch.benchmark.stop();
        benchStop.disabled = true;
        if (benchStart) benchStart.disabled = false;
        app.log.add('Benchmark stopped manually', 'warn');
      });
    }

    /* ─ Micro-calibration reset ─ */
    const microReset = document.querySelector('#p2-reset-micro');
    if (microReset) {
      microReset.addEventListener('click', () => {
        orch.dynCalib.resetMicro();
        const countEl = document.querySelector('#p2-micro-count');
        const biasEl  = document.querySelector('#p2-bias-val');
        if (countEl) countEl.textContent = '0';
        if (biasEl)  biasEl.textContent  = '0 / 0';
        app.log.add('Dynamic micro-calibration reset', 'warn');
        app.toast.show('Micro-Calib Reset', 'Interaction-based drift corrections cleared.', 'warn', 'fas fa-undo', 2500);
      });
    }

    /* ─ Update micro-calib UI on events ─ */
    orch.dynCalib.on('microCalib', (d) => {
      const countEl = document.querySelector('#p2-micro-count');
      const biasEl  = document.querySelector('#p2-bias-val');
      if (countEl) countEl.textContent = d.sampleCount;
      if (biasEl)  biasEl.textContent  = `${d.biasX.toFixed(3)} / ${d.biasY.toFixed(3)}`;
    });

    /* ─ Benchmark complete: restore buttons ─ */
    orch.benchmark.on('complete', () => {
      if (benchStart) benchStart.disabled = false;
      if (benchStop)  benchStop.disabled  = true;
    });

    /* ─ Saccade/fixation event logging (lightweight) ─ */
    orch.saccade.on('fixation', (fix) => {
      // Already handled in orchestrator; just refresh UI stats
      const fixEl = document.querySelector('#p2-fixation');
      if (fixEl) {
        fixEl.textContent = `Fixated ${Math.round(fix.duration)}ms`;
        fixEl.style.color = 'var(--accent-green)';
      }
    });

    orch.saccade.on('saccade', () => {
      const fixEl = document.querySelector('#p2-fixation');
      if (fixEl) {
        fixEl.textContent = 'Scanning';
        fixEl.style.color = 'var(--accent-yellow)';
      }
    });

    /* ─ Confidence low-light & glasses warnings ─ */
    let _lastWarning = 0;
    orch.confidence.score = (function(origScore) {
      return function(lm, headPose, irisData) {
        const result = origScore.call(this, lm, headPose, irisData);
        const now = performance.now();
        if (now - _lastWarning > 5000) {
          if (result.lowLight) {
            _lastWarning = now;
            app.toast.show(
              '⚠ Low Light',
              'Gaze accuracy reduced — improve lighting for better tracking.',
              'warn', 'fas fa-sun', 4000
            );
          } else if (result.glassesDetected) {
            _lastWarning = now;
            app.toast.show(
              '⚠ Glare Detected',
              'Glasses reflection may reduce iris detection accuracy.',
              'warn', 'fas fa-glasses', 4000
            );
          }
        }
        return result;
      };
    })(orch.confidence.score.bind(orch.confidence));
  }

  /* ── Expose public Phase 2 API on window.AccessEye ── */
  _exposeAPI(app) {
    const orch = this.orchestrator;
    if (window.AccessEye) {
      window.AccessEye.phase2 = {
        /** Activate Phase 2 manually */
        activate: async (videoEl, canvasEl) => {
          return orch.activate(videoEl, canvasEl);
        },

        /** Get current gaze state with Phase 2 fields */
        getGazeState: () => ({
          screen:            orch.hybridGaze.smoothGaze,
          raw:               orch.hybridGaze.rawGaze,
          confidence:        orch.confidence.lastScore,
          headPose:          {
            yaw:   orch.headPose.yaw,
            pitch: orch.headPose.pitch,
            roll:  orch.headPose.roll
          },
          fixation: {
            isFixated:   orch.saccade.isFixated,
            fixationAge: orch.saccade.fixationAge,
            stats:       orch.saccade.getStats()
          },
          cameraFPS: orch.cameraFPS
        }),

        /** Enable/disable AI intent prediction */
        setIntentEnabled: (val) => orch.intent.setEnabled(val),

        /** Run benchmark */
        runBenchmark: (durationMs = 30000) => {
          orch.benchmark.duration = durationMs;
          orch.benchmark.start();
          setTimeout(() => orch.benchmark.running && orch.benchmark.stop(), durationMs + 500);
        },

        /** Get benchmark report */
        getBenchmarkReport: () => orch.benchmark.generateReport(),

        /** Reset micro-calibration */
        resetMicroCalib: () => orch.dynCalib.resetMicro(),

        /** Get orchestrator reference */
        orchestrator: orch
      };
    }
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   BOOTSTRAP — run after DOMContentLoaded
───────────────────────────────────────────────────────────────────────── */
const _p2init = new Phase2InitController();
// FIX CAM-1: Expose globally so app.js _startCamera can reset activation state on stop/restart
window._p2InitController = _p2init;

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => _p2init.init());
} else {
  _p2init.init();
}

console.log('%c Phase 2 Init Script Loaded ✅', 'color:#a78bfa;font-weight:bold;font-size:12px;');
