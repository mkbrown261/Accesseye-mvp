/**
 * ═══════════════════════════════════════════════════════════════════════════
 *  AccessEye — Phase 3 Initialization & UI Wiring
 *  phase3-init.js
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  Responsibilities:
 *   1. Wait for Phase 1 app + Phase 2 orchestrator + Phase 3 engine
 *   2. Instantiate Phase3Orchestrator
 *   3. Wire Phase 3 UI controls:
 *      - Dwell preset selector (Fast/Normal/Accessible/Extended)
 *      - Smooth pursuit calibration button
 *      - Post-calibration validation button
 *      - PACE reset button
 *      - Head-free stabilization toggle
 *   4. Auto-activate Phase 3 after Phase 2 activates
 *   5. Extend window.AccessEye.phase3 public API
 *
 *  Load order (index.tsx script tags):
 *    1. app.js            — Phase 1 core engine
 *    2. phase2-engine.js  — Phase 2 modules
 *    3. phase2-init.js    — Phase 2 wiring
 *    4. phase3-engine.js  — Phase 3 modules
 *    5. phase3-init.js    — THIS FILE
 * ═══════════════════════════════════════════════════════════════════════════
 */

'use strict';

class Phase3InitController {
  constructor() {
    this.orchestrator = null;
    this._ready = false;
  }

  init() {
    const check = () => {
      if (window.app && window.Phase2?.Phase2Orchestrator && window.Phase3?.Phase3Orchestrator) {
        // Also wait for Phase 2 to be instantiated on app
        if (window.app.phase2) {
          this._attach(window.app);
        } else {
          setTimeout(check, 150);
        }
      } else {
        setTimeout(check, 100);
      }
    };
    setTimeout(check, 400);
  }

  _attach(app) {
    if (this._ready) return;
    this._ready = true;

    try {
      this.orchestrator = new window.Phase3.Phase3Orchestrator(app.phase2, app);

      // Attach to app
      app.phase3 = this.orchestrator;

      // Patch Phase 2 camera activation to also activate Phase 3
      this._patchPhase2Activation(app);

      // Wire UI controls
      this._wireUI(app);

      // Expose public API
      this._exposeAPI(app);

      console.log('%c Phase 3 Init ✅ — Orchestrator attached to AccessEyeApp',
                  'color:#00d4ff;font-weight:bold;font-size:12px;');
    } catch (e) {
      console.error('[Phase3Init] Attach failed:', e);
    }
  }

  /**
   * Patch Phase 2's activate method to also activate Phase 3 afterwards.
   */
  _patchPhase2Activation(app) {
    const p2Orch = app.phase2;
    const p3Orch = this.orchestrator;
    const origActivate = p2Orch.activate.bind(p2Orch);

    p2Orch.activate = async function(videoEl, canvasEl) {
      await origActivate(videoEl, canvasEl);
      // Activate Phase 3 after Phase 2 is running
      if (p3Orch && !p3Orch.active) {
        // Small delay to ensure Phase 2 pipeline is fully wired
        setTimeout(() => {
          // Update One Euro freq to actual camera FPS
          p3Orch.oneEuro.setFreq(p2Orch.cameraFPS || 30);
          p3Orch.activate();
        }, 300);
      }
    };
  }

  _wireUI(app) {
    const orch = this.orchestrator;

    /* ─── Dwell Preset Selector ─── */
    const presetBtns = document.querySelectorAll('[data-dwell-preset]');
    presetBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const preset = btn.dataset.dwellPreset;
        orch.dwell.setPreset(preset);
        // Update active state
        presetBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const info = orch.dwell.getPresetInfo();
        app.toast.show(
          `Dwell: ${info.label}`,
          `Dwell time set to ${orch.dwell.baseMs}ms`,
          'info', 'fas fa-clock', 2000
        );
        app.log.add(`Dwell preset: ${info.label} (${orch.dwell.baseMs}ms)`, 'info');
      });
    });

    /* ─── Smooth Pursuit Calibration ─── */
    const pursuitBtn = document.getElementById('p3-pursuit-btn');
    if (pursuitBtn) {
      pursuitBtn.addEventListener('click', () => {
        if (!app.phase2?.active) {
          app.toast.show('Camera Required', 'Start camera before running pursuit calibration.', 'warn');
          return;
        }
        app.log.add('Starting smooth pursuit calibration…', 'info');
        app.toast.show(
          'Smooth Pursuit Starting',
          'Follow the moving dot with your eyes for 9 seconds.',
          'info', 'fas fa-route', 4000
        );
        orch.pursuit.start(document.body);
      });
    }

    /* ─── Post-Calibration Validation ─── */
    const validateBtn = document.getElementById('p3-validate-btn');
    if (validateBtn) {
      validateBtn.addEventListener('click', async () => {
        if (!app.calibration?.isCalibrated) {
          app.toast.show('Not Calibrated', 'Run calibration first before validating.', 'warn');
          return;
        }
        validateBtn.disabled = true;
        validateBtn.textContent = 'Validating…';
        app.log.add('Running post-calibration validation (5 points)…', 'info');
        app.toast.show('Validation Starting', 'Look at each dot as it appears.', 'info', 'fas fa-crosshairs', 3000);

        await orch.validator.run();

        validateBtn.disabled = false;
        validateBtn.textContent = '✓ Validate Accuracy';
      });
    }

    /* ─── PACE Reset ─── */
    const paceResetBtn = document.getElementById('p3-pace-reset');
    if (paceResetBtn) {
      paceResetBtn.addEventListener('click', () => {
        orch.pace.reset();
        localStorage.removeItem('accesseye_pace');
        const paceEl = document.getElementById('p3-pace-count');
        if (paceEl) paceEl.textContent = '0';
        app.log.add('PACE recalibration buffer cleared', 'warn');
        app.toast.show('PACE Reset', 'Passive recalibration buffer cleared.', 'warn', 'fas fa-undo', 2500);
      });
    }

    /* ─── Head-Free Toggle ─── */
    const headFreeBtn = document.getElementById('p3-headfree-toggle');
    let headFreeEnabled = true;
    if (headFreeBtn) {
      headFreeBtn.addEventListener('click', () => {
        headFreeEnabled = !headFreeEnabled;
        // Enable/disable head-free stabilizer by resetting it
        if (!headFreeEnabled) {
          orch.headFree.reset();
          headFreeBtn.classList.remove('active');
          headFreeBtn.textContent = 'Enable Head-Free';
          app.log.add('Head-free stabilization: OFF', 'warn');
        } else {
          headFreeBtn.classList.add('active');
          headFreeBtn.textContent = 'Disable Head-Free';
          app.log.add('Head-free stabilization: ON', 'success');
          // Re-set reference from current face center
          const lm = orch._lastLandmarks;
          if (lm) {
            const fc = window.Phase3.HeadFreeStabilizer.faceCenter(lm);
            orch.headFree.setReference(fc.x, fc.y);
          }
        }
        app.toast.show(
          `Head-Free: ${headFreeEnabled ? 'ON' : 'OFF'}`,
          headFreeEnabled
            ? 'Head movement compensation active.'
            : 'Head-free stabilization disabled.',
          headFreeEnabled ? 'success' : 'info',
          'fas fa-arrows-alt', 2500
        );
      });
    }

    /* ─── Update calibration button to show 13-point count ─── */
    this._updateCalibUI(app);

    /* ─── Run validation after each calibration ─── */
    // Listen for existing calibration completion
    const origOnComplete = app._calibOnComplete;
    app._calibOnComplete = function(success) {
      if (origOnComplete) origOnComplete(success);
      if (success && orch.active) {
        // Auto-validate after 1s delay
        setTimeout(async () => {
          app.log.add('Auto-running post-calibration validation…', 'info');
          await orch.validator.run();
        }, 1000);
        // Set head-free reference after new calibration
        const lm = orch._lastLandmarks;
        if (lm) {
          const fc = window.Phase3.HeadFreeStabilizer.faceCenter(lm);
          orch.headFree.setReference(fc.x, fc.y);
        }
      }
    };

    /* ─── IVT status display ─── */
    orch.ivt.on('saccade', () => {
      const el = document.getElementById('p3-ivt-status');
      if (el) { el.textContent = '↗ Saccade'; el.style.color = '#ffd32a'; }
    });

    /* ─── PACE refit status ─── */
    orch.pace.on('refit', (d) => {
      const el = document.getElementById('p3-pace-count');
      if (el) el.textContent = d.pace;
    });
  }

  _updateCalibUI(app) {
    // Update calibration step counter to show 13 points
    const stepLabel = document.getElementById('calib-step-label');
    if (stepLabel && app.calibration?.CALIB_POINTS?.length === 13) {
      // Will be updated dynamically during calibration
    }
    // Update any static "5 point" text in UI
    document.querySelectorAll('[data-calib-point-count]').forEach(el => {
      el.textContent = '13';
    });
  }

  _exposeAPI(app) {
    const orch = this.orchestrator;
    if (window.AccessEye) {
      window.AccessEye.phase3 = {
        /** Set dwell preset */
        setDwellPreset: (key) => orch.dwell.setPreset(key),

        /** Get available dwell presets */
        getDwellPresets: () => window.Phase3.AdaptiveDwellTimer.listPresets(),

        /** Start smooth pursuit calibration */
        runPursuitCalib: () => orch.pursuit.start(document.body),

        /** Run post-calibration validation */
        runValidation: () => orch.validator.run(),

        /** Get last validation report */
        getValidationReport: () => orch.validator.lastReport,

        /** PACE recalibrator reference */
        pace: orch.pace,

        /** IVT detector state */
        ivt: orch.ivt,

        /** Head-free stabilizer */
        headFree: orch.headFree,

        /** Get orchestrator */
        orchestrator: orch
      };
    }
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   BOOTSTRAP
───────────────────────────────────────────────────────────────────────── */
const _p3init = new Phase3InitController();

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => _p3init.init());
} else {
  _p3init.init();
}

console.log('%c Phase 3 Init Script Loaded ✅', 'color:#00d4ff;font-weight:bold;font-size:12px;');
