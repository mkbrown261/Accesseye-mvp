/**
 * ═══════════════════════════════════════════════════════════════════════════
 *  AccessEye — Accuracy Engine  v2.0
 *  accuracy-engine.js
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  WHAT PHASES 1–3 ALREADY HAVE:
 *   ✅ One Euro Filter        (Phase 3 — minCutoff 0.3, β 0.05)
 *   ✅ IVT saccade classifier (Phase 3 — 35px/frame threshold)
 *   ✅ Adaptive dwell timer   (Phase 3 — Fast/Normal/Accessible/Extended)
 *   ✅ Kalman + EMA + trimmed-mean stabilizer (Phase 2)
 *   ✅ Binocular iris fusion  (Phase 2)
 *   ✅ Dynamic calibration + bias correction (Phase 2)
 *   ✅ Snap-to engine + target predictor (snap-engine)
 *
 *  WHAT IS MISSING (this file adds):
 *
 *   ACC.1  GravitySnapEngine   — Gravity-model attractor pull
 *     Ref: Grossman & Balakrishnan (2005); Phase 6 AccessEye report
 *     25–40% mis-selection reduction vs distance-only snap
 *     Difference from SnapToEngine: snap-engine does HARD SNAP (cursor jumps
 *     to nearest element within threshold). Gravity does a SOFT PULL (cursor
 *     is nudged toward the highest-force attractor — feels natural, not jumpy).
 *     The two work together: gravity pre-aligns, then snap locks.
 *
 *   ACC.2  PerSessionDriftCorrector   — Accumulating drift compensation
 *     Ref: Phase 5 AccessEye audit — "missing drift correction"
 *     Over a session, gaze systematically drifts 2–5% of screen width
 *     (eye fatigue + head settling). Phase 2 has a PACE-style bias correction
 *     on INTERACTION events. This adds a slower passive correction on FIXATION
 *     events — runs ~5×/sec during stable fixations and nudges the displayed
 *     cursor back toward the fixation centroid.
 *
 *  SAFE INTEGRATION:
 *   • Wraps app._updateGazeCursor (the FINAL output step only)
 *   • All Phase 2/3 logic still runs first
 *   • Both modules are optional and individually toggleable
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
   * @returns {{ x: number, y: number }}
   */
  update(gx, gy, isFixated, conf) {
    if (!isFixated || conf < this.MIN_CONF) {
      this.lastForce = 0;
      return { x: gx, y: gy };
    }

    const els = this._getElements();
    if (!els.length) return { x: gx, y: gy };

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

    if (!bestEl || this.lastForce < 0.05) return { x: gx, y: gy };

    return {
      x: gx + (bestCx - gx) * this.PULL_STRENGTH * this.lastForce,
      y: gy + (bestCy - gy) * this.PULL_STRENGTH * this.lastForce
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
    const W = window.innerWidth  || 1920;
    const H = window.innerHeight || 1080;

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
   ACCURACY ORCHESTRATOR
   ═══════════════════════════════════════════════════════════════════════════
   Wires ACC.1 and ACC.2 into Phase 2's output step.
   Polls for Phase2Orchestrator availability, then installs a safe wrapper
   on app._updateGazeCursor.
*/
class AccuracyOrchestrator {
  constructor() {
    this.gravity    = new _AccGravitySnap();
    this.driftCorr  = new _AccDriftCorrector();

    this._active    = false;
    this._installed = false;
    this._attempts  = 0;

    this.config = {
      enableGravity:   true,
      enableDriftCorr: true
    };

    // Diagnostics
    this.diag = {
      frames:       0,
      gravityPulls: 0,
      driftCorrs:   0,
      lastPullForce:  0,
      lastDriftMag:   0
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
        self._updateStatusUI();
        console.log('%c[AccuracyEngine v2] Active — GravitySnap + DriftCorrector', 'color:#00ff88;font-weight:bold');
        return r;
      };
    }

    if (origDeactivate) {
      p2.deactivate = function() {
        self._active = false;
        self.gravity.reset();
        self.driftCorr.reset();
        return origDeactivate();
      };
    }

    // ── CORE PATCH: Wrap app._updateGazeCursor ──
    const origUpdate = app._updateGazeCursor?.bind(app);
    if (!origUpdate) {
      console.warn('[AccuracyEngine] app._updateGazeCursor not found — skipping patch');
      return;
    }
    app._updateGazeCursor = function(sx, sy) {
      if (!self._active) return origUpdate(sx, sy);

      try {
        const W    = window.innerWidth  || 1920;
        const H    = window.innerHeight || 1080;
        const conf = p2.confidence?.lastScore?.total ?? app.gazeEngine?.confidence ?? 0.5;

        // Determine fixation state from best available source
        const ivt   = window.app?.phase3?.ivt;
        const sacc  = p2.saccade;
        const isFixated = (ivt?.isFixating ?? sacc?.isFixated) ?? false;

        let px = sx, py = sy;

        // ── ACC.1: Gravity snap (soft pull toward likely target) ──
        if (self.config.enableGravity) {
          const g = self.gravity.update(px, py, isFixated, conf);
          if (g.x !== px || g.y !== py) self.diag.gravityPulls++;
          px = g.x; py = g.y;
          self.diag.lastPullForce = self.gravity.lastForce;
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

    console.log('%c[AccuracyEngine v2] Patch installed on app._updateGazeCursor', 'color:#00d4ff;font-size:11px');
  }

  _updateStatusUI() {
    const panel = document.getElementById('p2-status-panel');
    if (!panel || document.getElementById('acc-status-row')) return;

    const row = document.createElement('div');
    row.id = 'acc-status-row';
    row.style.cssText = [
      'margin-top:8px', 'padding:6px 8px',
      'background:rgba(0,255,136,0.07)', 'border:1px solid rgba(0,255,136,0.22)',
      'border-radius:6px', 'font-size:11px', 'color:#94a3b8'
    ].join(';');
    row.innerHTML = `
      <div style="color:#00ff88;font-weight:600;margin-bottom:4px;font-size:11px;">
        <i class="fas fa-crosshairs" style="margin-right:4px;"></i>Accuracy Engine v2
      </div>
      <div style="display:flex;gap:12px;flex-wrap:wrap;">
        <span title="Soft gravity pull toward likely targets">
          <i class="fas fa-magnet" style="color:#00d4ff;margin-right:3px;"></i>
          Gravity <span id="acc-pull-val" style="color:#fbbf24">—</span>
        </span>
        <span title="Session drift correction">
          <i class="fas fa-compress-arrows-alt" style="color:#00d4ff;margin-right:3px;"></i>
          Drift <span id="acc-drift-val" style="color:#fbbf24">—</span>
        </span>
      </div>`;
    panel.appendChild(row);
  }

  _updateLiveUI() {
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
  }

  getDiag() {
    return {
      ...this.diag,
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
      '%c[AccuracyEngine v2] Loaded — GravitySnap + DriftCorrector',
      'color:#00ff88;font-weight:bold;font-size:12px'
    );
    console.log(
      '%c  Research: Grossman & Balakrishnan 2005 | Phase 5-6 AccessEye Report',
      'color:#94a3b8;font-size:10px'
    );
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    setTimeout(init, 0);
  }
})();
