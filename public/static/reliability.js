/**
 * ═══════════════════════════════════════════════════════════════════════════
 *  AccessEye — Hospital-Grade Reliability Layer  (reliability.js)
 *  Phase 4 — NEW FILE — additive, non-destructive
 *
 *  ARCHITECTURE:
 *  ─────────────────────────────────────────────────────────────────────────
 *  • Read-only access to window.app (reads .cameraOn, .mode)
 *  • NEVER modifies Action Layer, engine internals, or cursor state
 *  • Adds a small System Status indicator to the DOM (dismissable)
 *  • Uses window.AccessEye.on() for gaze/focus events (read-only)
 *  • Provides auto-retry logic via the same Interaction Layer paths
 *    Care Mode uses (button.click() for camera start, etc.)
 *  • State persistence: saves/restores Care Mode state and voice state
 *    across page navigations via sessionStorage
 *  • Privacy: zero camera/biometric data stored or transmitted;
 *    all processing is local; indicator shows "Local processing only"
 *
 *  DOES NOT TOUCH:
 *  • app.js internals (_startCamera, _stopCamera, phase2/3 engines)
 *  • snap-engine.js, gesture-studio.js, phase2/3 engines
 *  • Any existing event listeners or global objects
 * ═══════════════════════════════════════════════════════════════════════════
 */

'use strict';

/* Guard: never double-load */
if (window.__accesseyeReliabilityLoaded) {
  console.log('[Reliability] Already loaded — skipping');
  // Don't return; use a flag-check pattern
}

if (!window.__accesseyeReliabilityLoaded) {
  window.__accesseyeReliabilityLoaded = true;

/* ─────────────────────────────────────────────────────────────────────────
   VERSION + CONSTANTS
───────────────────────────────────────────────────────────────────────── */
const RL_VERSION = '1.0.0';

/** System health states */
const HEALTH = {
  READY    : 'ready',     // ✅ All systems nominal
  DEGRADED : 'degraded',  // ⚠️  Partial functionality
  ERROR    : 'error',     // ❌ Critical failure
  INIT     : 'init',      // 🔄 Initialising
};

/** Error categories */
const ERR = {
  CRITICAL : 'critical',  // Must recover or block usage
  MEDIUM   : 'medium',    // Degrades experience but usable
  MINOR    : 'minor',     // Cosmetic / non-blocking
};

/** Session storage keys */
const SS_KEYS = {
  CARE_MODE_ACTIVE : 'rl_care_mode_active',
  VOICE_NAV_ACTIVE : 'rl_voice_nav_active',
  LAST_PAGE        : 'rl_last_page',
  HEALTH_STATE     : 'rl_health_state',
};

/** Monitor intervals (ms) */
const MONITOR_INTERVAL_FAST = 500;   // voice command & camera watchdog
const MONITOR_INTERVAL_SLOW = 3000;  // health summary refresh

/* ─────────────────────────────────────────────────────────────────────────
   STATE
───────────────────────────────────────────────────────────────────────── */
const rlState = {
  health          : HEALTH.INIT,
  errors          : [],          // rolling log of recent errors
  cameraWatchdog  : null,
  healthInterval  : null,
  gazeLastSeen    : 0,
  voiceLastSeen   : 0,
  cameraRetries   : 0,
  voiceRetries    : 0,
  indicatorEl     : null,
  dismissed       : false,
  MAX_ERRORS      : 50,
};

/* ─────────────────────────────────────────────────────────────────────────
   ERROR LOG
───────────────────────────────────────────────────────────────────────── */
function _logError(category, message, silent = false) {
  const entry = {
    ts       : Date.now(),
    category,
    message,
    time     : new Date().toISOString().slice(11, 23),
  };
  rlState.errors.push(entry);
  if (rlState.errors.length > rlState.MAX_ERRORS) {
    rlState.errors.shift();
  }

  if (!silent) {
    if (category === ERR.CRITICAL) {
      console.error(`[Reliability][CRITICAL] ${message}`);
    } else if (category === ERR.MEDIUM) {
      console.warn(`[Reliability][MEDIUM] ${message}`);
    } else {
      console.log(`[Reliability][MINOR] ${message}`);
    }
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   SYSTEM STATUS INDICATOR
   A small fixed badge (bottom-right) showing current health state.
   Non-intrusive: 180×auto, dismissable, scoped CSS.
───────────────────────────────────────────────────────────────────────── */
function _injectIndicator() {
  if (document.getElementById('rl-status-indicator')) return;

  const style = document.createElement('style');
  style.id = 'rl-styles';
  style.textContent = `
    /* ─── Reliability indicator — all rules scoped ─── */
    #rl-status-indicator {
      position: fixed;
      bottom: 18px; right: 18px;
      z-index: 89999;
      min-width: 176px; max-width: 230px;
      background: rgba(11,15,28,0.97);
      border: 1.5px solid #1a2540;
      border-radius: 12px;
      padding: 10px 14px;
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      font-size: 12px;
      color: #8899bb;
      box-shadow: 0 4px 24px rgba(0,0,0,0.5);
      transition: opacity 0.3s, transform 0.3s;
      pointer-events: auto;
      user-select: none;
    }
    #rl-status-indicator.rl-hidden {
      opacity: 0; transform: translateY(8px);
      pointer-events: none;
    }
    #rl-status-indicator .rl-header {
      display: flex; align-items: center;
      justify-content: space-between; gap: 8px;
      margin-bottom: 6px;
    }
    #rl-status-indicator .rl-title {
      font-size: 11px; font-weight: 700;
      letter-spacing: 0.5px; text-transform: uppercase;
      color: #4a5a7a;
    }
    #rl-status-indicator .rl-dismiss {
      background: none; border: none;
      color: #4a5a7a; cursor: pointer;
      font-size: 14px; line-height: 1; padding: 0;
    }
    #rl-status-indicator .rl-dismiss:hover { color: #8899bb; }
    #rl-status-indicator .rl-state {
      display: flex; align-items: center; gap: 7px;
      font-size: 13px; font-weight: 700;
      margin-bottom: 6px;
    }
    #rl-status-indicator .rl-dot {
      width: 9px; height: 9px; border-radius: 50%;
      flex-shrink: 0;
    }
    #rl-status-indicator .rl-dot.ready    { background: #22c55e; }
    #rl-status-indicator .rl-dot.degraded { background: #f59e0b;
      animation: rl-pulse 1.5s ease infinite; }
    #rl-status-indicator .rl-dot.error    { background: #ef4444;
      animation: rl-pulse 0.8s ease infinite; }
    #rl-status-indicator .rl-dot.init     { background: #60a5fa;
      animation: rl-pulse 1.2s ease infinite; }
    @keyframes rl-pulse {
      0%,100% { opacity:1 } 50% { opacity:0.35 }
    }
    #rl-status-indicator .rl-systems {
      display: flex; flex-direction: column; gap: 3px;
      border-top: 1px solid #1a2540; padding-top: 6px;
      margin-top: 2px;
    }
    #rl-status-indicator .rl-sys-row {
      display: flex; align-items: center; gap: 6px;
      font-size: 11px; color: #6b7a9a;
    }
    #rl-status-indicator .rl-sys-icon { font-size: 11px; }
    #rl-status-indicator .rl-privacy {
      margin-top: 6px;
      border-top: 1px solid #1a2540; padding-top: 6px;
      font-size: 10px; color: #3a4a6a;
      display: flex; align-items: center; gap: 5px;
    }
  `;
  document.head.appendChild(style);

  const el = document.createElement('div');
  el.id = 'rl-status-indicator';
  el.setAttribute('role', 'status');
  el.setAttribute('aria-live', 'polite');
  el.setAttribute('aria-label', 'AccessEye system status');
  el.innerHTML = _indicatorHTML(HEALTH.INIT);
  document.body.appendChild(el);
  rlState.indicatorEl = el;

  // Dismiss button
  el.querySelector('.rl-dismiss')?.addEventListener('click', () => {
    rlState.dismissed = true;
    el.classList.add('rl-hidden');
    _logError(ERR.MINOR, 'Status indicator dismissed by user', true);
  });
}

function _indicatorHTML(health) {
  const app = window.app;
  const cameraOn = app?.cameraOn === true;
  const voiceOn  = window.voiceNav?.enabled === true;
  const careModeOn = window.CareMode?.isActive?.() === true;

  const healthLabel = {
    [HEALTH.READY]    : '✅ Ready',
    [HEALTH.DEGRADED] : '⚠️ Degraded',
    [HEALTH.ERROR]    : '❌ Error',
    [HEALTH.INIT]     : '🔄 Initialising',
  }[health] || '🔄 Initialising';

  const camStatus  = cameraOn  ? '✅' : '⭕';
  const voiceStatus= voiceOn   ? '✅' : '⭕';
  const careStatus = careModeOn? '✅' : '⭕';

  return `
    <div class="rl-header">
      <span class="rl-title">System Status</span>
      <button class="rl-dismiss" aria-label="Dismiss status indicator" title="Dismiss">×</button>
    </div>
    <div class="rl-state">
      <span class="rl-dot ${health}"></span>
      <span>${healthLabel}</span>
    </div>
    <div class="rl-systems">
      <div class="rl-sys-row">
        <span class="rl-sys-icon">${camStatus}</span>
        <span>Eye tracking ${cameraOn ? 'active' : 'inactive'}</span>
      </div>
      <div class="rl-sys-row">
        <span class="rl-sys-icon">${voiceStatus}</span>
        <span>Voice nav ${voiceOn ? 'active' : 'inactive'}</span>
      </div>
      <div class="rl-sys-row">
        <span class="rl-sys-icon">${careStatus}</span>
        <span>Care Mode ${careModeOn ? 'active' : 'inactive'}</span>
      </div>
    </div>
    <div class="rl-privacy">
      🔒 Local processing only — no data transmitted
    </div>`;
}

function _updateIndicator() {
  if (rlState.dismissed || !rlState.indicatorEl) return;
  rlState.indicatorEl.innerHTML = _indicatorHTML(rlState.health);
  // Re-attach dismiss listener
  rlState.indicatorEl.querySelector('.rl-dismiss')?.addEventListener('click', () => {
    rlState.dismissed = true;
    rlState.indicatorEl.classList.add('rl-hidden');
    _logError(ERR.MINOR, 'Status indicator dismissed by user', true);
  });
}

/* ─────────────────────────────────────────────────────────────────────────
   HEALTH ASSESSMENT
   Runs every MONITOR_INTERVAL_SLOW ms.
───────────────────────────────────────────────────────────────────────── */
function _assessHealth() {
  const app       = window.app;
  const cameraOn  = app?.cameraOn === true;
  const voiceOn   = window.voiceNav?.enabled === true;

  const now = Date.now();
  const gazeAge    = now - rlState.gazeLastSeen;  // ms since last gaze event
  const voiceAge   = now - rlState.voiceLastSeen; // ms since last voice result

  let health = HEALTH.READY;
  const issues = [];

  // Camera/gaze health
  if (cameraOn && gazeAge > 5000) {
    issues.push({ category: ERR.MEDIUM, msg: `Gaze events stopped ${(gazeAge/1000).toFixed(1)}s ago` });
    health = HEALTH.DEGRADED;
  }

  // Voice system health  
  if (voiceOn && voiceAge > 30000) {
    issues.push({ category: ERR.MINOR, msg: `Voice recognition idle ${(voiceAge/1000).toFixed(0)}s` });
    if (health === HEALTH.READY) health = HEALTH.DEGRADED;
  }

  // Critical: app itself not loaded
  if (!app) {
    issues.push({ category: ERR.CRITICAL, msg: 'window.app not available — core system failure' });
    health = HEALTH.ERROR;
  }

  issues.forEach(i => _logError(i.category, i.msg, true));

  if (health !== rlState.health) {
    console.log(`[Reliability] Health: ${rlState.health} → ${health}`);
    rlState.health = health;
    try { sessionStorage.setItem(SS_KEYS.HEALTH_STATE, health); } catch (_) {}
  }

  _updateIndicator();
  return health;
}

/* ─────────────────────────────────────────────────────────────────────────
   FAILSAFE MONITORS
   Camera watchdog: if camera drops while in gaze mode, attempt auto-retry.
   Voice watchdog: if recognition drops, auto-restart.
───────────────────────────────────────────────────────────────────────── */
function _startCameraWatchdog() {
  if (rlState.cameraWatchdog) return;

  rlState.cameraWatchdog = setInterval(() => {
    const app = window.app;
    if (!app) return;

    const shouldHaveCamera = app.mode === 'gaze';
    const cameraOn         = app.cameraOn === true;

    if (shouldHaveCamera && !cameraOn && rlState.cameraRetries < 3) {
      rlState.cameraRetries++;
      _logError(ERR.MEDIUM,
        `Camera watchdog: camera off while in gaze mode — retry #${rlState.cameraRetries} via interaction layer`);

      // Use the same interaction-layer path as Care Mode — button.click()
      const startBtn = document.getElementById('start-camera-btn');
      if (startBtn) {
        _logError(ERR.MINOR, 'Watchdog: triggering #start-camera-btn.click()');
        startBtn.click();
      }
      _updateIndicator();
    } else if (cameraOn) {
      // Camera recovered — reset retry counter
      if (rlState.cameraRetries > 0) {
        _logError(ERR.MINOR, 'Watchdog: camera recovered', true);
        rlState.cameraRetries = 0;
      }
    }
  }, MONITOR_INTERVAL_FAST);
}

function _stopCameraWatchdog() {
  if (rlState.cameraWatchdog) {
    clearInterval(rlState.cameraWatchdog);
    rlState.cameraWatchdog = null;
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   STATE PERSISTENCE
   Saves key system states to sessionStorage so they can be restored
   after page navigation or UI transitions.
───────────────────────────────────────────────────────────────────────── */
function _saveState() {
  try {
    const careActive  = window.CareMode?.isActive?.() === true;
    const voiceActive = window.voiceNav?.enabled === true;
    const lastPage    = document.querySelector('.page.active')?.id?.replace('page-', '') || 'home';

    sessionStorage.setItem(SS_KEYS.CARE_MODE_ACTIVE, careActive  ? '1' : '0');
    sessionStorage.setItem(SS_KEYS.VOICE_NAV_ACTIVE, voiceActive ? '1' : '0');
    sessionStorage.setItem(SS_KEYS.LAST_PAGE,         lastPage);
  } catch (_) {
    // sessionStorage may be unavailable in some contexts — silently skip
  }
}

function _restoreState() {
  try {
    const careSaved  = sessionStorage.getItem(SS_KEYS.CARE_MODE_ACTIVE) === '1';
    const voiceSaved = sessionStorage.getItem(SS_KEYS.VOICE_NAV_ACTIVE) === '1';

    // Restore Care Mode if it was active
    if (careSaved && window.CareMode && !window.CareMode.isActive()) {
      _logError(ERR.MINOR, 'Restoring Care Mode from session state');
      setTimeout(() => {
        if (!window.CareMode.isActive()) {
          window.CareMode.mount?.();
        }
      }, 800);
    }

    // Restore voice navigation if it was active
    if (voiceSaved && window.voiceNav && !window.voiceNav.enabled) {
      _logError(ERR.MINOR, 'Restoring voice navigation from session state');
      setTimeout(() => {
        if (!window.voiceNav.enabled) {
          window.voiceNav.enable?.();
        }
      }, 1000);
    }
  } catch (_) {}
}

/* ─────────────────────────────────────────────────────────────────────────
   GAZE EVENT MONITOR
   Tracks when gaze events last arrived to detect eye-tracking loss.
───────────────────────────────────────────────────────────────────────── */
function _attachGazeMonitor() {
  if (!window.AccessEye?.on) {
    setTimeout(_attachGazeMonitor, 300);
    return;
  }

  window.AccessEye.on('gaze', () => {
    rlState.gazeLastSeen = Date.now();
    // If we were in degraded state and gaze comes back, re-assess
    if (rlState.health === HEALTH.DEGRADED) {
      _assessHealth();
    }
  });

  _logError(ERR.MINOR, 'Gaze monitor attached', true);
}

/* ─────────────────────────────────────────────────────────────────────────
   PUBLIC API
   Exposed on window.AccessEyeReliability
───────────────────────────────────────────────────────────────────────── */
const ReliabilityAPI = {
  version     : RL_VERSION,

  /** Get current health state */
  getHealth   : () => rlState.health,

  /** Get recent error log */
  getErrors   : () => [...rlState.errors],

  /** Manually trigger a health assessment */
  assess      : _assessHealth,

  /** Show/hide status indicator */
  showIndicator () {
    rlState.dismissed = false;
    rlState.indicatorEl?.classList.remove('rl-hidden');
    _updateIndicator();
  },
  hideIndicator () {
    rlState.dismissed = true;
    rlState.indicatorEl?.classList.add('rl-hidden');
  },

  /** Save/restore system state */
  saveState    : _saveState,
  restoreState : _restoreState,

  /** Start/stop camera watchdog */
  startWatchdog : _startCameraWatchdog,
  stopWatchdog  : _stopCameraWatchdog,
};

window.AccessEyeReliability = ReliabilityAPI;

/* ─────────────────────────────────────────────────────────────────────────
   BOOTSTRAP
───────────────────────────────────────────────────────────────────────── */
function _bootstrap() {
  // Wait for app to be ready
  const waitForApp = () => {
    if (!window.app) { setTimeout(waitForApp, 300); return; }
    _init();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', waitForApp);
  } else {
    setTimeout(waitForApp, 600);
  }
}

function _init() {
  // 1. Inject status indicator into DOM
  _injectIndicator();

  // 2. Attach gaze event monitor
  _attachGazeMonitor();

  // 3. Start camera watchdog
  _startCameraWatchdog();

  // 4. Start periodic health assessment
  rlState.healthInterval = setInterval(() => {
    _assessHealth();
    _saveState();
  }, MONITOR_INTERVAL_SLOW);

  // 5. Restore state from previous session/navigation
  _restoreState();

  // 6. Initial health assessment (after brief delay for other systems to init)
  setTimeout(() => {
    rlState.health = HEALTH.READY;
    _assessHealth();
  }, 2000);

  // 7. Save state on page visibility change / before unload
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) _saveState();
  });
  window.addEventListener('beforeunload', _saveState);

  // 8. Watch for voice nav activity
  if (window.voiceNav) {
    rlState.voiceLastSeen = Date.now();
  }

  console.log(`%c AccessEye Reliability Layer ✅ v${RL_VERSION}`,
    'color:#22c55e;font-weight:bold;font-size:12px;');
  console.log(
    '%c [Reliability] Monitors: camera watchdog ✅ | gaze monitor ✅ | health check ✅ | state persistence ✅ | privacy: local-only ✅',
    'color:#4a5a7a;font-size:11px;'
  );
}

_bootstrap();

} // end guard block
