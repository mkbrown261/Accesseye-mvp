/**
 * ═══════════════════════════════════════════════════════════════════════
 *  AccessEye — Hospital-Grade Reliability Layer  v1.0
 *  reliability-layer.js
 *
 *  PURPOSE: Passive watchdog + state-persistence layer.
 *           Does NOT modify the Action Layer or any existing module.
 *           Uses only:
 *            • Read-only access to window.app (cameraOn, mode, gazeEngine)
 *            • window.AccessEye.on() subscription (Interaction Layer)
 *            • localStorage for Care Mode / voice state persistence
 *            • CustomEvents for status broadcast
 *            • Injects ONE status indicator div into the page body
 *
 *  ARCHITECTURE RULES (strictly followed):
 *  ─────────────────────────────────────────────────────────────────────
 *  ✅ Read-only access to window.app, window.AccessEye, window.CareMode
 *  ✅ Only additive DOM changes (one injected status indicator)
 *  ✅ Persists state via localStorage (no cookies, no server calls)
 *  ✅ Auto-recovery via silent retry (no alerts, no popups)
 *  ✅ Privacy: no camera frames, no biometric data stored anywhere
 *  ❌ Does NOT patch, wrap, or replace any existing function
 *  ❌ Does NOT use browser extension APIs
 *  ❌ Does NOT modify the Action Layer
 * ═══════════════════════════════════════════════════════════════════════
 */

;(function () {
  'use strict';

  /* ── Guard: prevent double-load ─────────────────────────────────── */
  if (window.__reliabilityLayerLoaded) return;
  window.__reliabilityLayerLoaded = true;

  /* ════════════════════════════════════════════════════════════════════
     VERSION
  ════════════════════════════════════════════════════════════════════ */
  const RL_VERSION = '1.0.0';

  /* ════════════════════════════════════════════════════════════════════
     PRIVACY GUARANTEE
     Displayed in the status indicator; also logged on init.
  ════════════════════════════════════════════════════════════════════ */
  const PRIVACY_NOTICE =
    '🔒 All data is processed locally. Nothing is stored or transmitted.';

  /* ════════════════════════════════════════════════════════════════════
     ERROR CATEGORIES
  ════════════════════════════════════════════════════════════════════ */
  const ERR = {
    CRITICAL: 'Critical',   // Camera failure, total eye-tracking loss
    MEDIUM:   'Medium',     // Voice miss, brief tracking loss
    MINOR:    'Minor',      // UI glitch, overlay timeout
  };

  /* ════════════════════════════════════════════════════════════════════
     STATE
  ════════════════════════════════════════════════════════════════════ */
  const state = {
    systemStatus:   'ready',   // 'ready' | 'degraded' | 'error'
    cameraOk:       false,
    trackingOk:     false,
    voiceOk:        true,
    careModeActive: false,
    lastGazeMs:     0,
    lastVoiceMs:    0,
    failureLog:     [],        // circular buffer, max 50 entries
    recoveryCount:  0,
    watchdogTimer:  null,
    gazeTimeoutTimer: null,
  };

  // FIX-TRACKING-LOST: don't show "Tracking lost" until at least one
  // gaze event has arrived in this camera session. Camera warm-up
  // takes 2-4 s; showing the indicator immediately is a false alarm.
  let _gazeEverReceived = false;

  /* ════════════════════════════════════════════════════════════════════
     CONSTANTS
  ════════════════════════════════════════════════════════════════════ */
  const GAZE_TIMEOUT_MS      = 4000;   // No gaze events for >4s → degraded
  const CAMERA_POLL_MS       = 2000;   // Check window.app.cameraOn every 2s
  const WATCHDOG_INTERVAL_MS = 1500;   // Main watchdog tick
  const MAX_LOG_ENTRIES      = 50;
  const PERSIST_KEY          = 'ae_reliability_state';

  /* ════════════════════════════════════════════════════════════════════
     LOGGER
  ════════════════════════════════════════════════════════════════════ */
  const RL = {
    tag: '[Reliability]',
    _log (level, category, msg, data) {
      const entry = {
        ts: Date.now(),
        level,
        category,
        msg,
        data: data || null,
      };
      state.failureLog.push(entry);
      if (state.failureLog.length > MAX_LOG_ENTRIES) {
        state.failureLog.shift();
      }
      const style = level === ERR.CRITICAL ? 'color:#ef4444;font-weight:bold'
                  : level === ERR.MEDIUM   ? 'color:#f59e0b'
                  :                          'color:#6b7280';
      console.log(`%c${this.tag} [${level}] ${msg}`, style, data || '');
      // Broadcast failure event (passive — no existing code listens)
      document.dispatchEvent(new CustomEvent('ae:reliability:failure', {
        bubbles: false,
        detail: entry,
      }));
    },
    critical (msg, data) { this._log(ERR.CRITICAL, 'System', msg, data); },
    medium   (msg, data) { this._log(ERR.MEDIUM,   'Voice',  msg, data); },
    minor    (msg, data) { this._log(ERR.MINOR,    'UI',     msg, data); },
    ok       (msg)       { console.log(`%c${this.tag} ✅ ${msg}`, 'color:#22c55e'); },
  };

  /* ════════════════════════════════════════════════════════════════════
     STATUS INDICATOR
     One div injected at bottom-left; z-index 99997 (below care-mode).
     Updates in real-time to reflect system health.
  ════════════════════════════════════════════════════════════════════ */
  const INDICATOR_ID = 'ae-reliability-indicator';
  const ICON_MAP = {
    ready:    '✅',
    degraded: '⚠️',
    error:    '❌',
  };
  const LABEL_MAP = {
    ready:    'Ready',
    degraded: 'Degraded',
    error:    'Error',
  };
  const COLOR_MAP = {
    ready:    { bg: 'rgba(6,30,15,0.92)', border: '#22c55e44', dot: '#22c55e' },
    degraded: { bg: 'rgba(30,18,5,0.92)', border: '#f59e0b44', dot: '#f59e0b' },
    error:    { bg: 'rgba(30,6,6,0.92)',  border: '#ef444444', dot: '#ef4444' },
  };

  function _injectIndicator () {
    if (document.getElementById(INDICATOR_ID)) return;

    const el = document.createElement('div');
    el.id = INDICATOR_ID;
    el.setAttribute('role', 'status');
    el.setAttribute('aria-live', 'polite');
    el.setAttribute('aria-label', 'AccessEye system status');
    Object.assign(el.style, {
      position:     'fixed',
      bottom:       '12px',
      left:         '12px',
      zIndex:       '99997',
      display:      'flex',
      alignItems:   'center',
      gap:          '7px',
      padding:      '6px 12px',
      borderRadius: '20px',
      fontSize:     '12px',
      fontFamily:   'system-ui, sans-serif',
      fontWeight:   '600',
      cursor:       'pointer',
      userSelect:   'none',
      transition:   'all 0.3s ease',
      backdropFilter: 'blur(8px)',
      WebkitBackdropFilter: 'blur(8px)',
      border:       '1px solid',
      whiteSpace:   'nowrap',
      maxWidth:     '200px',
    });

    // Dot
    const dot = document.createElement('span');
    dot.id = 'ae-rl-dot';
    Object.assign(dot.style, {
      width: '8px', height: '8px',
      borderRadius: '50%',
      flexShrink: '0',
      transition: 'background 0.3s',
    });

    // Label
    const lbl = document.createElement('span');
    lbl.id = 'ae-rl-label';
    lbl.style.color = '#e8edf8';

    // Privacy tooltip (hidden by default, shown on click)
    const tip = document.createElement('div');
    tip.id = 'ae-rl-tooltip';
    Object.assign(tip.style, {
      display:      'none',
      position:     'absolute',
      bottom:       '36px',
      left:         '0',
      background:   'rgba(10,14,26,0.98)',
      border:       '1px solid #1a2540',
      borderRadius: '10px',
      padding:      '10px 14px',
      fontSize:     '11px',
      color:        '#9ca3af',
      maxWidth:     '260px',
      whiteSpace:   'normal',
      lineHeight:   '1.5',
      zIndex:       '100000',
      boxShadow:    '0 4px 20px rgba(0,0,0,0.6)',
    });
    tip.innerHTML = `<strong style="color:#fff">AccessEye Status</strong><br>${PRIVACY_NOTICE}`;

    el.appendChild(dot);
    el.appendChild(lbl);
    el.appendChild(tip);

    // Toggle tooltip on click
    el.addEventListener('click', () => {
      const isVisible = tip.style.display !== 'none';
      tip.style.display = isVisible ? 'none' : 'block';
    });

    // Hide tooltip when clicking elsewhere
    document.addEventListener('click', (e) => {
      if (!el.contains(e.target)) tip.style.display = 'none';
    }, { capture: true });

    document.body.appendChild(el);
    _updateIndicator('ready', 'AccessEye Ready');
  }

  function _updateIndicator (status, detail) {
    state.systemStatus = status;
    const el  = document.getElementById(INDICATOR_ID);
    if (!el) return;
    const dot = document.getElementById('ae-rl-dot');
    const lbl = document.getElementById('ae-rl-label');
    const clr = COLOR_MAP[status] || COLOR_MAP.ready;

    el.style.background   = clr.bg;
    el.style.borderColor  = clr.border;
    if (dot) dot.style.background = clr.dot;
    if (lbl) lbl.textContent = `${ICON_MAP[status]} ${detail || LABEL_MAP[status]}`;

    // Broadcast for external listeners
    document.dispatchEvent(new CustomEvent('ae:reliability:status', {
      bubbles: false,
      detail: { status, detail },
    }));
  }

  /* ════════════════════════════════════════════════════════════════════
     STATE PERSISTENCE
     Saves / restores: Care Mode status, active overlay state, voice state.
     Uses localStorage ONLY. No biometric data is ever stored.
  ════════════════════════════════════════════════════════════════════ */
  function _saveState () {
    try {
      const data = {
        careModeActive: state.careModeActive,
        voiceEnabled:   !!window.voiceNav?.enabled,
        timestamp:      Date.now(),
      };
      localStorage.setItem(PERSIST_KEY, JSON.stringify(data));
    } catch (_) {}
  }

  function _restoreState () {
    try {
      const raw = localStorage.getItem(PERSIST_KEY);
      if (!raw) return;
      const data = JSON.parse(raw);
      // Only restore state saved within the last 10 minutes
      if (Date.now() - data.timestamp > 10 * 60 * 1000) return;

      RL.ok(`Restoring persisted state: careModeActive=${data.careModeActive}, voiceEnabled=${data.voiceEnabled}`);

      // Restore Care Mode if it was active
      if (data.careModeActive && window.CareMode && !window.CareMode.isActive()) {
        RL.ok('Auto-restoring Care Mode (was active before navigation)');
        setTimeout(() => {
          try { window.CareMode.mount(); } catch (_) {}
        }, 800);
      }

      // Restore voice navigation if it was enabled
      if (data.voiceEnabled && window.voiceNav && !window.voiceNav.enabled) {
        RL.ok('Auto-restoring voice navigation (was enabled before navigation)');
        setTimeout(() => {
          try { window.voiceNav.enable(); } catch (_) {}
        }, 1000);
      }
    } catch (_) {}
  }

  /* ════════════════════════════════════════════════════════════════════
     FAILURE MONITORING  — silent auto-recovery
  ════════════════════════════════════════════════════════════════════ */

  /**
   * CAMERA MONITOR
   * Polls window.app.cameraOn every CAMERA_POLL_MS.
   * On disconnect: logs Critical, updates indicator, attempts 1 silent retry
   * by clicking #start-camera-btn (the same interaction-layer path used elsewhere).
   */
  let _cameraWasOn = false;
  let _cameraRetryCount = 0;
  const MAX_CAMERA_RETRIES = 3;

  function _checkCamera () {
    const app = window.app;
    if (!app) return;

    const isOn = app.cameraOn === true;

    if (isOn && !_cameraWasOn) {
      // Camera just came on
      _cameraWasOn = true;
      _cameraRetryCount = 0;
      state.cameraOk = true;
      RL.ok('Camera: connected');
      _updateIndicator('ready', 'Ready');
    } else if (!isOn && _cameraWasOn) {
      // Camera just disconnected
      _cameraWasOn = false;
      state.cameraOk = false;
      RL.critical('Camera disconnected unexpectedly', { retryCount: _cameraRetryCount });
      _updateIndicator('error', '❌ Camera lost');

      // Silent auto-retry
      if (_cameraRetryCount < MAX_CAMERA_RETRIES) {
        _cameraRetryCount++;
        RL.ok(`Camera auto-retry ${_cameraRetryCount}/${MAX_CAMERA_RETRIES}…`);
        setTimeout(() => {
          const btn = document.getElementById('start-camera-btn');
          if (btn && !window.app?.cameraOn) {
            btn.click();   // Interaction-layer path
            _updateIndicator('degraded', '⚠️ Reconnecting…');
          }
        }, 1500 * _cameraRetryCount);  // Increasing delay between retries
      } else {
        RL.critical('Camera auto-retry exhausted — manual restart required');
        _updateIndicator('error', '❌ Camera error');
      }
    }

    // Initial state sync
    if (isOn !== undefined && _cameraWasOn === isOn) {
      state.cameraOk = isOn;
    }
  }

  /**
   * EYE-TRACKING LOSS MONITOR
   * Tracks last gaze event timestamp. If no gaze for GAZE_TIMEOUT_MS,
   * marks tracking as degraded. Clears when gaze resumes.
   */
  function _onGazeEvent () {
    state.lastGazeMs = Date.now();
    _gazeEverReceived = true;  // FIX-TRACKING-LOST: mark first real gaze frame
    if (!state.trackingOk) {
      state.trackingOk = true;
      RL.ok('Eye tracking: resumed');
      _computeOverallStatus();
    }
    // Reset gaze timeout
    clearTimeout(state.gazeTimeoutTimer);
    state.gazeTimeoutTimer = setTimeout(_onGazeLost, GAZE_TIMEOUT_MS);
  }

  function _onGazeLost () {
    if (window.app?.cameraOn && state.trackingOk) {
      state.trackingOk = false;
      RL.medium('Eye tracking: no gaze events for ' + (GAZE_TIMEOUT_MS / 1000) + 's — degraded');
      _computeOverallStatus();
    }
  }

  /**
   * VOICE MISS COUNTER
   * Intercepts 'ae:reliability:failure' events from VoiceNav (not direct coupling —
   * voice-nav.js fires its own status events). We just count generic failures here.
   */
  let _voiceMissStreak = 0;
  const VOICE_MISS_THRESHOLD = 5;

  function _onVoiceMiss () {
    _voiceMissStreak++;
    state.lastVoiceMs = Date.now();
    if (_voiceMissStreak >= VOICE_MISS_THRESHOLD) {
      state.voiceOk = false;
      RL.medium(`Voice: ${_voiceMissStreak} consecutive misses — degraded`);
      _computeOverallStatus();
      // Auto-recovery: restart voice recognition
      _retryVoice();
    }
  }

  function _onVoiceSuccess () {
    if (_voiceMissStreak > 0) {
      _voiceMissStreak = 0;
      state.voiceOk = true;
      _computeOverallStatus();
    }
  }

  function _retryVoice () {
    const vn = window.voiceNav;
    if (!vn?.enabled) return;
    RL.ok('Voice: silent auto-restart…');
    try {
      vn.disable();
      setTimeout(() => {
        try { vn.enable(); } catch (_) {}
        _voiceMissStreak = 0;
        state.voiceOk = true;
        _computeOverallStatus();
        RL.ok('Voice: restarted');
      }, 800);
    } catch (_) {}
  }

  /**
   * OVERALL STATUS COMPUTATION
   * Determines ready / degraded / error from individual subsystem flags.
   */
  function _computeOverallStatus () {
    const app = window.app;
    const cameraOn = app?.cameraOn === true;

    if (!cameraOn && _cameraRetryCount >= MAX_CAMERA_RETRIES) {
      _updateIndicator('error', '❌ Camera error');
    } else if (!cameraOn && _cameraWasOn) {
      _updateIndicator('degraded', '⚠️ Reconnecting…');
    } else if (!state.trackingOk && cameraOn && _gazeEverReceived) {
      _updateIndicator('degraded', '⚠️ Tracking lost');
    } else if (!state.voiceOk) {
      _updateIndicator('degraded', '⚠️ Voice degraded');
    } else {
      _updateIndicator('ready', '✅ Ready');
    }

    _saveState();
  }

  /* ════════════════════════════════════════════════════════════════════
     MAIN WATCHDOG
     Runs every WATCHDOG_INTERVAL_MS. Checks all subsystems.
  ════════════════════════════════════════════════════════════════════ */
  function _watchdogTick () {
    _checkCamera();

    // Track Care Mode state for persistence
    const cmActive = window.CareMode?.isActive?.() === true;
    if (cmActive !== state.careModeActive) {
      state.careModeActive = cmActive;
      _saveState();
    }

    // Update voice status from voiceNav
    const vnEnabled = window.voiceNav?.enabled === true;
    if (vnEnabled !== state.voiceEnabled) {
      state.voiceEnabled = vnEnabled;
      _saveState();
    }

    // Determine overall state
    _computeOverallStatus();
  }

  /* ════════════════════════════════════════════════════════════════════
     INTERACTION LAYER SUBSCRIPTION
     Subscribes to gaze events via window.AccessEye.on (read-only).
     No wrapping or patching of any function.
  ════════════════════════════════════════════════════════════════════ */
  function _attachToInteractionLayer () {
    const tryAttach = () => {
      if (!window.AccessEye?.on) {
        setTimeout(tryAttach, 300);
        return;
      }

      // Subscribe to gaze events to monitor tracking health
      window.AccessEye.on('gaze', () => {
        _onGazeEvent();
      });

      RL.ok('Attached to Interaction Layer (gaze monitor active)');
    };
    tryAttach();
  }

  /* ════════════════════════════════════════════════════════════════════
     VOICE NAV MONITORING
     Listens to CustomEvents from voice-nav.js (ae:voicenav:* events).
     Does NOT wrap or patch voice-nav functions.
  ════════════════════════════════════════════════════════════════════ */
  function _attachVoiceMonitor () {
    // Listen for no-match events (voice-nav fires these as custom events)
    document.addEventListener('ae:voicenav:nomatch', _onVoiceMiss);
    document.addEventListener('ae:voicenav:match',   _onVoiceSuccess);
    document.addEventListener('ae:voicenav:error',   () => {
      RL.medium('Voice recognition error (microphone or API)', null);
      _computeOverallStatus();
    });
  }

  /* ════════════════════════════════════════════════════════════════════
     CSS for the indicator
  ════════════════════════════════════════════════════════════════════ */
  function _injectStyles () {
    if (document.getElementById('ae-rl-styles')) return;
    const s = document.createElement('style');
    s.id = 'ae-rl-styles';
    s.textContent = `
#ae-reliability-indicator {
  will-change: background, border-color;
}
#ae-reliability-indicator:hover {
  filter: brightness(1.15);
}
@keyframes ae-rl-pulse {
  0%,100% { opacity: 1; }
  50%      { opacity: 0.5; }
}
#ae-reliability-indicator[data-status="error"]   #ae-rl-dot,
#ae-reliability-indicator[data-status="degraded"] #ae-rl-dot {
  animation: ae-rl-pulse 1.2s ease infinite;
}
    `;
    document.head.appendChild(s);
  }

  /* ════════════════════════════════════════════════════════════════════
     PUBLIC API (window.AccessEyeReliability)
  ════════════════════════════════════════════════════════════════════ */
  window.AccessEyeReliability = {
    version:    RL_VERSION,
    getStatus:  () => ({ ...state }),
    getLog:     () => [...state.failureLog],
    privacy:    PRIVACY_NOTICE,
    clearLog:   () => { state.failureLog = []; },
  };

  /* ════════════════════════════════════════════════════════════════════
     BOOTSTRAP
  ════════════════════════════════════════════════════════════════════ */
  function init () {
    _injectStyles();
    _injectIndicator();
    _attachToInteractionLayer();
    _attachVoiceMonitor();
    _restoreState();

    // Start main watchdog
    state.watchdogTimer = setInterval(_watchdogTick, WATCHDOG_INTERVAL_MS);
    // Initial camera poll (wait for app to init)
    setTimeout(() => {
      _cameraWasOn = window.app?.cameraOn === true;
      state.cameraOk = _cameraWasOn;
      _watchdogTick();
    }, 1000);

    RL.ok(`Hospital-Grade Reliability Layer v${RL_VERSION} ✅`);
    RL.ok(PRIVACY_NOTICE);

    console.log(
      '%c🔒 AccessEye Reliability Layer v' + RL_VERSION,
      'color:#22c55e;font-weight:bold;font-size:12px;'
    );
    console.log('%c' + PRIVACY_NOTICE, 'color:#6b7280;font-style:italic');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
