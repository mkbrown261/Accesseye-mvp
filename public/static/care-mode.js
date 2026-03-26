/**
 * ═══════════════════════════════════════════════════════════════════
 *  AccessEye — care-mode.js  v2.2
 *  Patient Interface — Care Mode
 *
 *  ARCHITECTURE RULES (strictly followed):
 *  ─────────────────────────────────────────────────────────────────
 *  • Reads from interaction layer ONLY via:
 *      window.AccessEye.on('focus', ...)
 *      window.AccessEye.on('activate', ...)
 *      window.AccessEye.on('gaze', ...)
 *      window.AccessEye.registerElement(...)
 *      window.AccessEye.unregisterElement(...)
 *
 *  • Camera init: triggered by clicking #start-camera-btn — the
 *    exact same path used when the button is gaze-activated by the
 *    base system (uiRegistry routes el.click()).  window.app.cameraOn
 *    is read (read-only) to detect camera state. Zero action-layer
 *    calls.
 *
 *  • Does NOT modify, patch, or call any engine internals
 *  • Does NOT mutate any existing global variables / listeners
 *  • All DOM lives in isolated #care-mode-root (z-index 99999)
 *  • Mount / unmount is fully clean — no trace left when OFF
 *  • Default Mode is 100 % unaffected when Care Mode is OFF
 * ═══════════════════════════════════════════════════════════════════
 */

;(function () {
  'use strict';

  /* ── Guard: don't double-load ─────────────────────────────────── */
  if (window.__careModeLoaded) return;
  window.__careModeLoaded = true;

  /* ════════════════════════════════════════════════════════════════
     CONSTANTS
  ════════════════════════════════════════════════════════════════ */
  const CM_ROOT_ID   = 'care-mode-root';
  const CM_TOGGLE_ID = 'care-mode-toggle-btn';
  const DWELL_MS     = 1600;   // ms gaze must hold to activate
  const DWELL_TICK   = 40;     // progress-arc refresh interval (ms)
  const CIRC         = 276.5;  // SVG arc circumference for r=44
  const CM_VERSION   = '3.0';  // Phase 3+: camera permission, gaze alignment, larger buttons

  /* ════════════════════════════════════════════════════════════════
     LOCAL STATE  — never touches window.app or any global
  ════════════════════════════════════════════════════════════════ */
  const state = {
    active:       false,
    screen:       'main',
    focusedBtn:   null,
    dwellTimer:   null,
    dwellStart:   0,
    nurseAlerted: false,
    // Phase 3: camera permission state
    camPermission: null,  // null | 'granted' | 'denied' | 'prompt'
  };

  /* ════════════════════════════════════════════════════════════════
     LIFECYCLE LOGGER  — full audit trail as requested
  ════════════════════════════════════════════════════════════════ */
  const LC = {
    tag:  '[CareMode]',
    step (n, msg) { console.log(`${this.tag} [LC-${n}] ${msg}`); },
    warn (msg)    { console.warn(`${this.tag} ⚠  ${msg}`); },
    ok   (msg)    { console.log(`${this.tag} ✅ ${msg}`); },
    err  (msg)    { console.error(`${this.tag} ❌ ${msg}`); },
  };

  /* ════════════════════════════════════════════════════════════════
     PART 1 — CAMERA INITIALIZATION  (via interaction layer only)
  ════════════════════════════════════════════════════════════════ */

  /**
   * Phase 3: Request camera permission FIRST, then trigger the camera.
   * Uses the Permissions API where available, falls back to getUserMedia probe.
   * On denial, shows an informative in-overlay message.
   */
  async function _requestCameraPermission () {
    LC.step('P3-1', 'Checking/requesting camera permission…');

    // Check existing permission status without requesting
    if (navigator.permissions) {
      try {
        const status = await navigator.permissions.query({ name: 'camera' });
        LC.step('P3-1', `Permissions API: camera state = ${status.state}`);
        state.camPermission = status.state;

        if (status.state === 'denied') {
          _showPermissionDenied();
          return;
        }
        // 'granted' or 'prompt' → proceed to camera start
        _ensureCameraRunning();
        return;
      } catch (_) {
        // Permissions API not available for 'camera' on this browser — probe directly
        LC.warn('P3-1: Permissions API unavailable, probing getUserMedia');
      }
    }

    // Fallback: probe getUserMedia (shows browser permission dialog if needed)
    try {
      _updateStatusBar('waiting');
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      // Permission granted — stop the probe stream immediately (app.js will open its own)
      stream.getTracks().forEach(t => t.stop());
      LC.ok('P3-1: Camera permission granted via getUserMedia probe');
      state.camPermission = 'granted';
      _ensureCameraRunning();
    } catch (err) {
      LC.err(`P3-1: Camera permission denied/error: ${err.name}`);
      state.camPermission = 'denied';
      _showPermissionDenied();
    }
  }

  function _showPermissionDenied () {
    _updateStatusBar('denied');
    const bar = document.getElementById('cm-status-bar');
    if (bar) bar.querySelector('.cm-status-txt').textContent =
      '🚫 Camera permission denied — eye tracking unavailable';
    LC.warn('Camera permission denied — showing in-overlay fallback guidance');
    // Show a dismissable hint inside the content area
    const root = document.getElementById(CM_ROOT_ID);
    if (!root) return;
    const existing = root.querySelector('.cm-cam-denied');
    if (existing) return;
    const hint = document.createElement('div');
    hint.className = 'cm-cam-denied';
    hint.innerHTML = `
      <div class="cm-cam-denied-icon">📷</div>
      <div class="cm-cam-denied-title">Camera Access Required</div>
      <div class="cm-cam-denied-body">
        Eye tracking needs camera access.<br>
        Please allow camera permission in your browser settings,
        then reload the page.
      </div>
      <div class="cm-cam-denied-hint">
        <strong>You can still use Care Mode</strong> — touch or click the buttons below.
      </div>`;
    // Insert above content
    const header = root.querySelector('.cm-header');
    if (header) header.insertAdjacentElement('afterend', hint);
    else root.querySelector('.cm-overlay')?.prepend(hint);
  }

  /**
   * Checks whether the camera is already running via the authoritative
   * flag window.app.cameraOn (read-only access — no mutation).
   * If not running:
   *   1. Request camera permission directly (graceful fallback if denied)
   *   2. Trigger start via #start-camera-btn.click() — the interaction-layer path
   */
  function _ensureCameraRunning () {
    LC.step(1, 'Checking camera state via window.app.cameraOn…');

    /* ── LC-1: check authoritative flag ─────────────────────────── */
    const alreadyOn = window.app?.cameraOn === true;
    LC.step(1, `window.app.cameraOn = ${alreadyOn}`);

    if (alreadyOn) {
      LC.step(2, 'Camera already running — skipping init, running lifecycle audit');
      _updateStatusBar('tracking');
      _lifecycleAudit();
      return;
    }

    /* ── Phase 3: Request camera permission proactively ─────────── */
    _updateStatusBar('waiting');
    LC.step('1b', 'Requesting camera permission via getUserMedia (Phase 3)…');

    if (navigator.mediaDevices?.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then(stream => {
          /* Permission granted — release the test stream immediately, let
             app.js own the real camera; just trigger the start button.    */
          stream.getTracks().forEach(t => t.stop());
          LC.ok('LC-1b: camera permission GRANTED — proceeding to start button');
          _triggerCameraStartBtn();
        })
        .catch(err => {
          /* Permission denied or hardware error */
          const denied = (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError');
          LC.warn(`LC-1b: camera permission ${denied ? 'DENIED' : 'ERROR'} — ${err.name}`);
          if (denied) {
            _updateStatusBar('hint');
            _showCameraHint('Camera permission denied. Allow camera access in browser settings.');
            _showPermissionBanner();
          } else {
            // Hardware or constraint error — still try the start button
            _triggerCameraStartBtn();
          }
        });
    } else {
      // Browser doesn't support getUserMedia — attempt direct trigger
      LC.warn('LC-1b: navigator.mediaDevices.getUserMedia not available — trying button');
      _triggerCameraStartBtn();
    }
  }

  function _showPermissionBanner () {
    const bar = document.getElementById('cm-status-bar');
    if (!bar) return;
    const txt = bar.querySelector('.cm-status-txt');
    if (txt) txt.textContent = '📷 Camera access denied — touch buttons work normally';
    // Insert a visible denied banner above the content
    const existing = document.getElementById('cm-cam-denied-banner');
    if (existing) return;
    const banner = document.createElement('div');
    banner.id = 'cm-cam-denied-banner';
    banner.className = 'cm-cam-denied';
    banner.innerHTML = `
      <span class="cm-cam-denied-icon">📷</span>
      <span class="cm-cam-denied-title">Camera Access Denied</span>
      <span class="cm-cam-denied-body">Allow camera in browser settings to enable eye tracking.<br>You can still use Care Mode by tapping/clicking buttons.</span>
      <span class="cm-cam-denied-hint">🔒 All data is processed locally. Nothing is stored or transmitted.</span>`;
    const content = document.getElementById('cm-content');
    if (content?.parentElement) {
      content.parentElement.insertBefore(banner, content);
    }
  }

  function _triggerCameraStartBtn () {
    /* ── LC-2: locate #start-camera-btn ─────────────────────────── */
    const startBtn = document.getElementById('start-camera-btn');
    if (!startBtn) {
      LC.warn('LC-2: #start-camera-btn not found — user is not on Demo page');
      _showCameraHint('Navigate to Live Demo page and start the camera first.');
      return;
    }

    /* ── Optionally navigate to demo tab first ───────────────────── */
    const demoPage = document.getElementById('page-demo');
    const isDemoVisible = demoPage
      ? (demoPage.style.display !== 'none' && !demoPage.classList.contains('hidden'))
      : false;

    if (!isDemoVisible) {
      LC.step('2a', 'Demo page not visible — navigating via nav click (interaction layer)');
      const navDemo = document.querySelector('[data-page="demo"]') ||
                      document.getElementById('nav-demo');
      if (navDemo) {
        navDemo.click();  // existing nav click handler — interaction layer path
        LC.ok('Nav-demo clicked');
      } else {
        LC.warn('LC-2a: demo nav button not found');
      }
      setTimeout(_doTriggerStart, 450);   // wait for page transition
    } else {
      _doTriggerStart();
    }
  }

  function _doTriggerStart () {
    const startBtn = document.getElementById('start-camera-btn');
    if (!startBtn) {
      LC.err('LC-3: #start-camera-btn still not found after navigation');
      _showCameraHint('Open the Live Demo tab and press Start Camera.');
      return;
    }

    LC.step(3, 'Triggering camera via #start-camera-btn.click() — interaction layer path');
    startBtn.click();   // ← mirrors what gaze-activate does (uiRegistry → el.click())

    /* ── Poll until window.app.cameraOn is true (max 3 s) ───────── */
    let attempts = 0;
    const poll = setInterval(() => {
      attempts++;
      const running = window.app?.cameraOn === true;
      LC.step(`3-poll-${attempts}`, `window.app.cameraOn=${running}`);

      if (running) {
        clearInterval(poll);
        LC.step(4, 'Camera confirmed ON — proceeding to lifecycle audit');
        _lifecycleAudit();
        return;
      }
      if (attempts >= 30) {   // 3 s
        clearInterval(poll);
        LC.warn('LC-3: camera did not start within 3 s — audit will proceed anyway');
        _lifecycleAudit();
      }
    }, 100);
  }

  /* ════════════════════════════════════════════════════════════════
     PART 2 — LIFECYCLE AUDIT
     Verifies every stage of the pipeline with a debug log per step.
  ════════════════════════════════════════════════════════════════ */
  function _lifecycleAudit () {

    /* LC-4 Camera on flag */
    const cameraOn = window.app?.cameraOn === true;
    cameraOn
      ? LC.ok('LC-4 Camera ON (window.app.cameraOn)')
      : LC.warn('LC-4 Camera OFF — eye tracking will use simulation mode');

    /* LC-5 Video stream */
    const video = document.getElementById('demo-video');
    if (video?.srcObject) {
      const tracks = video.srcObject.getTracks?.() ?? [];
      LC.ok(`LC-5 Video stream active — ${tracks.length} track(s), readyState=${video.readyState}`);
    } else {
      LC.warn('LC-5 video.srcObject not yet set (may still be initialising)');
    }

    /* LC-6 Tracking loop — mpController */
    if (window.app?.mpController) {
      LC.ok('LC-6 mpController present — requestAnimationFrame loop active');
    } else {
      LC.warn('LC-6 mpController not found — simulation / Phase-2 mode likely active');
    }

    /* LC-7 Gaze data */
    if (window.app?.gazeEngine) {
      const g = window.app.gazeEngine.smoothGaze;
      LC.ok(`LC-7 gazeEngine present — last smoothGaze=(${g?.x?.toFixed(3)},${g?.y?.toFixed(3)})`);
    } else {
      LC.warn('LC-7 gazeEngine not found on window.app');
    }

    /* LC-8 Interaction layer */
    if (window.AccessEye?.on) {
      LC.ok('LC-8 window.AccessEye.on available — events routing to Care Mode');
    } else {
      LC.err('LC-8 window.AccessEye.on NOT available — bridge will retry');
    }

    /* LC-9 Registered buttons */
    LC.ok(`LC-9 Registered buttons: ${_registeredIds.length} — [${_registeredIds.join(', ')}]`);

    /* LC-10 DOM isolation */
    const root = document.getElementById(CM_ROOT_ID);
    LC.ok(`LC-10 #care-mode-root in DOM: ${!!root} — z-index 99999, pointer-events all`);

    /* LC-11 Gaze hover + dwell */
    LC.step(11, 'Gaze hover + dwell: bridge listens to AccessEye focus/activate/gaze events ' +
               `— dwell threshold ${DWELL_MS} ms`);

    /* Update status bar */
    _updateStatusBar(cameraOn ? 'tracking' : 'hint');
    LC.ok('━━ Lifecycle audit complete ━━');
  }

  /* ════════════════════════════════════════════════════════════════
     INTERACTION LAYER BRIDGE
     Subscribes to AccessEye public events once (guarded by flag).
     Every callback is a no-op when Care Mode is off.
  ════════════════════════════════════════════════════════════════ */
  let _bridgeAttached = false;

  function _attachBridge () {
    if (_bridgeAttached) return;
    _bridgeAttached = true;

    const tryAttach = () => {
      if (!window.AccessEye?.on) { setTimeout(tryAttach, 200); return; }
      LC.ok('Interaction-layer bridge attached');

      /* focus → start dwell arc */
      window.AccessEye.on('focus', ({ id }) => {
        if (!state.active) return;
        const el = document.getElementById(id);
        if (!el?.closest('#' + CM_ROOT_ID)) return;
        LC.step('IL-F', `focus → ${id}`);
        _startDwell(id);
      });

      /* activate → onActivate already called by uiRegistry; this handles
         the case where the base dwell fires first (gesture-mode).        */
      window.AccessEye.on('activate', ({ id }) => {
        if (!state.active) return;
        const el = document.getElementById(id);
        if (!el?.closest('#' + CM_ROOT_ID)) return;
        LC.step('IL-A', `activate → ${id}`);
        _stopDwell();
        _handleAction(id);
      });

      /*
       * gaze → cancel dwell when gaze leaves the button.
       * Phase 3: Increased tolerance for screen corners (40px) where
       * gaze accuracy degrades. Also uses visualViewport for correct
       * pixel mapping on high-DPR/zoomed displays.
       */
      window.AccessEye.on('gaze', ({ screen }) => {
        if (!state.active || !state.focusedBtn || !screen) return;
        const el = document.getElementById(state.focusedBtn);
        if (!el) return;
        const r    = el.getBoundingClientRect();
        // Use visualViewport dimensions for correct mapping on zoomed/mobile
        const vw   = window.visualViewport?.width  ?? window.innerWidth;
        const vh   = window.visualViewport?.height ?? window.innerHeight;
        const px   = screen.x * vw;
        const py   = screen.y * vh;
        // Phase 3: larger tolerance near screen edges (corners have worst accuracy)
        const isCorner = (px < 120 || px > vw - 120) && (py < 120 || py > vh - 120);
        const pad  = isCorner ? 52 : 40; // 40px general, 52px in corners
        const inBounds = px >= r.left - pad && px <= r.right  + pad &&
                         py >= r.top  - pad && py <= r.bottom + pad;
        if (!inBounds) _stopDwell();
      });
    };

    tryAttach();
  }

  /* ════════════════════════════════════════════════════════════════
     DWELL ENGINE  — isolated, does not touch uiRegistry dwell
     Provides the visible progress-arc animation.
     NOTE: onActivate callback on each registered element is the
     primary trigger; this engine ensures smooth visual feedback.
  ════════════════════════════════════════════════════════════════ */
  function _startDwell (btnId) {
    if (state.focusedBtn === btnId) return; // already dwelling here
    _stopDwell();

    state.focusedBtn = btnId;
    state.dwellStart = performance.now();
    _setFocusCls(btnId, true);

    state.dwellTimer = setInterval(() => {
      const pct = Math.min((performance.now() - state.dwellStart) / DWELL_MS, 1);
      _updateArc(btnId, pct);
      if (pct >= 1) {
        _stopDwell();
        _handleAction(btnId);
      }
    }, DWELL_TICK);
  }

  function _stopDwell () {
    if (state.dwellTimer) { clearInterval(state.dwellTimer); state.dwellTimer = null; }
    if (state.focusedBtn) {
      _setFocusCls(state.focusedBtn, false);
      _updateArc(state.focusedBtn, 0);
    }
    state.focusedBtn = null;
  }

  function _setFocusCls (id, on) {
    document.getElementById(id)?.classList.toggle('cm-focused', on);
  }

  function _updateArc (id, p) {
    const arc = document.getElementById(id)?.querySelector('.cm-dwell-arc');
    if (arc) arc.style.strokeDashoffset = String(CIRC * (1 - p));
  }

  /* ════════════════════════════════════════════════════════════════
     REGISTER / UNREGISTER with AccessEye interaction layer
  ════════════════════════════════════════════════════════════════ */
  let _registeredIds = [];

  function _registerButtons () {
    if (!window.AccessEye?.registerElement) return;
    _unregisterButtons();   // clean slate before each screen render

    const root = document.getElementById(CM_ROOT_ID);
    if (!root) return;

    root.querySelectorAll('.cm-btn[id]').forEach(el => {
      _registeredIds.push(el.id);
      window.AccessEye.registerElement({
        id:         el.id,
        element:    el,
        label:      el.dataset.label || el.textContent.trim().slice(0, 30),
        onActivate: () => {
          LC.step('REG-OA', `onActivate fired for ${el.id}`);
          _handleAction(el.id);
        },
      });
    });

    LC.ok(`Registered ${_registeredIds.length} Care Mode buttons with interaction layer`);
  }

  function _unregisterButtons () {
    if (!window.AccessEye?.unregisterElement) return;
    _registeredIds.forEach(id => {
      try { window.AccessEye.unregisterElement(id); } catch (_) {}
    });
    _registeredIds = [];
  }

  /* ════════════════════════════════════════════════════════════════
     ACTION STATE MACHINE  — all flows contained locally
  ════════════════════════════════════════════════════════════════ */
  function _handleAction (btnId) {
    const action = btnId.replace(/^cm-/, '');
    LC.ok(`Action dispatched: ${action}`);

    switch (action) {
      /* ── Main menu ────────────────────────────────────────────── */
      case 'pain':          _goScreen('pain');          break;
      case 'needs':         _goScreen('needs');         break;
      case 'communication': _goScreen('communication'); break;
      case 'call-nurse':    _goScreen('nurse_confirm'); break;

      /* ── Pain sub-items ───────────────────────────────────────── */
      case 'pain-mild':     _signal('pain', 'Mild pain reported');            break;
      case 'pain-moderate': _signal('pain', 'Moderate pain reported');        break;
      case 'pain-severe':   _signal('pain', 'Severe pain — needs attention'); break;
      case 'pain-chest':    _signal('pain', 'Chest pain — URGENT');           break;
      case 'pain-head':     _signal('pain', 'Headache reported');             break;
      case 'pain-stomach':  _signal('pain', 'Stomach pain reported');         break;

      /* ── Needs sub-items ──────────────────────────────────────── */
      case 'needs-water':      _signal('needs', 'Patient needs water');         break;
      case 'needs-blanket':    _signal('needs', 'Patient needs a blanket');     break;
      case 'needs-bathroom':   _signal('needs', 'Patient needs the bathroom');  break;
      case 'needs-medication': _signal('needs', 'Patient needs medication');    break;
      case 'needs-position':   _signal('needs', 'Patient needs repositioning'); break;
      case 'needs-quiet':      _signal('needs', 'Patient requests quiet');      break;

      /* ── Communication sub-items ──────────────────────────────── */
      case 'comm-yes':    _signal('communication', 'Patient says: YES');            break;
      case 'comm-no':     _signal('communication', 'Patient says: NO');             break;
      case 'comm-help':   _signal('communication', 'Patient says: HELP');           break;
      case 'comm-thanks': _signal('communication', 'Patient says: THANK YOU');      break;
      case 'comm-pain':   _goScreen('pain');                                        break;
      case 'comm-family': _signal('communication', 'Patient wants family contact'); break;

      /* ── Nurse call ───────────────────────────────────────────── */
      case 'nurse-yes':
        state.nurseAlerted = true;
        _signal('nurse', 'NURSE CALL — Patient needs assistance');
        _showNurseConfirmed();
        break;
      case 'nurse-no':
        _goScreen('main');
        break;

      /* ── Navigation ───────────────────────────────────────────── */
      case 'back':
        _goScreen('main');
        break;

      default:
        LC.warn('Unknown action: ' + action);
    }
  }

  /* ════════════════════════════════════════════════════════════════
     SIGNAL EMITTER — CustomEvent only, no action-layer contact
  ════════════════════════════════════════════════════════════════ */
  function _signal (category, message) {
    document.dispatchEvent(new CustomEvent('caremode:signal', {
      bubbles: true,
      detail: { category, message, timestamp: Date.now() },
    }));
    _showFeedback(message);
    _speak(message);
    setTimeout(() => _goScreen('main'), 2800);
  }

  function _speak (text) {
    try {
      const s = window.speechSynthesis;
      if (!s) return;
      s.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 0.88; u.volume = 1;
      s.speak(u);
    } catch (_) {}
  }

  /* ════════════════════════════════════════════════════════════════
     SCREEN ROUTER
  ════════════════════════════════════════════════════════════════ */
  function _goScreen (screen) {
    _stopDwell();
    state.screen = screen;
    _render();
    _registerButtons();
  }

  function _render () {
    const content = document.getElementById('cm-content');
    if (!content) return;
    switch (state.screen) {
      case 'main':          content.innerHTML = _sMain();         break;
      case 'pain':          content.innerHTML = _sPain();         break;
      case 'needs':         content.innerHTML = _sNeeds();        break;
      case 'communication': content.innerHTML = _sComm();         break;
      case 'nurse_confirm': content.innerHTML = _sNurseConfirm(); break;
      default:              content.innerHTML = _sMain();
    }
  }

  /* ════════════════════════════════════════════════════════════════
     BUTTON BUILDER
     Each button carries its own SVG dwell-progress arc.
     Part 3: sizing via CSS (see _injectStyles).
  ════════════════════════════════════════════════════════════════ */
  function _btn (id, icon, label, cls = '') {
    return `
      <button id="cm-${id}" class="cm-btn ${cls}" data-label="${label}" aria-label="${label}">
        <svg class="cm-dwell-ring" viewBox="0 0 100 100" aria-hidden="true">
          <circle class="cm-dwell-track" cx="50" cy="50" r="44"/>
          <circle class="cm-dwell-arc"   cx="50" cy="50" r="44"
            style="stroke-dasharray:${CIRC};stroke-dashoffset:${CIRC}"/>
        </svg>
        <span class="cm-btn-icon">${icon}</span>
        <span class="cm-btn-label">${label}</span>
      </button>`;
  }

  function _backBtn () { return _btn('back', '←', 'Back', 'cm-btn-back'); }

  /* ════════════════════════════════════════════════════════════════
     SCREENS
  ════════════════════════════════════════════════════════════════ */
  function _sMain () {
    return `
      <p class="cm-title">How can we help?</p>
      <div class="cm-grid cm-2x2">
        ${_btn('pain',          '😣', 'Pain',          'cm-red')}
        ${_btn('needs',         '🙏', 'Needs',         'cm-blue')}
        ${_btn('call-nurse',    '🔔', 'Call Nurse',    'cm-amber')}
        ${_btn('communication', '💬', 'Communication', 'cm-green')}
      </div>`;
  }

  function _sPain () {
    return `
      <p class="cm-title">Where / How bad?</p>
      <div class="cm-grid cm-2x3">
        ${_btn('pain-mild',     '😌', 'Mild',    'cm-pain-lvl')}
        ${_btn('pain-moderate', '😟', 'Moderate','cm-pain-lvl')}
        ${_btn('pain-severe',   '😣', 'Severe',  'cm-pain-lvl cm-urgent')}
        ${_btn('pain-chest',    '❤️', 'Chest',   'cm-pain-lvl cm-urgent')}
        ${_btn('pain-head',     '🤕', 'Head',    'cm-pain-lvl')}
        ${_btn('pain-stomach',  '🤢', 'Stomach', 'cm-pain-lvl')}
      </div>
      <div class="cm-back-row">${_backBtn()}</div>`;
  }

  function _sNeeds () {
    return `
      <p class="cm-title">What do you need?</p>
      <div class="cm-grid cm-2x3">
        ${_btn('needs-water',      '💧', 'Water',      'cm-blue')}
        ${_btn('needs-blanket',    '🛏', 'Blanket',    'cm-blue')}
        ${_btn('needs-bathroom',   '🚻', 'Bathroom',   'cm-blue')}
        ${_btn('needs-medication', '💊', 'Medication', 'cm-blue')}
        ${_btn('needs-position',   '🔄', 'Reposition', 'cm-blue')}
        ${_btn('needs-quiet',      '🤫', 'Quiet',      'cm-blue')}
      </div>
      <div class="cm-back-row">${_backBtn()}</div>`;
  }

  function _sComm () {
    return `
      <p class="cm-title">Communication</p>
      <div class="cm-grid cm-2x3">
        ${_btn('comm-yes',    '✅', 'YES',          'cm-green cm-large-txt')}
        ${_btn('comm-no',     '❌', 'NO',           'cm-red   cm-large-txt')}
        ${_btn('comm-help',   '🆘', 'Help',         'cm-amber')}
        ${_btn('comm-thanks', '🙏', 'Thank You',    'cm-green')}
        ${_btn('comm-pain',   '😣', 'I have pain',  'cm-red')}
        ${_btn('comm-family', '👪', 'Call Family',  'cm-blue')}
      </div>
      <div class="cm-back-row">${_backBtn()}</div>`;
  }

  function _sNurseConfirm () {
    return `
      <p class="cm-title cm-urgent-title">🔔 Call the Nurse?</p>
      <p class="cm-sub">A nurse will be alerted right away.</p>
      <div class="cm-grid cm-confirm">
        ${_btn('nurse-yes', '✅', 'Yes — Call Nurse', 'cm-green cm-large-txt')}
        ${_btn('nurse-no',  '❌', 'Cancel',           'cm-grey')}
      </div>`;
  }

  /* ════════════════════════════════════════════════════════════════
     STATUS BAR
  ════════════════════════════════════════════════════════════════ */
  function _updateStatusBar (st) {
    const bar = document.getElementById('cm-status-bar');
    if (!bar) return;
    const map = {
      tracking: { dot: '#22c55e', text: '👁 Eye tracking active'                            },
      waiting:  { dot: '#f59e0b', text: '⏳ Starting camera…'                               },
      hint:     { dot: '#ef4444', text: '📷 Go to Live Demo → Start Camera to track'        },
      denied:   { dot: '#ef4444', text: '🚫 Camera permission denied — touch to select'    },
      permission: { dot: '#f59e0b', text: '📷 Requesting camera permission…'                },
    };
    const s = map[st] || map.waiting;
    const dot = bar.querySelector('.cm-status-dot');
    const txt = bar.querySelector('.cm-status-txt');
    if (dot) { dot.style.background = s.dot; dot.style.animationPlayState = st === 'tracking' ? 'paused' : 'running'; }
    if (txt) txt.textContent = s.text;
  }

  function _showCameraHint (msg) {
    _updateStatusBar('hint');
    const bar = document.getElementById('cm-status-bar');
    if (bar) bar.querySelector('.cm-status-txt').textContent = '📷 ' + msg;
  }

  /* ════════════════════════════════════════════════════════════════
     FEEDBACK TOAST (inside overlay only)
  ════════════════════════════════════════════════════════════════ */
  function _showFeedback (msg) {
    const root = document.getElementById(CM_ROOT_ID);
    if (!root) return;
    let fb = root.querySelector('.cm-feedback');
    if (!fb) {
      fb = document.createElement('div');
      fb.className = 'cm-feedback';
      root.appendChild(fb);
    }
    fb.textContent = '✅  ' + msg;
    fb.classList.add('cm-fb-on');
    clearTimeout(fb._t);
    fb._t = setTimeout(() => fb.classList.remove('cm-fb-on'), 2600);
  }

  function _showNurseConfirmed () {
    const c = document.getElementById('cm-content');
    if (!c) return;
    c.innerHTML = `
      <div class="cm-nurse-done">
        <div class="cm-nurse-bell">🔔</div>
        <div class="cm-nurse-title">Nurse Called</div>
        <div class="cm-nurse-sub">Help is on the way. Stay calm.</div>
      </div>`;
    _speak('Nurse has been called. Help is on the way.');
    setTimeout(() => _goScreen('main'), 4000);
  }

  /* ════════════════════════════════════════════════════════════════
     MOUNT / UNMOUNT
  ════════════════════════════════════════════════════════════════ */
  function mount () {
    if (document.getElementById(CM_ROOT_ID)) return;
    LC.ok('Mounting Care Mode…');

    _injectStyles();

    const root = document.createElement('div');
    root.id = CM_ROOT_ID;
    root.innerHTML = `
      <div class="cm-overlay">
        <div class="cm-header">
          <div class="cm-hdr-left">
            <span class="cm-hdr-icon">🏥</span>
            <span class="cm-hdr-title">Care Mode</span>
          </div>
          <div id="cm-status-bar" class="cm-status-bar">
            <span class="cm-status-dot"></span>
            <span class="cm-status-txt">Initialising…</span>
          </div>
          <button id="cm-close-btn" class="cm-close-btn" aria-label="Exit Care Mode">✕ Exit</button>
        </div>
        <div class="cm-content" id="cm-content"></div>
        <div class="cm-footer">
          <span class="cm-footer-hint">
            👁&nbsp; Look at a button and hold your gaze to select
            &nbsp;·&nbsp; ${DWELL_MS / 1000}s dwell time
          </span>
        </div>
      </div>`;
    document.body.appendChild(root);

    root.querySelector('#cm-close-btn').addEventListener('click', toggle);

    state.active = true;
    _updateToggle();
    _updateStatusBar('waiting');
    _goScreen('main');

    /* Phase 3: request permission first, then ensure camera running */
    setTimeout(() => {
      // Check if already running
      if (window.app?.cameraOn === true) {
        _ensureCameraRunning();
      } else {
        _updateStatusBar('permission');
        _requestCameraPermission();
      }
    }, 300);

    LC.ok('Care Mode mounted ✅');
  }

  function unmount () {
    LC.ok('Unmounting Care Mode…');
    _stopDwell();
    _unregisterButtons();
    document.getElementById(CM_ROOT_ID)?.remove();
    document.getElementById('care-mode-styles')?.remove();
    state.active = false;
    state.screen = 'main';
    _updateToggle();
    LC.ok('Care Mode unmounted 🔴');
  }

  function toggle () {
    state.active ? unmount() : mount();
  }

  function _updateToggle () {
    const btn = document.getElementById(CM_TOGGLE_ID);
    if (!btn) return;
    btn.classList.toggle('cm-toggle-on', state.active);
    btn.title = state.active ? 'Exit Care Mode' : 'Enter Care Mode (Patient Interface)';
  }

  /* ════════════════════════════════════════════════════════════════
     TOGGLE BUTTON INJECTOR
     Injects a single button into the nav — nothing else changed.
     Part 3: button is ≈35% larger than v2.0 (see CSS below).
  ════════════════════════════════════════════════════════════════ */
  function _injectToggle () {
    if (document.getElementById(CM_TOGGLE_ID)) return;
    const btn = document.createElement('button');
    btn.id        = CM_TOGGLE_ID;
    btn.innerHTML = '🏥 Care Mode';
    btn.title     = 'Enter Care Mode (Patient Interface)';
    btn.setAttribute('aria-label', 'Toggle Care Mode');
    btn.addEventListener('click', toggle);

    const anchor = document.querySelector('.nav-links') ||
                   document.querySelector('#main-nav')  ||
                   document.querySelector('nav');
    if (anchor) {
      anchor.insertAdjacentElement('afterend', btn);
    } else {
      btn.style.cssText = 'position:fixed;top:14px;right:14px;z-index:99998;';
      document.body.appendChild(btn);
    }
  }

  /* ════════════════════════════════════════════════════════════════
     CSS — every rule scoped to #care-mode-root or #CM_TOGGLE_ID
     so there is zero bleed into the base app.
  ════════════════════════════════════════════════════════════════ */
  function _injectStyles () {
    if (document.getElementById('care-mode-styles')) return;
    const s = document.createElement('style');
    s.id = 'care-mode-styles';
    s.textContent = `

/* ───── Isolation root ──────────────────────────────────────── */
#care-mode-root {
  position: fixed; inset: 0;
  z-index: 99999;
  pointer-events: all;
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}
#care-mode-root *, #care-mode-root *::before, #care-mode-root *::after {
  box-sizing: border-box; margin: 0; padding: 0;
}

/* ───── Overlay shell ───────────────────────────────────────── */
#care-mode-root .cm-overlay {
  display: flex; flex-direction: column;
  width: 100%; height: 100%;
  background: #07090f;
  color: #f0f4ff; overflow: hidden;
}

/* ───── Header ──────────────────────────────────────────────── */
#care-mode-root .cm-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 28px;
  background: #0b0f1c;
  border-bottom: 2px solid #1a2540;
  flex-shrink: 0; gap: 16px;
}
#care-mode-root .cm-hdr-left { display: flex; align-items: center; gap: 10px; }
#care-mode-root .cm-hdr-icon { font-size: 28px; }
#care-mode-root .cm-hdr-title {
  font-size: 22px; font-weight: 800; color: #fff; letter-spacing: 0.3px;
}
#care-mode-root .cm-close-btn {
  background: #1a2540; border: 1px solid #2a3a5e;
  color: #8899bb; border-radius: 8px;
  padding: 10px 20px; font-size: 14px; font-weight: 600;
  cursor: pointer; transition: all 0.2s; flex-shrink: 0;
}
#care-mode-root .cm-close-btn:hover { background: #2a3a5e; color: #fff; }

/* ───── Status bar ──────────────────────────────────────────── */
#care-mode-root .cm-status-bar {
  display: flex; align-items: center; gap: 8px;
  background: #0d1220; border: 1px solid #1a2540;
  border-radius: 20px; padding: 6px 14px;
  flex: 1; max-width: 360px;
}
#care-mode-root .cm-status-dot {
  width: 10px; height: 10px; border-radius: 50%;
  background: #f59e0b; flex-shrink: 0;
  animation: cm-blink 1.4s ease infinite;
}
@keyframes cm-blink { 0%,100%{opacity:1} 50%{opacity:.35} }
#care-mode-root .cm-status-txt {
  font-size: 13px; color: #8899bb; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis;
}

/* ───── Content area ────────────────────────────────────────── */
#care-mode-root .cm-content {
  flex: 1;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  padding: 20px 28px;
  overflow-y: auto; gap: 0;
}

/* ───── Screen title ────────────────────────────────────────── */
#care-mode-root .cm-title {
  font-size: clamp(22px, 3.2vw, 38px);
  font-weight: 800; color: #e8edf8;
  text-align: center; margin-bottom: 20px;
  letter-spacing: -0.3px; line-height: 1.2;
}
#care-mode-root .cm-urgent-title {
  color: #f87171; font-size: clamp(24px, 3.8vw, 42px);
}
#care-mode-root .cm-sub {
  font-size: clamp(14px, 1.8vw, 18px); color: #6b7a9a;
  text-align: center; margin-top: -12px; margin-bottom: 24px;
}

/* ───── GRID LAYOUTS ────────────────────────────────────────── */
#care-mode-root .cm-grid {
  display: grid; gap: 18px;
  width: 100%; max-width: 920px;
}
/* 2×2 main screen — Phase 3: buttons ~40-45% viewport height */
#care-mode-root .cm-2x2 {
  grid-template-columns: repeat(2, 1fr);
  grid-template-rows: repeat(2, minmax(0, 1fr));
  height: min(84vh, 760px);
  max-width: 880px;
}
/* 2×3 pain / needs / comm — Phase 3: slightly taller */
#care-mode-root .cm-2x3 {
  grid-template-columns: repeat(3, 1fr);
  grid-template-rows: repeat(2, minmax(0, 1fr));
  height: min(76vh, 650px);
}
/* confirm — 2 wide buttons */
#care-mode-root .cm-confirm {
  grid-template-columns: repeat(2, 1fr);
  grid-template-rows: 1fr;
  height: min(44vh, 340px);
  max-width: 780px;
}

/* ═══ BUTTONS — Phase 3: +15-20% larger from v2.1 for better eye-targeting ═══
   v2.1 baseline: padding 44px/28px, icon clamp(40,6vw,68), label clamp(20,2.6vw,30)
   v2.2 target:   padding 52px/32px, icon clamp(46,7vw,78), label clamp(22,3vw,34)
   • Each main-menu button ≈ 38-45% viewport height
   • Hit area expanded: min-height enforced per grid cell
   • High contrast: border 3px → 3.5px, radius 22 → 24px             */
#care-mode-root .cm-btn {
  position: relative;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 18px;
  /* Phase 3: +18% vertical padding, +14% horizontal */
  padding: 52px 32px 48px;
  border-radius: 24px;
  border: 3.5px solid #1a2540;
  background: #0d1220;
  color: #e8edf8;
  cursor: pointer;
  width: 100%; height: 100%;
  transition: border-color 0.18s, background 0.18s,
              transform 0.15s, box-shadow 0.18s;
  overflow: hidden;
  -webkit-tap-highlight-color: transparent;
  font-family: inherit;
  /* Ensure minimum hit area for eye tracking */
  min-height: 120px;
}
#care-mode-root .cm-btn:hover,
#care-mode-root .cm-btn.cm-focused {
  transform: scale(1.04);
  box-shadow: 0 0 0 5px #3b82f644, 0 10px 36px #00000060;
}

/* Phase 3: icon — clamp from 46px (small) → 78px (wide) — ~15% increase */
#care-mode-root .cm-btn-icon {
  font-size: clamp(46px, 7vw, 78px);
  line-height: 1; pointer-events: none;
  filter: drop-shadow(0 2px 6px #00000066);
}
/* Phase 3: label — clamp from 22px → 34px — ~15% increase */
#care-mode-root .cm-btn-label {
  font-size: clamp(22px, 3vw, 34px);
  font-weight: 800; text-align: center;
  pointer-events: none; line-height: 1.2;
  letter-spacing: 0.2px;
}
/* YES / NO / Confirm — even larger */
#care-mode-root .cm-btn.cm-large-txt .cm-btn-label {
  font-size: clamp(26px, 3.6vw, 40px);
}

/* ───── Colour variants ─────────────────────────────────────── */
#care-mode-root .cm-red   { border-color: #7f1d1d; }
#care-mode-root .cm-red:hover, #care-mode-root .cm-red.cm-focused
  { border-color: #ef4444; background: #1c0808;
    box-shadow: 0 0 0 4px #ef444430, 0 8px 32px #00000060; }

#care-mode-root .cm-blue  { border-color: #1d3a6e; }
#care-mode-root .cm-blue:hover, #care-mode-root .cm-blue.cm-focused
  { border-color: #3b82f6; background: #090f20;
    box-shadow: 0 0 0 4px #3b82f630, 0 8px 32px #00000060; }

#care-mode-root .cm-amber { border-color: #78350f; }
#care-mode-root .cm-amber:hover, #care-mode-root .cm-amber.cm-focused
  { border-color: #f59e0b; background: #180e00;
    box-shadow: 0 0 0 4px #f59e0b30, 0 8px 32px #00000060; }

#care-mode-root .cm-green { border-color: #14532d; }
#care-mode-root .cm-green:hover, #care-mode-root .cm-green.cm-focused
  { border-color: #22c55e; background: #071510;
    box-shadow: 0 0 0 4px #22c55e30, 0 8px 32px #00000060; }

#care-mode-root .cm-grey  { border-color: #374151; }
#care-mode-root .cm-grey:hover, #care-mode-root .cm-grey.cm-focused
  { border-color: #6b7280; background: #111827;
    box-shadow: 0 0 0 4px #6b728030, 0 8px 32px #00000060; }

#care-mode-root .cm-pain-lvl { border-color: #4b1d1d; }
#care-mode-root .cm-pain-lvl:hover, #care-mode-root .cm-pain-lvl.cm-focused
  { border-color: #f87171; background: #160808;
    box-shadow: 0 0 0 4px #f8717130, 0 8px 32px #00000060; }

#care-mode-root .cm-urgent { border-color: #991b1b !important; }
#care-mode-root .cm-urgent:hover, #care-mode-root .cm-urgent.cm-focused
  { border-color: #ef4444 !important; background: #200808 !important;
    box-shadow: 0 0 0 6px #ef444440, 0 8px 32px #00000080 !important; }

/* ───── Dwell ring (SVG arc) ────────────────────────────────── */
#care-mode-root .cm-dwell-ring {
  position: absolute; inset: 0;
  width: 100%; height: 100%;
  pointer-events: none; overflow: visible;
  opacity: 0; transition: opacity 0.2s;
}
#care-mode-root .cm-btn.cm-focused .cm-dwell-ring { opacity: 1; }
#care-mode-root .cm-dwell-track {
  fill: none; stroke: #1a2540; stroke-width: 6;
  vector-effect: non-scaling-stroke;
}
#care-mode-root .cm-dwell-arc {
  fill: none; stroke: #60a5fa; stroke-width: 6;
  stroke-linecap: round; vector-effect: non-scaling-stroke;
  transform: rotate(-90deg); transform-origin: 50% 50%;
  transition: stroke-dashoffset ${DWELL_TICK}ms linear;
  filter: drop-shadow(0 0 4px #3b82f6);
}

/* ───── Back row ────────────────────────────────────────────── */
#care-mode-root .cm-back-row {
  margin-top: 14px; width: 100%; max-width: 920px;
}
#care-mode-root .cm-btn-back {
  flex-direction: row; gap: 8px;
  min-height: 56px; height: auto; padding: 14px 28px;
  border-color: #1a2540; background: transparent; color: #6b7a9a;
  font-size: clamp(14px, 1.8vw, 18px); width: auto;
}
#care-mode-root .cm-btn-back:hover,
#care-mode-root .cm-btn-back.cm-focused
  { border-color: #3b82f6; color: #e8edf8; background: #090f20; }

/* ───── Nurse confirmed ─────────────────────────────────────── */
#care-mode-root .cm-nurse-done {
  display: flex; flex-direction: column;
  align-items: center; gap: 20px; padding: 48px 24px;
}
#care-mode-root .cm-nurse-bell {
  font-size: clamp(64px, 10vw, 96px);
  animation: cm-ring 0.9s ease infinite;
}
@keyframes cm-ring {
  0%,100% { transform: rotate(0deg); }
  20%     { transform: rotate(-18deg); }
  40%     { transform: rotate(18deg); }
  60%     { transform: rotate(-10deg); }
  80%     { transform: rotate(10deg); }
}
#care-mode-root .cm-nurse-title {
  font-size: clamp(28px, 4vw, 48px); font-weight: 800; color: #f59e0b;
}
#care-mode-root .cm-nurse-sub {
  font-size: clamp(16px, 2vw, 22px); color: #94a3b8;
}

/* ───── Feedback toast ──────────────────────────────────────── */
#care-mode-root .cm-feedback {
  position: absolute; bottom: 72px;
  left: 50%; transform: translateX(-50%);
  background: rgba(20,32,60,0.97);
  border: 1.5px solid #3b82f6;
  color: #e8edf8; padding: 16px 32px;
  border-radius: 14px; font-size: 17px; font-weight: 700;
  text-align: center; pointer-events: none;
  opacity: 0; transition: opacity 0.3s;
  max-width: 480px; white-space: nowrap; z-index: 10;
  box-shadow: 0 4px 24px #3b82f640;
}
#care-mode-root .cm-feedback.cm-fb-on { opacity: 1; }

/* ───── Footer ──────────────────────────────────────────────── */
#care-mode-root .cm-footer {
  padding: 12px 28px;
  background: #0b0f1c;
  border-top: 1px solid #1a2540;
  text-align: center; flex-shrink: 0;
}
#care-mode-root .cm-footer-hint {
  font-size: 13px; color: #374151; letter-spacing: 0.2px;
}

/* ═══ Nav toggle button — Part 3: ≈35 % larger than v2.0 ═══
   Old:  padding 7px 14px, font-size 12px
   New:  padding 10px 20px, font-size 15px               */
#${CM_TOGGLE_ID} {
  background: #0d1220; border: 2px solid #1a2540;
  color: #8899bb; border-radius: 9px;
  /* ↓ ~35% bigger */
  padding: 10px 20px;
  font-size: 15px; font-weight: 700;
  cursor: pointer; display: flex; align-items: center; gap: 7px;
  transition: all 0.2s; white-space: nowrap;
  line-height: 1;
}
#${CM_TOGGLE_ID}:hover {
  background: #1a2540; color: #e8edf8; border-color: #3b82f6;
}
#${CM_TOGGLE_ID}.cm-toggle-on {
  background: #071510; border-color: #22c55e; color: #22c55e;
}

/* ───── Responsive ──────────────────────────────────────────── */
@media (max-width: 700px) {
  #care-mode-root .cm-2x3 {
    grid-template-columns: repeat(2, 1fr);
    height: min(76vh, 580px);
  }
  #care-mode-root .cm-confirm { grid-template-columns: 1fr; height: auto; }
  #care-mode-root .cm-confirm .cm-btn { min-height: 120px; height: auto; }
  #care-mode-root .cm-btn-icon { font-size: clamp(36px, 9vw, 58px); }
  #care-mode-root .cm-btn { padding: 36px 20px 32px; }
}
@media (max-height: 640px) {
  #care-mode-root .cm-2x2 { height: min(82vh, 520px); }
  #care-mode-root .cm-2x3 { height: min(74vh, 460px); }
  #care-mode-root .cm-btn { padding: 28px 18px 24px; gap: 12px; }
  #care-mode-root .cm-title { font-size: 19px; margin-bottom: 12px; }
}

/* ───── Camera permission denied banner ─────────────────────── */
#care-mode-root .cm-cam-denied {
  display: flex; flex-direction: column;
  align-items: center; gap: 8px;
  background: #1a0808; border: 1.5px solid #7f1d1d;
  border-radius: 12px; padding: 14px 24px;
  margin: 0 28px; text-align: center; flex-shrink: 0;
}
#care-mode-root .cm-cam-denied-icon  { font-size: 28px; }
#care-mode-root .cm-cam-denied-title {
  font-size: 16px; font-weight: 800; color: #f87171;
}
#care-mode-root .cm-cam-denied-body  { font-size: 13px; color: #9ca3af; line-height: 1.5; }
#care-mode-root .cm-cam-denied-hint  {
  font-size: 13px; color: #60a5fa; font-style: italic;
}

    `;
    document.head.appendChild(s);
  }

  /* ════════════════════════════════════════════════════════════════
     BOOTSTRAP
  ════════════════════════════════════════════════════════════════ */
  function init () {
    _injectToggle();
    _attachBridge();
    document.addEventListener('caremode:signal', e =>
      LC.ok('Signal dispatched: ' + JSON.stringify(e.detail)));
    LC.ok(`Care Mode v${CM_VERSION} initialised 👁🏥`);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  /* ════════════════════════════════════════════════════════════════
     PUBLIC API
  ════════════════════════════════════════════════════════════════ */
  window.CareMode = {
    mount,
    unmount,
    toggle,
    isActive:  () => state.active,
    onSignal:  cb => document.addEventListener('caremode:signal', e => cb(e.detail)),
    runAudit:  _lifecycleAudit,  // expose for manual testing in console
    version:   CM_VERSION,
  };

})();
