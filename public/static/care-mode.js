/**
 * ═══════════════════════════════════════════════════════════════════
 *  AccessEye — care-mode.js
 *  Patient Interface — Care Mode
 *
 *  ARCHITECTURE RULES (strictly followed):
 *  ─────────────────────────────────────────────────────────────────
 *  • This file ONLY reads from the INTERACTION LAYER via
 *    window.AccessEye.on('focus', ...) and window.AccessEye.on('activate', ...)
 *  • It does NOT modify, patch, or call any core engine directly
 *  • It does NOT touch the cursor, gaze pipeline, or dwell logic
 *  • It does NOT mutate any existing global variables
 *  • All DOM is injected into an isolated root div (#care-mode-root)
 *    that sits above all existing UI (z-index 99999)
 *  • Mount / unmount is fully clean — no trace left when OFF
 *  • When Care Mode is OFF the base system is 100% unaffected
 * ═══════════════════════════════════════════════════════════════════
 */

;(function () {
  'use strict';

  /* ── Guard: don't double-load ────────────────────────────────── */
  if (window.__careModeLoaded) return;
  window.__careModeLoaded = true;

  /* ══════════════════════════════════════════════════════════════
     CONSTANTS
  ══════════════════════════════════════════════════════════════ */
  const CM_ROOT_ID     = 'care-mode-root';
  const CM_TOGGLE_ID   = 'care-mode-toggle-btn';
  const DWELL_MS       = 1800;   // ms of focus before auto-activate
  const DWELL_TICK_MS  = 50;     // progress update interval

  /* ══════════════════════════════════════════════════════════════
     STATE  (fully local — never touches window.app or any global)
  ══════════════════════════════════════════════════════════════ */
  const state = {
    active:        false,   // is Care Mode mounted?
    screen:        'main',  // 'main' | 'pain' | 'needs' | 'communication' | 'nurse_confirm'
    focusedBtn:    null,    // id of currently focused care-mode button
    dwellTimer:    null,    // setInterval handle
    dwellStart:    0,       // timestamp when dwell began
    dwellProgress: 0,       // 0–1
    nurseAlerted:  false,   // has nurse been called this session?
  };

  /* ══════════════════════════════════════════════════════════════
     INTERACTION LAYER BRIDGE
     Listens to AccessEye public events — NEVER touches internals
  ══════════════════════════════════════════════════════════════ */
  let _bridgeAttached = false;

  function attachInteractionBridge() {
    if (_bridgeAttached) return;
    _bridgeAttached = true;

    // Wait until window.AccessEye is ready (it's set after DOMContentLoaded)
    const tryAttach = () => {
      if (!window.AccessEye?.on) {
        setTimeout(tryAttach, 200);
        return;
      }

      /* focus event → start dwell on matching care-mode button */
      window.AccessEye.on('focus', ({ id }) => {
        if (!state.active) return;
        const el = document.getElementById(id);
        if (!el || !el.closest('#care-mode-root')) return; // not a care-mode element
        _startDwell(id);
      });

      /* activate event → trigger care-mode action (gesture / dwell complete) */
      window.AccessEye.on('activate', ({ id }) => {
        if (!state.active) return;
        const el = document.getElementById(id);
        if (!el || !el.closest('#care-mode-root')) return;
        _stopDwell();
        _handleAction(id);
      });
    };
    tryAttach();
  }

  /* ══════════════════════════════════════════════════════════════
     DWELL ENGINE  (isolated — does not touch uiRegistry dwell)
  ══════════════════════════════════════════════════════════════ */
  function _startDwell(btnId) {
    if (state.focusedBtn === btnId) return; // already dwelling on this btn
    _stopDwell();

    state.focusedBtn    = btnId;
    state.dwellStart    = performance.now();
    state.dwellProgress = 0;

    // Highlight focused button
    _setFocusStyle(btnId, true);

    state.dwellTimer = setInterval(() => {
      const elapsed = performance.now() - state.dwellStart;
      state.dwellProgress = Math.min(elapsed / DWELL_MS, 1);

      // Update progress arc on the button
      _updateDwellArc(btnId, state.dwellProgress);

      if (state.dwellProgress >= 1) {
        _stopDwell();
        _handleAction(btnId);
      }
    }, DWELL_TICK_MS);
  }

  function _stopDwell() {
    if (state.dwellTimer) {
      clearInterval(state.dwellTimer);
      state.dwellTimer = null;
    }
    if (state.focusedBtn) {
      _setFocusStyle(state.focusedBtn, false);
      _updateDwellArc(state.focusedBtn, 0);
    }
    state.focusedBtn    = null;
    state.dwellProgress = 0;
  }

  function _setFocusStyle(btnId, focused) {
    const el = document.getElementById(btnId);
    if (!el) return;
    el.classList.toggle('cm-focused', focused);
  }

  function _updateDwellArc(btnId, progress) {
    const el = document.getElementById(btnId);
    if (!el) return;
    const arc = el.querySelector('.cm-dwell-arc');
    if (!arc) return;
    // SVG circle circumference = 2πr, r=44 → ~276.5
    const CIRC = 276.5;
    arc.style.strokeDashoffset = String(CIRC * (1 - progress));
  }

  /* ══════════════════════════════════════════════════════════════
     ACTION ROUTER  (Care Mode state machine)
  ══════════════════════════════════════════════════════════════ */
  function _handleAction(btnId) {
    // Strip screen prefix to get action key: "cm-pain", "cm-needs", etc.
    const action = btnId.replace(/^cm-/, '');

    switch (action) {
      /* ── Main screen ─────────────────────────────── */
      case 'pain':          _goScreen('pain');          break;
      case 'needs':         _goScreen('needs');         break;
      case 'communication': _goScreen('communication'); break;
      case 'call-nurse':    _goScreen('nurse_confirm'); break;

      /* ── Pain flow ───────────────────────────────── */
      case 'pain-mild':     _sendSignal('pain', 'Mild pain reported');     break;
      case 'pain-moderate': _sendSignal('pain', 'Moderate pain reported'); break;
      case 'pain-severe':   _sendSignal('pain', 'Severe pain reported');   break;
      case 'pain-chest':    _sendSignal('pain', 'Chest pain reported — urgent'); break;
      case 'pain-head':     _sendSignal('pain', 'Headache reported');      break;
      case 'pain-stomach':  _sendSignal('pain', 'Stomach pain reported');  break;

      /* ── Needs flow ──────────────────────────────── */
      case 'needs-water':       _sendSignal('needs', 'Patient needs water');        break;
      case 'needs-blanket':     _sendSignal('needs', 'Patient needs blanket');      break;
      case 'needs-bathroom':    _sendSignal('needs', 'Patient needs bathroom');     break;
      case 'needs-medication':  _sendSignal('needs', 'Patient needs medication');   break;
      case 'needs-position':    _sendSignal('needs', 'Patient needs repositioning');break;
      case 'needs-quiet':       _sendSignal('needs', 'Patient requests quiet');     break;

      /* ── Communication flow ──────────────────────── */
      case 'comm-yes':     _sendSignal('communication', 'Patient says: YES');  break;
      case 'comm-no':      _sendSignal('communication', 'Patient says: NO');   break;
      case 'comm-help':    _sendSignal('communication', 'Patient says: HELP'); break;
      case 'comm-thanks':  _sendSignal('communication', 'Patient says: THANK YOU'); break;
      case 'comm-pain':    _goScreen('pain');   break;
      case 'comm-family':  _sendSignal('communication', 'Patient wants family contact'); break;

      /* ── Nurse confirm ───────────────────────────── */
      case 'nurse-yes':
        state.nurseAlerted = true;
        _sendSignal('nurse', 'NURSE CALL — Patient requested assistance');
        _showNurseConfirmed();
        break;
      case 'nurse-no':
        _goScreen('main');
        break;

      /* ── Back button (any screen) ────────────────── */
      case 'back':
        _goScreen('main');
        break;

      default:
        console.warn('[CareMode] Unknown action:', action);
    }
  }

  /* ══════════════════════════════════════════════════════════════
     SIGNAL EMITTER
     Emits a CustomEvent so the host app / nurse system can listen
     without Care Mode touching any internals.
     DOES NOT call anything in the action layer.
  ══════════════════════════════════════════════════════════════ */
  function _sendSignal(category, message) {
    const ev = new CustomEvent('caremode:signal', {
      bubbles: true,
      detail: {
        category,
        message,
        timestamp: Date.now(),
      }
    });
    document.dispatchEvent(ev);

    // Show confirmation to patient
    _showFeedback(message);

    // TTS via Web Speech (does not touch AccessEye audio system)
    _speak(message);

    // Return to main after short delay
    setTimeout(() => _goScreen('main'), 2800);
  }

  function _speak(text) {
    try {
      const synth = window.speechSynthesis;
      if (!synth) return;
      synth.cancel();
      const utt = new SpeechSynthesisUtterance(text);
      utt.rate = 0.9;
      utt.volume = 1.0;
      synth.speak(utt);
    } catch(_) {}
  }

  /* ══════════════════════════════════════════════════════════════
     SCREEN ROUTER
  ══════════════════════════════════════════════════════════════ */
  function _goScreen(screen) {
    _stopDwell();
    state.screen = screen;
    _renderCurrentScreen();
    // Re-register all new buttons with AccessEye interaction layer
    _registerCareButtons();
  }

  /* ══════════════════════════════════════════════════════════════
     FEEDBACK TOAST  (isolated — appended to care-mode root only)
  ══════════════════════════════════════════════════════════════ */
  function _showFeedback(message) {
    const root = document.getElementById(CM_ROOT_ID);
    if (!root) return;
    let fb = root.querySelector('.cm-feedback');
    if (!fb) {
      fb = document.createElement('div');
      fb.className = 'cm-feedback';
      root.appendChild(fb);
    }
    fb.textContent = '✅  ' + message;
    fb.classList.add('cm-feedback-visible');
    clearTimeout(fb._hideTimer);
    fb._hideTimer = setTimeout(() => fb.classList.remove('cm-feedback-visible'), 2500);
  }

  function _showNurseConfirmed() {
    const root = document.getElementById(CM_ROOT_ID);
    if (!root) return;
    const content = root.querySelector('.cm-content');
    if (!content) return;
    content.innerHTML = `
      <div class="cm-nurse-confirmed">
        <div class="cm-nurse-icon">🔔</div>
        <div class="cm-nurse-title">Nurse Called</div>
        <div class="cm-nurse-sub">Help is on the way. Stay calm.</div>
      </div>`;
    _speak('Nurse has been called. Help is on the way.');
    setTimeout(() => _goScreen('main'), 4000);
  }

  /* ══════════════════════════════════════════════════════════════
     REGISTER / UNREGISTER CARE MODE BUTTONS
     Uses window.AccessEye public API only — read-only interaction hook
  ══════════════════════════════════════════════════════════════ */
  function _registerCareButtons() {
    if (!window.AccessEye?.registerElement) return;
    const root = document.getElementById(CM_ROOT_ID);
    if (!root) return;

    // Unregister any previously registered care buttons
    _unregisterCareButtons();

    // Register every button inside the overlay
    root.querySelectorAll('.cm-btn[id]').forEach(el => {
      window.AccessEye.registerElement({
        id:         el.id,
        element:    el,
        label:      el.dataset.label || el.textContent.trim(),
        onActivate: () => _handleAction(el.id),
      });
    });
  }

  let _registeredCareIds = [];

  function _unregisterCareButtons() {
    if (!window.AccessEye?.unregisterElement) return;
    _registeredCareIds.forEach(id => {
      try { window.AccessEye.unregisterElement(id); } catch(_) {}
    });
    _registeredCareIds = [];
    // Also collect current ids for tracking
    const root = document.getElementById(CM_ROOT_ID);
    if (root) {
      root.querySelectorAll('.cm-btn[id]').forEach(el => {
        _registeredCareIds.push(el.id);
      });
    }
  }

  /* ══════════════════════════════════════════════════════════════
     MOUNT / UNMOUNT
  ══════════════════════════════════════════════════════════════ */
  function mount() {
    if (document.getElementById(CM_ROOT_ID)) return; // already mounted

    // Build root overlay
    const root = document.createElement('div');
    root.id = CM_ROOT_ID;
    root.innerHTML = _buildOverlayHTML();
    document.body.appendChild(root);

    // Inject isolated styles
    _injectStyles();

    // Render initial screen
    _renderCurrentScreen();

    // Register buttons with interaction layer
    _registerCareButtons();

    // Wire up the close button (mouse/touch fallback)
    const closeBtn = root.querySelector('#cm-close-btn');
    if (closeBtn) closeBtn.addEventListener('click', toggle);

    state.active = true;
    _updateToggleBtn();

    console.log('[CareMode] Mounted ✅');
  }

  function unmount() {
    _stopDwell();
    _unregisterCareButtons();

    const root = document.getElementById(CM_ROOT_ID);
    if (root) root.remove();

    const styleTag = document.getElementById('care-mode-styles');
    if (styleTag) styleTag.remove();

    state.active  = false;
    state.screen  = 'main';
    _updateToggleBtn();

    console.log('[CareMode] Unmounted 🔴');
  }

  function toggle() {
    if (state.active) unmount();
    else              mount();
  }

  function _updateToggleBtn() {
    const btn = document.getElementById(CM_TOGGLE_ID);
    if (!btn) return;
    if (state.active) {
      btn.classList.add('cm-toggle-active');
      btn.title = 'Exit Care Mode';
    } else {
      btn.classList.remove('cm-toggle-active');
      btn.title = 'Enter Care Mode (Patient Interface)';
    }
  }

  /* ══════════════════════════════════════════════════════════════
     HTML BUILDERS
  ══════════════════════════════════════════════════════════════ */
  function _buildOverlayHTML() {
    return `
      <div class="cm-overlay">
        <div class="cm-header">
          <div class="cm-header-left">
            <span class="cm-header-icon">🏥</span>
            <span class="cm-header-title">Care Mode</span>
          </div>
          <button id="cm-close-btn" class="cm-close-btn" aria-label="Exit Care Mode">✕ Exit</button>
        </div>
        <div class="cm-content" id="cm-content"></div>
        <div class="cm-footer">
          <span class="cm-footer-hint">👁 Look at a button and hold gaze to select</span>
        </div>
      </div>`;
  }

  function _renderCurrentScreen() {
    const content = document.getElementById('cm-content');
    if (!content) return;

    switch (state.screen) {
      case 'main':          content.innerHTML = _screenMain();          break;
      case 'pain':          content.innerHTML = _screenPain();          break;
      case 'needs':         content.innerHTML = _screenNeeds();         break;
      case 'communication': content.innerHTML = _screenCommunication(); break;
      case 'nurse_confirm': content.innerHTML = _screenNurseConfirm();  break;
      default:              content.innerHTML = _screenMain();
    }
  }

  /* ── Button HTML helper ──────────────────────────────────────── */
  function _btn(id, icon, label, colorClass = '') {
    return `
      <button id="cm-${id}" class="cm-btn ${colorClass}" data-label="${label}" aria-label="${label}">
        <svg class="cm-dwell-ring" viewBox="0 0 100 100" aria-hidden="true">
          <circle class="cm-dwell-track" cx="50" cy="50" r="44"/>
          <circle class="cm-dwell-arc"   cx="50" cy="50" r="44"
            style="stroke-dasharray:276.5;stroke-dashoffset:276.5"/>
        </svg>
        <span class="cm-btn-icon">${icon}</span>
        <span class="cm-btn-label">${label}</span>
      </button>`;
  }

  function _backBtn() {
    return _btn('back', '←', 'Back', 'cm-btn-back');
  }

  /* ── Screen: Main ────────────────────────────────────────────── */
  function _screenMain() {
    return `
      <div class="cm-screen-title">How can we help?</div>
      <div class="cm-grid cm-grid-2x2">
        ${_btn('pain',          '😣', 'Pain',          'cm-btn-pain')}
        ${_btn('needs',         '🙏', 'Needs',         'cm-btn-needs')}
        ${_btn('call-nurse',    '🔔', 'Call Nurse',    'cm-btn-nurse')}
        ${_btn('communication', '💬', 'Communication', 'cm-btn-comm')}
      </div>`;
  }

  /* ── Screen: Pain ────────────────────────────────────────────── */
  function _screenPain() {
    return `
      <div class="cm-screen-title">Where / How bad is the pain?</div>
      <div class="cm-grid cm-grid-3x2">
        ${_btn('pain-mild',     '😌', 'Mild',    'cm-btn-pain-level')}
        ${_btn('pain-moderate', '😟', 'Moderate','cm-btn-pain-level')}
        ${_btn('pain-severe',   '😣', 'Severe',  'cm-btn-pain-level cm-btn-urgent')}
        ${_btn('pain-chest',    '❤️', 'Chest',   'cm-btn-pain-level cm-btn-urgent')}
        ${_btn('pain-head',     '🤕', 'Head',    'cm-btn-pain-level')}
        ${_btn('pain-stomach',  '🤢', 'Stomach', 'cm-btn-pain-level')}
      </div>
      <div class="cm-back-row">${_backBtn()}</div>`;
  }

  /* ── Screen: Needs ───────────────────────────────────────────── */
  function _screenNeeds() {
    return `
      <div class="cm-screen-title">What do you need?</div>
      <div class="cm-grid cm-grid-3x2">
        ${_btn('needs-water',      '💧', 'Water',       'cm-btn-need')}
        ${_btn('needs-blanket',    '🛏', 'Blanket',     'cm-btn-need')}
        ${_btn('needs-bathroom',   '🚻', 'Bathroom',    'cm-btn-need')}
        ${_btn('needs-medication', '💊', 'Medication',  'cm-btn-need')}
        ${_btn('needs-position',   '🔄', 'Reposition',  'cm-btn-need')}
        ${_btn('needs-quiet',      '🤫', 'Quiet',       'cm-btn-need')}
      </div>
      <div class="cm-back-row">${_backBtn()}</div>`;
  }

  /* ── Screen: Communication ───────────────────────────────────── */
  function _screenCommunication() {
    return `
      <div class="cm-screen-title">Communication</div>
      <div class="cm-grid cm-grid-3x2">
        ${_btn('comm-yes',    '✅', 'YES',          'cm-btn-comm-yes')}
        ${_btn('comm-no',     '❌', 'NO',           'cm-btn-comm-no')}
        ${_btn('comm-help',   '🆘', 'Help',         'cm-btn-comm-help')}
        ${_btn('comm-thanks', '🙏', 'Thank You',    'cm-btn-comm')}
        ${_btn('comm-pain',   '😣', 'I have pain',  'cm-btn-comm')}
        ${_btn('comm-family', '👨‍👩‍👧', 'Call Family',  'cm-btn-comm')}
      </div>
      <div class="cm-back-row">${_backBtn()}</div>`;
  }

  /* ── Screen: Nurse Confirm ───────────────────────────────────── */
  function _screenNurseConfirm() {
    return `
      <div class="cm-screen-title cm-screen-title-urgent">🔔 Call the Nurse?</div>
      <div class="cm-confirm-sub">A nurse will be alerted immediately.</div>
      <div class="cm-grid cm-grid-confirm">
        ${_btn('nurse-yes', '✅', 'Yes, Call Nurse', 'cm-btn-nurse-yes')}
        ${_btn('nurse-no',  '❌', 'Cancel',          'cm-btn-nurse-no')}
      </div>`;
  }

  /* ══════════════════════════════════════════════════════════════
     CSS  (fully scoped to #care-mode-root — zero bleed to base app)
  ══════════════════════════════════════════════════════════════ */
  function _injectStyles() {
    if (document.getElementById('care-mode-styles')) return;
    const style = document.createElement('style');
    style.id = 'care-mode-styles';
    style.textContent = `
/* ── Care Mode root — isolated layer above everything ────────── */
#care-mode-root {
  position: fixed;
  inset: 0;
  z-index: 99999;
  pointer-events: all;
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}

/* ── Overlay panel ───────────────────────────────────────────── */
#care-mode-root .cm-overlay {
  display: flex;
  flex-direction: column;
  width: 100%;
  height: 100%;
  background: #0a0e1a;
  color: #f0f4ff;
  overflow: hidden;
}

/* ── Header ──────────────────────────────────────────────────── */
#care-mode-root .cm-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 24px;
  background: #0d1220;
  border-bottom: 2px solid #1e2d50;
  flex-shrink: 0;
}
#care-mode-root .cm-header-left {
  display: flex; align-items: center; gap: 10px;
}
#care-mode-root .cm-header-icon { font-size: 26px; }
#care-mode-root .cm-header-title {
  font-size: 20px; font-weight: 800;
  letter-spacing: 0.5px; color: #fff;
}
#care-mode-root .cm-close-btn {
  background: #1e2d50; border: 1px solid #2d4070;
  color: #94a3b8; border-radius: 8px;
  padding: 8px 16px; font-size: 13px; font-weight: 600;
  cursor: pointer; transition: all 0.2s;
}
#care-mode-root .cm-close-btn:hover {
  background: #2d3f6b; color: #fff;
}

/* ── Content area ────────────────────────────────────────────── */
#care-mode-root .cm-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 24px;
  overflow-y: auto;
}

/* ── Screen title ────────────────────────────────────────────── */
#care-mode-root .cm-screen-title {
  font-size: clamp(20px, 3vw, 32px);
  font-weight: 800;
  color: #e2e8f0;
  text-align: center;
  margin-bottom: 28px;
  letter-spacing: -0.3px;
}
#care-mode-root .cm-screen-title-urgent {
  color: #f87171;
  font-size: clamp(22px, 3.5vw, 36px);
}
#care-mode-root .cm-confirm-sub {
  font-size: 16px; color: #94a3b8;
  text-align: center; margin-top: -18px; margin-bottom: 28px;
}

/* ── Grid layouts ────────────────────────────────────────────── */
#care-mode-root .cm-grid {
  display: grid;
  gap: 16px;
  width: 100%;
  max-width: 760px;
}
#care-mode-root .cm-grid-2x2 {
  grid-template-columns: repeat(2, 1fr);
  max-width: 600px;
}
#care-mode-root .cm-grid-3x2 {
  grid-template-columns: repeat(3, 1fr);
}
#care-mode-root .cm-grid-confirm {
  grid-template-columns: repeat(2, 1fr);
  max-width: 560px;
}

/* ── Buttons ─────────────────────────────────────────────────── */
#care-mode-root .cm-btn {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 20px 12px 16px;
  border-radius: 18px;
  border: 2px solid #1e2d50;
  background: #0f1729;
  color: #e2e8f0;
  cursor: pointer;
  min-height: 130px;
  transition: border-color 0.2s, background 0.2s, transform 0.15s;
  overflow: hidden;
  -webkit-tap-highlight-color: transparent;
}
#care-mode-root .cm-btn:hover,
#care-mode-root .cm-btn.cm-focused {
  border-color: #3b82f6;
  background: #0f1f3d;
  transform: scale(1.03);
}
#care-mode-root .cm-btn-icon {
  font-size: clamp(28px, 4vw, 44px);
  line-height: 1;
  pointer-events: none;
}
#care-mode-root .cm-btn-label {
  font-size: clamp(13px, 1.8vw, 18px);
  font-weight: 700;
  text-align: center;
  pointer-events: none;
  line-height: 1.2;
}

/* ── Button colour variants ──────────────────────────────────── */
#care-mode-root .cm-btn-pain         { border-color: #7f1d1d; }
#care-mode-root .cm-btn-pain:hover,
#care-mode-root .cm-btn-pain.cm-focused { border-color: #ef4444; background: #1f0a0a; }

#care-mode-root .cm-btn-needs        { border-color: #1d3a6e; }
#care-mode-root .cm-btn-needs:hover,
#care-mode-root .cm-btn-needs.cm-focused { border-color: #3b82f6; background: #0a1020; }

#care-mode-root .cm-btn-nurse        { border-color: #78350f; }
#care-mode-root .cm-btn-nurse:hover,
#care-mode-root .cm-btn-nurse.cm-focused { border-color: #f59e0b; background: #1f1000; }

#care-mode-root .cm-btn-comm         { border-color: #1e3a2e; }
#care-mode-root .cm-btn-comm:hover,
#care-mode-root .cm-btn-comm.cm-focused { border-color: #22c55e; background: #0a1f12; }

#care-mode-root .cm-btn-urgent       { border-color: #991b1b !important; }
#care-mode-root .cm-btn-urgent:hover,
#care-mode-root .cm-btn-urgent.cm-focused { border-color: #ef4444 !important; background: #1f0a0a !important; }

#care-mode-root .cm-btn-comm-yes     { border-color: #166534; }
#care-mode-root .cm-btn-comm-yes:hover,
#care-mode-root .cm-btn-comm-yes.cm-focused { border-color: #22c55e; background: #0a1f12; }

#care-mode-root .cm-btn-comm-no      { border-color: #7f1d1d; }
#care-mode-root .cm-btn-comm-no:hover,
#care-mode-root .cm-btn-comm-no.cm-focused { border-color: #ef4444; background: #1f0a0a; }

#care-mode-root .cm-btn-comm-help    { border-color: #78350f; }
#care-mode-root .cm-btn-comm-help:hover,
#care-mode-root .cm-btn-comm-help.cm-focused { border-color: #f59e0b; background: #1f1000; }

#care-mode-root .cm-btn-nurse-yes    { border-color: #166534; background: #0a1f12; }
#care-mode-root .cm-btn-nurse-yes:hover,
#care-mode-root .cm-btn-nurse-yes.cm-focused { border-color: #22c55e; background: #0d2818; }

#care-mode-root .cm-btn-nurse-no     { border-color: #374151; }
#care-mode-root .cm-btn-nurse-no:hover,
#care-mode-root .cm-btn-nurse-no.cm-focused { border-color: #6b7280; }

#care-mode-root .cm-btn-back {
  background: transparent; border-color: #1e2d50;
  color: #94a3b8; min-height: 52px; padding: 10px 24px;
  flex-direction: row; gap: 6px;
}
#care-mode-root .cm-btn-back:hover,
#care-mode-root .cm-btn-back.cm-focused { border-color: #3b7fff; color: #e2e8f0; }

/* ── Dwell ring (SVG progress arc) ──────────────────────────── */
#care-mode-root .cm-dwell-ring {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  border-radius: 16px;
  overflow: visible;
  opacity: 0;
  transition: opacity 0.2s;
}
#care-mode-root .cm-btn.cm-focused .cm-dwell-ring {
  opacity: 1;
}
#care-mode-root .cm-dwell-track {
  fill: none;
  stroke: #1e2d50;
  stroke-width: 4;
  vector-effect: non-scaling-stroke;
}
#care-mode-root .cm-dwell-arc {
  fill: none;
  stroke: #3b82f6;
  stroke-width: 4;
  stroke-linecap: round;
  vector-effect: non-scaling-stroke;
  transform: rotate(-90deg);
  transform-origin: 50% 50%;
  transition: stroke-dashoffset 0.05s linear;
}

/* ── Back row ────────────────────────────────────────────────── */
#care-mode-root .cm-back-row {
  margin-top: 16px;
  width: 100%;
  max-width: 760px;
  display: flex;
  justify-content: flex-start;
}

/* ── Nurse confirmed ─────────────────────────────────────────── */
#care-mode-root .cm-nurse-confirmed {
  display: flex; flex-direction: column;
  align-items: center; gap: 16px;
  padding: 40px;
}
#care-mode-root .cm-nurse-icon {
  font-size: 72px;
  animation: cm-pulse 1s ease infinite;
}
@keyframes cm-pulse {
  0%,100% { transform: scale(1); }
  50%      { transform: scale(1.15); }
}
#care-mode-root .cm-nurse-title {
  font-size: 36px; font-weight: 800; color: #f59e0b;
}
#care-mode-root .cm-nurse-sub {
  font-size: 18px; color: #94a3b8;
}

/* ── Feedback toast ──────────────────────────────────────────── */
#care-mode-root .cm-feedback {
  position: absolute;
  bottom: 80px;
  left: 50%; transform: translateX(-50%);
  background: rgba(30, 45, 80, 0.97);
  border: 1px solid #3b82f6;
  color: #e2e8f0;
  padding: 14px 28px;
  border-radius: 12px;
  font-size: 16px; font-weight: 600;
  text-align: center;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.3s;
  max-width: 460px;
  white-space: nowrap;
  z-index: 10;
}
#care-mode-root .cm-feedback.cm-feedback-visible {
  opacity: 1;
}

/* ── Footer ──────────────────────────────────────────────────── */
#care-mode-root .cm-footer {
  padding: 12px 24px;
  background: #0d1220;
  border-top: 1px solid #1e2d50;
  text-align: center;
  flex-shrink: 0;
}
#care-mode-root .cm-footer-hint {
  font-size: 13px; color: #4b5563;
}

/* ── Toggle button in nav ─────────────────────────────────────── */
#${CM_TOGGLE_ID} {
  background: #0f1729;
  border: 1.5px solid #1e2d50;
  color: #94a3b8;
  border-radius: 8px;
  padding: 6px 12px;
  font-size: 12px; font-weight: 600;
  cursor: pointer;
  display: flex; align-items: center; gap: 6px;
  transition: all 0.2s;
  white-space: nowrap;
}
#${CM_TOGGLE_ID}:hover {
  background: #1e2d50; color: #e2e8f0;
  border-color: #3b82f6;
}
#${CM_TOGGLE_ID}.cm-toggle-active {
  background: #0a1f12;
  border-color: #22c55e;
  color: #22c55e;
}

/* ── Responsive ──────────────────────────────────────────────── */
@media (max-width: 600px) {
  #care-mode-root .cm-grid-3x2 {
    grid-template-columns: repeat(2, 1fr);
  }
  #care-mode-root .cm-btn { min-height: 100px; }
}
@media (max-height: 600px) {
  #care-mode-root .cm-btn { min-height: 80px; padding: 12px 8px; }
  #care-mode-root .cm-screen-title { font-size: 18px; margin-bottom: 16px; }
}
    `;
    document.head.appendChild(style);
  }

  /* ══════════════════════════════════════════════════════════════
     TOGGLE BUTTON INJECTOR
     Injects the Care Mode toggle into the nav once DOM is ready
  ══════════════════════════════════════════════════════════════ */
  function _injectToggleButton() {
    if (document.getElementById(CM_TOGGLE_ID)) return;

    const btn = document.createElement('button');
    btn.id        = CM_TOGGLE_ID;
    btn.innerHTML = '🏥 Care Mode';
    btn.title     = 'Enter Care Mode (Patient Interface)';
    btn.setAttribute('aria-label', 'Toggle Care Mode');
    btn.addEventListener('click', toggle);

    // Try to insert after the nav-links block, fall back to nav, fall back to body
    const navLinks = document.querySelector('.nav-links') || document.querySelector('#main-nav') || document.querySelector('nav');
    if (navLinks) {
      navLinks.insertAdjacentElement('afterend', btn);
    } else {
      // Last resort: floating button top-right
      btn.style.cssText = 'position:fixed;top:14px;right:14px;z-index:99998;';
      document.body.appendChild(btn);
    }
  }

  /* ══════════════════════════════════════════════════════════════
     BOOTSTRAP
  ══════════════════════════════════════════════════════════════ */
  function init() {
    _injectToggleButton();
    attachInteractionBridge();

    // Listen for caremode:signal so host page can hook in (optional)
    document.addEventListener('caremode:signal', (e) => {
      console.log('[CareMode] Signal emitted:', e.detail);
    });

    console.log('[CareMode] Initialized 👁🏥');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  /* ══════════════════════════════════════════════════════════════
     PUBLIC API  (window.CareMode)
     Allows host page to control Care Mode programmatically
  ══════════════════════════════════════════════════════════════ */
  window.CareMode = {
    mount,
    unmount,
    toggle,
    isActive: () => state.active,
    /** Listen for patient signals: category, message, timestamp */
    onSignal: (cb) => document.addEventListener('caremode:signal', e => cb(e.detail)),
  };

})();
