# AccessEye MVP

**Hands-free, eye-tracking browser interface — deployed on Cloudflare Pages**

- **Production:** https://accesseye-mvp.pages.dev
- **GitHub:** https://github.com/mkbrown261/Accesseye-mvp

---

## What It Does

AccessEye lets you control a computer entirely with your eyes — no hands, no mouse, no touch. It runs 100 % in the browser using your webcam. No data ever leaves the device; all processing is local.

Look at any button or link and dwell on it for ~350 ms to activate it. An optional voice layer lets you speak commands ("click", "scroll down", "show clickable items") as a parallel input channel. A gesture layer lets you trigger actions with facial movements (lip-tap, tongue-out, custom trained gestures).

**Target users:** people with motor impairments, ALS, spinal cord injury, or any condition that makes traditional input devices inaccessible.

---

## Feature Summary

| Feature | Status |
|---|---|
| Real-time eye tracking (MediaPipe FaceMesh, 478 landmarks) | ✅ Live |
| Quality-gated 9-point calibration | ✅ Live |
| Phase 2 — binocular iris + head-pose + pupil fusion | ✅ Live |
| Phase 3 — One-Euro filter, IVT, PACE, adaptive dwell | ✅ Live |
| Snap-To engine — gravity-model target assist | ✅ Live |
| Accuracy Engine — gain remap, drift correction, center gravity | ✅ Live |
| Voice Navigation + Intent Fusion | ✅ Live |
| Gesture Studio — facial gesture detection + custom training | ✅ Live |
| Accessibility Control Mode (WCAG 2.1 AA / ADA / Section 508) | ✅ Live |
| AI Intent Prediction (OpenAI-backed, server-side) | ✅ Live |

---

## Architecture Overview

```
Browser (all client-side)
├── app.js                Phase 1 — MediaPipe wiring, CalibrationEngine v8,
│                         CalibrationUI, GazeEngine, UIElementRegistry, dwell
├── phase2-engine.js      Phase 2 — HybridGazeEngine, HeadPoseEstimator,
│                         TemporalStabilizer, DynamicCalibrationEngine,
│                         MicroSaccadeFilter, GazeConfidenceScorer,
│                         IntentPredictionEngine, GazeBenchmark
├── phase2-init.js        Phase 2 bootstrap — camera patching, UI wiring,
│                         intent-engine toggle, benchmark controls
├── phase3-engine.js      Phase 3 — OneEuroFilter, IVTSaccadeDetector,
│                         AdaptiveDwellTimer, PACERecalibrator,
│                         SmoothPursuitCalibrator, CalibrationValidator,
│                         HeadFreeStabilizer
├── phase3-init.js        Phase 3 bootstrap — dwell presets, smooth-pursuit
│                         calibration UI, post-calib validation
├── snap-engine.js        Snap-To Engine — TargetPredictor, AdaptiveGazeLearner,
│                         SnapToEngine (gravity-model soft attractor)
├── accuracy-engine.js    Accuracy Engine v3 — GazeGainRemapper,
│                         CenterGravity, GravitySnapEngine,
│                         PerSessionDriftCorrector
├── voice-nav.js          Voice Navigation + Intent Fusion — Web Speech API,
│                         NavigationElementList, VoiceNavigationController,
│                         show-clickable overlay, focus-on, all 40+ commands
├── gesture-studio.js     Gesture Studio — FacialGestureEngine (lip-tap,
│                         tongue-out), CustomGestureRecorder/Recognizer,
│                         GestureStudio lifecycle manager
├── a11y-mode.js          Accessibility Control Mode (ACM) — WCAG 2.1 AA/AAA,
│                         ADA Title III, Section 508 — gaze-dwell focus loop,
│                         skip-nav, live region, dictation, hint overlay
└── a11y-logger.js        Interaction logger — timestamped CSV + PDF export
                          of all ACM modality events

Edge (Cloudflare Pages / Workers)
└── src/index.tsx         Hono app — serves HTML shell, /api/intent endpoint
```

---

## Full Gaze Pipeline (per frame)

```
Webcam (30 fps, 640×480)
  ↓
MediaPipe FaceMesh (refineLandmarks = true, 478 pts)
  ↓
HeadPoseEstimator  ← 6-DOF Euler angles from 6 stable anchor points
  ↓
HybridGazeEngine
  ├── iris offset signal  (binocular centroid, EMA-stabilised eye span)
  ├── head-pose signal    (yaw/pitch compensation, gated on |headMag| > 15°)
  └── pupil boundary signal (centroid relative to eye centre)
       ↓ confidence-weighted fusion
  ↓
GazeConfidenceScorer  ← brightness, occlusion, glare, landmark quality
  ↓
[Phase 3] OneEuroFilter  (minCutoff 1 Hz, β 0.007)  ← inserted by P3 init
  ↓
TemporalStabilizer  (Kalman R=0.005, Q=0.0001 + EMA α=0.22 + trimmed mean)
  ↓
MicroSaccadeFilter  (12 px / 200 ms fixation stability window)
  ↓
DynamicCalibrationEngine.applyBiasCorrection  ← slow EMA drift offset (±12%)
  ↓
CalibrationEngine.mapGaze
  (degree-3 polynomial ridge regression, λ=0.01, 10 terms, iris-only model)
  ↓ [GazeGainRemapper patch, accuracy-engine]
  (power-law remap u′ = sign(u)×|u|^0.70 — expands centre, compresses edges)
  ↓
AccuracyEngine patches (_updateGazeCursor)
  ├── GravitySnapEngine  (Grossman & Balakrishnan gravity model)
  ├── CenterGravity      (idle return pull, 0.8%/frame)
  └── PerSessionDriftCorrector  (fixation-centroid comparison, ±2.5% max)
  ↓
app._updateGazeCursor  → pixel position (lastScreenX, lastScreenY)
  ↓
SnapToEngine.update  (optional) → snapped pixel position
  ↓
UIElementRegistry.updateGaze → dwell check → 'activate' event
  ↓
[Phase 3 AdaptiveDwellTimer] → IVT-gated dwell (Fast / Normal / Accessible / Extended)
```

---

## Calibration (v8 — quality-gated)

**How it works:**

1. Open the app → click the **Calibrate** tab in the top mode bar
2. Click **Start Calibration**
3. For each of the 9 dots (4 corners, 4 inner ring, centre):
   - 🔵 **Blue** — move your eyes to the dot
   - 🟠 **Amber** — gaze is locking on — hold still
   - 🟢 **Green** — locked, sampling — the ring fills automatically
   - ✨ **Flash + shrink** — point captured, advances automatically
4. After all 9 points the polynomial model is built. The cursor appears.

**Signal used for calibration:** iris-only offset (head-pose and pupil signals deliberately excluded during calibration to avoid adding fusion noise while the user is stationary).

**Calibration model specs:**

| Property | Value |
|---|---|
| Points | 9 (4 corners @ 8%/92%, 4 inner @ 25%/75%, centre) |
| Regression | Degree-3 polynomial ridge, λ = 0.01, 10 basis terms |
| Corner weight | 4× vs 1× interior |
| Gaze normalisation | Observed range + 18% padding for edge extrapolation |
| Quality gate | 0.45 threshold; 8 consecutive lock-frames before sampling |
| Samples per point | 40 |
| Storage | localStorage key `accesseye_calib` (version 8) |

**Tips:**
- Sit 50–70 cm from the screen, face roughly centred in the camera frame
- For corner dots, move your eyes only — small head movement is fine
- If a dot stays amber, blink once to reset then re-fixate
- Recalibrate any time via the Calibrate tab or voice command "calibrate"

---

## Snap-To Engine

Snap-To is an optional cursor-assist layer that pulls the gaze cursor toward the nearest interactive element when you look close enough.

**Three modules:**

| Module | Role |
|---|---|
| `TargetPredictor` | Scores candidates by distance, size, element type, and interaction history |
| `AdaptiveGazeLearner` | Builds a per-user profile from every dwell event; auto-tunes snap distance, dwell time, smoothing, and prediction weight |
| `SnapToEngine` | Smooth-interpolates cursor to winning target; emits snap/release events |

**Controls (Live Demo → Snap-To panel):**
- On/Off toggle
- Auto-dwell toggle
- Snap threshold distance (px)
- Dwell time (ms)
- Cursor smoothing (%)
- Prediction weight (%)

---

## Voice Navigation

Voice navigation runs via the Web Speech API (Chrome/Edge). Enable it with the microphone toggle in the Phase 2 panel.

**How it works:**
1. Utterance is transcribed
2. Raw lowercase text is checked against multi-word ACTION_COMMANDS first (longest match wins)
3. Tokenised words are checked for single-word action commands
4. Embedded action + element combos are tried ("click home", "open demo")
5. Named element matching against the live DOM scan
6. If nothing matches — "No match" status displayed

**Command categories:**

| Category | Example commands |
|---|---|
| Discovery | "show clickable items", "hide clickable", "what can i click" |
| Focus | "focus on [name]", "next item", "previous item" |
| Navigation | "home", "demo", "architecture", "docs", "studio" |
| Mode | "calibrate", "gaze mode", "mouse mode" |
| Camera | "start camera", "restart", "stop" |
| Scrolling | "scroll up", "scroll down", "scroll to top", "stop scrolling" |
| Zoom | "zoom in", "zoom out", "reset zoom" |
| Interaction | "click", "press", "double click", "right click" |
| Control recovery | "pause control", "resume control", "reset cursor", "exit mode" |
| Editing | "select all", "copy", "paste", "cut", "focus search" |
| Voice control | "pause voice", "resume voice" |
| Media | "play", "pause", "stop" |

**"Show Clickable Items"** highlights every interactive element with a numbered cyan badge and outline ring. Auto-clears after 8 seconds or on "hide clickable items".

---

## Gesture Studio

Facial gesture control, accessible from the **Gesture Studio** tab.

**Built-in gestures:**

| Gesture | Action |
|---|---|
| Lip-tap (open-close-open mouth quickly) | Scroll up |
| Tongue-out (hold) | Scroll down continuously |
| Smile (stretch ≥ 38% of face width) | Configurable |
| Blow | Configurable |

**Custom gestures:**
1. Click **New Gesture** in Gesture Studio
2. Name it and assign an action (click, scroll, navigate, run custom script, etc.)
3. Click **Record** — hold your chosen face position for 2–4 seconds
4. The profile is saved to localStorage and fires in real-time

**Actions available for custom gestures:** click focused element, scroll up/down, navigate to page, open settings, run arbitrary JavaScript.

---

## Accessibility Control Mode (ACM)

ACM is a WCAG 2.1 AA/AAA + ADA Title III + Section 508 compliance layer that sits entirely on top of the gaze pipeline without modifying it.

**Enable:** ACM toggle button in the Phase 2 panel.

**Features:**
- Gaze-dwell focus loop (10 fps — CPU-lightweight)
- Skip-navigation link (WCAG 2.4.1)
- Screen-reader live region announcements (WCAG 4.1.3)
- Focus outlines and ARIA role support
- Voice command integration (speaks element labels aloud)
- Keyboard Tab/Enter/Space event tracking
- Dwell time control (300–2000 ms slider)
- Real-time stat counters: gaze, voice, keyboard, intent events
- **CSV export** — timestamped log of every interaction
- **PDF export** — same log formatted for reporting/compliance

---

## Accuracy Engine (v3)

Four additive post-processing patches applied after `CalibrationEngine.mapGaze`, before the cursor pixel position is written:

| Module | What it does |
|---|---|
| **GazeGainRemapper** (ACC.3) | Power-law remap (γ=0.70) expands centre region ~18%, compresses edges; shrinks centre dead-zone from ~20% → ~8% of screen |
| **CenterGravity** (ACC.4) | Gentle 0.8%/frame return pull when cursor is idle at screen edge |
| **GravitySnapEngine** (ACC.1) | Gravity-model attractor (Grossman & Balakrishnan 2005) — 25–40% mis-selection reduction vs distance-only snap |
| **PerSessionDriftCorrector** (ACC.2) | Watches 50-fixation sliding window; infers and corrects systematic drift up to ±2.5% screen over ~60 s |

---

## Dynamic Calibration (Micro-Recalibration)

The `DynamicCalibrationEngine` (Phase 2) continuously refines the calibration model during normal use — no explicit recalibration step needed.

**Two mechanisms:**

1. **Interaction-triggered (click-based):** Every time a UI element is activated by dwell or gesture, the current raw iris position is recorded as a micro-sample paired with the known screen target. Weight-decay (0.995/frame, PACE-style) ages old samples out. The polynomial model is rebuilt and the bias is recalculated.

2. **Passive drift EMA:** A slow EMA bias correction (clamped to ±12% of screen) is applied every frame during stable fixations, independently of explicit interactions.

Micro-samples are persisted to localStorage and reloaded on the next session.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Edge framework | Hono 4.x on Cloudflare Pages/Workers |
| Build | Vite 6 + @hono/vite-cloudflare-pages |
| Eye tracking | MediaPipe FaceMesh + Hands (CDN, no install) |
| Styling | Tailwind CSS (CDN) + custom CSS |
| Icons | Font Awesome 6 (CDN) |
| Voice | Web Speech API (built-in, Chrome/Edge) |
| AI intent | OpenAI API (server-side Hono route, key never exposed) |
| Storage | localStorage (calibration, micro-samples, gesture profiles, PACE data) |
| Runtime | Cloudflare Workers (edge, no Node.js, no server) |

---

## Deployment

```bash
# Build
npm run build

# Deploy to Cloudflare Pages
npx wrangler pages deploy dist --project-name accesseye-mvp

# Local development (sandbox)
npm run build
pm2 start ecosystem.config.cjs

# Test
curl http://localhost:3000
```

**Production URL:** https://accesseye-mvp.pages.dev  
**Platform:** Cloudflare Pages  
**Status:** ✅ Active

---

## Module File Reference

| File | Size (approx) | Role |
|---|---|---|
| `app.js` | ~2 800 lines | Phase 1 core — MediaPipe, calibration, UI registry, dwell |
| `phase2-engine.js` | ~2 400 lines | Phase 2 — hybrid gaze, head pose, dynamic calibration, intent |
| `phase2-init.js` | ~320 lines | Phase 2 bootstrap and UI wiring |
| `phase3-engine.js` | ~1 700 lines | Phase 3 — filtering, saccade, PACE, smooth pursuit, head-free |
| `phase3-init.js` | ~300 lines | Phase 3 bootstrap, dwell presets, validation UI |
| `snap-engine.js` | ~600 lines | Snap-To — target prediction, adaptive learner, engine |
| `accuracy-engine.js` | ~930 lines | Accuracy layer — gain remap, gravity snap, drift correction |
| `voice-nav.js` | ~990 lines | Voice navigation, intent fusion, show-clickable overlays |
| `gesture-studio.js` | ~780 lines | Facial gesture engine + custom gesture lifecycle |
| `a11y-mode.js` | ~530 lines | ACM — WCAG/ADA/508 compliance layer |
| `a11y-logger.js` | ~250 lines | Interaction logger, CSV/PDF export |
| `src/index.tsx` | ~1 600 lines | Hono edge server — HTML shell + /api/intent |

---

## Known Bugs Fixed (This Session)

| ID | File | Description |
|---|---|---|
| CALIB-BTN-1 | app.js | "Start Calibration" button re-opened the overlay instead of starting calibration — two-phase handler added |
| RESTART-5 | phase2-engine.js | Camera restart corrupted gaze pipeline — `hybridGaze.processResults` was not restored on deactivation |
| VOICE-1 | voice-nav.js | Nav/mode/camera commands routed through intent fusion and silently failed — `DIRECT_ID_ACTIONS` set added |
| VOICE-2 | voice-nav.js | `_extractEmbeddedAction` treated nav words as verbs — `skipAsVerb` set added |
| VOICE-3 | voice-nav.js | "Reset cursor" targeted wrong element ID — corrected to `#global-gaze-cursor` |
| VOICE-4 | voice-nav.js | "Show Clickable Items" and 16 other commands were never in `ACTION_COMMANDS` and had no implementation — full `_showClickableOverlays`, `_hideClickableOverlays`, `_handleFocusOn`, `_extractMultiWordAction`, and all missing switch cases added |

---

## Development History (Key Milestones)

| Commit | Change |
|---|---|
| `8531b3f` | VOICE-4: Show Clickable Items + 16 missing voice commands fully implemented |
| `74eafc9` | VOICE-1/2/3: nav commands, embedded-action parsing, reset-cursor target fixed |
| `fabef82` | CALIB-BTN-1 + RESTART-5: calibration start button + camera restart gaze loss fixed |
| `53e9b8c` | Fix dwell never completing — 5 targeted fixes |
| `585876d` | Fix edge-bias / centre dead-zone — 4 additive accuracy improvements |
| `1a7f2b6` | Accuracy Engine v2: GravitySnap + SessionDriftCorrector |
| `82af62b` | ACM live counters — event bus hooks for gaze/voice/keyboard/intent |
| `6dd602f` | Re-add ACM: WCAG/ADA/508 layer, safe 10 fps dwell loop |
| `b0cc00c` | Voice Navigation + Intent Fusion system |
| `7a41806` | Camera restart, dwell toggle, smile gesture, P2 panel layout |
| `b3f374c` | Fix gaze tracking after restart — Phase 2 activates cleanly at 30 fps |
| `7de9a41` | Phase 3: ridge regression, 13-pt calib, IVT, PACE, head-free |
| `35319bd` | 12 precision improvements — jitter + accuracy |
| `305740b` | Calibration v8: quality-gated sampling, auto-advance on stable fixation |
| `0e218e8` | Fix calibration stuck at pt1: separate recentFrames buffer for lock phase |

---

## What Has Not Been Built Yet

The following techniques were evaluated but not yet implemented:

| # | Technique | Notes |
|---|---|---|
| 3 | Breathing notch filter | Would remove ~0.25 Hz physiological oscillation from gaze signal |
| 5 | Iris ellipse aspect ratio correction | Iris is currently tracked as centroid only; shape ignored |
| 8 | 3D corneal surface reconstruction | Would need depth/stereo camera or glint-based PCCR |
| 9 | Zero-shot implicit calibration | Explicit 9-point calibration is always required |
| 10 | Multi-anchor ensemble gaze fusion | Single polynomial model; no mixture-of-experts |

---

## License

MIT — built by mkbrown261
