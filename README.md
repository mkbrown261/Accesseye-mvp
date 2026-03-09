# AccessEye — Eye + Gesture Control System (MVP)

## Project Overview
- **Name**: AccessEye MVP
- **Goal**: Touchless smartphone interaction for users with limited motor control using eye tracking + hand gesture detection
- **Privacy**: 100% on-device processing — zero camera data transmitted

## Live URLs
- **Production**: https://accesseye-mvp.pages.dev
- **Sandbox Dev**: http://localhost:3000

## Features Implemented

### ✅ Layer 1 — Vision Input
- MediaPipe FaceMesh (468 + iris landmarks 468–477)
- MediaPipe Hands (21-point hand tracking)
- Head pose estimation

### ✅ Layer 2 — Gaze Mapping Engine
- Kalman Filter (2-state: position + velocity)
- Exponential Moving Average (α=0.3)
- Calibrated screen mapping via polynomial regression

### ✅ Layer 3 — 5-Point Calibration System
- TopLeft, TopRight, Center, BottomLeft, BottomRight
- 25 samples per point, < 10 seconds total
- Polynomial regression model (a0 + a1·gx + a2·gy + a3·gx·gy)
- Saved to localStorage

### ✅ Layer 4 — UI Target Detection
- DOM element registry with bounding boxes
- 350ms dwell timer for focus
- Dwell progress bar visual feedback
- Gaze highlight (cyan glow)

### ✅ Layer 5 — Gesture Engine
- **Pinch** (thumb–index < 0.06 normalized) → Select
- **Air Tap** (index Z-delta > 0.06 in 8 frames) → Click
- **Open Palm** (avg tip spread > 0.35) → Cancel/Back
- Debounce: 500–800ms per gesture

### ✅ Layer 6 — Accessibility Layer
- Web Speech API TTS confirmation
- Visual ripple on activation
- Gesture indicator overlay
- Toast notification system
- Interaction log

### ✅ Layer 7 — Mouse Simulation Mode
- Full mouse-cursor simulation for demo/testing
- Seamless fallback when camera unavailable

### ✅ Four Pages
1. **Home** — Hero, features, user flow
2. **Architecture** — Layered diagram, tech stack, perf gauges
3. **Live Demo** — Camera + gesture control + messaging app
4. **API Docs** — Integration guide, code samples, testing plan

## Performance Targets
| Metric | Target | Status |
|--------|--------|--------|
| System Latency | <100ms | ✅ |
| Camera FPS | 30 FPS | ✅ |
| Gesture Detection | <150ms | ✅ |
| Gaze Accuracy | >85% post-calibration | ✅ |

## Architecture
```
Camera → FaceMesh → Iris Landmarks → Kalman+EMA → Calibration Model → Screen Coords
Camera → Hands    → 21 Landmarks  → Gesture Classifier → Debounce → Action
Screen Coords → UI Registry → Dwell Timer → Focus → Gesture → Activate → TTS
```

## Tech Stack
- **Backend**: Hono + TypeScript on Cloudflare Workers
- **Vision AI**: MediaPipe FaceMesh + Hands (WebGL/WASM)
- **Signal Processing**: Kalman Filter, EMA, Polynomial Regression
- **Frontend**: Vanilla JS, CSS Animations, Web Speech API
- **Deployment**: Cloudflare Pages (Edge CDN)

## Public API
```js
window.AccessEye.registerElement({ id, element, label, onActivate })
window.AccessEye.registerElements([...])
window.AccessEye.calibrate()
window.AccessEye.on('gaze' | 'gesture' | 'focus' | 'activate', cb)
window.AccessEye.toggleAudio()
window.AccessEye.getGaze()
```

## Deployment Status
- **Platform**: Cloudflare Pages
- **Status**: ✅ Active
- **Project**: accesseye-mvp
- **Last Deployed**: 2026-03-09
