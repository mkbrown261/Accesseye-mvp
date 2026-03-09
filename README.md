# AccessEye — Hybrid Gaze + Gesture Control System (v2)

## Project Overview
- **Name**: AccessEye
- **Version**: v2 (Phase 2 Hybrid Engine)
- **Goal**: Touchless mobile/browser interaction for users with motor impairments using eye tracking + hand gesture detection — 100% on-device
- **Key upgrade**: Phase 2 replaces the basic iris-center engine with a full **Hybrid Gaze Estimation System**

## Live URLs
- **Production**: https://accesseye-mvp.pages.dev
- **Sandbox (dev)**: https://3000-i576bnl6bd9ekefqjmae6-0e616f0a.sandbox.novita.ai

---

## Phase 2 Architecture Pipeline

```
Camera (120/60/30 FPS auto-negotiate)
  ↓
MediaPipe FaceMesh (468 + iris landmarks, refineLandmarks=true)
  ↓
P2.2  HeadPoseEstimator     — 6-DOF (yaw, pitch, roll) via solvePnP approximation
  ↓
P2.3  HybridGazeEngine      — iris offset (55%) + head pose (30%) + pupil boundary (15%)
                               weighted fusion with eyelid aperture confidence
  ↓
P2.6  GazeConfidenceScorer  — brightness, occlusion, glare, head angle → 0–1 score
  ↓
P2.4  TemporalStabilizer    — Adaptive Kalman (R scales with confidence)
                             + Velocity-adaptive EMA
                             + Sliding window trimmed mean (7 frames)
  ↓
P2.5  MicroSaccadeFilter    — Stability window: 12px radius / 200ms
                               Emits fixation/saccade events
  ↓
P2.7  DynamicCalibrationEngine — 9-point polynomial regression
                                  + micro-update from confirmed interactions
                                  + rolling bias drift correction
  ↓
P2.8  IntentPredictionEngine  — OpenAI gpt-5-mini via /api/intent (server-side proxy)
                                 Inputs: gaze_coords, stability, head_pose, confidence, fixation_event
  ↓
UIElementRegistry — screen coords → bounding box hit detection + dwell timer
```

## Currently Completed Features

### Phase 1 (MVP Foundation)
- [x] MediaPipe FaceMesh + Hands initialization (30 FPS, WebGL)
- [x] Iris center gaze extraction (binocular average)
- [x] Kalman 2D filter + EMA filter for smoothing
- [x] 5-point polynomial regression calibration
- [x] UIElementRegistry with dwell timer (350ms)
- [x] Gesture engine: pinch, air-tap, open-palm with debounce
- [x] Audio TTS feedback (Web Speech API)
- [x] Mouse cursor simulation mode
- [x] 4-page SPA: Home, Architecture, Live Demo, Docs
- [x] Toast notifications, interaction log

### Phase 2 (Hybrid Engine — NEW)
- [x] **P2.1** HighFPSCameraController — 120→60→30 FPS fallback chain
- [x] **P2.2** HeadPoseEstimator — 6-DOF solvePnP-approximation (yaw/pitch/roll)
- [x] **P2.3** HybridGazeEngine — binocular iris + head pose + pupil boundary + eyelid aperture
- [x] **P2.4** TemporalStabilizer — adaptive Kalman + velocity-EMA + trimmed-mean window
- [x] **P2.5** MicroSaccadeFilter — 12px/200ms fixation stability, saccade suppression
- [x] **P2.6** GazeConfidenceScorer — brightness/occlusion/glare/head-angle multi-factor score
- [x] **P2.7** DynamicCalibrationEngine — 9-point + micro-update from interactions + bias correction
- [x] **P2.8** IntentPredictionEngine — OpenAI /api/intent proxy with behavioral context payload
- [x] **P2.9** GazeBenchmark — Phase 1 vs Phase 2 jitter/latency comparison (30s window)
- [x] **Phase2Orchestrator** — wires all modules, patches MediaPipe face handler
- [x] **Phase2InitController** — attaches orchestrator to AccessEyeApp at runtime
- [x] **Phase 2 Status Panel** in Live Demo — confidence bar, head pose 6-DOF, fixation stats, AI intent, benchmark, micro-calib controls
- [x] Low-light and glasses/glare warnings via toast notifications
- [x] Dynamic calibration UI controls (reset micro-calib)
- [x] API route `/api/intent` — server-side AI intent prediction proxy
- [x] API route `/api/benchmark` — in-memory benchmark storage
- [x] SVG favicon + updated nav badge (v2)

## Functional Entry Points (API)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main SPA (all 4 pages) |
| `/static/app.js` | GET | Phase 1 core engine |
| `/static/phase2-engine.js` | GET | Phase 2 modules |
| `/static/phase2-init.js` | GET | Phase 2 wiring layer |
| `/static/styles.css` | GET | Full stylesheet |
| `/api/intent` | POST | AI intent prediction proxy |
| `/api/benchmark` | POST | Store benchmark result |
| `/api/benchmark/latest` | GET | Get latest benchmark |

### /api/intent payload schema
```json
{
  "gaze_coordinates": { "x": 0.5, "y": 0.5 },
  "gaze_stability_duration": 350,
  "head_pose_angle": { "yaw": -5.2, "pitch": 3.1, "roll": 0.8 },
  "gaze_confidence": 0.82,
  "fixation_event": true,
  "focused_element": { "id": "btn-send", "label": "Send Message" },
  "recent_elements": [...],
  "gesture_history": [...],
  "session_context": { "duration_s": 45, "activation_count": 3 }
}
```

## Performance Targets (Phase 2)

| Metric | Target | Notes |
|--------|--------|-------|
| Camera FPS | 60 FPS ideal, 30 fallback | HighFPSCameraController auto-negotiates |
| Gaze latency | < 80ms | Adaptive Kalman |
| Focus detection | < 200ms | Micro-saccade 12px/200ms window |
| Confidence threshold | > 0.6 | Below this: slow dwell / disable auto-select |
| Accuracy (calibrated) | > 85% | 9-point + dynamic micro-update |

## Data Architecture

| Storage | Key | Data |
|---------|-----|------|
| localStorage | `accesseye_calib` | Polynomial model, 9-pt calibration data |
| localStorage | `accesseye_micro` | Micro-calib bias, interaction samples |
| In-memory | `benchmarkResults[]` | Session benchmark reports |

## User Guide

1. **Launch**: Open https://accesseye-mvp.pages.dev → click "Launch Live Demo"
2. **Start Camera**: Click "Start Camera" — grants webcam access, MediaPipe loads (~5s), Phase 2 activates automatically
3. **Mouse Simulation**: Default mode — move cursor over buttons, click to activate
4. **Gaze Mode**: Switch to "Gaze" tab — look at buttons, dwell 350ms to focus, pinch/air-tap to activate
5. **Calibration**: Switch to "Calibrate" tab → "Start Calibration" — follow 5 on-screen dots
6. **Phase 2 Panel** (scroll down in demo): Live confidence score, head pose angles, fixation stats, AI intent, benchmark controls
7. **Benchmark**: Click "Run 30s Benchmark" — compare Phase 1 vs Phase 2 jitter/latency

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Hono v4 + TypeScript on Cloudflare Workers |
| Vision AI | MediaPipe FaceMesh (468+iris) + Hands (21pt) |
| Signal processing | Custom Kalman, EMA, sliding window, solvePnP |
| AI Intent | OpenAI gpt-5-mini via Genspark LLM proxy |
| Frontend | Vanilla JS (SPA), TailwindCSS-inspired custom CSS |
| Deployment | Cloudflare Pages (edge, HTTPS, zero-latency CDN) |
| Privacy | 100% on-device — zero video/data egress |

## Deployment Status
- **Platform**: Cloudflare Pages
- **Status**: ✅ Active
- **Project**: accesseye-mvp
- **Last Deployed**: 2026-03-09
