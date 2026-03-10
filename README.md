# AccessEye MVP

**Hands-free eye-tracking interface for Cloudflare Pages**
Live: https://accesseye-mvp.pages.dev
GitHub: https://github.com/mkbrown261/Accesseye-mvp

---

## What it does

AccessEye lets you control a computer with only your eyes. Look at a button and dwell on it to click — no hands needed. It runs entirely in the browser using your webcam, with no server-side processing.

Key features:
- **Real-time eye tracking** via MediaPipe FaceMesh (478 landmarks)
- **Quality-gated calibration** — dot auto-advances only when gaze is stable; bad frames (blinks, saccades) are silently skipped
- **9-point polynomial calibration** with iris-only signal for maximum accuracy
- **Dwell-to-click** — look at any registered UI element for 350 ms to activate it
- **AI intent prediction** — optional endpoint integration for predicting intended actions
- **PACE micro-recalibration** — continuously refines the model during use
- **Phase 2 gaze engine** — binocular iris tracking, head-pose compensation, OneEuro filtering
- **Phase 3 upgrades** — IVT saccade detection, adaptive dwell, smooth pursuit calibration, head-free stabilization
- **Gesture engine** — pinch and air-tap detection via hand landmarks

---

## Calibration (v8 — quality-gated)

1. Open the app and click **Calibrate** (top bar)
2. Click **Start Calibration**
3. For each of the 9 dots:
   - **Blue** = move your eyes to the dot
   - **Amber** = gaze is stabilising — keep still
   - **Green** = locked, collecting samples — the ring fills up automatically
   - **Flash + shrink** = point captured, moves to next automatically
4. After all 9 points the model is built. The cursor appears and tracks your eyes.

**Tips:**
- Sit 50–70 cm from screen, face centred in frame
- For corner dots, move only your eyes (small head movement is fine)
- If a dot stays amber, blink once to reset and try again
- Recalibrate any time from the top bar

---

## Architecture

```
Browser
├── app.js                  Phase 1 — MediaPipe wiring, GazeEngine, CalibrationEngine (v8), CalibrationUI, UI registry, dwell
├── phase2-engine.js        Phase 2 — HybridGazeEngine (binocular iris + head pose + pupil fusion)
├── phase2-init.js          Phase 2 initialisation, camera patching, UI wiring
└── phase3-engine.js        Phase 3 — OneEuro filter, IVT, PACE, adaptive dwell, head-free stabilizer

Edge (Cloudflare Pages)
└── src/index.tsx           Hono app — serves HTML shell + /api/intent endpoint
```

**Gaze pipeline (per frame):**
```
MediaPipe FaceMesh
  → iris landmark extraction (_computeIrisSignal)
  → binocular average with per-eye quality weighting
  → head-pose compensation (yaw/pitch)
  → OneEuro filter (minCutoff 0.45 Hz, beta 0.05)
  → CalibrationEngine.mapGaze (degree-3 ridge regression, 9-point iris-only model)
  → HeadFreeStabilizer (EMA, compScale 0.60)
  → TemporalStabilizer (Kalman R=0.005, EMA α=0.22)
  → _updateGazeCursor (pixel position)
  → UIElementRegistry dwell check
```

**Calibration model:**
- Signal: iris-only offset (no head/pupil fusion noise)
- Points: 9 (4 corners at 0.08/0.92, 4 inner ring at 0.25/0.75, center)
- Regression: degree-3 polynomial ridge (λ=0.01, 10 terms)
- Normalisation: observed gaze range padded 18 % for edge extrapolation
- Weights: corners 4×, interior 1×
- Storage: localStorage key `accesseye_calib` (version 8)

---

## Tech stack

| Layer | Technology |
|---|---|
| Framework | Hono 4.x on Cloudflare Pages |
| Build | Vite 6 + @hono/vite-cloudflare-pages |
| Eye tracking | MediaPipe FaceMesh + Hands (CDN) |
| Styling | Tailwind CSS (CDN) + custom CSS |
| Icons | Font Awesome 6 (CDN) |
| Runtime | Cloudflare Workers (edge, no Node.js) |

---

## Deployment

**Production URL:** https://accesseye-mvp.pages.dev
**Platform:** Cloudflare Pages
**Status:** ✅ Active

```bash
# Build
npm run build

# Deploy to Cloudflare Pages
npx wrangler pages deploy dist --project-name accesseye-mvp

# Local development
npm run build
pm2 start ecosystem.config.cjs
```

---

## Development history (key milestones)

| Commit | Change |
|---|---|
| `0e218e8` | Fix calibration stuck at pt1: separate recentFrames buffer for lock phase |
| `305740b` | Calibration v8: quality-gated sampling, auto-advance on stable fixation |
| `886406a` | Fix ptLabel ReferenceError + hide cursor during calibration |
| `33d1184` | 9-point grid, shorter timing, head-movement tolerance |
| `0401dc4` | Iris-only calibration signal, relaxed edge clamp, faster OneEuro |
| `81e3f1b` | 7 peripheral accuracy improvements |
| `f50c4f8` | 6 pipeline diagnostic fixes |
| `7db2c7a` | Camera restart, recalibration, corner-stuck, over-scaling fixes |
| `35319bd` | 12 precision improvements — jitter + accuracy |
| `7de9a41` | Phase 3: ridge regression, 13-pt calib, IVT, PACE, head-free |

---

## License

MIT — built by mkbrown261
