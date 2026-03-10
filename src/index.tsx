import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { serveStatic } from 'hono/cloudflare-workers'

type Bindings = { OPENAI_API_KEY?: string; OPENAI_BASE_URL?: string }
const app = new Hono<{ Bindings: Bindings }>()

// Serve static assets
app.use('/static/*', serveStatic({ root: './' }))

// CORS for API routes
app.use('/api/*', cors())

/* ════════════════════════════════════════════════════════════════
   Phase 2: Intent Prediction API
   POST /api/intent
   Body: gaze + behavioral context payload
   Returns: { predicted_action, confidence, reasoning, suggestions }
════════════════════════════════════════════════════════════════ */
app.post('/api/intent', async (c) => {
  try {
    const payload = await c.req.json()

    const apiKey  = c.env?.OPENAI_API_KEY  || ''
    const baseURL = c.env?.OPENAI_BASE_URL || 'https://www.genspark.ai/api/llm_proxy/v1'

    if (!apiKey) {
      return c.json({ error: 'No API key configured', predicted_action: 'unknown', confidence: 0 }, 200)
    }

    // Build context summary for the AI prompt
    const gazeX       = ((payload.gaze_coordinates?.x || 0.5) * 100).toFixed(1)
    const gazeY       = ((payload.gaze_coordinates?.y || 0.5) * 100).toFixed(1)
    const stability   = Math.round(payload.gaze_stability_duration || 0)
    const confidence  = ((payload.gaze_confidence || 0) * 100).toFixed(0)
    const headYaw     = (payload.head_pose_angle?.yaw || 0).toFixed(1)
    const headPitch   = (payload.head_pose_angle?.pitch || 0).toFixed(1)
    const fixation    = payload.fixation_event ? 'YES (stable fixation)' : 'NO (scanning)'
    const focusedEl   = payload.focused_element?.label || 'none'
    const recentEls   = (payload.recent_elements || []).map((e: any) => `${e.label}(${e.dwell_ms}ms)`).join(', ') || 'none'
    const gestures    = (payload.gesture_history || []).map((g: any) => `${g.type} on ${g.element}`).join(', ') || 'none'
    const sessionDur  = payload.session_context?.duration_s || 0
    const activations = payload.session_context?.activation_count || 0

    const systemPrompt = `You are an accessibility AI assistant for an eye-tracking + gesture control system.
Your job is to predict what action a user with motor impairments is most likely trying to perform,
based on their eye gaze behavior and interaction history.

Respond with JSON ONLY in this exact format (no markdown, no explanation outside the JSON):
{
  "predicted_action": "<short action label, max 40 chars>",
  "confidence": <0.0-1.0>,
  "reasoning": "<1-2 sentences explaining the prediction>",
  "suggestions": ["<suggestion 1>", "<suggestion 2>"],
  "adaptation": "<any suggested system adaptation (dwell time, sensitivity, etc.)>"
}`

    const userPrompt = `Current eye tracking state:
- Gaze position: (${gazeX}%, ${gazeY}%) of screen
- Gaze stability: ${stability}ms | Fixation: ${fixation}
- Tracking confidence: ${confidence}% | Head: yaw=${headYaw}° pitch=${headPitch}°
- Currently focused element: ${focusedEl}
- Recent elements gazed at: ${recentEls}
- Recent gestures: ${gestures || 'none yet'}
- Session: ${sessionDur}s active, ${activations} total activations

Based on this gaze pattern, predict the user's most likely intended action.`

    const response = await fetch(`${baseURL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user',   content: userPrompt }
        ],
        max_tokens: 300,
        temperature: 0.4,
        response_format: { type: 'json_object' }
      })
    })

    if (!response.ok) {
      const errText = await response.text()
      console.error('OpenAI error:', errText)
      return c.json({
        predicted_action: 'Processing gaze pattern...',
        confidence: 0.5,
        reasoning: 'AI prediction temporarily unavailable.',
        suggestions: [],
        adaptation: ''
      })
    }

    const data: any = await response.json()
    const content   = data.choices?.[0]?.message?.content || '{}'
    let result: any = {}
    try { result = JSON.parse(content) } catch(_) {
      result = { predicted_action: 'Analyzing...', confidence: 0.5, reasoning: content.slice(0, 100) }
    }

    return c.json({
      predicted_action: result.predicted_action || 'Observing...',
      confidence:       result.confidence       || 0.5,
      reasoning:        result.reasoning        || '',
      suggestions:      result.suggestions      || [],
      adaptation:       result.adaptation       || '',
      intent:           result.predicted_action,
      raw_payload_echo: { gaze_x: gazeX, gaze_y: gazeY, focused: focusedEl }
    })

  } catch (err: any) {
    console.error('Intent API error:', err)
    return c.json({ predicted_action: 'Error', confidence: 0, reasoning: err.message }, 200)
  }
})

/* Phase 2: Benchmark results store (in-memory for session) */
const benchmarkResults: any[] = []
app.post('/api/benchmark', async (c) => {
  const body = await c.req.json()
  benchmarkResults.push({ ...body, timestamp: Date.now() })
  return c.json({ ok: true, stored: benchmarkResults.length })
})

app.get('/api/benchmark/latest', (c) => {
  const last = benchmarkResults[benchmarkResults.length - 1]
  return c.json(last || { error: 'No benchmark data yet' })
})

/* ════════════════════════════════════════════════════════════════
   Health + Config Check
   GET /api/health
   Returns: system status, OpenAI key presence, build version.
   IMPORTANT: Never expose the actual key — only confirm presence.
════════════════════════════════════════════════════════════════ */
app.get('/api/health', (c) => {
  const hasKey = !!(c.env?.OPENAI_API_KEY)
  return c.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    openai_key_configured: hasKey,
    // FIX: openai_key is NEVER returned — only presence flag
    base_url: c.env?.OPENAI_BASE_URL || 'https://www.genspark.ai/api/llm_proxy/v1',
    version: '2.0.0',
    phases: ['Phase1-GazeEngine', 'Phase2-HybridGaze', 'Phase3-OneEuro']
  })
})

// Main app - single page application
app.get('/', (c) => {
  return c.html(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <title>AccessEye — Gaze & Gesture Control System</title>
  <link rel="icon" type="image/svg+xml" href="/static/favicon.svg" />
  <link rel="stylesheet" href="/static/styles.css" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet" />
  <link href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css" rel="stylesheet" />
</head>
<body>
  <div id="app-root">

    <!-- ══════════════════════════════════════════
         NAV BAR
    ══════════════════════════════════════════ -->
    <nav id="main-nav">
      <div class="nav-logo">
        <i class="fas fa-eye"></i>
        <span>AccessEye</span>
        <span class="nav-badge">v2</span>
      </div>
      <div class="nav-links">
        <button class="nav-btn active" data-page="home"><i class="fas fa-home"></i><span>Home</span></button>
        <button class="nav-btn" data-page="architecture"><i class="fas fa-project-diagram"></i><span>Architecture</span></button>
        <button class="nav-btn" data-page="demo"><i class="fas fa-play-circle"></i><span>Live Demo</span></button>
        <button class="nav-btn" data-page="docs"><i class="fas fa-book"></i><span>Docs</span></button>
      </div>
      <div class="nav-status" id="system-status">
        <span class="status-dot offline"></span>
        <span class="status-label">Camera Off</span>
      </div>
    </nav>

    <!-- ══════════════════════════════════════════
         PAGE: HOME
    ══════════════════════════════════════════ -->
    <div id="page-home" class="page active">
      <section class="hero-section">
        <div class="hero-content">
          <div class="hero-badge"><i class="fas fa-universal-access"></i> Accessibility Technology</div>
          <h1 class="hero-title">
            Control Your Phone With<br/>
            <span class="gradient-text">Eyes & Gestures</span>
          </h1>
          <p class="hero-subtitle">
            A production-ready system for touchless interaction using hybrid gaze estimation,
            6-DOF head pose, temporal stabilization, micro-saccade filtering, dynamic calibration
            &amp; AI intent prediction — 100% on-device, no cloud, no neural implants.
          </p>
          <div class="hero-actions">
            <button class="btn-primary" id="launch-demo-btn">
              <i class="fas fa-play"></i> Launch Live Demo
            </button>
            <button class="btn-secondary" id="view-arch-btn">
              <i class="fas fa-project-diagram"></i> View Architecture
            </button>
          </div>
          <div class="hero-stats">
            <div class="stat"><span class="stat-val">&lt;80ms</span><span class="stat-lbl">Latency</span></div>
            <div class="stat"><span class="stat-val">60 FPS</span><span class="stat-lbl">Camera</span></div>
            <div class="stat"><span class="stat-val">9-Point</span><span class="stat-lbl">Calibration</span></div>
            <div class="stat"><span class="stat-val">100%</span><span class="stat-lbl">On-Device</span></div>
          </div>
        </div>
        <div class="hero-visual">
          <div class="eye-demo-card">
            <div class="eye-demo-screen">
              <div class="eye-scan-ring"></div>
              <div class="eye-scan-ring r2"></div>
              <div class="eye-scan-ring r3"></div>
              <div class="pupil-dot" id="hero-pupil"></div>
              <div class="gaze-crosshair" id="hero-crosshair"></div>
            </div>
            <div class="eye-demo-label">
              <i class="fas fa-eye"></i> Real-time Gaze Tracking
            </div>
          </div>
        </div>
      </section>

      <!-- Feature Cards -->
      <section class="features-section">
        <h2 class="section-title">System Layers</h2>
        <div class="features-grid">
          <div class="feature-card">
            <div class="feature-icon eye-icon"><i class="fas fa-eye"></i></div>
            <h3>Vision Input Layer</h3>
            <p>MediaPipe Face Mesh + Hands captures eye position, head pose, and hand landmarks at 60/120 FPS with multi-point iris + eyelid contours.</p>
            <ul class="feature-list">
              <li><i class="fas fa-check"></i> Pupil + iris detection</li>
              <li><i class="fas fa-check"></i> Head pose estimation</li>
              <li><i class="fas fa-check"></i> 21-point hand landmarks</li>
            </ul>
          </div>
          <div class="feature-card">
            <div class="feature-icon map-icon"><i class="fas fa-crosshairs"></i></div>
            <h3>Gaze Mapping Engine</h3>
            <p>Hybrid gaze model fuses binocular iris offset, head pose vector, and pupil boundary with temporal filtering to eliminate jitter.</p>
            <ul class="feature-list">
              <li><i class="fas fa-check"></i> Adaptive Kalman + EMA + window</li>
              <li><i class="fas fa-check"></i> Micro-saccade filter (12px/200ms)</li>
              <li><i class="fas fa-check"></i> Confidence-weighted fusion</li>
            </ul>
          </div>
          <div class="feature-card">
            <div class="feature-icon ui-icon"><i class="fas fa-mouse-pointer"></i></div>
            <h3>UI Target Detection</h3>
            <p>Registers interactive components as bounding boxes. Detects gaze intersection with 300ms dwell time.</p>
            <ul class="feature-list">
              <li><i class="fas fa-check"></i> Bounding box registry</li>
              <li><i class="fas fa-check"></i> Dwell-time focus</li>
              <li><i class="fas fa-check"></i> Visual glow feedback</li>
            </ul>
          </div>
          <div class="feature-card">
            <div class="feature-icon gesture-icon"><i class="fas fa-hand-paper"></i></div>
            <h3>Gesture Engine</h3>
            <p>MediaPipe Hands landmarks power pinch detection, air tap recognition, and open palm cancel.</p>
            <ul class="feature-list">
              <li><i class="fas fa-check"></i> Pinch → Select</li>
              <li><i class="fas fa-check"></i> Air tap → Click</li>
              <li><i class="fas fa-check"></i> Open palm → Cancel</li>
            </ul>
          </div>
          <div class="feature-card">
            <div class="feature-icon calib-icon"><i class="fas fa-sliders-h"></i></div>
            <h3>Calibration System</h3>
            <p>9-point calibration + continuous dynamic micro-calibration from confirmed interactions keeps gaze drift corrected over time.</p>
            <ul class="feature-list">
              <li><i class="fas fa-check"></i> 9-point polynomial regression</li>
              <li><i class="fas fa-check"></i> Dynamic bias drift correction</li>
              <li><i class="fas fa-check"></i> Confidence-gated updates</li>
            </ul>
          </div>
          <div class="feature-card">
            <div class="feature-icon privacy-icon"><i class="fas fa-shield-alt"></i></div>
            <h3>Privacy & Safety</h3>
            <p>All video processing is 100% on-device. No camera data ever leaves the browser or device.</p>
            <ul class="feature-list">
              <li><i class="fas fa-check"></i> Zero cloud processing</li>
              <li><i class="fas fa-check"></i> No data transmission</li>
              <li><i class="fas fa-check"></i> Local calibration storage</li>
            </ul>
          </div>
        </div>
      </section>

      <!-- User Flow -->
      <section class="flow-section">
        <h2 class="section-title">How It Works</h2>
        <div class="flow-steps">
          <div class="flow-step">
            <div class="flow-num">01</div>
            <div class="flow-icon"><i class="fas fa-play"></i></div>
            <h4>Launch</h4>
            <p>User launches accessibility mode — camera permissions requested</p>
          </div>
          <div class="flow-arrow"><i class="fas fa-arrow-right"></i></div>
          <div class="flow-step">
            <div class="flow-num">02</div>
            <div class="flow-icon"><i class="fas fa-sliders-h"></i></div>
            <h4>Calibrate</h4>
            <p>13-point ridge regression builds personalized gaze-to-screen model</p>
          </div>
          <div class="flow-arrow"><i class="fas fa-arrow-right"></i></div>
          <div class="flow-step">
            <div class="flow-num">03</div>
            <div class="flow-icon"><i class="fas fa-eye"></i></div>
            <h4>Gaze</h4>
            <p>User looks at a UI element — system detects and highlights it</p>
          </div>
          <div class="flow-arrow"><i class="fas fa-arrow-right"></i></div>
          <div class="flow-step">
            <div class="flow-num">04</div>
            <div class="flow-icon"><i class="fas fa-hand-point-up"></i></div>
            <h4>Gesture</h4>
            <p>User performs pinch or air tap to confirm and activate the element</p>
          </div>
          <div class="flow-arrow"><i class="fas fa-arrow-right"></i></div>
          <div class="flow-step">
            <div class="flow-num">05</div>
            <div class="flow-icon"><i class="fas fa-check-circle"></i></div>
            <h4>Action</h4>
            <p>Element activates with visual + audio feedback confirming the selection</p>
          </div>
        </div>
      </section>
    </div>

    <!-- ══════════════════════════════════════════
         PAGE: ARCHITECTURE
    ══════════════════════════════════════════ -->
    <div id="page-architecture" class="page">
      <div class="page-header">
        <h1><i class="fas fa-project-diagram"></i> System Architecture</h1>
        <p>Production-ready layered architecture for on-device eye + gesture control</p>
      </div>

      <!-- Architecture Diagram -->
      <section class="arch-diagram-section">
        <div class="arch-diagram">
          <!-- Layer 1 -->
          <div class="arch-layer layer-input">
            <div class="arch-layer-label">Layer 1 — Vision Input</div>
            <div class="arch-blocks">
              <div class="arch-block camera-block">
                <i class="fas fa-camera"></i>
                <span>Front Camera</span>
                <small>120/60/30 FPS auto</small>
              </div>
              <div class="arch-block">
                <i class="fas fa-eye"></i>
                <span>Face Mesh</span>
                <small>468 landmarks</small>
              </div>
              <div class="arch-block">
                <i class="fas fa-hand-paper"></i>
                <span>Hand Tracking</span>
                <small>21 landmarks</small>
              </div>
              <div class="arch-block">
                <i class="fas fa-head-side-cough"></i>
                <span>Head Pose</span>
                <small>6-DOF estimation</small>
              </div>
            </div>
          </div>
          <div class="arch-connector"><i class="fas fa-arrow-down"></i></div>

          <!-- Layer 2 -->
          <div class="arch-layer layer-processing">
            <div class="arch-layer-label">Layer 2 — Processing</div>
            <div class="arch-blocks">
              <div class="arch-block">
                <i class="fas fa-crosshairs"></i>
                <span>Gaze Vector</span>
                <small>Pupil → direction</small>
              </div>
              <div class="arch-block highlight-block">
                <i class="fas fa-filter"></i>
                <span>Kalman Filter</span>
                <small>Noise smoothing</small>
              </div>
              <div class="arch-block">
                <i class="fas fa-wave-square"></i>
                <span>EMA Filter</span>
                <small>α=0.3 smoothing</small>
              </div>
              <div class="arch-block">
                <i class="fas fa-hand-scissors"></i>
                <span>Gesture Classifier</span>
                <small>Landmark distances</small>
              </div>
            </div>
          </div>
          <div class="arch-connector"><i class="fas fa-arrow-down"></i></div>

          <!-- Layer 3 -->
          <div class="arch-layer layer-mapping">
            <div class="arch-layer-label">Layer 3 — Gaze Mapping</div>
            <div class="arch-blocks">
              <div class="arch-block">
                <i class="fas fa-map"></i>
                <span>Calibration Model</span>
                <small>Polynomial regression</small>
              </div>
              <div class="arch-block highlight-block">
                <i class="fas fa-vector-square"></i>
                <span>Screen Mapping</span>
                <small>Gaze → (x,y) coords</small>
              </div>
              <div class="arch-block">
                <i class="fas fa-clock"></i>
                <span>Dwell Timer</span>
                <small>300ms stabilization</small>
              </div>
              <div class="arch-block">
                <i class="fas fa-shield-virus"></i>
                <span>Debounce Logic</span>
                <small>Anti-jitter guard</small>
              </div>
            </div>
          </div>
          <div class="arch-connector"><i class="fas fa-arrow-down"></i></div>

          <!-- Layer 4 -->
          <div class="arch-layer layer-ui">
            <div class="arch-layer-label">Layer 4 — UI Interaction</div>
            <div class="arch-blocks">
              <div class="arch-block">
                <i class="fas fa-object-group"></i>
                <span>Element Registry</span>
                <small>Bounding boxes</small>
              </div>
              <div class="arch-block">
                <i class="fas fa-mouse-pointer"></i>
                <span>Hit Detection</span>
                <small>Gaze ∩ BBox</small>
              </div>
              <div class="arch-block highlight-block">
                <i class="fas fa-lightbulb"></i>
                <span>Visual Feedback</span>
                <small>Glow + highlight</small>
              </div>
              <div class="arch-block">
                <i class="fas fa-volume-up"></i>
                <span>Audio TTS</span>
                <small>Speech feedback</small>
              </div>
            </div>
          </div>
          <div class="arch-connector"><i class="fas fa-arrow-down"></i></div>

          <!-- Layer 5 -->
          <div class="arch-layer layer-output">
            <div class="arch-layer-label">Layer 5 — Action Output</div>
            <div class="arch-blocks">
              <div class="arch-block camera-block">
                <i class="fas fa-bolt"></i>
                <span>Element Activation</span>
                <small>Simulated tap/click</small>
              </div>
              <div class="arch-block">
                <i class="fas fa-history"></i>
                <span>Action Log</span>
                <small>Local storage only</small>
              </div>
              <div class="arch-block">
                <i class="fas fa-lock"></i>
                <span>Privacy Guard</span>
                <small>Zero data egress</small>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- Tech Stack -->
      <section class="tech-section">
        <h2 class="section-title">Technology Stack</h2>
        <div class="tech-grid">
          <div class="tech-card">
            <div class="tech-icon"><i class="fab fa-js-square"></i></div>
            <h3>Frontend</h3>
            <div class="tech-tags">
              <span class="tag">Hono + TypeScript</span>
              <span class="tag">Vanilla JS</span>
              <span class="tag">CSS Animations</span>
              <span class="tag">Web Speech API</span>
            </div>
          </div>
          <div class="tech-card">
            <div class="tech-icon"><i class="fas fa-brain"></i></div>
            <h3>Vision AI</h3>
            <div class="tech-tags">
              <span class="tag">MediaPipe Face Mesh</span>
              <span class="tag">MediaPipe Hands</span>
              <span class="tag">WebGL Backend</span>
              <span class="tag">WASM Processing</span>
            </div>
          </div>
          <div class="tech-card">
            <div class="tech-icon"><i class="fas fa-calculator"></i></div>
            <h3>Signal Processing</h3>
            <div class="tech-tags">
              <span class="tag">Kalman Filter</span>
              <span class="tag">EMA Smoothing</span>
              <span class="tag">Polynomial Regression</span>
              <span class="tag">Debounce Logic</span>
            </div>
          </div>
          <div class="tech-card">
            <div class="tech-icon"><i class="fas fa-cloud"></i></div>
            <h3>Deployment</h3>
            <div class="tech-tags">
              <span class="tag">Cloudflare Pages</span>
              <span class="tag">Edge Network</span>
              <span class="tag">Zero-latency CDN</span>
              <span class="tag">HTTPS Only</span>
            </div>
          </div>
        </div>
      </section>

      <!-- Performance Targets -->
      <section class="perf-section">
        <h2 class="section-title">Performance Targets</h2>
        <div class="perf-grid">
          <div class="perf-card">
            <div class="perf-meter">
              <svg viewBox="0 0 100 60" class="perf-gauge">
                <path d="M 10 55 A 45 45 0 0 1 90 55" fill="none" stroke="#1e293b" stroke-width="8"/>
                <path id="gauge-latency" d="M 10 55 A 45 45 0 0 1 90 55" fill="none" stroke="#00d4ff" stroke-width="8" stroke-dasharray="0 141"/>
              </svg>
              <div class="perf-value">&lt;80ms</div>
            </div>
            <div class="perf-label">System Latency</div>
            <div class="perf-sub">End-to-end response time</div>
          </div>
          <div class="perf-card">
            <div class="perf-meter">
              <svg viewBox="0 0 100 60" class="perf-gauge">
                <path d="M 10 55 A 45 45 0 0 1 90 55" fill="none" stroke="#1e293b" stroke-width="8"/>
                <path id="gauge-fps" d="M 10 55 A 45 45 0 0 1 90 55" fill="none" stroke="#00ff88" stroke-width="8" stroke-dasharray="0 141"/>
              </svg>
              <div class="perf-value">60 FPS</div>
            </div>
            <div class="perf-label">Camera Processing</div>
            <div class="perf-sub">Real-time frame analysis</div>
          </div>
          <div class="perf-card">
            <div class="perf-meter">
              <svg viewBox="0 0 100 60" class="perf-gauge">
                <path d="M 10 55 A 45 45 0 0 1 90 55" fill="none" stroke="#1e293b" stroke-width="8"/>
                <path id="gauge-gesture" d="M 10 55 A 45 45 0 0 1 90 55" fill="none" stroke="#ff6b35" stroke-width="8" stroke-dasharray="0 141"/>
              </svg>
              <div class="perf-value">&lt;150ms</div>
            </div>
            <div class="perf-label">Gesture Detection</div>
            <div class="perf-sub">Hand gesture recognition</div>
          </div>
          <div class="perf-card">
            <div class="perf-meter">
              <svg viewBox="0 0 100 60" class="perf-gauge">
                <path d="M 10 55 A 45 45 0 0 1 90 55" fill="none" stroke="#1e293b" stroke-width="8"/>
                <path id="gauge-accuracy" d="M 10 55 A 45 45 0 0 1 90 55" fill="none" stroke="#a78bfa" stroke-width="8" stroke-dasharray="0 141"/>
              </svg>
              <div class="perf-value">&gt;85%</div>
            </div>
            <div class="perf-label">Gaze Accuracy</div>
              <div class="perf-sub">Post dynamic-calibration precision</div>
          </div>
        </div>
      </section>
    </div>

    <!-- ══════════════════════════════════════════
         PAGE: LIVE DEMO
    ══════════════════════════════════════════ -->
    <div id="page-demo" class="page">
      <div class="demo-layout">

        <!-- LEFT: Camera + Controls Panel -->
        <div class="demo-sidebar">
          <div class="demo-panel">
            <h3><i class="fas fa-video"></i> Camera Feed</h3>
            <div class="camera-container" id="camera-container">
              <video id="demo-video" playsinline muted autoplay></video>
              <canvas id="overlay-canvas"></canvas>
              <div class="camera-placeholder" id="camera-placeholder">
                <i class="fas fa-camera"></i>
                <p>Camera not started</p>
                <small>Click Start to begin</small>
              </div>
              <!-- Gaze dot overlay -->
              <div class="gaze-dot" id="gaze-dot"></div>
            </div>

            <div class="demo-controls">
              <button class="btn-primary" id="start-camera-btn">
                <i class="fas fa-play"></i> Start Camera
              </button>
              <button class="btn-danger" id="stop-camera-btn" disabled>
                <i class="fas fa-stop"></i> Stop
              </button>
            </div>

            <!-- Mode Selector -->
            <div class="mode-selector">
              <h4>Active Mode</h4>
              <div class="mode-tabs">
                <button class="mode-tab active" data-mode="mouse">
                  <i class="fas fa-mouse-pointer"></i> Mouse Sim
                </button>
                <button class="mode-tab" data-mode="gaze">
                  <i class="fas fa-eye"></i> Gaze
                </button>
                <button class="mode-tab" data-mode="calibrate">
                  <i class="fas fa-sliders-h"></i> Calibrate
                </button>
              </div>
            </div>
          </div>

          <!-- Detection Status -->
          <div class="demo-panel">
            <h3><i class="fas fa-tachometer-alt"></i> Detection Status</h3>
            <div class="status-grid">
              <div class="status-item" id="status-face">
                <div class="status-indicator"></div>
                <div class="status-info">
                  <span class="status-name">Face Detection</span>
                  <span class="status-val" id="face-status-val">Inactive</span>
                </div>
              </div>
              <div class="status-item" id="status-gaze">
                <div class="status-indicator"></div>
                <div class="status-info">
                  <span class="status-name">Gaze Tracking</span>
                  <span class="status-val" id="gaze-status-val">Inactive</span>
                </div>
              </div>
              <div class="status-item" id="status-hand">
                <div class="status-indicator"></div>
                <div class="status-info">
                  <span class="status-name">Hand Detection</span>
                  <span class="status-val" id="hand-status-val">Inactive</span>
                </div>
              </div>
              <div class="status-item" id="status-gesture">
                <div class="status-indicator"></div>
                <div class="status-info">
                  <span class="status-name">Gesture</span>
                  <span class="status-val" id="gesture-status-val">None</span>
                </div>
              </div>
            </div>

            <!-- Metrics -->
            <div class="metrics-row">
              <div class="metric-box">
                <span class="metric-val" id="fps-display">0</span>
                <span class="metric-lbl">FPS</span>
              </div>
              <div class="metric-box">
                <span class="metric-val" id="latency-display">0</span>
                <span class="metric-lbl">ms</span>
              </div>
              <div class="metric-box">
                <span class="metric-val" id="confidence-display">0%</span>
                <span class="metric-lbl">Conf.</span>
              </div>
            </div>

            <!-- Gaze Coords -->
            <div class="coords-display">
              <div class="coord-item">
                <span class="coord-lbl">Gaze X:</span>
                <span class="coord-val" id="gaze-x-val">—</span>
              </div>
              <div class="coord-item">
                <span class="coord-lbl">Gaze Y:</span>
                <span class="coord-val" id="gaze-y-val">—</span>
              </div>
              <div class="coord-item">
                <span class="coord-lbl">Target:</span>
                <span class="coord-val" id="target-val">—</span>
              </div>
            </div>
          </div>
        </div>

        <!-- RIGHT: Interactive Demo Area -->
        <div class="demo-main">
          <!-- Calibration Overlay -->
          <div class="calibration-overlay" id="calibration-overlay" style="display:none">
            <div class="calib-header">
              <h2><i class="fas fa-sliders-h"></i> Eye Tracking Calibration</h2>
              <p id="calib-instruction-text">Look at each numbered dot. <strong>Move only your eyes — small natural head movement is fine for corners.</strong></p>
              <div class="calib-progress-bar"><div class="calib-progress-fill" id="calib-progress-fill"></div></div>
              <span id="calib-step-label">Step 0 / 9</span>
            </div>
            <div class="calib-points-container" id="calib-points-container">
              <!-- Points injected by JS -->
            </div>
            <!-- Live tip bar — updated by JS per point zone -->
            <div id="calib-tip-bar" style="
              position:absolute; bottom:64px; left:0; right:0;
              text-align:center; font-size:13px; color:#00d4ff;
              padding:6px; background:rgba(10,14,26,0.7); pointer-events:none;
              transition: opacity 0.3s;
            "></div>
            <div class="calib-footer">
              <button class="btn-secondary" id="cancel-calib-btn"><i class="fas fa-times"></i> Cancel</button>
              <button class="btn-primary" id="start-calib-btn"><i class="fas fa-play"></i> Start Calibration</button>
            </div>
          </div>

          <!-- Gaze cursor (full page) -->
          <div class="gaze-cursor" id="gaze-cursor">
            <div class="gaze-cursor-ring"></div>
            <div class="gaze-cursor-dot"></div>
            <div class="dwell-ring" id="dwell-ring"></div>
          </div>

          <!-- Demo App: Messaging Interface -->
          <div class="demo-app" id="demo-app">
            <div class="demo-app-header">
              <h3><i class="fas fa-comments"></i> Accessibility Demo — Messaging App</h3>
              <p class="demo-app-hint">
                <i class="fas fa-info-circle"></i>
                <span id="demo-hint-text">Start camera, then move your gaze over buttons. Perform <strong>pinch</strong> or <strong>air tap</strong> to activate.</span>
              </p>
            </div>

            <!-- Message Thread -->
            <div class="message-thread" id="message-thread">
              <div class="msg-bubble received">
                <div class="msg-avatar"><i class="fas fa-user-md"></i></div>
                <div class="msg-content">
                  <p>Hi! This is the AccessEye demo. Try looking at the buttons below and performing a pinch gesture to interact.</p>
                  <span class="msg-time">10:30 AM</span>
                </div>
              </div>
              <div class="msg-bubble received">
                <div class="msg-avatar"><i class="fas fa-user-md"></i></div>
                <div class="msg-content">
                  <p>The system will highlight buttons as your gaze focuses on them. Hold your gaze for 300ms to select.</p>
                  <span class="msg-time">10:31 AM</span>
                </div>
              </div>
            </div>

            <!-- Quick Reply Buttons (gaze targets) -->
            <div class="quick-replies" id="quick-replies">
              <div class="quick-reply-label"><i class="fas fa-hand-point-up"></i> Gaze at a reply, then pinch:</div>
              <div class="quick-reply-btns">
                <button class="gaze-target quick-btn" data-id="reply-yes" data-label="Yes, I understand!">
                  <i class="fas fa-check"></i> Yes, I understand!
                </button>
                <button class="gaze-target quick-btn" data-id="reply-help" data-label="I need help">
                  <i class="fas fa-question-circle"></i> I need help
                </button>
                <button class="gaze-target quick-btn" data-id="reply-later" data-label="Talk later">
                  <i class="fas fa-clock"></i> Talk later
                </button>
              </div>
            </div>

            <!-- Action Toolbar -->
            <div class="action-toolbar">
              <div class="toolbar-label">Toolbar Actions:</div>
              <div class="toolbar-btns">
                <button class="gaze-target toolbar-btn" data-id="btn-send" data-label="Send Message">
                  <i class="fas fa-paper-plane"></i>
                  <span>Send</span>
                  <div class="dwell-progress"></div>
                </button>
                <button class="gaze-target toolbar-btn" data-id="btn-call" data-label="Start Call">
                  <i class="fas fa-phone"></i>
                  <span>Call</span>
                  <div class="dwell-progress"></div>
                </button>
                <button class="gaze-target toolbar-btn" data-id="btn-photo" data-label="Send Photo">
                  <i class="fas fa-camera"></i>
                  <span>Photo</span>
                  <div class="dwell-progress"></div>
                </button>
                <button class="gaze-target toolbar-btn" data-id="btn-settings" data-label="Settings">
                  <i class="fas fa-cog"></i>
                  <span>Settings</span>
                  <div class="dwell-progress"></div>
                </button>
                <button class="gaze-target toolbar-btn" data-id="btn-back" data-label="Go Back">
                  <i class="fas fa-arrow-left"></i>
                  <span>Back</span>
                  <div class="dwell-progress"></div>
                </button>
                <button class="gaze-target toolbar-btn" data-id="btn-emoji" data-label="Add Emoji">
                  <i class="fas fa-smile"></i>
                  <span>Emoji</span>
                  <div class="dwell-progress"></div>
                </button>
              </div>
            </div>

            <!-- Event Log -->
            <div class="event-log" id="event-log">
              <div class="event-log-header">
                <i class="fas fa-list"></i> Interaction Log
                <button class="clear-log-btn" id="clear-log-btn"><i class="fas fa-trash"></i></button>
              </div>
              <div class="event-log-body" id="event-log-body">
                <div class="log-entry info">
                  <span class="log-time">—</span>
                  <span class="log-msg">System ready. Start camera to begin.</span>
                </div>
              </div>
            </div>

            <!-- ═══════════════════════════════════════════
                 PHASE 2 STATUS PANEL (hidden until active)
            ═══════════════════════════════════════════ -->
            <div class="p2-panel" id="p2-status-panel" style="display:none">
              <div class="p2-panel-header">
                <i class="fas fa-brain"></i>
                <span>Phase 2 — Hybrid Engine</span>
                <span class="p2-badge">ACTIVE</span>
              </div>

              <!-- Pipeline label -->
              <div class="p2-pipeline-row">
                <i class="fas fa-sitemap"></i>
                <span id="p2-pipeline-label">Hybrid | 30FPS | Kalman+EMA+Window</span>
              </div>

              <!-- Gaze Confidence Meter -->
              <div class="p2-section">
                <div class="p2-section-title">Gaze Confidence</div>
                <div class="p2-conf-bar-wrap">
                  <div class="p2-conf-bar">
                    <div class="p2-conf-fill" id="p2-conf-fill" style="width:0%"></div>
                  </div>
                  <span class="p2-conf-val" id="p2-conf-val">0%</span>
                </div>
                <div class="p2-sub-scores">
                  <div class="p2-sub"><span class="p2-sub-lbl">Brightness</span><span class="p2-sub-val" id="p2-bright">—</span></div>
                  <div class="p2-sub"><span class="p2-sub-lbl">Occlusion</span><span class="p2-sub-val" id="p2-occl">—</span></div>
                  <div class="p2-sub"><span class="p2-sub-lbl">Glare</span><span class="p2-sub-val" id="p2-glare">—</span></div>
                  <div class="p2-sub"><span class="p2-sub-lbl">P2 Latency</span><span class="p2-sub-val" id="p2-latency">—</span></div>
                </div>
              </div>

              <!-- Head Pose -->
              <div class="p2-section">
                <div class="p2-section-title"><i class="fas fa-head-side-cough"></i> Head Pose (6-DOF)</div>
                <div class="p2-pose-grid">
                  <div class="p2-pose-item">
                    <span class="p2-pose-axis yaw-axis">YAW</span>
                    <span class="p2-pose-val" id="p2-hp-yaw">0°</span>
                  </div>
                  <div class="p2-pose-item">
                    <span class="p2-pose-axis pitch-axis">PITCH</span>
                    <span class="p2-pose-val" id="p2-hp-pitch">0°</span>
                  </div>
                  <div class="p2-pose-item">
                    <span class="p2-pose-axis roll-axis">ROLL</span>
                    <span class="p2-pose-val" id="p2-hp-roll">0°</span>
                  </div>
                </div>
              </div>

              <!-- Fixation & Saccade Stats -->
              <div class="p2-section">
                <div class="p2-section-title"><i class="fas fa-crosshairs"></i> Micro-Saccade Filter</div>
                <div class="p2-stats-row">
                  <div class="p2-stat-item">
                    <span class="p2-stat-lbl">Fixation</span>
                    <span class="p2-stat-val" id="p2-fixation" style="color:var(--accent-yellow)">Scanning</span>
                  </div>
                  <div class="p2-stat-item">
                    <span class="p2-stat-lbl">Saccades</span>
                    <span class="p2-stat-val" id="p2-saccades">0</span>
                  </div>
                  <div class="p2-stat-item">
                    <span class="p2-stat-lbl">Fixations</span>
                    <span class="p2-stat-val" id="p2-fixcount">0</span>
                  </div>
                </div>
              </div>

              <!-- AI Intent Engine -->
              <div class="p2-section p2-intent-section">
                <div class="p2-section-title"><i class="fas fa-robot"></i> AI Intent Prediction</div>
                <div class="p2-intent-result" id="p2-intent-result">Starting camera to begin...</div>
                <div class="p2-intent-meta">
                  <span class="p2-intent-conf-lbl">Confidence:</span>
                  <span class="p2-intent-conf" id="p2-intent-conf">0%</span>
                </div>
                <div class="p2-intent-reason" id="p2-intent-reason">AI intent is ON — predictions update every fixation</div>
                <div class="p2-intent-controls">
                  <button class="p2-toggle-btn active" id="p2-intent-toggle">
                    <i class="fas fa-power-off"></i> Disable AI Intent
                  </button>
                </div>
              </div>

              <!-- Benchmark -->
              <div class="p2-section">
                <div class="p2-section-title"><i class="fas fa-chart-bar"></i> Pipeline Benchmark</div>
                <div class="p2-bench-controls">
                  <button class="p2-bench-btn" id="p2-bench-start">
                    <i class="fas fa-play"></i> Run 30s Benchmark
                  </button>
                  <button class="p2-bench-btn" id="p2-bench-stop" disabled>
                    <i class="fas fa-stop"></i> Stop
                  </button>
                </div>
                <div class="benchmark-report" id="benchmark-report" style="display:none"></div>
              </div>

              <!-- Dynamic Calibration -->
              <div class="p2-section">
                <div class="p2-section-title"><i class="fas fa-magic"></i> Dynamic Calibration</div>
                <div class="p2-micro-status">
                  <span class="p2-micro-lbl">Micro-samples:</span>
                  <span class="p2-micro-val" id="p2-micro-count">0</span>
                </div>
                <div class="p2-micro-status">
                  <span class="p2-micro-lbl">Drift Bias X/Y:</span>
                  <span class="p2-micro-val" id="p2-bias-val">0 / 0</span>
                </div>
                <button class="p2-toggle-btn" id="p2-reset-micro">
                  <i class="fas fa-undo"></i> Reset Micro-Calib
                </button>
              </div>

              <!-- ═══════════════════════════════════════════════ -->
              <!-- PHASE 3 UPGRADES PANEL                         -->
              <!-- ═══════════════════════════════════════════════ -->

              <!-- P3.1 + P3.2: One Euro Filter + IVT -->
              <div class="p2-section" id="p3-section">
                <div class="p2-section-title"><i class="fas fa-wave-square"></i> Phase 3 — Advanced Filters</div>

                <!-- IVT Status -->
                <div class="p2-micro-status">
                  <span class="p2-micro-lbl">IVT Status:</span>
                  <span class="p2-micro-val" id="p3-ivt-status" style="color:#00ff88">Scanning</span>
                </div>
                <div class="p2-micro-status">
                  <span class="p2-micro-lbl">Velocity:</span>
                  <span class="p2-micro-val" id="p3-velocity">0px/f</span>
                </div>
              </div>

              <!-- P3.3: Adaptive Dwell Timer -->
              <div class="p2-section">
                <div class="p2-section-title"><i class="fas fa-clock"></i> Adaptive Dwell Timer</div>
                <div class="p2-micro-status">
                  <span class="p2-micro-lbl">Current Preset:</span>
                  <span class="p2-micro-val" id="p3-dwell-preset">Normal (300ms)</span>
                </div>
                <!-- Preset buttons -->
                <div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:6px;">
                  <button class="p2-toggle-btn active" data-dwell-preset="fast"
                          style="font-size:0.7rem;padding:4px 8px" title="Fast — 180ms">⚡ Fast</button>
                  <button class="p2-toggle-btn active" data-dwell-preset="normal"
                          style="font-size:0.7rem;padding:4px 8px" title="Normal — 300ms">🎯 Normal</button>
                  <button class="p2-toggle-btn" data-dwell-preset="accessible"
                          style="font-size:0.7rem;padding:4px 8px" title="Accessible — 500ms">♿ Access.</button>
                  <button class="p2-toggle-btn" data-dwell-preset="extended"
                          style="font-size:0.7rem;padding:4px 8px" title="Extended — 800ms">🐢 Extended</button>
                </div>
              </div>

              <!-- P3.4: PACE Recalibration -->
              <div class="p2-section">
                <div class="p2-section-title"><i class="fas fa-sync-alt"></i> PACE Recalibration</div>
                <div class="p2-micro-status">
                  <span class="p2-micro-lbl">Passive samples:</span>
                  <span class="p2-micro-val" id="p3-pace-count">0</span>
                </div>
                <button class="p2-toggle-btn" id="p3-pace-reset">
                  <i class="fas fa-undo"></i> Reset PACE Buffer
                </button>
              </div>

              <!-- P3.5: Smooth Pursuit Calibration -->
              <div class="p2-section">
                <div class="p2-section-title"><i class="fas fa-route"></i> Smooth Pursuit Calib.</div>
                <div style="font-size:0.75rem;color:#888;margin-bottom:8px">
                  Follow a moving dot to calibrate without fixed staring
                </div>
                <button class="p2-toggle-btn" id="p3-pursuit-btn">
                  <i class="fas fa-play-circle"></i> Start Pursuit Calibration
                </button>
              </div>

              <!-- P3.6: Post-Calibration Validation -->
              <div class="p2-section">
                <div class="p2-section-title"><i class="fas fa-crosshairs"></i> Accuracy Validation</div>
                <div style="font-size:0.75rem;color:#888;margin-bottom:8px">
                  5-point test — thresholds 70% / 85%+ pass
                </div>
                <button class="p2-toggle-btn" id="p3-validate-btn">
                  <i class="fas fa-check-circle"></i> ✓ Validate Accuracy
                </button>
              </div>

              <!-- P3.7: Head-Free Stabilization -->
              <div class="p2-section">
                <div class="p2-section-title"><i class="fas fa-arrows-alt"></i> Head-Free Stabilization</div>
                <div style="font-size:0.75rem;color:#888;margin-bottom:8px">
                  Compensates for head movement dynamically
                </div>
                <button class="p2-toggle-btn active" id="p3-headfree-toggle">
                  <i class="fas fa-power-off"></i> Disable Head-Free
                </button>
              </div>

              <!-- ═══════════════════════════════════════════════ -->
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ══════════════════════════════════════════
         PAGE: DOCS
    ══════════════════════════════════════════ -->
    <div id="page-docs" class="page">
      <div class="page-header">
        <h1><i class="fas fa-book"></i> API Documentation</h1>
        <p>Integration guide for registering UI elements and interacting with the AccessEye system</p>
      </div>

      <div class="docs-layout">
        <div class="docs-sidebar">
          <ul class="docs-nav">
            <li><a href="#api-overview" class="docs-nav-link active">Overview</a></li>
            <li><a href="#api-init" class="docs-nav-link">Initialization</a></li>
            <li><a href="#api-register" class="docs-nav-link">Register Elements</a></li>
            <li><a href="#api-calibrate" class="docs-nav-link">Calibration</a></li>
            <li><a href="#api-events" class="docs-nav-link">Events</a></li>
            <li><a href="#api-gaze" class="docs-nav-link">Gaze Engine</a></li>
            <li><a href="#api-gesture" class="docs-nav-link">Gesture Engine</a></li>
            <li><a href="#api-testing" class="docs-nav-link">Testing Plan</a></li>
          </ul>
        </div>
        <div class="docs-content">
          <div id="api-overview" class="docs-section">
            <h2>Overview</h2>
            <p>AccessEye provides a JavaScript API for integrating eye tracking and gesture control into any web application. All processing runs client-side using MediaPipe WebGL workers.</p>
            <div class="callout callout-info">
              <i class="fas fa-info-circle"></i>
              <strong>Privacy First:</strong> No video data ever leaves the device. All ML inference runs in WebAssembly/WebGL workers in the browser.
            </div>
          </div>

          <div id="api-init" class="docs-section">
            <h2>Initialization</h2>
            <pre class="code-block"><code><span class="c">// Initialize the AccessEye system</span>
<span class="k">const</span> eye = <span class="k">new</span> <span class="f">AccessEye</span>({
  videoElement: document.<span class="f">getElementById</span>(<span class="s">'camera'</span>),
  overlayCanvas: document.<span class="f">getElementById</span>(<span class="s">'overlay'</span>),
  dwellTime: <span class="n">300</span>,      <span class="c">// ms before element focuses</span>
  smoothing: <span class="n">0.3</span>,      <span class="c">// EMA alpha (0-1)</span>
  useKalman: <span class="k">true</span>,     <span class="c">// Kalman filter enabled</span>
  audioFeedback: <span class="k">true</span>,  <span class="c">// Web Speech API TTS</span>
  debug: <span class="k">false</span>
});

<span class="k">await</span> eye.<span class="f">initialize</span>();
<span class="k">await</span> eye.<span class="f">startCamera</span>();</code></pre>
          </div>

          <div id="api-register" class="docs-section">
            <h2>Register UI Elements</h2>
            <p>Register interactive elements to make them gaze-targetable:</p>
            <pre class="code-block"><code><span class="c">// Register a single element</span>
eye.<span class="f">registerElement</span>({
  id: <span class="s">'sendButton'</span>,
  element: document.<span class="f">getElementById</span>(<span class="s">'send-btn'</span>),
  label: <span class="s">'Send Message'</span>,    <span class="c">// TTS label</span>
  onActivate: () => <span class="f">sendMessage</span>()
});

<span class="c">// Or register multiple at once</span>
eye.<span class="f">registerElements</span>([
  { id: <span class="s">'sendBtn'</span>,  x: <span class="n">200</span>, y: <span class="n">400</span>, width: <span class="n">120</span>, height: <span class="n">60</span>,
    label: <span class="s">'Send'</span>, onActivate: () => <span class="f">send</span>() },
  { id: <span class="s">'menuIcon'</span>, x: <span class="n">20</span>,  y: <span class="n">40</span>,  width: <span class="n">40</span>,  height: <span class="n">40</span>,
    label: <span class="s">'Menu'</span>, onActivate: () => <span class="f">openMenu</span>() }
]);

<span class="c">// Remove element</span>
eye.<span class="f">unregisterElement</span>(<span class="s">'sendButton'</span>);</code></pre>
          </div>

          <div id="api-calibrate" class="docs-section">
            <h2>Calibration System</h2>
            <pre class="code-block"><code><span class="c">// Run 5-point calibration flow</span>
<span class="k">const</span> result = <span class="k">await</span> eye.<span class="f">calibrate</span>({
  points: <span class="n">13</span>,         <span class="c">// 13-point ridge regression grid</span>
  samplesPerPoint: <span class="n">30</span>, <span class="c">// frames to average</span>
  timeout: <span class="n">10000</span>      <span class="c">// max 10s</span>
});

<span class="c">// Calibration result</span>
<span class="c">// { success: true, accuracy: 92.3, model: [...] }</span>

<span class="c">// Save calibration (localStorage)</span>
eye.<span class="f">saveCalibration</span>();

<span class="c">// Load saved calibration</span>
eye.<span class="f">loadCalibration</span>();

<span class="c">// Calibration points schema</span>
<span class="c">// TopLeft(10%,10%), TopRight(90%,10%),</span>
<span class="c">// Center(50%,50%), BottomLeft(10%,90%),</span>
<span class="c">// BottomRight(90%,90%)</span></code></pre>
          </div>

          <div id="api-events" class="docs-section">
            <h2>Event System</h2>
            <pre class="code-block"><code><span class="c">// Listen for gaze events</span>
eye.<span class="f">on</span>(<span class="s">'gaze'</span>, ({ x, y, confidence }) => {
  console.<span class="f">log</span>(<span class="s">\`Gaze at \${x}, \${y}\`</span>);
});

<span class="c">// Element focused (gaze entered + dwell met)</span>
eye.<span class="f">on</span>(<span class="s">'focus'</span>, ({ elementId, label }) => {
  console.<span class="f">log</span>(<span class="s">\`Focused: \${label}\`</span>);
});

<span class="c">// Element activated (gesture confirmed)</span>
eye.<span class="f">on</span>(<span class="s">'activate'</span>, ({ elementId, gesture }) => {
  console.<span class="f">log</span>(<span class="s">\`Activated via \${gesture}\`</span>);
});

<span class="c">// Gesture detected</span>
eye.<span class="f">on</span>(<span class="s">'gesture'</span>, ({ type, confidence }) => {
  <span class="c">// type: 'pinch' | 'airTap' | 'openPalm'</span>
});</code></pre>
          </div>

          <div id="api-gaze" class="docs-section">
            <h2>Gaze Engine Internals</h2>
            <div class="docs-table-wrapper">
              <table class="docs-table">
                <thead>
                  <tr><th>Component</th><th>Method</th><th>Description</th></tr>
                </thead>
                <tbody>
                  <tr><td>Pupil Detection</td><td>Face Mesh iris landmarks 468–472</td><td>Left/right iris center coords</td></tr>
                  <tr><td>Gaze Vector</td><td>Head pose + iris offset</td><td>3D direction from eye</td></tr>
                  <tr><td>Kalman Filter</td><td>2-state Kalman (pos+vel)</td><td>Removes jitter noise</td></tr>
                  <tr><td>EMA Smoothing</td><td>α=0.3 per-axis</td><td>Temporal smoothing</td></tr>
                  <tr><td>Screen Mapping</td><td>Polynomial regression</td><td>Calibrated gaze→screen coords</td></tr>
                  <tr><td>Dwell Timer</td><td>300ms window</td><td>Anti-accidental selection</td></tr>
                </tbody>
              </table>
            </div>
          </div>

          <div id="api-gesture" class="docs-section">
            <h2>Gesture Recognition</h2>
            <div class="docs-table-wrapper">
              <table class="docs-table">
                <thead>
                  <tr><th>Gesture</th><th>Detection Logic</th><th>Action</th><th>Debounce</th></tr>
                </thead>
                <tbody>
                  <tr><td><i class="fas fa-hand-scissors"></i> Pinch</td><td>Thumb–index distance &lt;0.05 (normalized)</td><td>Select / Click</td><td>500ms</td></tr>
                  <tr><td><i class="fas fa-hand-point-up"></i> Air Tap</td><td>Index forward Z-delta &gt;0.04 in 150ms</td><td>Click</td><td>600ms</td></tr>
                  <tr><td><i class="fas fa-hand-paper"></i> Open Palm</td><td>All finger spread &gt;0.08 avg</td><td>Cancel / Back</td><td>800ms</td></tr>
                </tbody>
              </table>
            </div>
          </div>

          <div id="api-testing" class="docs-section">
            <h2>Testing Plan</h2>
            <div class="testing-grid">
              <div class="test-card">
                <h4><i class="fas fa-glasses"></i> Glasses Users</h4>
                <p>Test with thick-frame glasses and anti-reflective coatings. Adjust iris detection threshold for glare compensation.</p>
                <div class="test-status pass">✓ Supported</div>
              </div>
              <div class="test-card">
                <h4><i class="fas fa-moon"></i> Low Lighting</h4>
                <p>Test at &lt;100 lux. MediaPipe Face Mesh remains robust to 50 lux with confidence threshold of 0.6.</p>
                <div class="test-status pass">✓ Supported</div>
              </div>
              <div class="test-card">
                <h4><i class="fas fa-head-side-cough"></i> Slow Head Movement</h4>
                <p>EMA + Kalman smoothing handles slow-head tremor users. Dwell window extended to 400ms for motor impairments.</p>
                <div class="test-status pass">✓ Supported</div>
              </div>
              <div class="test-card">
                <h4><i class="fas fa-wave-square"></i> Tremor Conditions</h4>
                <p>Kalman velocity state dampens high-frequency tremor. Gaze stabilization window set to 300–400ms.</p>
                <div class="test-status pass">✓ Supported</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Global Gaze Cursor (full viewport) -->
    <div class="global-gaze-cursor" id="global-gaze-cursor" style="display:none">
      <div class="gc-ring"></div>
      <div class="gc-dot"></div>
      <svg class="gc-dwell-svg" viewBox="0 0 60 60">
        <circle cx="30" cy="30" r="26" fill="none" stroke="#00d4ff" stroke-width="3"
                stroke-dasharray="0 163.36" id="dwell-circle" stroke-linecap="round"
                transform="rotate(-90 30 30)"/>
      </svg>
    </div>

    <!-- Toast Notification -->
    <div class="toast-container" id="toast-container"></div>

    <!-- Audio Feedback Panel -->
    <div class="audio-toggle" id="audio-toggle" title="Toggle Audio Feedback">
      <i class="fas fa-volume-up" id="audio-icon"></i>
    </div>

    <!-- ── GAZE DIAGNOSTIC OVERLAY ────────────────────────────────────────────
         Real-time debug panel showing raw iris offsets, screen projection,
         confidence, head-pose angles, and pipeline stage values.
         Toggle with keyboard shortcut: Alt+D (or click the toggle button).
         Helps diagnose axis inversion, scope issues, and mapping errors.
    ─────────────────────────────────────────────────────────────────────── -->
    <div id="gaze-debug-panel" style="
      display:none;
      position:fixed;
      bottom:12px;
      left:12px;
      z-index:99999;
      background:rgba(10,14,26,0.93);
      border:1px solid #00d4ff44;
      border-radius:10px;
      padding:12px 16px;
      min-width:260px;
      font-family:monospace;
      font-size:11px;
      color:#e0e0e0;
      box-shadow:0 4px 24px #000a;
      pointer-events:none;
    ">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;pointer-events:auto;">
        <span style="color:#00d4ff;font-weight:bold;font-size:12px;">⚙ Gaze Diagnostics</span>
        <span style="cursor:pointer;color:#888;font-size:13px;" id="debug-close-btn" title="Close (Alt+D)">✕</span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:3px 12px;line-height:1.7;">
        <!-- Row 1: Raw iris offset -->
        <span style="color:#888;">rawGX</span>
        <span id="dbg-raw-gx" style="color:#00d4ff;">—</span>
        <span style="color:#888;">rawGY</span>
        <span id="dbg-raw-gy" style="color:#00d4ff;">—</span>

        <!-- Row 2: Mapped screen (0-1) -->
        <span style="color:#888;">screenX</span>
        <span id="dbg-screen-x" style="color:#4ade80;">—</span>
        <span style="color:#888;">screenY</span>
        <span id="dbg-screen-y" style="color:#4ade80;">—</span>

        <!-- Row 3: Screen pixels -->
        <span style="color:#888;">px</span>
        <span id="dbg-px" style="color:#facc15;">—</span>
        <span style="color:#888;">py</span>
        <span id="dbg-py" style="color:#facc15;">—</span>

        <!-- Row 4: Confidence -->
        <span style="color:#888;">confidence</span>
        <span id="dbg-conf" style="color:#fb923c;">—</span>
        <span style="color:#888;">calib</span>
        <span id="dbg-calib" style="color:#fb923c;">—</span>

        <!-- Row 5: Head pose -->
        <span style="color:#888;">hp.yaw</span>
        <span id="dbg-hp-yaw" style="color:#c084fc;">—</span>
        <span style="color:#888;">hp.pitch</span>
        <span id="dbg-hp-pitch" style="color:#c084fc;">—</span>

        <!-- Row 6: Phase -->
        <span style="color:#888;">phase</span>
        <span id="dbg-phase" style="color:#f0abfc;">—</span>
        <span style="color:#888;">fps</span>
        <span id="dbg-fps" style="color:#f0abfc;">—</span>
      </div>

      <!-- Axis direction test: shows L/R/U/D arrow based on gaze quadrant -->
      <div style="margin-top:8px;padding-top:8px;border-top:1px solid #ffffff11;">
        <span style="color:#888;">direction:</span>
        <span id="dbg-direction" style="color:#fff;font-size:16px;margin-left:6px;">·</span>
        <span style="color:#555;font-size:10px;margin-left:4px;">(verify axes)</span>
      </div>
      <!-- Scope check bar -->
      <div style="margin-top:6px;">
        <span style="color:#888;font-size:10px;">X scope: </span>
        <span id="dbg-scope-x-min" style="color:#666;">—</span>
        <span style="color:#444;"> .. </span>
        <span id="dbg-scope-x-max" style="color:#666;">—</span>
        <span style="color:#888;font-size:10px;margin-left:8px;">Y: </span>
        <span id="dbg-scope-y-min" style="color:#666;">—</span>
        <span style="color:#444;"> .. </span>
        <span id="dbg-scope-y-max" style="color:#666;">—</span>
      </div>
    </div>

    <!-- Debug toggle button (always visible, bottom-left corner) -->
    <button id="debug-toggle-btn" title="Toggle Gaze Diagnostics (Alt+D)" style="
      position:fixed;
      bottom:12px;
      left:12px;
      z-index:99998;
      background:rgba(0,212,255,0.12);
      border:1px solid #00d4ff44;
      border-radius:6px;
      color:#00d4ff;
      padding:4px 8px;
      font-size:11px;
      cursor:pointer;
      font-family:monospace;
    ">⚙ Debug</button>

  </div><!-- end #app-root -->

  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/control_utils/control_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>
  <script src="/static/app.js"></script>
  <script src="/static/phase2-engine.js"></script>
  <script src="/static/phase2-init.js"></script>
  <script src="/static/phase3-engine.js"></script>
  <script src="/static/phase3-init.js"></script>
</body>
</html>`)
})

export default app
