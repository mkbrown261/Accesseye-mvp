# AccessEye — Chrome Extension

Persistent voice control layer for the browser. Built on the AccessEye MVP.

## File Structure

```
extension/
├── manifest.json      — MV3 manifest
├── background.js      — Service worker: voice routing + tab control
├── voiceEngine.js     — Command parser + intent mapper (module)
├── offscreen.html     — Offscreen document host (Chrome 109+)
├── offscreen.js       — Web Speech API lives here (MV3 workaround)
├── content.js         — Injected into all pages: scroll + click
├── popup.html         — Extension popup UI
├── popup.js           — Popup logic
└── icons/             — Extension icons
```

## Architecture

```
Mic → offscreen.js (SpeechRecognition)
         ↓ VOICE_RESULT message
      background.js (service worker)
         ↓ processVoiceInput() via voiceEngine.js
         ↓ intent routing
    ┌────┴─────────────┐
    │                  │
 Tab APIs          sendToContent()
 (chrome.tabs)         ↓
                   content.js
                   (scroll/click)
```

## Voice Commands

| Say | Action |
|-----|--------|
| "New tab" | Open + focus new tab |
| "Next tab" | Switch right |
| "Previous tab" | Switch left |
| "Close tab" | Close current tab |
| "Scroll down" | Scroll 600px down |
| "Scroll up" | Scroll 600px up |
| "Scroll to top" | Jump to top |
| "Scroll to bottom" | Jump to bottom |
| "Go back" | Browser back |
| "Go forward" | Browser forward |
| "Reload" | Refresh page |
| "Click [text]" | Click element by text |

## Installation

1. Open Chrome → `chrome://extensions`
2. Enable **Developer Mode**
3. Click **Load Unpacked**
4. Select the `extension/` folder
5. Click the AccessEye icon → **Start Listening**

## Notes

- Uses Chrome Offscreen API (Chrome 109+) for persistent voice recognition
- Zero `window.open()` — all tab ops via `chrome.tabs` API
- No "double lip tab" — removed entirely
- Modular: voiceEngine.js is standalone, easy to extend
