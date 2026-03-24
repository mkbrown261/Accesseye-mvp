/**
 * ═══════════════════════════════════════════════════════════════
 *  AccessEye — content.js
 *  Injected into all pages. Handles scroll + click commands.
 * ═══════════════════════════════════════════════════════════════
 */

const SCROLL_AMOUNT = 600;

// ── Scroll handler ────────────────────────────────────────────
function handleScroll(direction) {
  switch (direction) {
    case 'down':
      window.scrollBy({ top: SCROLL_AMOUNT, behavior: 'smooth' });
      showToast('⬇ Scrolling down');
      break;
    case 'up':
      window.scrollBy({ top: -SCROLL_AMOUNT, behavior: 'smooth' });
      showToast('⬆ Scrolling up');
      break;
    case 'top':
      window.scrollTo({ top: 0, behavior: 'smooth' });
      showToast('⬆ Top of page');
      break;
    case 'bottom':
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
      showToast('⬇ Bottom of page');
      break;
  }
}

// ── Click handler ─────────────────────────────────────────────
function handleClick(target) {
  if (!target) return;

  // Try matching by visible text content
  const allElements = document.querySelectorAll(
    'a, button, input[type="button"], input[type="submit"], [role="button"], [role="link"]'
  );

  const lowerTarget = target.toLowerCase();

  for (const el of allElements) {
    const text = (el.innerText || el.value || el.getAttribute('aria-label') || '').toLowerCase();
    if (text.includes(lowerTarget)) {
      el.click();
      el.focus();
      showToast(`🖱 Clicked: ${el.innerText || el.value || target}`);
      return;
    }
  }

  showToast(`❓ Could not find: "${target}"`);
}

// ── Toast feedback ────────────────────────────────────────────
let toastEl = null;

function showToast(message) {
  if (!toastEl) {
    toastEl = document.createElement('div');
    toastEl.id = '__accesseye_toast__';
    Object.assign(toastEl.style, {
      position: 'fixed',
      bottom: '24px',
      right: '24px',
      background: 'rgba(0, 0, 0, 0.82)',
      color: '#fff',
      padding: '10px 18px',
      borderRadius: '8px',
      fontSize: '14px',
      fontFamily: 'system-ui, sans-serif',
      zIndex: '2147483647',
      pointerEvents: 'none',
      transition: 'opacity 0.3s ease',
      opacity: '0',
      maxWidth: '280px',
      boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
    });
    document.documentElement.appendChild(toastEl);
  }

  toastEl.textContent = message;
  toastEl.style.opacity = '1';

  clearTimeout(toastEl._hideTimer);
  toastEl._hideTimer = setTimeout(() => {
    toastEl.style.opacity = '0';
  }, 1800);
}

// ── Message listener ──────────────────────────────────────────
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  switch (message.type) {
    case 'SCROLL':
      handleScroll(message.direction);
      sendResponse({ success: true });
      break;

    case 'CLICK':
      handleClick(message.target);
      sendResponse({ success: true });
      break;

    case 'PING':
      sendResponse({ success: true, url: window.location.href });
      break;

    default:
      break;
  }
});
