/**
 * ═══════════════════════════════════════════════════════════════
 *  AccessEye — extension.js
 *  Extension page nav + onboarding interaction
 * ═══════════════════════════════════════════════════════════════
 */

(function () {
  'use strict';

  // ── Page navigation (wire up the new Extension nav button) ────
  // The main app.js handles nav-btn clicks, but it may initialise
  // before this script runs. We patch in after DOMContentLoaded.

  function showPage(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));

    const page = document.getElementById('page-' + pageId);
    if (page) page.classList.add('active');

    const btn = document.querySelector('[data-page="' + pageId + '"]');
    if (btn) btn.classList.add('active');

    window.scrollTo(0, 0);
  }

  function hookNav() {
    // Extension nav button
    const extBtn = document.querySelector('.nav-btn[data-page="extension"]');
    if (extBtn) {
      extBtn.addEventListener('click', () => showPage('extension'));
    }

    // "Get Browser Extension" hero button
    const heroExtBtn = document.getElementById('get-extension-btn');
    if (heroExtBtn) {
      heroExtBtn.addEventListener('click', () => showPage('extension'));
    }

    // Any other element with data-page="extension"
    document.querySelectorAll('[data-page="extension"]').forEach(el => {
      if (!el.classList.contains('page')) {
        el.addEventListener('click', () => showPage('extension'));
      }
    });
  }

  // ── Onboarding step interaction ──────────────────────────────
  function initOnboarding() {
    // Step 1: mark complete on download
    const downloadBtns = document.querySelectorAll('a[href*="accesseye-extension.zip"]');
    downloadBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        markStepDone(1);
        // Auto-advance to step 2 highlight after short delay
        setTimeout(() => highlightStep(2), 500);
      });
    });

    // Step 3: open chrome://extensions — can't actually navigate there
    // but we show the copy-paste helper
    const openExtBtn = document.getElementById('open-extensions-btn');
    if (openExtBtn) {
      openExtBtn.addEventListener('click', (e) => {
        e.preventDefault();
        // Copy to clipboard and show feedback
        navigator.clipboard.writeText('chrome://extensions').then(() => {
          openExtBtn.innerHTML = '<i class="fas fa-check"></i> Copied! Paste in Chrome address bar';
          openExtBtn.style.background = '#166534';
          openExtBtn.style.color = '#4ade80';
          setTimeout(() => {
            openExtBtn.innerHTML = '<i class="fas fa-external-link-alt"></i> Open chrome://extensions';
            openExtBtn.style.background = '';
            openExtBtn.style.color = '';
          }, 3000);
        }).catch(() => {
          // Fallback: show instruction
          openExtBtn.innerHTML = '<i class="fas fa-info-circle"></i> Type chrome://extensions in your address bar';
        });
      });
    }
  }

  function markStepDone(stepNum) {
    const check = document.getElementById('check-' + stepNum);
    if (check) check.classList.add('visible');
  }

  function highlightStep(stepNum) {
    const step = document.getElementById('step-' + stepNum);
    if (step) {
      step.style.borderColor = '#2563eb';
      step.style.boxShadow = '0 0 0 1px #2563eb44';
      setTimeout(() => {
        step.style.borderColor = '';
        step.style.boxShadow = '';
      }, 2000);
    }
  }

  // ── Deep-link support: ?page=extension ───────────────────────
  function checkDeepLink() {
    const params = new URLSearchParams(window.location.search);
    const page = params.get('page');
    if (page === 'extension') {
      setTimeout(() => showPage('extension'), 100);
    }
  }

  // ── Hash link support: #extension ────────────────────────────
  function checkHash() {
    if (window.location.hash === '#extension') {
      setTimeout(() => showPage('extension'), 100);
    }
  }

  // ── Init ─────────────────────────────────────────────────────
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      hookNav();
      initOnboarding();
      checkDeepLink();
      checkHash();
    });
  } else {
    // DOM already ready
    hookNav();
    initOnboarding();
    checkDeepLink();
    checkHash();
  }

})();
