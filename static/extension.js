/**
 * ═══════════════════════════════════════════════════════════════
 *  AccessEye — extension.js
 *  Install wizard + page navigation
 * ═══════════════════════════════════════════════════════════════
 */

(function () {
  'use strict';

  /* ── Page navigation ─────────────────────────────────────── */
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
    document.querySelectorAll('[data-page="extension"]').forEach(el => {
      if (!el.classList.contains('page')) {
        el.addEventListener('click', () => showPage('extension'));
      }
    });
    const heroExtBtn = document.getElementById('get-extension-btn');
    if (heroExtBtn) heroExtBtn.addEventListener('click', () => showPage('extension'));
  }

  /* ── Wizard state ────────────────────────────────────────── */
  let currentStep = 1;
  const TOTAL_STEPS = 4;

  function goToStep(n) {
    // Hide all cards
    for (let i = 1; i <= TOTAL_STEPS + 1; i++) {
      const card = document.getElementById('wiz-step-' + i);
      if (card) card.classList.remove('active');
    }

    // Show target
    const target = document.getElementById('wiz-step-' + n);
    if (target) target.classList.add('active');

    // Update crumbs
    document.querySelectorAll('.wiz-crumb').forEach(c => {
      const s = parseInt(c.dataset.step);
      c.classList.remove('active', 'done');
      if (s === n) c.classList.add('active');
      else if (s < n) c.classList.add('done');
    });

    // Update progress fill
    const fill = document.getElementById('wiz-track-fill');
    if (fill) {
      const pct = n <= TOTAL_STEPS ? ((n - 1) / (TOTAL_STEPS - 1)) * 100 : 100;
      fill.style.width = pct + '%';
    }

    currentStep = n;
    triggerStepAnimation(n);
    window.scrollTo({ top: document.querySelector('.install-wizard-section')?.offsetTop - 80 || 0, behavior: 'smooth' });
  }

  /* ── Per-step animations ─────────────────────────────────── */
  function triggerStepAnimation(step) {
    if (step === 1) {
      // Animate download bar after short delay
      setTimeout(() => {
        const fill = document.getElementById('wiz-dl-bar-fill');
        const status = document.getElementById('wiz-dl-status');
        if (fill) fill.style.width = '0%';
        if (status) status.textContent = 'Ready to download';
      }, 100);
    }

    if (step === 2) {
      // Show context menu animation, then show extracted folder
      const menu = document.getElementById('wiz-fe-menu');
      const result = document.getElementById('wiz-fe-result');
      if (menu) menu.style.display = 'block';
      if (result) {
        result.style.display = 'none';
        setTimeout(() => { result.style.display = 'flex'; }, 1800);
      }
    }

    if (step === 3) {
      // Animate: toggle appears, then load unpacked appears, then ext card
      const toggle = document.getElementById('wiz-toggle-switch');
      const loadBtn = document.getElementById('wiz-load-unpacked-btn');
      const extCard = document.getElementById('wiz-ext-card-mock');
      if (toggle) toggle.classList.remove('on');
      if (loadBtn) loadBtn.style.display = 'none';
      if (extCard) extCard.style.display = 'none';

      setTimeout(() => { if (toggle) toggle.classList.add('on'); }, 800);
      setTimeout(() => { if (loadBtn) loadBtn.style.display = 'inline-flex'; }, 1400);
      setTimeout(() => { if (extCard) extCard.style.display = 'flex'; }, 2400);
    }

    if (step === 4) {
      // Animate: popup dropdown appears after icon pulse
      const dropdown = document.getElementById('wiz-popup-dropdown');
      if (dropdown) {
        dropdown.style.display = 'none';
        setTimeout(() => { dropdown.style.display = 'block'; }, 1000);
      }
    }

    if (step === 5) {
      // Show success on done card
      const success = document.getElementById('wiz-success');
      if (success) success.style.display = 'flex';
    }
  }

  /* ── Wizard wiring ───────────────────────────────────────── */
  function initWizard() {
    // Download button → animate bar + advance hint
    const dlBtn = document.getElementById('wiz-dl-btn');
    if (dlBtn) {
      dlBtn.addEventListener('click', () => {
        const fill = document.getElementById('wiz-dl-bar-fill');
        const status = document.getElementById('wiz-dl-status');
        if (fill) fill.style.width = '100%';
        if (status) {
          status.textContent = 'Downloading…';
          setTimeout(() => { status.textContent = '✓ Downloaded!'; }, 1200);
        }
      });
    }

    // Next buttons
    document.getElementById('wiz-next-1')?.addEventListener('click', () => goToStep(2));
    document.getElementById('wiz-next-2')?.addEventListener('click', () => goToStep(3));
    document.getElementById('wiz-next-3')?.addEventListener('click', () => goToStep(4));

    // Finish button
    document.getElementById('wiz-finish-btn')?.addEventListener('click', () => {
      const success = document.getElementById('wiz-success');
      if (success) success.style.display = 'flex';
      setTimeout(() => goToStep(5), 600);
    });

    // Restart
    document.getElementById('wiz-restart-btn')?.addEventListener('click', () => goToStep(1));

    // See all commands → scroll to commands section
    document.getElementById('wiz-see-commands-btn')?.addEventListener('click', (e) => {
      e.preventDefault();
      const sec = document.querySelector('.ext-commands-section');
      if (sec) sec.scrollIntoView({ behavior: 'smooth' });
    });

    // Back buttons
    document.querySelectorAll('.wiz-back-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const n = parseInt(btn.dataset.goto);
        if (n) goToStep(n);
      });
    });

    // Crumb navigation (click to go back to a done step)
    document.querySelectorAll('.wiz-crumb').forEach(c => {
      c.addEventListener('click', () => {
        const s = parseInt(c.dataset.step);
        if (s <= currentStep) goToStep(s);
      });
    });

    // Copy chrome://extensions
    document.getElementById('wiz-copy-url-btn')?.addEventListener('click', () => {
      const hint = document.getElementById('wiz-copy-hint');
      navigator.clipboard.writeText('chrome://extensions').then(() => {
        if (hint) hint.style.display = 'inline';
        setTimeout(() => { if (hint) hint.style.display = 'none'; }, 4000);
      }).catch(() => {
        if (hint) {
          hint.textContent = 'Type chrome://extensions in your address bar';
          hint.style.display = 'inline';
        }
      });
    });

    // OS tab switcher (Step 2)
    document.querySelectorAll('.wiz-os-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        const os = tab.dataset.os;
        document.querySelectorAll('.wiz-os-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.wiz-os-inst').forEach(i => i.classList.remove('active'));
        tab.classList.add('active');
        const inst = document.querySelector('.wiz-os-inst[data-os="' + os + '"]');
        if (inst) inst.classList.add('active');
      });
    });

    // Devmode toggle click (Step 3 visual)
    document.getElementById('wiz-devmode-toggle')?.addEventListener('click', () => {
      const toggle = document.getElementById('wiz-toggle-switch');
      const loadBtn = document.getElementById('wiz-load-unpacked-btn');
      if (toggle) {
        toggle.classList.toggle('on');
        if (toggle.classList.contains('on')) {
          if (loadBtn) loadBtn.style.display = 'inline-flex';
        } else {
          if (loadBtn) loadBtn.style.display = 'none';
        }
      }
    });

    // Initial step animation
    triggerStepAnimation(1);
  }

  /* ── Deep-link / hash support ─────────────────────────────── */
  function checkDeepLink() {
    const params = new URLSearchParams(window.location.search);
    if (params.get('page') === 'extension') setTimeout(() => showPage('extension'), 100);
    if (window.location.hash === '#extension') setTimeout(() => showPage('extension'), 100);
  }

  /* ── Init ────────────────────────────────────────────────── */
  function init() {
    hookNav();
    initWizard();
    checkDeepLink();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
