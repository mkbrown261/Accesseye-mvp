/**
 * AccessEye — Accessibility Control Mode (ACM)  v1.0.2
 *
 * WCAG 2.1 AA/AAA · ADA Title III · Section 508
 *
 * CRITICAL FIX vs v1.0.0:
 *   - NO requestAnimationFrame / setInterval runs on page load.
 *   - The gaze-dwell loop ONLY starts when the user clicks "ACM ON".
 *   - The loop runs at ~10 fps (100 ms throttle), not 60 fps.
 *   - document.elementFromPoint / querySelectorAll are never called
 *     while ACM is disabled.
 *
 * ADDITIVE ONLY — does not modify any core engine, cursor logic,
 * tracking pipeline, calibration, gesture recognition, or snap-to.
 */
'use strict';

const ACM_VERSION = '1.0.2';

const FOCUSABLE = [
  'a[href]','button:not([disabled])',
  'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])','textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
  '[role="button"],[role="link"],[role="menuitem"]',
  '[role="option"],[role="tab"],[role="checkbox"]',
  '[role="radio"],[role="switch"],[contenteditable="true"]',
].join(',');

/* ── Skip-nav link (WCAG 2.4.1) ───────────────────────────────────────── */
function injectSkipNav() {
  if (document.getElementById('a11y-skip-nav')) return;
  const a = document.createElement('a');
  a.id='a11y-skip-nav'; a.href='#demo-main';
  a.textContent='Skip to main content';
  a.setAttribute('aria-label','Skip to main content');
  document.body.insertBefore(a, document.body.firstChild);
}

/* ── Screen-reader live region (WCAG 4.1.3) ──────────────────────────── */
class Announcer {
  constructor() {
    this._el = document.createElement('div');
    Object.assign(this._el, {id:'a11y-sr-announce'});
    this._el.setAttribute('aria-live','assertive');
    this._el.setAttribute('aria-atomic','true');
    this._el.style.cssText='position:absolute;left:-9999px;width:1px;height:1px;overflow:hidden;';
    document.body.appendChild(this._el);
  }
  say(text) {
    this._el.textContent='';
    requestAnimationFrame(()=>{ this._el.textContent=text; });
  }
}

/* ── Enhanced focus ring ──────────────────────────────────────────────── */
class FocusRing {
  constructor() {
    this._el = document.createElement('div');
    this._el.id='a11y-focus-ring';
    this._el.setAttribute('aria-hidden','true');
    this._el.style.cssText=`position:fixed;pointer-events:none;z-index:999990;
      border:3px solid #00d4ff;border-radius:4px;
      box-shadow:0 0 0 2px rgba(0,212,255,.25),0 0 16px 4px rgba(0,212,255,.4);
      transition:all .12s ease;opacity:0;display:none;`;
    document.body.appendChild(this._el);
    this._cur=null;
  }
  show(el) {
    if(!el) return; this._cur=el;
    const r=el.getBoundingClientRect(), p=3;
    Object.assign(this._el.style,{
      display:'block',opacity:'1',
      left:(r.left-p)+'px',top:(r.top-p)+'px',
      width:(r.width+p*2)+'px',height:(r.height+p*2)+'px',
    });
  }
  hide() {
    this._el.style.opacity='0';
    setTimeout(()=>{this._el.style.display='none';},150);
    this._cur=null;
  }
}

/* ── First-time hint overlay ──────────────────────────────────────────── */
class HintOverlay {
  constructor() {
    this._el=null;
    this._gone=localStorage.getItem('acm_hint_v2')==='1';
  }
  show() {
    if(this._gone||this._el) return;
    this._el=document.createElement('div');
    this._el.id='a11y-hint-overlay';
    this._el.innerHTML=`
      <div class="a11y-hint-card">
        <div class="a11y-hint-header">
          <i class="fas fa-universal-access"></i>
          <span>Accessibility Control Mode Active</span>
          <button class="a11y-hint-close" id="a11y-hint-close" aria-label="Dismiss"><i class="fas fa-times"></i></button>
        </div>
        <div class="a11y-hint-body">
          <div class="a11y-hint-row"><span class="a11y-hint-icon"><i class="fas fa-eye"></i></span><span><strong>Gaze</strong> — Look at any element for ~800 ms to activate it</span></div>
          <div class="a11y-hint-row"><span class="a11y-hint-icon"><i class="fas fa-microphone"></i></span><span><strong>Voice</strong> — Say element name or command (e.g. "Click Send")</span></div>
          <div class="a11y-hint-row"><span class="a11y-hint-icon"><i class="fas fa-crosshairs"></i></span><span><strong>Intent Fusion</strong> — Gaze at target + speak action together</span></div>
          <div class="a11y-hint-row"><span class="a11y-hint-icon"><i class="fas fa-magnet"></i></span><span><strong>Snap-To</strong> — Cursor auto-snaps to nearest interactive element</span></div>
          <div class="a11y-hint-row"><span class="a11y-hint-icon"><i class="fas fa-keyboard"></i></span><span><strong>Tab / Arrow keys</strong> work normally for keyboard nav</span></div>
        </div>
        <div class="a11y-hint-footer">
          <label class="a11y-hint-noshow"><input type="checkbox" id="a11y-hint-cb"> Don't show again</label>
          <button class="a11y-hint-btn" id="a11y-hint-ok">Got it</button>
        </div>
      </div>`;
    document.body.appendChild(this._el);
    const dismiss=()=>{
      if(document.getElementById('a11y-hint-cb')?.checked) localStorage.setItem('acm_hint_v2','1');
      this._el.style.opacity='0'; this._el.style.transform='translateY(10px)';
      setTimeout(()=>{this._el?.remove();this._el=null;},300);
      this._gone=true;
    };
    document.getElementById('a11y-hint-close').addEventListener('click',dismiss);
    document.getElementById('a11y-hint-ok').addEventListener('click',dismiss);
    setTimeout(dismiss,15000);
  }
  reset() { this._gone=false; this._el=null; this.show(); }
}

/* ── Gaze-dwell activator ─────────────────────────────────────────────
   SAFE: only ticks while _active===true (set by ACM enable/disable).
   Throttled to ~10 fps via setTimeout, NOT continuous rAF.
   Does zero DOM work while disabled.
─────────────────────────────────────────────────────────────────────── */
class DwellActivator {
  constructor(ring, logger) {
    this._ring=ring; this._logger=logger;
    this._active=false;
    this._dwellMs=800;
    this._dwellEl=null; this._dwellStart=0;
    this._timer=null;
    /* dwell arc indicator */
    this._ind=document.createElement('div');
    this._ind.id='a11y-dwell-ind';
    this._ind.setAttribute('aria-hidden','true');
    this._ind.style.cssText=`position:fixed;pointer-events:none;z-index:999989;
      width:50px;height:50px;display:none;opacity:0;transition:opacity .1s;`;
    this._ind.innerHTML=`<svg width="50" height="50" viewBox="0 0 50 50">
      <circle id="a11y-dwell-arc" cx="25" cy="25" r="21"
        fill="none" stroke="#00ff88" stroke-width="3"
        stroke-dasharray="0 132" stroke-linecap="round"
        transform="rotate(-90 25 25)"/></svg>`;
    document.body.appendChild(this._ind);
  }

  /* Called ONLY from ACM.enable() */
  start() { this._active=true; this._schedule(); }

  /* Called ONLY from ACM.disable() */
  stop()  {
    this._active=false;
    clearTimeout(this._timer); this._timer=null;
    this._dwellEl=null; this._dwellStart=0;
    this._ring.hide(); this._hideInd();
  }

  setDwellMs(ms) { this._dwellMs=Math.max(300,Math.min(2000,ms)); }

  /* ~10 fps tick — only while _active */
  _schedule() {
    if(!this._active) return;
    this._timer=setTimeout(()=>{ this._tick(); this._schedule(); }, 100);
  }

  _tick() {
    const app=window.app;
    const gx=app?._lastScreenX, gy=app?._lastScreenY;
    if(typeof gx!=='number'||typeof gy!=='number') return;

    const hit=document.elementFromPoint(gx,gy);
    const target=hit?.closest(FOCUSABLE)||null;

    if(target && target===this._dwellEl) {
      const prog=Math.min((performance.now()-this._dwellStart)/this._dwellMs,1);
      this._updateInd(gx,gy,prog);
      if(prog>=1){ this._activate(target); this._dwellEl=null; this._dwellStart=0; }
    } else if(target) {
      this._dwellEl=target; this._dwellStart=performance.now();
      this._ring.show(target); this._showInd(gx,gy);
    } else {
      this._dwellEl=null; this._dwellStart=0;
      this._ring.hide(); this._hideInd();
    }
  }

  _activate(el) {
    this._ring.hide(); this._hideInd();
    el.focus();
    el.dispatchEvent(new MouseEvent('click',{bubbles:true,cancelable:true}));
    const lbl=el.getAttribute('aria-label')||el.textContent?.trim().slice(0,50)||el.tagName;
    this._logger?.log(`User activated "${lbl}" via gaze dwell`,'gaze','activate',el);
    this._beep(880,80);
  }

  _showInd(x,y) {
    Object.assign(this._ind.style,{display:'block',opacity:'1',left:(x-25)+'px',top:(y-25)+'px'});
  }
  _updateInd(x,y,p) {
    const arc=document.getElementById('a11y-dwell-arc');
    if(arc) arc.setAttribute('stroke-dasharray',`${132*p} 132`);
    Object.assign(this._ind.style,{left:(x-25)+'px',top:(y-25)+'px'});
  }
  _hideInd() {
    this._ind.style.opacity='0';
    setTimeout(()=>{this._ind.style.display='none';},100);
    const arc=document.getElementById('a11y-dwell-arc');
    if(arc) arc.setAttribute('stroke-dasharray','0 132');
  }
  _beep(freq,ms) {
    try {
      const ctx=new(window.AudioContext||window.webkitAudioContext)();
      const o=ctx.createOscillator(),g=ctx.createGain();
      o.connect(g);g.connect(ctx.destination);
      o.frequency.value=freq;o.type='sine';
      g.gain.setValueAtTime(.12,ctx.currentTime);
      g.gain.exponentialRampToValueAtTime(.001,ctx.currentTime+ms/1000);
      o.start();o.stop(ctx.currentTime+ms/1000);
    } catch(_){}
  }
}

/* ── Dictation (fires only on explicit voice command) ─────────────────── */
class Dictation {
  constructor(logger) {
    this._logger=logger; this._rec=null;
    this._ind=document.createElement('div');
    this._ind.id='a11y-dict-ind';
    this._ind.setAttribute('aria-live','polite');
    this._ind.style.cssText=`position:fixed;bottom:70px;left:50%;transform:translateX(-50%);
      background:rgba(0,255,136,.12);border:1px solid rgba(0,255,136,.4);
      color:#00ff88;padding:6px 16px;border-radius:20px;font-size:.78rem;
      font-weight:600;z-index:999995;display:none;pointer-events:none;`;
    document.body.appendChild(this._ind);
    window.addEventListener('acm:dictate:start',()=>this._start());
    window.addEventListener('acm:dictate:stop', ()=>this._stop());
  }
  _start() {
    const el=document.activeElement;
    const ok=el&&(el.tagName==='INPUT'||el.tagName==='TEXTAREA'||el.isContentEditable);
    if(!ok){ window.app?.toast?.show?.('Dictation','Focus a text field first','warn','fas fa-keyboard',3000); return; }
    if(!('SpeechRecognition' in window||'webkitSpeechRecognition' in window)){
      window.app?.toast?.show?.('Dictation','Speech API not available','error','fas fa-microphone-slash',3000); return;
    }
    this._ind.style.display='block'; this._ind.textContent='🎙 Dictating…';
    const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
    this._rec=new SR();
    this._rec.continuous=false; this._rec.interimResults=false; this._rec.lang='en-US';
    this._rec.onresult=(e)=>{
      const t=e.results[0][0].transcript;
      if(el.tagName==='INPUT'||el.tagName==='TEXTAREA'){
        const s=el.selectionStart??el.value.length, en=el.selectionEnd??el.value.length;
        el.value=el.value.slice(0,s)+t+el.value.slice(en);
        el.selectionStart=el.selectionEnd=s+t.length;
        el.dispatchEvent(new Event('input',{bubbles:true}));
      } else {
        const sel=window.getSelection();
        if(sel.rangeCount){sel.deleteFromDocument();sel.getRangeAt(0).insertNode(document.createTextNode(t));sel.collapseToEnd();}
      }
      this._logger?.log(`User dictated: "${t.slice(0,40)}"`, 'voice','dictate',el);
    };
    this._rec.onend=()=>this._stop();
    this._rec.onerror=()=>this._stop();
    this._rec.start();
  }
  _stop() {
    this._ind.style.display='none';
    try{this._rec?.stop();}catch(_){}
    this._rec=null;
  }
}

/* ── Main ACM Controller ──────────────────────────────────────────────── */
class AccessibilityControlMode {
  constructor() {
    this._enabled=false;
    this._standards=['WCAG 2.1 AA','ADA Title III','Section 508'];
    this._ring=new FocusRing();
    this._ann=new Announcer();
    this._hint=new HintOverlay();
    this._logger=null;
    this._dwell=null;
    this._dict=null;
    this._keyFn=this._onKey.bind(this);
  }

  init(logger) {
    this._logger=logger;
    this._dwell=new DwellActivator(this._ring, logger);
    this._dict=new Dictation(logger);
    injectSkipNav();
    this._wireUI();
    this._watchFocus();
    this._patchVoiceNav();
    this._updateUI();
    window.dispatchEvent(new CustomEvent('acm:ready',{detail:{version:ACM_VERSION}}));
    console.log('%c Accessibility Control Mode ✅ v'+ACM_VERSION+' — WCAG/ADA/508','color:#00ff88;font-weight:bold;font-size:12px;');
  }

  enable() {
    if(this._enabled) return;
    this._enabled=true;
    this._dwell.start();                          // ← loop starts HERE, not on load
    document.addEventListener('keydown',this._keyFn,true);
    document.body.classList.add('a11y-mode-active');
    this._updateUI();
    this._hint.show();
    const cnt=document.querySelectorAll(FOCUSABLE).length;
    this._setCount(cnt);
    this._logger?.log(`ACM enabled — ${cnt} interactive elements indexed`,'system','mode_change',null,{standards:this._standards});
    this._ann.say('Accessibility Control Mode enabled. Gaze, voice, and intent fusion are active.');
    window.app?.toast?.show?.('Accessibility Mode',`Active — ${cnt} elements indexed`,'success','fas fa-universal-access',3000);
  }

  disable() {
    if(!this._enabled) return;
    this._enabled=false;
    this._dwell.stop();                           // ← loop stops HERE
    document.removeEventListener('keydown',this._keyFn,true);
    document.body.classList.remove('a11y-mode-active');
    this._ring.hide();
    this._updateUI();
    this._logger?.log('ACM disabled','system','mode_change');
    this._ann.say('Accessibility Control Mode disabled.');
    window.app?.toast?.show?.('Accessibility Mode','Disabled','info','fas fa-universal-access',2000);
  }

  toggle() { this._enabled ? this.disable() : this.enable(); }
  get enabled() { return this._enabled; }

  /* ── UI wiring ── */
  _wireUI() {
    const $=id=>document.getElementById(id);
    $('acm-toggle-btn')    ?.addEventListener('click',()=>this.toggle());
    $('acm-hint-btn')      ?.addEventListener('click',()=>this._hint.reset());
    $('acm-export-csv')    ?.addEventListener('click',()=>{ this._logger?.exportCSV(); this._logger?.log('CSV export','system','export'); });
    $('acm-export-pdf')    ?.addEventListener('click',()=>{ this._logger?.exportPDF(); this._logger?.log('PDF export','system','export'); });
    $('acm-dwell-slider')  ?.addEventListener('input',(e)=>{
      const ms=parseInt(e.target.value);
      this._dwell?.setDwellMs(ms);
      const v=$('acm-dwell-val'); if(v) v.textContent=ms+' ms';
    });
    /* standards badges */
    const sl=$('acm-standards-list');
    if(sl) sl.innerHTML=this._standards.map(s=>`<span class="a11y-std-badge">${s}</span>`).join('');

    /* live element count — rescan on DOM mutations, only while enabled */
    new MutationObserver(()=>{ if(this._enabled) this._setCount(document.querySelectorAll(FOCUSABLE).length); })
      .observe(document.body,{childList:true,subtree:true});
  }

  _updateUI() {
    const btn=document.getElementById('acm-toggle-btn');
    const badge=document.getElementById('acm-status-badge');
    if(btn){
      btn.classList.toggle('active',this._enabled);
      btn.innerHTML=this._enabled
        ?'<i class="fas fa-universal-access"></i> <span>ACM ON</span>'
        :'<i class="fas fa-universal-access"></i> <span>ACM OFF</span>';
    }
    if(badge){
      badge.textContent=this._enabled?'ACTIVE':'INACTIVE';
      badge.className='acm-status-badge'+(this._enabled?' active':'');
    }
  }

  _setCount(n) {
    const el=document.getElementById('acm-element-count');
    if(el) el.textContent=n;
  }

  updateLogCount(n) {
    const el=document.getElementById('acm-log-count');
    if(el) el.textContent=n;
  }

  /* ── Keyboard augmentation (WCAG 2.1.1) — only logs, never blocks ── */
  _onKey(e) {
    if(!this._enabled) return;
    const el=document.activeElement;
    if(!el||el===document.body) return;
    const lbl=el.getAttribute('aria-label')||el.textContent?.trim().slice(0,40)||el.tagName;
    if(e.key==='Tab'){
      this._ring.show(el);
      this._logger?.log(`Navigated to "${lbl}" via Tab`,'keyboard','focus',el);
      this._ann.say('Focused: '+lbl);
    } else if(e.key==='Enter'||e.key===' '){
      this._logger?.log(`Activated "${lbl}" via keyboard (${e.key})`,'keyboard','activate',el);
    }
  }

  /* ── Focus change listener ── */
  _watchFocus() {
    document.addEventListener('focusin',(e)=>{
      if(!this._enabled) return;
      this._ring.show(e.target);
      const lbl=e.target.getAttribute('aria-label')||e.target.textContent?.trim().slice(0,40)||e.target.tagName;
      this._ann.say(lbl);
    });
    document.addEventListener('focusout',()=>{ if(!this._enabled) return; this._ring.hide(); });
  }

  /* ── Voice-nav extension (additive patch) ── */
  _patchVoiceNav() {
    const tryPatch=()=>{
      const vn=window.voiceNav;
      if(!vn){ setTimeout(tryPatch,300); return; }

      /* Wrap _performAction to add ACM logging + new ACM-specific actions */
      const orig=vn._performAction.bind(vn);
      vn._performAction=(entry,action)=>{
        /* ACM-specific actions handled here */
        if(action==='acm:dictate:start'){ window.dispatchEvent(new Event('acm:dictate:start')); return; }
        if(action==='acm:dictate:stop') { window.dispatchEvent(new Event('acm:dictate:stop'));  return; }
        if(action==='acm:toggle')       { this.toggle(); return; }
        if(action==='acm:hint')         { this._hint.reset(); return; }
        if(action==='acm:csv')          { this._logger?.exportCSV(); return; }

        /* Log voice actions through ACM when enabled */
        if(this._enabled && this._logger && entry?.el){
          const lbl=entry.text||action;
          this._logger.log(`User activated "${lbl}" via voice`,'voice',action,entry.el);
        }
        orig(entry,action);
      };

      /* Inject ACM commands into voice multi-word lookup */
      const origMulti=vn._extractMultiWordAction.bind(vn);
      vn._extractMultiWordAction=(lower)=>{
        if(lower.includes('start dictation')||lower.includes('begin dictation')) return 'acm:dictate:start';
        if(lower.includes('stop dictation'))  return 'acm:dictate:stop';
        if(lower.includes('accessibility mode')) return 'acm:toggle';
        if(lower.includes('show guide')||lower.includes('show hint')) return 'acm:hint';
        if(lower.includes('export log')||lower.includes('download log')) return 'acm:csv';
        return origMulti(lower);
      };

      console.log('[ACM] Voice command extensions patched');
    };
    setTimeout(tryPatch,900);
  }
}

/* ── Bootstrap ────────────────────────────────────────────────────────── */
(function(){
  const acm=new AccessibilityControlMode();
  let attempts=0;

  const attach=()=>{
    attempts++;
    /* Merge check — window.AccessEye may have been replaced by app.js */
    const logger=window.AccessEye?.a11yLogger || window.a11yLogger;
    if(!window.app || !logger){
      if(attempts<60) setTimeout(attach,250);
      else console.warn('[ACM] Gave up waiting for app+logger after',attempts,'tries');
      return;
    }
    /* Re-attach logger ref in case AccessEye was replaced */
    window.AccessEye=window.AccessEye||{};
    window.AccessEye.a11yLogger=logger;

    acm.init(logger);
    window.AccessEye.acm=acm;
    window.acm=acm;

    window.addEventListener('a11y:log',()=>acm.updateLogCount(logger.count));
  };

  if(document.readyState==='loading'){
    document.addEventListener('DOMContentLoaded',()=>setTimeout(attach,600));
  } else {
    setTimeout(attach,600);
  }
})();
