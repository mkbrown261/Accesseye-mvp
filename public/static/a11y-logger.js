/**
 * AccessEye — Accessibility Interaction Logger  v1.0.1
 * Timestamped log of all ACM interactions. CSV + PDF export.
 * ADDITIVE ONLY — reads nothing from core engines.
 */
'use strict';

const LOGGER_VERSION = '1.0.1';

class AccessibilityLogger {
  constructor() {
    this._entries     = [];
    this._sessionId   = 'ses_' + Date.now();
    this._sessionStart= Date.now();
    this._nextId      = 1;
    this._standards   = ['WCAG 2.1 AA', 'ADA Title III', 'Section 508'];
    this._stats       = { gaze:0, voice:0, keyboard:0, intent_fusion:0, system:0, total:0 };
  }

  log(message, modality='system', action='event', el=null, extra=null) {
    const now = Date.now();
    const entry = {
      id:        this._nextId++,
      ts:        new Date(now).toISOString(),
      elapsed_s: ((now - this._sessionStart)/1000).toFixed(2),
      modality:  this._sanitise(modality),
      action,
      message,
      element:   el ? this._descEl(el) : null,
      standards: this._matchStds(modality, action),
      session_id:this._sessionId,
      extra:     extra || null,
    };
    this._entries.push(entry);
    this._stats[entry.modality] = (this._stats[entry.modality]||0)+1;
    this._stats.total++;
    window.dispatchEvent(new CustomEvent('a11y:log', { detail: entry }));
    window.app?.log?.add?.('[A11y/'+entry.modality+'] '+message, 'info');
    return entry;
  }

  get count() { return this._entries.length; }
  get stats()  { return {...this._stats}; }

  exportCSV() {
    const esc = v => {
      const s = String(v==null?'':v);
      return (s.includes(',')||s.includes('"')||s.includes('\n')) ? '"'+s.replace(/"/g,'""')+'"' : s;
    };
    const hdr = ['ID','Timestamp','Elapsed(s)','Modality','Action','Message','Element','Standards','SessionID'];
    const rows = this._entries.map(e=>[
      e.id,e.ts,e.elapsed_s,e.modality,e.action,e.message,e.element||'',(e.standards||[]).join('; '),e.session_id
    ].map(esc).join(','));
    rows.push('','--- Summary ---',
      'Session,'+esc(this._sessionId),
      'Total,'+this._stats.total,
      'Gaze,'+this._stats.gaze,
      'Voice,'+this._stats.voice,
      'Keyboard,'+this._stats.keyboard,
      'IntentFusion,'+this._stats.intent_fusion,
      'Start,'+new Date(this._sessionStart).toISOString(),
      'End,'+new Date().toISOString(),
    );
    this._dl([hdr.join(','),...rows].join('\r\n'),
      'accesseye-a11y-log_'+this._sessionId+'.csv', 'text/csv;charset=utf-8;');
    window.app?.toast?.show?.('Log Exported', this._entries.length+' entries → CSV', 'success','fas fa-file-csv',3000);
  }

  exportPDF() {
    const now = new Date().toLocaleString();
    const dur = ((Date.now()-this._sessionStart)/1000).toFixed(0);
    const mColor = {gaze:'#00b4d8',voice:'#00c853',keyboard:'#f59e0b',intent_fusion:'#7c4dff',system:'#94a3b8'};
    const rows = this._entries.map(e=>`<tr>
      <td style="color:#94a3b8">${e.id}</td>
      <td style="font-size:11px;white-space:nowrap">${e.ts.replace('T',' ').split('.')[0]}</td>
      <td style="color:${mColor[e.modality]||'#94a3b8'};font-weight:600;font-size:11px;text-transform:uppercase">${e.modality}</td>
      <td style="font-size:12px">${e.action}</td>
      <td style="font-size:12px">${this._he(e.message)}</td>
      <td style="font-size:10px;color:#94a3b8">${this._he(e.element||'—')}</td>
    </tr>`).join('');
    const html = `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>AccessEye A11y Log — ${this._sessionId}</title>
<style>*{box-sizing:border-box;margin:0;padding:0}body{font-family:Arial,sans-serif;padding:24px;color:#1a202c}
h1{font-size:20px;color:#00b4d8;margin-bottom:4px}h2{font-size:13px;color:#546e7a;font-weight:400;margin-bottom:16px}
.meta{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:20px}
.mc{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:8px 14px}
.ml{font-size:11px;color:#94a3b8;text-transform:uppercase}.mv{font-size:15px;font-weight:700;margin-top:2px}
.stds{display:flex;gap:6px;margin-bottom:16px}
.sb{background:#e0f7fa;color:#00796b;border:1px solid #b2dfdb;border-radius:12px;padding:3px 10px;font-size:11px;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:12px}
thead tr{background:#f1f5f9}th{padding:7px 9px;text-align:left;font-size:11px;text-transform:uppercase;color:#64748b;border-bottom:2px solid #e2e8f0}
td{padding:5px 9px;border-bottom:1px solid #f1f5f9;vertical-align:top}
.ft{margin-top:20px;font-size:11px;color:#94a3b8;border-top:1px solid #e2e8f0;padding-top:10px}
@media print{button{display:none}}</style></head><body>
<h1>♿ AccessEye — Accessibility Interaction Log</h1>
<h2>Compliance Report — WCAG 2.1 / ADA Title III / Section 508</h2>
<div class="stds"><span class="sb">✓ WCAG 2.1 AA</span><span class="sb">✓ ADA Title III</span><span class="sb">✓ Section 508</span></div>
<div class="meta">
  <div class="mc"><div class="ml">Session</div><div class="mv" style="font-size:12px">${this._sessionId}</div></div>
  <div class="mc"><div class="ml">Total Events</div><div class="mv">${this._stats.total}</div></div>
  <div class="mc"><div class="ml">Gaze</div><div class="mv" style="color:#00b4d8">${this._stats.gaze}</div></div>
  <div class="mc"><div class="ml">Voice</div><div class="mv" style="color:#00c853">${this._stats.voice}</div></div>
  <div class="mc"><div class="ml">Keyboard</div><div class="mv" style="color:#f59e0b">${this._stats.keyboard}</div></div>
  <div class="mc"><div class="ml">Intent Fusion</div><div class="mv" style="color:#7c4dff">${this._stats.intent_fusion}</div></div>
  <div class="mc"><div class="ml">Duration</div><div class="mv">${dur}s</div></div>
  <div class="mc"><div class="ml">Generated</div><div class="mv" style="font-size:11px">${now}</div></div>
</div>
<table><thead><tr><th>#</th><th>Timestamp</th><th>Modality</th><th>Action</th><th>Description</th><th>Element</th></tr></thead>
<tbody>${rows||'<tr><td colspan="6" style="text-align:center;padding:20px;color:#94a3b8">No events recorded yet</td></tr>'}</tbody></table>
<div class="ft">AccessEye ACM v${LOGGER_VERSION} | ${this._standards.join(' · ')} | ${now}</div>
<script>window.onload=()=>window.print()<\/script></body></html>`;
    const w = window.open('','_blank','width=1000,height=700');
    if(w){w.document.write(html);w.document.close();}
    else this._dl(html,'accesseye-a11y-report_'+this._sessionId+'.html','text/html;charset=utf-8;');
    window.app?.toast?.show?.('Report Ready', this._entries.length+' entries — print/save as PDF','success','fas fa-file-pdf',3000);
  }

  getSummary() {
    return {
      session_id:this._sessionId,
      start:new Date(this._sessionStart).toISOString(),
      end:new Date().toISOString(),
      duration_s:((Date.now()-this._sessionStart)/1000).toFixed(1),
      total_events:this._stats.total,
      by_modality:{...this._stats},
      standards:[...this._standards],
      version:LOGGER_VERSION,
    };
  }

  _sanitise(m) { return ['gaze','voice','keyboard','intent_fusion','system'].includes(m)?m:'system'; }
  _descEl(el)  {
    const tag=el.tagName?.toLowerCase()||'?';
    const id=el.id?'#'+el.id:'';
    const lbl=el.getAttribute('aria-label')||el.getAttribute('title')||el.textContent?.trim().replace(/\s+/g,' ').slice(0,40)||'';
    return `<${tag}${id}> "${lbl}"`.trim();
  }
  _matchStds(mod, act) {
    const s=[];
    if(['gaze','voice','intent_fusion','keyboard'].includes(mod)) s.push('WCAG 2.1 SC 2.1.1','Section 508 §1194.21(a)');
    if(act==='focus'||act==='activate') s.push('WCAG 2.1 SC 2.4.3','WCAG 2.1 SC 2.4.7');
    if(act==='dictate') s.push('WCAG 2.1 SC 3.3.2','ADA Title III §36.303');
    if(mod==='voice') s.push('WCAG 2.1 SC 1.3.1');
    return s.length?s:['WCAG 2.1 AA','ADA Title III'];
  }
  _dl(content, filename, mime) {
    const a=document.createElement('a');
    a.href=URL.createObjectURL(new Blob([content],{type:mime}));
    a.download=filename; a.style.display='none';
    document.body.appendChild(a); a.click();
    setTimeout(()=>{URL.revokeObjectURL(a.href);a.remove();},1000);
  }
  _he(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
}

/* Bootstrap — runs immediately, before DOMContentLoaded */
(function(){
  const logger = new AccessibilityLogger();
  window.AccessEye = window.AccessEye || {};
  window.AccessEye.a11yLogger = logger;
  window.a11yLogger = logger;
  console.log('%c Accessibility Logger ✅ v'+LOGGER_VERSION+' — CSV + PDF export ready','color:#00ff88;font-weight:bold;font-size:12px;');
})();
