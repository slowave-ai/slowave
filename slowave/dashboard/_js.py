"""App JS for the Slowave dashboard. Edit this file, not app.py."""

_APP_JS = r"""
const REFRESH_MS=__REFRESH_MS__;
const ALLOW_ACTIONS=__ALLOW_ACTIONS__;

const statusColor={active:"#3ecf6e",needs_review:"#f5b942",contradicted:"#f04e6a",superseded:"#9d71f0",archived:"#5a6e91",forgotten:"#3a3f4d",labile:"#f5b942"};
const relColor={relates_to:"#6366f1",coactivated_with:"#ec4899"};
const relLabel={relates_to:"relates to",coactivated_with:"coactivated"};

// Shared channel palette for the pulse graph and the creation histogram —
// keep in one place so the two views can never drift apart in color.
const CHANNELS=[
  {key:"raw_events",label:"raw events",color:"#10b981",gc:"rgba(16,185,129,"},
  {key:"episodes",  label:"episodes",  color:"#fbbf24",gc:"rgba(251,191,36,"},
  {key:"schemas",   label:"schemas",   color:"#3b82f6",gc:"rgba(59,130,246,"},
];

function truncContent(s,max){
  if(!s||s.length<=max)return esc(s);
  return '<span title="'+esc(s)+'">'+esc(s.slice(0,max-3))+'<span style="color:var(--muted)"> …</span></span>';
}
function esc(s){return String(s??"")
  .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
  .replace(/"/g,"&quot;").replace(/'/g,"&#39;");}
function fmtBytes(n){n=Number(n||0);if(n<1024)return n+" B";if(n<1048576)return (n/1024).toFixed(1)+" KB";return (n/1048576).toFixed(2)+" MB";}
function fmtTs(ts){if(!ts)return "—";return new Date(Number(ts)*1000).toLocaleString();}
function fmtTsCompact(ts){
  if(!ts)return "—";
  const d=new Date(Number(ts)*1000);
  return d.toLocaleTimeString();
}
function fmtTsCompactSub(ts){
  if(!ts)return "";
  return new Date(Number(ts)*1000).toLocaleDateString();
}
function fmtDate(ts){if(!ts)return "—";return new Date(Number(ts)*1000).toLocaleDateString();}
function age(s){s=Number(s||0);if(s<60)return s+"s";if(s<3600)return Math.floor(s/60)+"m";if(s<86400)return Math.floor(s/3600)+"h "+Math.floor((s%3600)/60)+"m";return Math.floor(s/86400)+"d";}
function dur(s){if(s==null||s===undefined)return "open";s=Number(s);if(s<1)return "<1s";return age(s);}
function num(n){return Number(n||0).toLocaleString();}
async function getJSON(url){const r=await fetch(url);return await r.json();}

// ── TOOLTIP ──
const ttEl=document.getElementById("tooltip");
function showTip(e,html){ttEl.innerHTML=html;ttEl.style.display="block";moveTip(e);}
function moveTip(e){if(!e||typeof e.clientX!="number"||typeof e.clientY!="number"){ttEl.style.left="50%";ttEl.style.top="50%";return}const x=e.clientX,y=e.clientY,w=ttEl.offsetWidth,h=ttEl.offsetHeight;ttEl.style.left=Math.min(x+12,window.innerWidth-w-8)+"px";ttEl.style.top=Math.min(y+12,window.innerHeight-h-8)+"px";}
function hideTip(){ttEl.style.display="none";}
document.addEventListener("mousemove",moveTip);

// ── TABS ──
document.querySelectorAll(".tab").forEach(b=>b.onclick=()=>{
  document.querySelectorAll(".tab").forEach(x=>x.classList.remove("active"));
  document.querySelectorAll(".section").forEach(x=>x.classList.remove("active"));
  b.classList.add("active");
  document.getElementById(b.dataset.tab).classList.add("active");
  const tab=b.dataset.tab;
  if(tab==="overview"){renderPulse();renderHistogram();}
  else if(tab==="schemas"){loadSchemas();loadGeneralizationStats();}
  else if(tab==="graph")loadGraph();
  else if(tab==="worker")loadWorker();
  else if(tab==="db")loadDbHealth();
});

// ── HELPERS ──
function pill(status){
  return `<span class="pill pill-${esc(status)}">${esc(status)}</span>`;
}
function salBar(val,max){
  const pct=Math.min(100,Math.round(val/Math.max(0.001,max)*100));
  return `<div class="sal-bar-wrap"><div class="sal-bar-track"><div class="sal-bar-fill" style="width:${pct}%"></div></div><span style="font-size:11px;color:var(--muted)">${Number(val||0).toFixed(2)}</span></div>`;
}
function confBar(val){
  const pct=Math.round(val*100);
  return `<div class="conf-bar"><div class="conf-track"><div class="conf-fill" style="width:${pct}%"></div></div><span style="font-size:11px;color:var(--muted)">${pct}%</span></div>`;
}
function table(head,rows,rawCols=[]){
  if(!rows.length)return emptyState("No data.");
  const ths=head.map(h=>`<th>${esc(h)}</th>`).join("");
  const trs=rows.map((r,ri)=>{
    const tds=r.map((c,ci)=>`<td>${rawCols.includes(ci)?c:esc(c)}</td>`).join("");
    return `<tr>${tds}</tr>`;
  }).join("");
  return `<div class="table-wrap"><table><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table></div>`;
}
function emptyState(msg,icon="📭"){
  return `<div class="empty-state"><div class="es-icon">${icon}</div><div class="es-text">${esc(msg)}</div></div>`;
}

// Shared by every schema-detail view (Schemas tab, Graph tab node panel) that
// renders a schema's evidence list from /api/schemas/{id} -- single source of
// truth so a fix here (like the session merge below) doesn't need repeating
// per call site. A session can produce BOTH a "micro" and a "macro" episode
// from the exact same underlying events (e.g. a 2-event session, where the
// micro window and the whole-session macro episode coincide) -- without
// merging, that shows as two near-identical rows for what's really one
// moment, so rows sharing a session are grouped into one with all their
// kinds badged together.
const EPISODE_KIND_COLOR={"macro":"var(--amber)","micro":"var(--blue)","decision":"var(--purple)","fact":"var(--green)","lesson":"var(--red)"};
function renderEvidenceList(evidence){
  if(!evidence||!evidence.length)return "<em style='color:var(--muted)'>No evidence.</em>";
  const bySession=new Map();
  const ungrouped=[];
  evidence.forEach(e=>{
    const sess=e.episode_session||e.event_session||"";
    if(!sess){ungrouped.push([e]);return;}
    if(!bySession.has(sess))bySession.set(sess,[]);
    bySession.get(sess).push(e);
  });
  const groups=[...bySession.values(),...ungrouped];
  groups.sort((a,b)=>Math.max(...b.map(e=>Number(e.weight||0)))-Math.max(...a.map(e=>Number(e.weight||0))));
  const rows=groups.map(group=>{
    const epIds=[...new Set(group.map(e=>e.episode_id).filter(Boolean))];
    const epLabel=epIds.length?epIds.map(id=>`epi_${id}`).join("+"):"—";
    const kinds=[...new Set(group.map(e=>e.episode_kind).filter(Boolean))];
    const kindBadges=`<span style="display:inline-flex;gap:6px">${kinds.map(k=>`<span style="color:${EPISODE_KIND_COLOR[k]||"var(--muted)"};font-size:10px;font-weight:500">${esc(k)}</span>`).join("")}</span>`;
    const sess=group[0].episode_session||group[0].event_session||"";
    const sessLink=sess?`<span onclick="loadSessionTimeline('${esc(sess)}')" style="cursor:pointer;color:var(--cyan);font-size:11px;white-space:nowrap" title="Open session timeline">${esc(sess)}</span>`:"";
    const weight=Math.max(...group.map(e=>Number(e.weight||0)));
    // Longest quote among the merged rows tends to be the most informative one.
    const quote=group.map(e=>e.quote||e.event_content||"").sort((a,b)=>b.length-a.length)[0]||"";
    return `<div style="margin-bottom:4px;padding:5px 8px;background:var(--panel2);border-radius:4px">
      <div style="display:grid;grid-template-columns:auto auto auto auto;align-items:center;gap:0 14px;font-size:11px;color:var(--muted)">
        <span>${epLabel}</span>${kindBadges}${sessLink}
        <span style="color:var(--green);font-size:11px;text-align:right">w${weight.toFixed(3).replace(/0+$/,'').replace(/\.$/,'.0')}</span>
      </div>
      ${quote?`<div style="color:var(--muted);line-height:1.4;font-size:11px;margin-top:3px;font-style:italic">${esc(quote.slice(0,200))}${quote.length>200?"…":""}</div>`:""}
    </div>`;
  }).join("");
  return `<div style="max-height:400px;overflow-y:auto"><div style="display:grid;grid-template-columns:auto auto auto auto;gap:0 14px;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;padding-bottom:6px;margin-bottom:2px;border-bottom:1px solid var(--line)"><span>Episode</span><span>Kind</span><span>Session</span><span style="text-align:right">Weight</span></div>${rows}</div>`;
}

// ── OVERVIEW ──
async function loadStatus(){
  const d=await getJSON("/api/status");
  window.lastStatus=d;
  document.getElementById("dbPath").textContent=d.db_path;
  document.getElementById("dbPath").title=d.db_path;
  document.getElementById("lastUpdated").textContent="Updated "+new Date().toLocaleTimeString();

  // Init salience slider once we have data
  if(!window.salienceSliderInitialized){initSalienceSlider(d);}

  // Populate scope dropdowns
  ["graphScope","schemaScope","procScope"].forEach(id=>{
    const scopeSel=document.getElementById(id);
    if(scopeSel&&d.scopes){
      const val=scopeSel.value;
      scopeSel.innerHTML='<option value="">(all scopes)</option>'+d.scopes.map(s=>`<option value="${esc(s.scope)}">${esc(s.scope)} (${s.sessions})</option>`).join("");
      scopeSel.value=val;
    }
  });

  const s=d.stats||{}, h=d.schema_health||{};
  const maxSal=Number(h?.active_salience?.max||1);

  // STAT CARDS
  const cards=[
    {icon:"💬",label:"Sessions",val:num(s.sessions),sub:"total",accent:"var(--cyan)"},
    {icon:"⚡",label:"Raw events",val:num(s.raw_events),sub:"logged",accent:"var(--blue)"},
    {icon:"🎞",label:"Episodes",val:num(s.episodes),sub:"formed",accent:"var(--blue)"},
    {icon:"🔵",label:"Prototypes",val:num(s.prototypes),sub:"semantic",accent:"var(--purple)"},
    {icon:"📖",label:"Schemas",val:num(s.schemas),sub:`<span style="display:inline-flex;align-items:center;gap:6px;flex-wrap:wrap"><span style="display:inline-flex;align-items:center;gap:4px"><span style="width:8px;height:8px;border-radius:50%;background:var(--green);display:inline-block;flex-shrink:0"></span><span style="font-size:11px;color:var(--text)">${num(h.active_schemas)} active</span></span><span style="display:inline-flex;align-items:center;gap:4px"><span style="width:8px;height:8px;border-radius:50%;background:var(--amber);display:inline-block;flex-shrink:0"></span><span style="font-size:11px;color:var(--text)">${num(h.needs_review_schemas)} needs review</span></span></span>`,accent:"var(--green)",raw:true},
    {icon:"🕸",label:"Edges",val:num(s.edges),sub:"prototype",accent:"var(--purple)"},
    {icon:"🔗",label:"Relations",val:num(s.schema_relations),sub:"schema",accent:"var(--purple)"},
    {icon:"🗣",label:"Feedback",val:num(s.feedback_events),sub:"events",accent:"var(--cyan)"},
    {icon:"🌐",label:"Promoted",val:num(s.promoted_schemas),sub:"stage ≥ 1",accent:"var(--amber)"},
    {icon:"✨",label:"Global",val:num(s.global_schemas),sub:"stage 3",accent:"var(--green)"},
    {icon:"🗺",label:"Known scopes",val:num(s.known_scopes),sub:"registered",accent:"var(--cyan)"},
    {icon:"💾",label:"DB size",val:fmtBytes(d.db_size_bytes),sub:d.wal_size_bytes>0?"WAL: "+fmtBytes(d.wal_size_bytes):"",accent:"var(--muted)"},
  ];
  document.getElementById("statGrid").innerHTML=cards.map(c=>{
    return `<div class="stat-card" style="--accent:${c.accent}">
      <div class="sc-icon">${c.icon}</div>
      <div class="sc-label">${esc(c.label)}</div>
      <div class="sc-value">${c.raw?c.val:esc(String(c.val??0))}</div>
      <div class="sc-sub">${c.raw?c.sub:esc(c.sub||"")} </div>
    </div>`;
  }).join("");

  // ALERTS
  const warns=d.warnings||[];
  let alertHtml="";
  if(!d.db_exists){
    alertHtml=`<div class="alert alert-error"><span class="alert-icon">❌</span><div><b>Database not found</b><br>Path: ${esc(d.db_path)}</div></div>`;
  } else if(warns.length){
    alertHtml=`<div class="alert alert-warn"><span class="alert-icon">⚠️</span><div><b>${warns.length} warning${warns.length>1?"s":""}</b><ul>${warns.map(w=>`<li>${esc(w)}</li>`).join("")}</ul></div></div>`;
    // Update badge
    
  } else {
    alertHtml=`<div class="alert alert-ok" style="align-items:center"><span class="alert-icon" style="display:flex;align-items:center;justify-content:center;width:16px;height:16px"><span style="width:10px;height:10px;border-radius:50%;background:var(--green);display:block;flex-shrink:0"></span></span><div><b>All systems healthy</b></div></div>`;
    
  }
  document.getElementById("alertArea").innerHTML=alertHtml;

  // SCHEMA HEALTH PANEL
  const byStatus=h.schemas_by_status||{};
  const total=Math.max(1,h.schemas_total||0);
  const statusOrder=["active","needs_review","contradicted","superseded","archived","forgotten"];
  const barSegs=statusOrder.map(st=>{
    const n=byStatus[st]||0;
    const pct=Math.round(n/total*100);
    return n>0?`<div class="status-bar-seg" style="width:${pct}%;background:${statusColor[st]||"#5a6e91"}"
      title="${st}: ${n}"></div>`:"";
  }).join("");
  const sal=h.active_salience||{};
  const avgSal=Number(sal.avg||0);
  const maxSalience=Number(sal.max||0);
  const salPct=maxSalience>0?Math.round(avgSal/maxSalience*100):0;
  const lastConsolidated=d.last_consolidation_ts?"Last session: "+fmtTs(d.last_consolidation_ts):"No sessions yet";
  document.getElementById("schemaHealthPanel").innerHTML=`
    <div class="status-bar">${barSegs||"<div class=\"status-bar-seg\" style=\"width:100%;background:var(--line)\"></div>"}</div>
    <div style="display:flex;flex-wrap:wrap;gap:6px;margin:8px 0">
      ${statusOrder.map(st=>{
        const n=byStatus[st]||0;
        return n>0?`<span class="pill pill-${st}">${st} ${num(n)}</span>`:"";
      }).join("")}
    </div>
    <div style="margin-top:10px;display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px">
      <div><span style="color:var(--muted)">Avg salience</span><br><b>${avgSal.toFixed(2)}</b></div>
      <div><span style="color:var(--muted)">Max salience</span><br><b>${maxSalience.toFixed(2)}</b></div>
      <div><span style="color:var(--muted)">Duplicates</span><br><b>${num(h.active_exact_duplicate_rows||0)}</b></div>
    </div>
    <div style="margin-top:10px;font-size:11px;color:var(--muted)">${esc(lastConsolidated)}</div>
  `;

  // DAEMON HEALTH PANEL
  const dmn=d.daemon||{};
  const daemonRunning=dmn.running===true;
  const daemonBadge=daemonRunning
    ?`<span class="pill pill-ok">&#x2022; running</span>`
    :`<span class="pill pill-orphan">&#x25cf; stopped</span>`;
  const daemonUrl=dmn.url||"http://127.0.0.1:8766/mcp";
  const daemonSessions=Number(dmn.active_sessions||0);
  const daemonVersion=dmn.version?"v"+dmn.version:"";
  document.getElementById("daemonPanel").innerHTML=daemonRunning?`
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
      ${daemonBadge}
      <span style="font-size:12px;color:var(--muted)">${esc(daemonVersion)}</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px;margin-bottom:10px">
      <div><span style="color:var(--muted)">Endpoint</span><br><code style="font-size:11px">${esc(daemonUrl)}</code></div>
      <div><span style="color:var(--muted)">Active sessions</span><br><b>${daemonSessions}</b></div>
    </div>
    <div style="font-size:11px;color:var(--muted)">
      <a href="${esc(dmn.health_url||'http://127.0.0.1:8766/health')}" target="_blank" style="color:var(--blue)">health endpoint &#x2197;</a>
    </div>`:daemonRunning===false?`
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
      ${daemonBadge}
    </div>
    <div style="color:var(--muted);font-size:13px;margin-bottom:8px">HTTP MCP daemon is not running.</div>
    <div style="font-size:12px">Start it with: <code>slowave serve start</code></div>`:"";

  // RECENT SESSIONS
  const sess=d.recent_sessions||[];
  if(!sess.length){
    document.getElementById("recentSessions").innerHTML=emptyState("No sessions yet.","💬");
  } else {
    document.getElementById("recentSessions").innerHTML=table(
      ["Session","Agent","Scope","Started","Duration","Events","Ep."],
      sess.map(r=>[
        `<span onclick="loadSessionTimeline('${esc(r.id)}')" style="cursor:pointer;color:var(--blue);font-family:monospace;font-size:11px" title="Click to replay session">${esc((r.id||"").slice(0,12))}…</span>`,
        r.agent||"—",r.scope_id||"(none)",
        fmtTs(r.started_ts),dur(r.duration_seconds),
        num(r.events),num(r.episodes)
      ]),
      [0]
    );
  }

  // SCOPES
  const scopes=d.scopes||[];
  if(!scopes.length){
    document.getElementById("scopesPanel").innerHTML=emptyState("No scopes.","🗂");
  } else {
    document.getElementById("scopesPanel").innerHTML=table(
      ["Scope","Sessions"],
      scopes.map(r=>[r.scope,num(r.sessions)])
    );
  }

  // PULSE GRAPH — refresh on every status poll cycle
  renderPulse();
}

async function renderPulse(){
  try{
    const d=await getJSON("/api/pulse?hours=3&bucket_m=15");
    const canvas=document.getElementById("pulseCanvas");
    const tooltip=document.getElementById("pulseTooltip");
    const stats=document.getElementById("pulseStats");
    if(!canvas)return;

    const DPR=window.devicePixelRatio||1;
    const W=canvas.parentElement.clientWidth;
    if(W<=0)return;
    const H=110;
    canvas.width=Math.round(W*DPR);
    canvas.height=Math.round(H*DPR);
    canvas.style.width=W+"px";
    canvas.style.height=H+"px";
    const ctx=canvas.getContext("2d");
    ctx.scale(DPR,DPR);
    ctx.clearRect(0,0,W,H);

    const channels=d.channels||{raw_events:d.buckets||[],episodes:[],schemas:[]};
    const allBuckets=channels[CHANNELS[0].key]||[];
    const N=allBuckets.length;
    // Combined signal per bucket — the heartbeat's strength is the sum across
    // all three channels, not any one of them.
    const bucketTotals=allBuckets.map((_,i)=>CHANNELS.reduce((s,ch)=>s+((channels[ch.key]||[])[i]?.n||0),0));
    const maxTotal=Math.max(...bucketTotals,0);

    if(!N||!maxTotal){
      const bl=H-18;
      ctx.strokeStyle="rgba(240,78,106,.18)";ctx.lineWidth=0.8;ctx.setLineDash([4,7]);
      ctx.beginPath();ctx.moveTo(0,bl);ctx.lineTo(W,bl);ctx.stroke();ctx.setLineDash([]);
      ctx.fillStyle="#8a5560";ctx.font="12px system-ui,sans-serif";ctx.textAlign="center";
      ctx.fillText("Flatline \u2014 no events in the last "+d.window_hours+"h",W/2,bl-10);
      canvas.onmousemove=null;canvas.onmouseleave=null;stats.innerHTML="";return;
    }

    // ── layout ──────────────────────────────────────────────────────────────────────────
    const PAD_L=4,PAD_R=4,PAD_T=8,PAD_B=18;
    const IW=W-PAD_L-PAD_R;
    const IH=H-PAD_T-PAD_B;
    const baseline=H-PAD_B;
    const step=IW/(N-1||1);
    const RED="#f04e6a",RED_GC="rgba(240,78,106,";
    const ampFor=v=>v>0?Math.sqrt(v/maxTotal)*IH*0.88:0;

    // ── grid + baseline ────────────────────────────────────────────────────────────
    ctx.strokeStyle="rgba(20,35,65,.8)";ctx.lineWidth=0.5;
    [0.25,0.5,0.75,1].forEach(f=>{
      const gy=baseline-Math.sqrt(f)*IH*0.88;
      ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(W,gy);ctx.stroke();
    });
    ctx.strokeStyle="rgba(56,189,248,.10)";ctx.lineWidth=0.5;
    ctx.beginPath();ctx.moveTo(0,baseline);ctx.lineTo(W,baseline);ctx.stroke();

    // ── ECG-style heartbeat trace ────────────────────────────────────────────────
    // One QRS-shaped blip per bucket, centered on that bucket's timestamp;
    // buckets with no activity are drawn as flat baseline. Blip height is
    // driven by ampFor(bucketTotals[i]) — more combined signal, taller heartbeat.
    const half=step/2;
    // Textbook EKG beat: small flutter, the tall R spike, a bit more flutter,
    // settling back to baseline. xf=fraction of beat width, yf=signed
    // fraction of peak (negative=up, since canvas y grows downward).
    const BEAT_SHAPE=[
      [0.00,0],[0.10,-0.08],[0.18,0.05],[0.27,-0.16],[0.36,0.14],
      [0.46,-1.00],[0.55,0.34],[0.63,-0.12],[0.71,0.07],[0.80,-0.05],[1.00,0],
    ];
    const verts=[];
    for(let i=0;i<N;i++){
      const cx=PAD_L+i*step,x1=cx+half;
      if(i===0)verts.push({x:cx-half,y:baseline});
      const peak=ampFor(bucketTotals[i]);
      if(peak<=0){verts.push({x:x1,y:baseline});continue;}
      // Beat width independent of amplitude but scales a bit with bucket
      // spacing, capped so dense charts stay readable.
      const beatW=Math.min(step*0.55,42),bx0=cx-beatW/2;
      BEAT_SHAPE.forEach(([xf,yf])=>verts.push({x:bx0+beatW*xf,y:baseline+peak*yf}));
      verts.push({x:x1,y:baseline});
    }
    function traceECG(){
      ctx.moveTo(verts[0].x,verts[0].y);
      for(let i=1;i<verts.length;i++)ctx.lineTo(verts[i].x,verts[i].y);
    }
    // Glow pass then crisp pass — same two-pass technique as the old
    // multi-channel spline, just one red neon trace instead of three.
    [{blur:9,w:2.2,a:.55},{blur:0,w:1.3,a:1}].forEach(({blur,w,a})=>{
      ctx.save();
      ctx.shadowColor=RED_GC+a+")";ctx.shadowBlur=blur;
      ctx.strokeStyle=RED;ctx.lineWidth=w;ctx.lineJoin="round";ctx.lineCap="round";
      ctx.beginPath();traceECG();ctx.stroke();
      ctx.restore();
    });

    // ── time axis ─────────────────────────────────────────────────────────────
    ctx.fillStyle="rgba(80,100,140,.6)";
    ctx.font="10px system-ui,sans-serif";ctx.textAlign="center";
    const labelStep=Math.ceil(N/6);
    allBuckets.forEach((b,i)=>{
      if(i%labelStep!==0&&i!==N-1)return;
      ctx.fillText(new Date(b.ts*1000).toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"}),
        Math.min(W-18,Math.max(18,PAD_L+i*step)),H-3);
    });

    // ── hover ─────────────────────────────────────────────────────────────────
    canvas.onmousemove=function(e){
      const rect=canvas.getBoundingClientRect();
      const mx=e.clientX-rect.left;
      const idx=Math.min(N-1,Math.max(0,Math.round((mx-PAD_L)/step)));
      const rows=CHANNELS.map(ch=>{
        const b=(channels[ch.key]||[])[idx];
        return`<span style='color:${ch.color}'>\u25cf</span> ${ch.label}: <b>${b?b.n:0}</b>`;
      }).join("<br>");
      const ts=allBuckets[idx]?.ts;
      const t2=ts?new Date(ts*1000).toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"}):"";
      tooltip.style.display="block";
      tooltip.style.left=Math.min(W-70,Math.max(40,PAD_L+idx*step))+"px";
      tooltip.style.top="4px";
      tooltip.style.transform="translateX(-50%)";
      tooltip.innerHTML=`<div style='font-size:10px;color:var(--muted);margin-bottom:3px'>${t2}</div>${rows}`;
    };
    canvas.onmouseleave=function(){tooltip.style.display="none";};

    // ── stats footer ──────────────────────────────────────────────────────────
    const totals=CHANNELS.map(ch=>{
      const sum=(channels[ch.key]||[]).reduce((a,b)=>a+b.n,0);
      return`<span style='color:${ch.color}'>\u25cf ${ch.label}</span> <b>${sum}</b>`;
    }).join(" &nbsp;\u00b7&nbsp; ");
    stats.innerHTML=`<span style='color:var(--muted)'>${d.window_hours}h &nbsp;\u00b7&nbsp; ${d.bucket_minutes}m buckets</span>&nbsp;&nbsp;${totals}`;
  }catch(e){console.error("pulse",e);}
}

// ── CREATION HISTOGRAM ──
let histogramRange="1w";
function formatBucketLabel(ts,bucketSeconds){
  const d=new Date(ts*1000);
  if(bucketSeconds>=30*86400)return d.toLocaleDateString([],{month:"short",year:"numeric"});
  if(bucketSeconds>=7*86400)return d.toLocaleDateString([],{month:"short",day:"numeric"});
  return d.toLocaleDateString([],{month:"short",day:"numeric"});
}
document.querySelectorAll("#histogramRangeToolbar .range-btn").forEach(b=>b.onclick=()=>{
  histogramRange=b.dataset.range;
  document.querySelectorAll("#histogramRangeToolbar .range-btn").forEach(x=>x.classList.toggle("active",x===b));
  renderHistogram();
});
document.querySelector(`#histogramRangeToolbar .range-btn[data-range="${histogramRange}"]`)?.classList.add("active");

async function renderHistogram(){
  try{
    const d=await getJSON("/api/histogram?range="+encodeURIComponent(histogramRange));
    const canvas=document.getElementById("histogramCanvas");
    const tooltip=document.getElementById("histogramTooltip");
    const stats=document.getElementById("histogramStats");
    if(!canvas)return;

    const DPR=window.devicePixelRatio||1;
    const W=canvas.parentElement.clientWidth;
    if(W<=0)return;
    const H=140;
    canvas.width=Math.round(W*DPR);
    canvas.height=Math.round(H*DPR);
    canvas.style.width=W+"px";
    canvas.style.height=H+"px";
    const ctx=canvas.getContext("2d");
    ctx.scale(DPR,DPR);
    ctx.clearRect(0,0,W,H);

    const channels=d.channels||{raw_events:[],episodes:[],schemas:[]};
    const allBuckets=channels[CHANNELS[0].key]||[];
    const N=allBuckets.length;
    const bucketSeconds=d.bucket_seconds||86400;

    if(!N||!d.stacked_max){
      const bl=H-22;
      ctx.strokeStyle="rgba(56,189,248,.18)";ctx.lineWidth=0.8;ctx.setLineDash([4,7]);
      ctx.beginPath();ctx.moveTo(0,bl);ctx.lineTo(W,bl);ctx.stroke();ctx.setLineDash([]);
      ctx.fillStyle="#5a7ab5";ctx.font="12px system-ui,sans-serif";ctx.textAlign="center";
      ctx.fillText("No activity in this window",W/2,bl-10);
      canvas.onmousemove=null;canvas.onmouseleave=null;stats.innerHTML="";return;
    }

    // ── layout ───────────────────────────────────────────────────────────────
    const PAD_L=4,PAD_R=4,PAD_T=8,PAD_B=20;
    const IW=W-PAD_L-PAD_R;
    const IH=H-PAD_T-PAD_B;
    const baseline=H-PAD_B;
    const step=IW/N;
    const barW=Math.max(1,step*0.64);
    const gmax=d.stacked_max||1;
    const scaleY=v=>(v/gmax)*IH*0.92;

    // ── grid ─────────────────────────────────────────────────────────────────
    ctx.strokeStyle="rgba(20,35,65,.8)";ctx.lineWidth=0.5;
    [0.25,0.5,0.75,1].forEach(f=>{
      const gy=baseline-f*IH*0.92;
      ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(W,gy);ctx.stroke();
    });
    ctx.strokeStyle="rgba(56,189,248,.10)";ctx.lineWidth=0.5;
    ctx.beginPath();ctx.moveTo(0,baseline);ctx.lineTo(W,baseline);ctx.stroke();

    // ── stacked bars ─────────────────────────────────────────────────────────
    const barTops=[];
    for(let i=0;i<N;i++){
      const bx=PAD_L+i*step+(step-barW)/2;
      let cum=0;
      const segs=[];
      CHANNELS.forEach(ch=>{
        const n=(channels[ch.key]||[])[i]?.n||0;
        cum+=n;
        segs.push({ch,n,top:cum});
      });
      barTops.push(cum);
      if(!cum)continue;
      segs.forEach(({ch,n,top})=>{
        if(!n)return;
        const y0=baseline-scaleY(top);
        const y1=baseline-scaleY(top-n);
        const gr=ctx.createLinearGradient(0,y0,0,y1);
        gr.addColorStop(0,ch.gc+"0.92)");gr.addColorStop(1,ch.gc+"0.62)");
        ctx.fillStyle=gr;
        ctx.fillRect(bx,y0,barW,Math.max(1,y1-y0));
      });
    }

    // ── time axis ─────────────────────────────────────────────────────────────
    ctx.fillStyle="rgba(80,100,140,.6)";
    ctx.font="10px system-ui,sans-serif";ctx.textAlign="center";
    const labelStep=Math.ceil(N/7);
    allBuckets.forEach((b,i)=>{
      if(i%labelStep!==0&&i!==N-1)return;
      ctx.fillText(formatBucketLabel(b.ts,bucketSeconds),
        Math.min(W-22,Math.max(22,PAD_L+i*step+step/2)),H-4);
    });

    // ── hover ─────────────────────────────────────────────────────────────────
    canvas.onmousemove=function(e){
      const rect=canvas.getBoundingClientRect();
      const mx=e.clientX-rect.left;
      const idx=Math.min(N-1,Math.max(0,Math.floor((mx-PAD_L)/step)));
      const rows=CHANNELS.map(ch=>{
        const b=(channels[ch.key]||[])[idx];
        return`<span style='color:${ch.color}'>●</span> ${ch.label}: <b>${b?b.n:0}</b>`;
      }).join("<br>");
      const ts=allBuckets[idx]?.ts;
      const label=ts?formatBucketLabel(ts,bucketSeconds):"";
      tooltip.style.display="block";
      tooltip.style.left=Math.min(W-70,Math.max(40,PAD_L+idx*step+step/2))+"px";
      tooltip.style.top="4px";
      tooltip.style.transform="translateX(-50%)";
      tooltip.innerHTML=`<div style='font-size:10px;color:var(--muted);margin-bottom:3px'>${label} · total <b>${barTops[idx]}</b></div>${rows}`;
    };
    canvas.onmouseleave=function(){tooltip.style.display="none";};

    // ── stats footer ──────────────────────────────────────────────────────────
    const totals=CHANNELS.map(ch=>{
      const sum=(channels[ch.key]||[]).reduce((a,b)=>a+b.n,0);
      return`<span style='color:${ch.color}'>● ${ch.label}</span> <b>${sum}</b>`;
    }).join(" &nbsp;·&nbsp; ");
    const bucketLabel=bucketSeconds>=30*86400?"monthly":bucketSeconds>=7*86400?"weekly":"daily";
    stats.innerHTML=`<span style='color:var(--muted)'>${bucketLabel} buckets</span>&nbsp;&nbsp;${totals}`;
  }catch(e){console.error("histogram",e);}
}

function statusBreakdown(by){
  const statusOrder=["active","needs_review","contradicted","superseded","archived","forgotten"];
  return statusOrder
    .filter(s=>by[s]>0)
    .map(s=>`<span class="pill pill-${s}" style="font-size:10px">${s[0].toUpperCase()} ${by[s]}</span>`)
    .join("");
}

// ── SCHEMAS ──
let schemaMaxSalience=1;
let schemaSort={col:null,dir:"asc"};
let cachedSchemas=[];
function sortBy(col){
  if(schemaSort.col===col){schemaSort.dir=schemaSort.dir==="asc"?"desc":"asc";}
  else{schemaSort.col=col;schemaSort.dir="asc";}
  if(!cachedSchemas.length)return;
  const sorted=cachedSchemas.slice();
  sorted.sort((a,b)=>{
    let va,vb;
    switch(col){
      case"id":va=a.schema_id;vb=b.schema_id;break;
      case"status":va=a.status||"";vb=b.status||"";break;
      case"salience":va=a.salience||0;vb=b.salience||0;break;
      case"confidence":va=a.confidence||0;vb=b.confidence||0;break;
      case"stage":va=a.generalization_stage||0;vb=b.generalization_stage||0;break;
      case"class":va=a.schema_class||"";vb=b.schema_class||"";break;
      case"scope":va=a.scope||"";vb=b.scope||"";break;
      case"support":va=a.support_count||0;vb=b.support_count||0;break;
      case"content":va=a.content||"";vb=b.content||"";break;
      default:return 0;
    }
    const mul=schemaSort.dir==="asc"?1:-1;
    if(typeof va==="string")return mul*va.localeCompare(vb);
    return mul*(va-vb);
  });
  const el=document.getElementById("schemaTable");
  el.innerHTML=renderSchemasTable(sorted);
  el.querySelectorAll("tr.expandable").forEach(tr=>{
    tr.addEventListener("click",()=>expandSchemaRow(tr,parseInt(tr.dataset.id)));
  });
}
function sortArrow(col){return schemaSort.col===col?(schemaSort.dir==="asc"?" ▲":" ▼"):"";}
function sortStyle(col){return schemaSort.col===col?"color:var(--blue);":"";}
async function loadSchemas(){
  const el=document.getElementById("schemaTable");
  const ld=document.getElementById("schemaLoading");
  ld.classList.add("show");el.innerHTML="";
  document.getElementById("schemaDetail").innerHTML="";
  try{
    const st=document.getElementById("schemaStatus").value;
    const sc=encodeURIComponent(document.getElementById("schemaScope").value);
    const q=encodeURIComponent(document.getElementById("schemaQ").value);
    const lim=document.getElementById("schemaLimit").value;
    const d=await getJSON(`/api/schemas?limit=${lim}&status=${encodeURIComponent(st)}&scope=${sc}&q=${q}`);
    schemaMaxSalience=Math.max(1,...(d.schemas||[]).map(s=>s.salience));
    cachedSchemas=d.schemas||[];
    const schemas=cachedSchemas.slice();
    if(schemaSort.col)schemas.sort((a,b)=>{let va,vb;switch(schemaSort.col){case"id":va=a.schema_id;vb=b.schema_id;break;case"status":va=a.status||"";vb=b.status||"";break;case"salience":va=a.salience||0;vb=b.salience||0;break;case"confidence":va=a.confidence||0;vb=b.confidence||0;break;case"stage":va=a.generalization_stage||0;vb=b.generalization_stage||0;break;case"class":va=a.schema_class||"";vb=b.schema_class||"";break;case"scope":va=a.scope||"";vb=b.scope||"";break;case"support":va=a.support_count||0;vb=b.support_count||0;break;case"content":va=a.content||"";vb=b.content||"";break;default:return 0;}const mul=schemaSort.dir==="asc"?1:-1;if(typeof va==="string")return mul*va.localeCompare(vb);return mul*(va-vb);});
    el.innerHTML=renderSchemasTable(schemas);
    // attach expand handlers
    el.querySelectorAll("tr.expandable").forEach(tr=>{
      tr.addEventListener("click",()=>expandSchemaRow(tr,parseInt(tr.dataset.id)));
    });
  } finally{ld.classList.remove("show");}
}
function renderSchemasTable(schemas){
  if(!schemas.length)return emptyState("No schemas found.","📖");
  const rows=schemas.map(s=>{
    const salPct=Math.round(s.salience/Math.max(0.001,schemaMaxSalience)*100);
    const salHtml=`<span style="font-size:12px;font-weight:600">${s.salience.toFixed(2)}</span>
      <div class="sal-bar-track" style="width:50px;display:inline-block;vertical-align:middle;margin-left:4px">
        <div class="sal-bar-fill" style="width:${salPct}%"></div></div>`;
    const confHtml=confBar(s.confidence);
    const tagsHtml=(s.tags||[]).map(t=>`<span class="pill" style="font-size:10px">${esc(t)}</span>`).join("");
    const stage=s.generalization_stage||0;
    const stageBadge=stage>0?`<span class="gen-badge gen-${stage}" style="font-size:10px">${GEN_LABELS[stage]||stage}</span>`:`<span class="gen-badge gen-0" style="font-size:10px">SCOPED</span>`;
    const labileStyle=s.is_labile?';border:2px solid #f5b942;border-radius:10px;padding:1px 4px':'';
    const labileTitle=s.is_labile?' title="Labile memory: under verification. Clears with sustained use — otherwise corrected or removed during background consolidation."':'';
    return `<tr class="expandable" data-id="${s.schema_id}">
      <td><code style="font-size:11px${labileStyle}"${labileTitle}>sch_${s.schema_id}</code></td>
      <td style="white-space:nowrap">${pill(s.status)}</td>
      <td>${salHtml}</td>
      <td>${confHtml}</td>
      <td>${stageBadge}</td>
      <td>${esc(s.schema_class||"—")}</td>
      <td>${esc(s.scope||"—")}</td>
      <td>${num(s.support_count)}</td>
      <td style="max-width:380px;word-break:break-word">${truncContent(s.content,120)}</td>
    </tr>`;
  }).join("");
  return `<table><thead><tr>
    <th style="cursor:pointer;user-select:none;${sortStyle('id')}" onclick="sortBy('id')">ID${sortArrow('id')}</th><th style="width:1%;white-space:nowrap;cursor:pointer;user-select:none;${sortStyle('status')}" onclick="sortBy('status')">Status${sortArrow('status')}</th><th style="cursor:pointer;user-select:none;${sortStyle('salience')}" onclick="sortBy('salience')">Salience${sortArrow('salience')}</th><th style="cursor:pointer;user-select:none;${sortStyle('confidence')}" onclick="sortBy('confidence')">Confidence${sortArrow('confidence')}</th>
    <th style="cursor:pointer;user-select:none;${sortStyle('stage')}" onclick="sortBy('stage')">Stage${sortArrow('stage')}</th><th style="cursor:pointer;user-select:none;${sortStyle('class')}" onclick="sortBy('class')">Class${sortArrow('class')}</th><th style="cursor:pointer;user-select:none;${sortStyle('scope')}" onclick="sortBy('scope')">Scope${sortArrow('scope')}</th><th style="cursor:pointer;user-select:none;${sortStyle('support')}" onclick="sortBy('support')">Support${sortArrow('support')}</th><th style="cursor:pointer;user-select:none;${sortStyle('content')}" onclick="sortBy('content')">Content${sortArrow('content')}</th>
  </tr></thead><tbody>${rows}</tbody></table>`;
}
async function expandSchemaRow(tr,schemaId){
  // Toggle: if already expanded, just collapse
  const nextTr=tr.nextElementSibling;
  if(nextTr&&nextTr.classList.contains("expand-row")){
    nextTr.remove();tr.classList.remove("schema-row-expanded");return;
  }
  // Collapse all other expanded rows first
  const allRows=tr.parentElement.querySelectorAll("tr.expand-row");
  allRows.forEach(r=>r.remove());
  // Clear highlight from any previously expanded row
  tr.parentElement.querySelectorAll("tr.schema-row-expanded").forEach(r=>r.classList.remove("schema-row-expanded"));
  const d=await getJSON(`/api/schemas/${schemaId}`);
  if(!d.schema){
    const errTr=document.createElement("tr");
    errTr.className="expand-row";
    errTr.innerHTML=`<td colspan="99"><em style='color:var(--muted)'>Could not load sch_${schemaId}: ${d.error||"not found"}</em></td>`;
    tr.parentElement.insertBefore(errTr,tr.nextElementSibling);
    tr.classList.add("schema-row-expanded");
    return;
  }
  const s=d.schema;
  const evHtml=renderEvidenceList(d.evidence);
const outHtml=d.outgoing&&d.outgoing.length?table(["To","Relation","Confidence","Reason"],
    d.outgoing.map(e=>[`sch_${e.dst_schema_id}`,e.relation,Number(e.confidence||0).toFixed(2),e.reason||"—"])):"<em style='color:var(--muted)'>None.</em>";
  const inHtml=d.incoming&&d.incoming.length?table(["From","Relation","Confidence","Reason"],
    d.incoming.map(e=>[`sch_${e.src_schema_id}`,e.relation,Number(e.confidence||0).toFixed(2),e.reason||"—"])):"<em style='color:var(--muted)'>None.</em>";
  const expTr=document.createElement("tr");
  expTr.className="expand-row";
  const stage=s.generalization_stage||0;
  const genHtml=`<div class="detail-section"><h4>Generalization</h4>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
      ${genBadge(stage)}
      <span style="font-size:12px;color:var(--muted)">${GEN_DESC[stage]||""}</span>
    </div>
    ${stage>0?`
      ${genBreadthBar((s.scope_breadth_pct||0),"scope breadth")}
      ${genBreadthBar((s.scope_kind_breadth_pct||0),"kind breadth")}
      <div style="font-size:12px;color:var(--muted);margin-top:6px">
        ${(s.distinct_scope_count||0)} scope${s.distinct_scope_count!==1?"s":""} ·
        ${(s.distinct_scope_kind_count||0)} kind${s.distinct_scope_kind_count!==1?"s":""} ·
        ${(s.cross_scope_recall_count||0)} cross-scope recall${s.cross_scope_recall_count!==1?"s":""}
        ${(s.recalled_scopes||[]).length?` <span class="scope-count-tip" data-tip="${esc((s.recalled_scopes||[]).join("\n"))}">(${(s.recalled_scopes||[]).length} scopes)</span>`:""}
      </div>`:"<div style='font-size:12px;color:var(--muted)'>Not yet recalled across multiple scopes.</div>"}
  </div>`;
  const forgetHtml=ALLOW_ACTIONS?(s.status==="forgotten"
    ?`<button class="btn btn-forget" onclick="unforgetSchema(${s.schema_id})">Unforget</button>`
    :`<button class="btn btn-forget" onclick="forgetSchema(${s.schema_id})">Forget</button>`):"";
  expTr.innerHTML=`<td colspan="9"><div class="expand-content">
    ${forgetHtml?`<div style="text-align:right;margin-bottom:8px">${forgetHtml}</div>`:""}
    <div class="detail-section" style="margin-bottom:14px">
      <h4>Content</h4>
      <div style="font-size:13px;line-height:1.6;white-space:pre-wrap;word-break:break-word;padding:10px 12px;background:var(--panel2);border:1px solid var(--line);border-radius:var(--radius-sm)">${esc(s.content)}</div>
    </div>
    <div class="three-col">
      <div>
        <div class="detail-section"><h4>Facets</h4><pre class="code-block">${esc(JSON.stringify(s.facets,null,2))}</pre></div>
        ${genHtml}
      </div>
      <div>
        <div class="detail-section"><h4>Tags</h4>${(s.tags||[]).map(t=>`<span class="pill">${esc(t)}</span>`).join("")||"<em style='color:var(--muted)'>none</em>"}</div>
        <div class="detail-section" style="margin-top:10px"><h4>Timestamps</h4>
          <div style="font-size:12px;color:var(--muted)">First formed: ${fmtTs(s.first_formed_ts)}<br>Updated: ${fmtTs(s.last_updated_ts)}</div>
        </div>
      </div>
      <div><div class="detail-section"><h4>Outgoing relations</h4>${outHtml}</div>
           <div class="detail-section" style="margin-top:10px"><h4>Incoming relations</h4>${inHtml}</div></div>
    </div>
    <div class="detail-section"><h4>Evidence</h4>${evHtml}</div>
  </div></td>`;
  tr.classList.add("schema-row-expanded");
  tr.after(expTr);
  tr.scrollIntoView({behavior:"smooth",block:"start"});
}

let _forgetModalSchemaId=null;
function forgetSchema(schemaId){
  _forgetModalSchemaId=schemaId;
  document.getElementById("forgetModalTitle").textContent=`Forget sch_${schemaId}?`;
  document.getElementById("forgetModalMessage").textContent=
    "This hides it from all future retrieval (reversible via Unforget).";
  document.getElementById("forgetModalReason").value="";
  document.getElementById("forgetModalOverlay").classList.add("show");
  document.getElementById("forgetModalReason").focus();
}
function closeForgetModal(){
  document.getElementById("forgetModalOverlay").classList.remove("show");
  _forgetModalSchemaId=null;
}
async function confirmForgetSchema(){
  const schemaId=_forgetModalSchemaId;
  if(schemaId===null)return;
  const reason=document.getElementById("forgetModalReason").value.trim();
  closeForgetModal();
  const res=await fetch(`/api/schemas/${schemaId}/forget`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({reason:reason||null})});
  const d=await res.json();
  if(d.error){alert("Error: "+d.error);return;}
  if(d.warning)alert(d.warning);
  loadSchemas();
}
async function unforgetSchema(schemaId){
  const res=await fetch(`/api/schemas/${schemaId}/unforget`,{method:"POST"});
  const d=await res.json();
  if(d.error){alert("Error: "+d.error);return;}
  loadSchemas();
}
window.forgetSchema=forgetSchema;
window.closeForgetModal=closeForgetModal;
window.confirmForgetSchema=confirmForgetSchema;
window.unforgetSchema=unforgetSchema;

async function loadEventInline(eventId, slotId){
  const slot=document.getElementById(slotId);
  if(!slot)return;
  // Toggle
  if(slot.innerHTML&&!slot.innerHTML.includes("Loading")){
    slot.innerHTML="";return;
  }
  slot.innerHTML=`<div style="color:var(--muted);font-size:10px;padding:4px 0">Loading event…</div>`;
  try{
    const d=await getJSON(`/api/events/${eventId}`);
    if(d.error){slot.innerHTML=`<div style="color:var(--red);font-size:10px">${esc(d.error)}</div>`;return;}
    const ev=d.event;
    slot.innerHTML=`<div style="background:var(--panel3);border:1px solid var(--line2);border-radius:4px;padding:6px 10px;margin-top:4px;font-size:11px">
      <span class="pill" style="font-size:9px">${esc(ev.type||"?")}</span>
      <span style="color:var(--muted)">${fmtTsCompact(ev.ts)}</span>
      <span style="color:var(--muted)"> · session ${esc((ev.session_id||"").slice(0,12))}…</span>
      <div style="margin-top:4px;line-height:1.4;white-space:pre-wrap;word-break:break-word">${esc(ev.content||"")}</div>
    </div>`;
  }catch(e){slot.innerHTML=`<div style="color:var(--red);font-size:10px">${esc(String(e))}</div>`;}
}
window.loadEventInline = loadEventInline;

// ── WORKER ──
async function loadWorker(){
  const ld=document.getElementById("workerLoading");
  const tbl=document.getElementById("workerTable");
  ld.classList.add("show");tbl.innerHTML="";
  try{
    const lim=document.getElementById("workerLimit").value;
    const d=await getJSON(`/api/worker/runs?limit=${lim}`);
    const runs=d.runs||[];
    const sum=d.summary||{};

    // Stat cards
    const cards=[
      {icon:"🔄",label:"Total passes",val:num(sum.total_passes),accent:"var(--blue)"},
      {icon:"📖",label:"Schemas created",val:num(sum.total_schemas_created),accent:"var(--green)"},
      {icon:"🔁",label:"Reinforced",val:num(sum.total_schemas_reinforced),accent:"var(--cyan)"},
      {icon:"⏱",label:"Avg duration",val:(sum.avg_duration_ms||0).toFixed(0)+"ms",accent:"var(--amber)"},
      {icon:"📉",label:"Decayed",val:num(sum.total_schemas_decayed),accent:"var(--red)"},
      {icon:"🕐",label:"Last run",val:fmtTsCompact(sum.last_run_ts),sub:fmtTsCompactSub(sum.last_run_ts),accent:"var(--muted)",raw:true},
    ];
    document.getElementById("workerStatGrid").innerHTML=cards.map(c=>
      `<div class="stat-card" style="--accent:${c.accent}">
        <div class="sc-icon">${c.icon}</div>
        <div class="sc-label">${esc(c.label)}</div>
        <div class="sc-value">${c.raw?c.val:esc(String(c.val))}</div>
        ${c.sub?`<div class="sc-sub">${c.raw?c.sub:esc(c.sub)}</div>`:""}
      </div>`
    ).join("");

    renderWorkerChart(runs);

    // Table
    if(!runs.length){
      tbl.innerHTML=emptyState("No worker runs recorded yet. Start the worker with: slowave worker","🧠");
      return;
    }
    const heads=["#","Started","Duration","Trigger","Schemas +","~Reinf.","Protos","Decayed","Eps proc","Status"];
    const rows=runs.map(r=>[
      r.id,
      fmtTs(r.started_ts),
      r.duration_ms!=null?r.duration_ms+"ms":"—",
      r.triggered_by||"worker",
      `<b style="color:var(--green)">+${r.schemas_created||0}</b>`,
      num(r.schemas_reinforced||0),
      num(r.prototypes_processed||0),
      `<span style="color:var(--red)">-${r.schemas_decayed||0}</span>`,
      num(r.episodes_processed||0),
      r.error_text
        ?`<span class="pill pill-contradicted" title="${esc(r.error_text)}">error</span>`
        :`<span class="pill pill-active">ok</span>`,
    ]);
    const rawCols=[4,7,9];
    tbl.innerHTML=table(heads,rows,rawCols);
  }finally{ld.classList.remove("show");}
}

// ── WORKER CHART ──
function renderWorkerChart(runs){
  const canvas=document.getElementById("workerCanvas");
  const tooltip=document.getElementById("workerTooltip");
  const stats=document.getElementById("workerChartStats");
  if(!canvas)return;
  const pts_runs=runs.slice(0,50).reverse();
  const N=pts_runs.length;
  const DPR=window.devicePixelRatio||1;
  const W=canvas.parentElement.clientWidth||600;
  const H=120;
  canvas.width=Math.round(W*DPR);canvas.height=Math.round(H*DPR);
  canvas.style.width=W+"px";canvas.style.height=H+"px";
  const ctx=canvas.getContext("2d");
  ctx.scale(DPR,DPR);ctx.clearRect(0,0,W,H);
  const CHANNELS=[
    {key:"schemas_created",    label:"schemas +",  color:"#3b82f6",gc:"rgba(59,130,246,"},
    {key:"schemas_reinforced", label:"reinforced", color:"#10b981",gc:"rgba(16,185,129,"},
    {key:"schemas_decayed",    label:"decayed",    color:"#f04e6a",gc:"rgba(240,78,106,"},
  ];
  const gmax=Math.max(1,...pts_runs.flatMap(r=>CHANNELS.map(ch=>r[ch.key]||0)));
  const PAD_L=4,PAD_R=4,PAD_T=8,PAD_B=18;
  const IW=W-PAD_L-PAD_R,IH=H-PAD_T-PAD_B;
  const baseline=H-PAD_B;
  const step=N>1?IW/(N-1):IW;
  const amp=v=>Math.sqrt(v/gmax)*IH*0.88;
  if(!N){
    ctx.strokeStyle="rgba(56,189,248,.18)";ctx.lineWidth=0.8;ctx.setLineDash([4,7]);
    ctx.beginPath();ctx.moveTo(0,baseline);ctx.lineTo(W,baseline);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle="#5a7ab5";ctx.font="12px system-ui,sans-serif";ctx.textAlign="center";
    ctx.fillText("No consolidation runs yet",W/2,baseline-10);
    stats.innerHTML="";return;
  }
  function buildSpline(pts){
    const pp=[pts[0],...pts,pts[pts.length-1]];
    ctx.moveTo(pp[1].x,pp[1].y);
    for(let i=1;i<pp.length-2;i++){
      const p0=pp[i-1],p1=pp[i],p2=pp[i+1],p3=pp[i+2];
      const d1=Math.hypot(p1.x-p0.x,p1.y-p0.y)||1;
      const d2=Math.hypot(p2.x-p1.x,p2.y-p1.y)||1;
      const d3=Math.hypot(p3.x-p2.x,p3.y-p2.y)||1;
      ctx.bezierCurveTo(
        p1.x+(p2.x-p0.x)/6*(d1/(d1+d2)),p1.y+(p2.y-p0.y)/6*(d1/(d1+d2)),
        p2.x-(p3.x-p1.x)/6*(d2/(d2+d3)),p2.y-(p3.y-p1.y)/6*(d2/(d2+d3)),
        p2.x,p2.y);
    }
  }
  function drawBase(){
    ctx.strokeStyle="rgba(20,35,65,.8)";ctx.lineWidth=0.5;
    [0.25,0.5,0.75,1].forEach(f=>{
      const gy=baseline-Math.sqrt(f)*IH*0.88;
      ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(W,gy);ctx.stroke();
    });
    ctx.strokeStyle="rgba(56,189,248,.10)";ctx.lineWidth=0.5;
    ctx.beginPath();ctx.moveTo(0,baseline);ctx.lineTo(W,baseline);ctx.stroke();
  }
  const allPts={};
  CHANNELS.forEach(ch=>{
    allPts[ch.key]=pts_runs.map((r,i)=>({x:PAD_L+i*step,y:baseline-amp(r[ch.key]||0),n:r[ch.key]||0}));
  });
  function drawChannels(hoverIdx){
    CHANNELS.forEach(ch=>{
      const pts=allPts[ch.key]||[];
      if(!pts.some(p=>p.n>0))return;
      const gr=ctx.createLinearGradient(0,PAD_T,0,baseline);
      gr.addColorStop(0,ch.gc+"0.06)");gr.addColorStop(1,ch.gc+"0)");
      ctx.beginPath();buildSpline(pts);
      ctx.lineTo(pts[pts.length-1].x,baseline);ctx.lineTo(pts[0].x,baseline);ctx.closePath();
      ctx.fillStyle=gr;ctx.fill();
      [{blur:5,w:1.2,a:.5},{blur:0,w:0.8,a:1}].forEach(({blur,w,a})=>{
        ctx.save();ctx.shadowColor=ch.gc+a+")";ctx.shadowBlur=blur;
        ctx.strokeStyle=ch.color;ctx.lineWidth=w;ctx.lineJoin="round";ctx.lineCap="round";
        ctx.beginPath();buildSpline(pts);ctx.stroke();ctx.restore();
      });
      if(hoverIdx!=null){const p=pts[hoverIdx];if(p&&p.n>0){ctx.beginPath();ctx.arc(p.x,p.y,3,0,2*Math.PI);ctx.fillStyle=ch.color;ctx.fill();}}
    });
  }
  const labelStep=Math.ceil(N/6);
  function drawLabels(){
    ctx.fillStyle="rgba(80,100,140,.6)";ctx.font="10px system-ui,sans-serif";ctx.textAlign="center";
    pts_runs.forEach((r,i)=>{
      if(i%labelStep!==0&&i!==N-1)return;
      ctx.fillText(new Date(r.started_ts*1000).toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"}),
        Math.min(W-18,Math.max(18,PAD_L+i*step)),H-3);
    });
  }
  function fullRedraw(hi){
    ctx.clearRect(0,0,W,H);drawBase();
    if(hi!=null){ctx.save();ctx.strokeStyle="rgba(255,255,255,.08)";ctx.lineWidth=1;ctx.setLineDash([3,5]);ctx.beginPath();ctx.moveTo(PAD_L+hi*step,PAD_T);ctx.lineTo(PAD_L+hi*step,baseline);ctx.stroke();ctx.setLineDash([]);ctx.restore();}
    drawChannels(hi);drawLabels();
  }
  fullRedraw(null);
  canvas.onmousemove=function(e){
    const rect=canvas.getBoundingClientRect();
    const mx=(e.clientX-rect.left)*(W/rect.width);
    const idx=Math.min(N-1,Math.max(0,Math.round((mx-PAD_L)/step)));
    const r=pts_runs[idx];if(!r){tooltip.style.display="none";return;}
    const time=new Date(r.started_ts*1000).toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"});
    const dur=r.duration_ms!=null?r.duration_ms+"ms":"—";
    const rows=CHANNELS.map(ch=>"<span style='color:"+ch.color+"'>&#9679;</span> "+ch.label+": <b>"+(r[ch.key]||0)+"</b>").join("<br>");
    const errRow=r.error_text?"<br><span style='color:var(--red)'>&#9888; "+esc(r.error_text.slice(0,60))+"</span>":"";
    tooltip.style.display="block";tooltip.style.left=Math.min(W-80,Math.max(40,PAD_L+idx*step))+"px";tooltip.style.top="4px";tooltip.style.transform="translateX(-50%)";
    tooltip.innerHTML="<div style='font-size:10px;color:var(--muted);margin-bottom:3px'>"+time+" &nbsp;·&nbsp; "+dur+"</div>"+rows+errRow;
    fullRedraw(idx);
  };
  canvas.onmouseleave=function(){tooltip.style.display="none";fullRedraw(null);};
  const totals=CHANNELS.map(ch=>{const sum=pts_runs.reduce((a,r)=>a+(r[ch.key]||0),0);return "<span style='color:"+ch.color+"'>&#9679; "+ch.label+"</span> <b>"+sum+"</b>";}).join(" &nbsp;·&nbsp; ");
  stats.innerHTML="<span style='color:var(--muted)'>"+N+" passes</span>&nbsp;&nbsp;"+totals;
}

// ── GRAPH ──
function initSalienceSlider(status){
  if(window.salienceSliderInitialized)return;
  const maxSal=Number(status?.schema_health?.active_salience?.max||25);
  // Slider's max must equal the true observed max, not a ceil()-rounded
  // value above it -- rounding up created a dead zone (e.g. real max 1.5
  // rounded to a slider ceiling of 2) where any selected threshold in that
  // gap guarantees zero results, with no visual cue that it's unreachable.
  const upper=Math.max(0.1,maxSal);
  const minEl=document.getElementById("graphMinSalience");
  if(!minEl)return;
  minEl.max=upper.toFixed(2);
  document.getElementById("graphObservedMaxSalienceLabel").textContent=maxSal.toFixed(2);
  window.salienceSliderInitialized=true;
  syncSalienceSlider(false);
}
function syncSalienceSlider(autoLoad=true){
  const minEl=document.getElementById("graphMinSalience");
  const min=Number(minEl.value);
  document.getElementById("graphMinSalienceLabel").textContent=min.toFixed(2);
  clearTimeout(window.salienceLoadTimer);
  if(autoLoad)window.salienceLoadTimer=setTimeout(loadGraph,400);
}
function resetSalienceSlider(){
  document.getElementById("graphMinSalience").value="0";
  syncSalienceSlider();
}
async function loadGraph(){
  const sts=[...document.querySelectorAll(".gstat:checked")].map(x=>x.value).join(",");
  const lim=document.getElementById("graphLimit").value;
  const scope=encodeURIComponent(document.getElementById("graphScope").value);
  const minSal=encodeURIComponent(document.getElementById("graphMinSalience").value);
  const d=await getJSON(`/api/graph/schemas?limit=${lim}&statuses=${sts}&scope=${scope}&min_salience=${minSal}`);
  renderLegend();
  drawGraph(d);
}
function renderLegend(){
  const el=document.getElementById("graphLegend");
  if(!el)return;
  const statusEntries=Object.entries(statusColor).filter(([k])=>k!=="labile").map(([k,v])=>`<div class="legend-item"><div class="legend-dot" style="background:${v}"></div>${k}</div>`).join("");
  const relEntries=Object.entries(relColor).map(([k,v])=>{
    const on=relVisible[k];
    return `<div class="legend-item legend-toggle${on?"":" off"}" onclick="toggleRel('${k}')" title="Click to ${on?"hide":"show"} ${relLabel[k]||k} edges" style="cursor:pointer"><div class="legend-line" style="background:${on?v:"#3a3f4d"}"></div>${relLabel[k]||k}</div>`;
  }).join("");
  document.getElementById("graphLegend").innerHTML=`
    <div style="font-size:11px;color:var(--muted);font-weight:600;margin-right:4px">Nodes:</div>${statusEntries}
    <div style="width:1px;background:var(--line);margin:0 6px"></div>
    <div style="font-size:11px;color:var(--muted);font-weight:600;margin-right:4px">Edges:</div>${relEntries}
  `;
}
function toggleRel(rel){
  relVisible[rel]=!relVisible[rel];
  renderLegend();
  if(lastGraphData)drawGraph(lastGraphData);
}
let schemaCy=null;
let graphLabelsForced=false;
let lastGraphData=null;
const relVisible={relates_to:true,coactivated_with:true};
function nodeColor(n){return statusColor[n.status||"active"]||"#5a6e91";}
function graphNodeSize(n){return 12+Math.min(34,Math.sqrt(Math.max(0,Number(n.salience||0)))*3.2);}
function graphNodeLabel(n){return `sch_${n.schema_id}`;}
function updateGraphLabels(){
  if(!schemaCy)return;
  const show=graphLabelsForced||schemaCy.zoom()>0.85||schemaCy.nodes().length<=80;
  schemaCy.nodes().forEach(n=>{
    n.style("label",show?n.data("label"):"");
  });
}
function fitGraph(){if(schemaCy){schemaCy.fit(undefined,40);updateGraphLabels();}}
function toggleGraphLabels(){graphLabelsForced=!graphLabelsForced;updateGraphLabels();}
function randomPointInDisk(radius){
  const r=radius*Math.sqrt(Math.random()),a=Math.random()*2*Math.PI;
  return {x:r*Math.cos(a),y:r*Math.sin(a)};
}
function rerunGraphLayout(){
  if(!schemaCy)return;
  // cose's per-iteration cost grows with node count; a fixed 150-iteration
  // animated layout on a large graph (e.g. the "all" node-count option) can
  // freeze the tab for a long time. Scale iterations down and drop the
  // animation past a size threshold instead of always paying the same cost.
  const n=schemaCy.nodes().length;
  const heavy=n>300;
  schemaCy.layout({
    name:"cose",animate:!heavy,animationDuration:400,fit:true,padding:50,gravity:0,
    numIter:heavy?Math.max(20,Math.round(150*300/n)):150,
  }).run();
}
// Co-activation line thickness is proportional to weight, scaled against
// the min/max weight across ALL co-activation edges currently loaded (not
// just the toggle-visible subset, so thickness stays stable when other
// relation types are shown/hidden via the legend). Returns a scaler
// function rather than a single value since both drawGraph (cytoscape)
// and drawGraphSvgFallback (no-cytoscape fallback) need their own
// min/max-to-pixel-range mapping.
function coactivationWidthScale(edges){
  const weights=(edges||[]).filter(e=>e.relation==="coactivated_with").map(e=>Number(e.confidence||0));
  const lo=weights.length?Math.min(...weights):0;
  const hi=weights.length?Math.max(...weights):1;
  return function(w,minPx,maxPx){
    if(hi<=lo)return (minPx+maxPx)/2;
    const t=(Number(w||0)-lo)/(hi-lo);
    return minPx+Math.max(0,Math.min(1,t))*(maxPx-minPx);
  };
}
function drawGraph(g){
  lastGraphData=g;
  const cyEl=document.getElementById("schemaGraphCy");
  const svg=document.getElementById("schemaGraph");
  const meta=document.getElementById("graphMeta");
  const nodes=g.nodes||[];
  const allEdges=g.edges||[];
  const edges=allEdges.filter(e=>relVisible[e.relation]!==false);
  const coactScale=coactivationWidthScale(allEdges);
  if(meta)meta.textContent=`${nodes.length} schemas · ${edges.length} relations · limit ${g.limit}`;
  if(!window.cytoscape){
    if(cyEl)cyEl.innerHTML=`<div class="graph-empty">Cytoscape failed to load; using SVG fallback.</div>`;
    if(svg)svg.style.display="block";
    drawGraphSvgFallback(g);
    return;
  }
  if(svg)svg.style.display="none";
  if(!nodes.length){
    if(schemaCy){schemaCy.destroy();schemaCy=null;}
    cyEl.innerHTML=`<div class="graph-empty">No nodes for selected filters</div>`;
    return;
  }
  cyEl.innerHTML="";
  if(schemaCy){schemaCy.destroy();schemaCy=null;}
  const elements=[];
  nodes.forEach(n=>{
    const stage=Number(n.generalization_stage||0);
    elements.push({data:{
      id:n.id,label:graphNodeLabel(n),schema_id:n.schema_id,content:n.content||n.label||"",
      status:n.status,effective_status:n.status||"active",scope:n.scope||"",schema_class:n.schema_class||"",
      salience:Number(n.salience||0),confidence:Number(n.confidence||0),stage,is_labile:!!n.is_labile,
      color:nodeColor(n),size:graphNodeSize(n),border:n.is_labile?"#f5b942":(stage>=3?"#6af5aa":stage===2?"#ffd580":stage===1?"#7ab5ff":"#253050"),
      borderWidth:n.is_labile?4:(stage>0?2.5:1.2)
    },position:randomPointInDisk(1600)});
  });
  edges.forEach(e=>{
    const conf=Number(e.confidence||0.5);
    const isCoact=e.relation==="coactivated_with";
    elements.push({data:{
      id:e.id,source:e.source,target:e.target,relation:e.relation,confidence:conf,
      color:relColor[e.relation]||"#5a6e91",width:isCoact?coactScale(conf,0.6,3.2):1+3*conf,
      label:relLabel[e.relation]||e.relation,reason:e.reason||"",src_schema_id:e.src_schema_id,dst_schema_id:e.dst_schema_id
    }});
  });
  schemaCy=cytoscape({
    container:cyEl,
    elements,
    minZoom:0.08,
    maxZoom:3.0,
    layout:{name:"preset",fit:true,padding:50},
    style:[
      {selector:"node",style:{
        "background-color":"data(color)","width":"data(size)","height":"data(size)",
        "border-color":"data(border)","border-width":"data(borderWidth)","border-opacity":0.95,
        "label":"","color":"#b8c8e8","font-size":10,"text-outline-color":"#050b18","text-outline-width":2,
        "text-valign":"bottom","text-halign":"right","text-margin-x":4,"text-margin-y":4,
        "overlay-padding":6,"transition-property":"border-width, opacity","transition-duration":"120ms"
      }},
      {selector:"node[effective_status = 'superseded']",style:{"opacity":0.55}},
      {selector:"node[effective_status = 'archived']",style:{"opacity":0.38}},
      {selector:"node[effective_status = 'forgotten']",style:{"opacity":0.22}},
      {selector:"edge",style:{
        "line-color":"data(color)","target-arrow-color":"data(color)","target-arrow-shape":"triangle",
        "width":"data(width)","curve-style":"bezier","opacity":0.48,
        "label":"","font-size":8,"color":"data(color)","text-background-color":"#050b18","text-background-opacity":0.75,"text-background-padding":2
      }},
      {selector:"edge[relation = 'relates_to']",style:{
        "source-arrow-shape":"triangle",
        "source-arrow-color":"data(color)",
        "target-arrow-shape":"triangle",
        "line-style":"dashed"
      }},
      {selector:"edge[relation = 'coactivated_with']",style:{
        "source-arrow-shape":"none",
        "target-arrow-shape":"none",
        "line-style":"dotted",
        "opacity":0.35,
        "width":"data(width)"
      }},
      {selector:"node:selected",style:{"border-color":"#fff","border-width":5,"z-index":20}},
      {selector:".faded",style:{"opacity":0.12}},
      {selector:".highlight",style:{"opacity":1,"z-index":30}},
      {selector:"edge.highlight",style:{"opacity":0.95,"width":4}}
    ]
  });
  // Gentle cose pass: pull connected nodes together while isolated ones stay scattered
  // PENDING EVAL - current test shows cose re-converges even at low iterations; disabled for now.
  // Use "Layout" button to run cose and see if edge clustering helps.
  schemaCy.on("zoom",updateGraphLabels);
  schemaCy.on("mouseover","node",ev=>{
    const n=ev.target.data();
    showTip(ev.originalEvent,`<b>${esc(n.label)}</b><br><span style="color:#7a8db5">${esc(n.effective_status)}</span> · sal ${Number(n.salience).toFixed(2)} · stage ${n.stage}<br><em>${esc(String(n.content||"").slice(0,180))}</em>`);
  });
  schemaCy.on("mouseout","node",hideTip);
  schemaCy.on("mouseover","edge",ev=>{
    const e=ev.target.data();
    const arrow = e.relation === "relates_to" ? "↔" : "—";
    const metric = e.relation === "coactivated_with" ? `weight: ${Number(e.confidence||0).toFixed(3)}` : `confidence: ${Number(e.confidence||0).toFixed(2)}`;
    showTip(ev.originalEvent,`<b>${esc(e.relation)}</b><br>sch_${e.src_schema_id} ${arrow} sch_${e.dst_schema_id}<br>${metric}${e.reason?`<br><em>${esc(e.reason)}</em>`:""}`);
  });
  schemaCy.on("mouseout","edge",hideTip);
  schemaCy.on("tap","node",ev=>{
    const node=ev.target;
    schemaCy.elements().removeClass("faded highlight");
    const neighborhood=node.closedNeighborhood();
    schemaCy.elements().not(neighborhood).addClass("faded");
    neighborhood.addClass("highlight");
    selectGraphNode(node.data(),null);
    updateGraphLabels();
  });
  schemaCy.on("tap",ev=>{
    if(ev.target===schemaCy){schemaCy.elements().removeClass("faded highlight");updateGraphLabels();}
  });
  updateGraphLabels();
}
function drawGraphSvgFallback(g){
  const svg=document.getElementById("schemaGraph");
  svg.innerHTML=`<defs>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L8,3 z" fill="#7a8db5" opacity="0.8"/>
    </marker>
    <marker id="arrow-rev" markerWidth="8" markerHeight="8" refX="1" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M8,0 L8,6 L0,3 z" fill="#7a8db5" opacity="0.8"/>
    </marker>
    <filter id="glow"><feGaussianBlur stdDeviation="2" result="coloredBlur"/>
      <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>`;
  const w=svg.clientWidth||900,h=svg.clientHeight||660,cx=w/2,cy=h/2;
  const nodes=g.nodes||[],edges=(g.edges||[]).filter(e=>relVisible[e.relation]!==false);
  const coactScale=coactivationWidthScale(g.edges||[]);
  if(!nodes.length){
    const t=document.createElementNS("http://www.w3.org/2000/svg","text");
    t.setAttribute("x",cx);t.setAttribute("y",cy);
    t.setAttribute("text-anchor","middle");t.setAttribute("fill","#5a6e91");t.setAttribute("font-size","14");
    t.textContent="No nodes for selected filters";
    svg.appendChild(t);return;
  }
  const byId=Object.fromEntries(nodes.map(n=>[n.id,n]));
  // Initial layout: spiral
  nodes.forEach((n,i)=>{
    const a=2*Math.PI*i/Math.max(1,nodes.length);
    const r=Math.min(w,h)*(0.25+0.2*((i%5)/5));
    n.x=cx+Math.cos(a)*r;n.y=cy+Math.sin(a)*r;
    n.vx=0;n.vy=0;
  });
  // Force-directed layout. The pairwise repulsion below is O(n^2) per
  // iteration -- a fixed 120 iterations on a large graph (e.g. the "all"
  // node-count option, up to _GRAPH_HARD_CAP=2000) can run for tens of
  // millions of operations and freeze the tab. Scale iterations down past a
  // size threshold instead of always paying the same fixed cost.
  const svgIters=nodes.length>300?Math.max(15,Math.round(120*300/nodes.length)):120;
  for(let iter=0;iter<svgIters;iter++){
    nodes.forEach(n=>{n.vx*=0.78;n.vy*=0.78;});
    for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){
      const a=nodes[i],b=nodes[j],dx=a.x-b.x,dy=a.y-b.y;
      const d2=dx*dx+dy*dy+0.01,d=Math.sqrt(d2),force=1100/d2;
      a.vx+=dx/d*force;b.vx-=dx/d*force;
      a.vy+=dy/d*force;b.vy-=dy/d*force;
    }
    edges.forEach(e=>{
      const a=byId[e.source],b=byId[e.target];
      if(!a||!b)return;
      const dx=b.x-a.x,dy=b.y-a.y,d=Math.sqrt(dx*dx+dy*dy)+0.01,force=(d-160)*0.003;
      a.vx+=dx/d*force;b.vx-=dx/d*force;
      a.vy+=dy/d*force;b.vy-=dy/d*force;
    });
    // center gravity
    nodes.forEach(n=>{
      n.vx+=(cx-n.x)*0.002;n.vy+=(cy-n.y)*0.002;
      n.x=Math.max(24,Math.min(w-24,n.x+n.vx));
      n.y=Math.max(24,Math.min(h-24,n.y+n.vy));
    });
  }
  // Draw edges
  edges.forEach(e=>{
    const a=byId[e.source],b=byId[e.target];
    if(!a||!b)return;
    const color=relColor[e.relation]||"#5a6e91";
    const sw=e.relation==="coactivated_with"?coactScale(e.confidence,0.5,2.8):1.5+2*(e.confidence||0.5);
    // offset for arrow
    const dx=b.x-a.x,dy=b.y-a.y,d=Math.sqrt(dx*dx+dy*dy)+0.01;
    const x2=b.x-dx/d*12,y2=b.y-dy/d*12;
    const isRelatesTo = e.relation === "relates_to";
    const isCoactivated = e.relation === "coactivated_with";
    const line=document.createElementNS("http://www.w3.org/2000/svg","line");
    line.setAttribute("x1",(isRelatesTo||isCoactivated) ? a.x+dx/d*12 : a.x);
    line.setAttribute("y1",(isRelatesTo||isCoactivated) ? a.y+dy/d*12 : a.y);
    line.setAttribute("x2",(isRelatesTo||isCoactivated) ? b.x-dx/d*12 : x2);
    line.setAttribute("y2",(isRelatesTo||isCoactivated) ? b.y-dy/d*12 : y2);
    line.setAttribute("class","edge");line.setAttribute("stroke",color);
    line.setAttribute("stroke-width",sw);
    if (isCoactivated) {
      line.setAttribute("stroke-dasharray","2,4");
      line.setAttribute("opacity","0.35");
    } else if (isRelatesTo) {
      line.setAttribute("marker-start","url(#arrow-rev)");
      line.setAttribute("marker-end","url(#arrow)");
      line.setAttribute("stroke-dasharray","6,3");
    } else {
      line.setAttribute("marker-end","url(#arrow)");
    }
    const arrowLabel = isRelatesTo ? "↔" : (isCoactivated ? "—" : "→");
    const tipMetric = isCoactivated ? `weight: ${Number(e.confidence||0).toFixed(3)}` : `confidence: ${Number(e.confidence||0).toFixed(2)}`;
    line.addEventListener("mouseenter",ev=>showTip(ev,`<b>${e.relation}</b><br>sch_${e.src_schema_id} ${arrowLabel} sch_${e.dst_schema_id}<br>${tipMetric}${e.reason?`<br><em>${esc(e.reason)}</em>`:""}`) );
    line.addEventListener("mouseleave",hideTip);
    svg.appendChild(line);
    // edge label mid-point
    const t=document.createElementNS("http://www.w3.org/2000/svg","text");
    t.setAttribute("x",(a.x+b.x)/2);t.setAttribute("y",(a.y+b.y)/2-3);
    t.setAttribute("class","edge-label");t.setAttribute("text-anchor","middle");
    t.setAttribute("fill",color);t.setAttribute("font-size",isCoactivated?"8":"9");
    t.textContent=relLabel[e.relation]||e.relation;
    svg.appendChild(t);
  });
  // Draw nodes
  nodes.forEach(n=>{
    const r=8+Math.min(18,Math.sqrt(Math.max(0,n.salience))*3.5);
    const color=statusColor[n.status]||"#5a6e91";
    const c=document.createElementNS("http://www.w3.org/2000/svg","circle");
    c.setAttribute("cx",n.x);c.setAttribute("cy",n.y);c.setAttribute("r",r);
    c.setAttribute("fill",color);c.setAttribute("fill-opacity","0.85");
    c.setAttribute("stroke",n.is_labile?"#f5b942":color);c.setAttribute("stroke-width",n.is_labile?"4":"1.5");
    c.setAttribute("class","node");
    c.addEventListener("mouseenter",ev=>showTip(ev,`<b>sch_${n.schema_id}</b><br><span style="color:#7a8db5">${esc(n.status)}</span> · sal ${Number(n.salience).toFixed(2)}<br><em>${esc(n.label)}</em>`));
    c.addEventListener("mouseleave",hideTip);
    c.onclick=()=>selectGraphNode(n,c);
    svg.appendChild(c);
    // short label
    const lab=document.createElementNS("http://www.w3.org/2000/svg","text");
    lab.setAttribute("x",n.x+r+3);lab.setAttribute("y",n.y+4);
    lab.setAttribute("class","node-label");lab.textContent=`sch_${n.schema_id}`;
    svg.appendChild(lab);
  });
}
async function selectGraphNode(n,el){
  if(el){document.querySelectorAll(".node").forEach(x=>x.classList.remove("selected"));el.classList.add("selected");}
  const d=await getJSON(`/api/schemas/${n.schema_id}`);
  const s=d.schema||n;
  const eStat=s.status||"active";
  const evHtml=renderEvidenceList(d.evidence);
  const outHtml=d.outgoing&&d.outgoing.length
    ?table(["To","Rel.","Conf.","Reason"],d.outgoing.map(e=>[
        `sch_${e.dst_schema_id}`,e.relation,Number(e.confidence||0).toFixed(2),e.reason||"—"]))
    :"<em style='color:var(--muted)'>None.</em>";
  const inHtml=d.incoming&&d.incoming.length
    ?table(["From","Rel.","Conf.","Reason"],d.incoming.map(e=>[
        `sch_${e.src_schema_id}`,e.relation,Number(e.confidence||0).toFixed(2),e.reason||"—"]))
    :"<em style='color:var(--muted)'>None.</em>";
  const gLabileStyle=s.is_labile?';border-left:3px solid #f5b942;padding-left:8px':'';
  const gLabileTitle=s.is_labile?' title="Labile memory: under verification. Clears with sustained use — otherwise corrected or removed during background consolidation."':'';

  document.getElementById("graphDetail").innerHTML=`
    <div style="margin-bottom:10px">
      <div style="font-size:16px;font-weight:700${gLabileStyle}"${gLabileTitle}>sch_${s.schema_id}</div>
      <div style="margin-top:6px">${pill(eStat)}
        <span class="pill">sal ${Number(s.salience).toFixed(2)}</span>
        <span class="pill">conf ${Number(s.confidence).toFixed(2)}</span>
        ${s.schema_class?`<span class="pill">${esc(s.schema_class)}</span>`:""}
      </div>
    </div>
    <div style="font-size:13px;color:var(--text);line-height:1.6;margin-bottom:12px;padding:10px;background:var(--panel2);border-radius:var(--radius-sm);border:1px solid var(--line)">${esc(s.content)}</div>
    <div class="detail-section"><h4>Facets</h4><pre class="code-block">${esc(JSON.stringify(s.facets,null,2))}</pre></div>
    <div class="detail-section"><h4>Tags</h4>${(s.tags||[]).map(t=>`<span class="pill">${esc(t)}</span>`).join("")||"<em style='color:var(--muted)'>none</em>"}</div>
    <div class="detail-section"><h4>Outgoing relations</h4>${outHtml}</div>
    <div class="detail-section"><h4>Incoming relations</h4>${inHtml}</div>
    <div class="detail-section"><h4>Evidence</h4>${evHtml}</div>
    <div style="margin-top:10px;font-size:11px;color:var(--muted)">First formed: ${fmtTs(s.first_formed_ts)}<br>Last updated: ${fmtTs(s.last_updated_ts)}</div>
  `;
}

// ── DB HEALTH ──
async function loadDbHealth(){
  const ld=document.getElementById("dbHealthLoading");
  const el=document.getElementById("dbHealth");
  ld.classList.add("show");el.innerHTML="";
  try{
    const d=await getJSON("/api/db/health");
    if(!d.db_exists){
      el.innerHTML=`<div class="alert alert-error"><span class="alert-icon">❌</span><div>Database not found at <code>${esc(d.db_path)}</code></div></div>`;
      return;
    }
    const integrityOk=d.integrity_check&&d.integrity_check.length===1&&d.integrity_check[0]==="ok";
    const fkOk=!d.foreign_key_check||d.foreign_key_check.length===0;
    let html=`
      <div class="two-col" style="margin-bottom:12px">
        <div class="panel">
          <div class="panel-title">Pragmas</div>
          <pre class="code-block">${esc(JSON.stringify(d.pragmas,null,2))}</pre>
        </div>
        <div class="panel">
          <div class="panel-title">Integrity</div>
          <div class="alert ${integrityOk?"alert-ok":"alert-error"}">
            <span class="alert-icon">${integrityOk?"✅":"❌"}</span>
            <div>${integrityOk?"Integrity check passed":`Issues: ${esc(JSON.stringify(d.integrity_check))}`}</div>
          </div>
          <div class="alert ${fkOk?"alert-ok":"alert-warn"}" style="margin-top:8px">
            <span class="alert-icon">${fkOk?"✅":"⚠️"}</span>
            <div>${fkOk?"No FK violations":"FK violations: "+esc(JSON.stringify(d.foreign_key_check))}</div>
          </div>
        </div>
      </div>
    `;
    html+=`<div class="panel"><div class="panel-title">Tables &amp; Views</div>`;
    html+=table(
      ["Name","Type","Row count"],
      (d.tables||[]).map(t=>[t.name,t.type,t.count!=null?num(t.count):"—"])
    );
    html+="</div>";
    el.innerHTML=html;
  }finally{ld.classList.remove("show");}
}
// ── SESSION REPLAY ──
async function loadSessionTimeline(sid){
  let detail=document.getElementById("sessionTimeline");
  if(!detail){
    detail=document.createElement("div");
    detail.id="sessionTimeline";
    detail.style.cssText="position:fixed;bottom:16px;right:16px;width:600px;max-height:70vh;padding:16px;background:var(--panel);border:2px solid var(--blue);border-radius:var(--radius);overflow-y:auto;display:none;z-index:9999;box-shadow:0 8px 32px rgba(0,0,0,.5)";
    document.body.appendChild(detail);
  }
  detail.innerHTML=`<div style=\"text-align:center;padding:12px;color:var(--muted)\">Loading timeline…</div>`;
  detail.style.display="block";
  try{
    const d=await getJSON(`/api/sessions/${encodeURIComponent(sid)}/timeline`);
    if(d.error){detail.innerHTML=emptyState(d.error,"⚠️");return;}
    const sess=d.session;
    let html=`<div style=\"background:var(--panel2);border:1px solid var(--line);padding:12px;margin-bottom:10px\">`;
    html+=`<div style=\"display:flex;justify-content:space-between;align-items:center\">`;
    html+=`<div><b>Session ${esc(sess.id).slice(0,16)}…</b></div>`;
    html+=`<button class=\"btn\" style=\"font-size:11px;padding:2px 8px\" onclick=\"document.getElementById('sessionTimeline').style.display='none'\">✕ Close</button></div>`;
    html+=`<div style=\"font-size:13px;color:var(--muted);margin-top:4px\">Agent: ${esc(sess.agent)} · Scope: ${esc(sess.scope_id||"—")} · Goal: ${esc(sess.goal||"—")}</div>`;
    html+=`<div style=\"font-size:12px;color:var(--muted)\">${fmtTs(sess.started_ts)} → ${sess.ended_ts?fmtTs(sess.ended_ts):"ongoing"} · Outcome: ${esc(sess.outcome||"—")}</div></div>`;
    const events=d.events||[];
    if(!events.length){
      html+=emptyState("No events in this session.","📭");
    } else {
      html+=`<div style=\"position:relative;padding-left:20px;border-left:2px solid var(--line2)\">`;
      events.forEach(ev=>{
        const icon=ev.type==="user_message"?"👤":ev.type==="assistant_message"?"🤖":ev.type==="activate"?"▶️":"📌";
        html+=`<div style=\"margin-bottom:10px;position:relative\">`;
        html+=`<span style=\"position:absolute;left:-29px;top:2px;font-size:14px\">${icon}</span>`;
        html+=`<div style=\"font-size:11px;color:var(--muted);margin-bottom:3px\">${fmtTsCompact(ev.ts)} · <span class=\"pill\" style=\"font-size:10px;padding:2px 5px\">${esc(ev.type)}</span></div>`;
        html+=`<div style=\"font-size:13px;line-height:1.5\">${truncContent(ev.content||"",300)}</div></div>`;
      });
      html+=`</div>`;
    }
    detail.innerHTML=html;detail.scrollIntoView({behavior:"smooth"});
  }catch(e){detail.innerHTML=emptyState("Error: "+esc(String(e)),"⚠️");}
}

// ── GENERALIZATION ──
const GEN_LABELS=['SCOPED','PORTABLE','CONTEXTUAL','GLOBAL'];
const GEN_COLORS=['var(--gray)','var(--blue)','var(--amber)','var(--green)'];
const GEN_DESC=[
  'Only retrieved within its origin scope',
  'Retrieved across same-kind scopes (e.g. project:* → project:*)',
  'Retrieved everywhere with a relevance floor',
  'Retrieved everywhere with no restriction',
];
function genBadge(stage){
  const lbl=GEN_LABELS[stage]||'SCOPED';
  return `<span class="gen-badge gen-${stage}">${lbl}</span>`;
}
function genBreadthBar(pct,label,color,count){
  const w=Math.round(Math.min(100,pct*100));
  const detail=count!=null?` · ${count}`:'';
  const fill=color||'var(--blue)';
  return `<div class="gen-bar-wrap">
    <div class="gen-bar-track"><div class="gen-bar" style="width:${w}%;background:${fill}"></div></div>
    <span style="font-size:11px;color:var(--muted);white-space:nowrap">${label}: ${(pct*100).toFixed(0)}%${detail}</span>
  </div>`;
}
async function loadGeneralizationStats(){
  try{
    const d=await getJSON("/api/generalization");
    const sum=d.summary||{};
    const dist=d.stage_distribution||{};
    // Stat cards
    const cards=[
      {icon:"📖",label:"Total active",val:num(sum.total_active_schemas),accent:"var(--green)"},
      {icon:"🚀",label:"Portable",val:num(dist[1]||0),sub:"stage 1",accent:"var(--blue)"},
      {icon:"🌍",label:"Contextual",val:num(dist[2]||0),sub:"stage 2",accent:"var(--amber)"},
      {icon:"✨",label:"Global",val:num(dist[3]||0),sub:"stage 3",accent:"var(--green)"},
    ];
    document.getElementById("genStatGrid").innerHTML=cards.map(c=>
      `<div class="stat-card" style="--accent:${c.accent}">
        <div class="sc-icon">${c.icon}</div>
        <div class="sc-label">${esc(c.label)}</div>
        <div class="sc-value">${esc(String(c.val))}</div>
        <div class="sc-sub">${esc(c.sub||"")}</div>
      </div>`
    ).join("");
  }catch(e){}
}

// ── PROCEDURES ──

async function loadProcedures(){
  const ld=document.getElementById("procLoading");
  const gate=document.getElementById("procGate");
  const list=document.getElementById("procList");
  ld.style.display="flex";gate.style.display="none";list.innerHTML="";
  try{
    const minS=document.getElementById("procMinSessions").value;
    const lim=document.getElementById("procLimit").value;
    const scope=document.getElementById("procScope").value;
    const params=new URLSearchParams({min_sessions:minS,limit:lim});
    if(scope)params.set("scope",scope);
    const d=await getJSON("/api/procedures?"+params.toString());
    const clusters=d.clusters||[];
    const gateVal=d.gate||"";
    const stepSessions=d.step_sessions||0;
    const nGood=d.n_good_clusters||0;
    const gateTarget=d.gate_target||5;
    const avgSuccess=clusters.length?Math.round(clusters.reduce((s,c)=>s+c.success_rate,0)/clusters.length*1000)/10:0;
    const cards=[
      {icon:"🔄",label:"Procedure clusters",val:num(d.total_clusters_found),sub:`≥${d.min_sessions||2} sess`,accent:"var(--cyan)"},
      {icon:"📈",label:"Step-sessions",val:num(stepSessions),sub:`of ${num(d.qualified_sessions||0)} with goal+outcome`,accent:"var(--blue)"},
      {icon:"✅",label:"Avg success rate",val:avgSuccess+"%",sub:"across clusters",accent:"var(--green)"},
      {icon:"🎯",label:"Phase 2 gate",val:`${nGood}/${gateTarget}`,sub:d.gate_pass?"PASS":"not yet",accent:d.gate_pass?"var(--green)":"var(--amber)"},
    ];
    document.getElementById("procStatGrid").innerHTML=cards.map(c=>
      `<div class="stat-card" style="--accent:${c.accent}">
        <div class="sc-icon">${c.icon}</div>
        <div class="sc-label">${esc(c.label)}</div>
        <div class="sc-value">${esc(String(c.val))}</div>
        <div class="sc-sub">${esc(c.sub||"")}</div>
      </div>`
    ).join("");
    if(gateVal==="insufficient_data"){
      gate.style.display="block";
      gate.innerHTML=`<div class="alert alert-warn" style="margin-bottom:12px">
        <span class="alert-icon">⚠️</span>
        <div><b>Insufficient signal</b><br>
        Needs ≥${d.min_for_detection||2} sessions with real <code>commit(steps=[...])</code> data
        to detect patterns. Currently: <b>${stepSessions}</b> step-sessions.
        <br>This tab populates as you dogfood <code>slowave_commit(steps=[...])</code> more.</div>
      </div>`;
      return;
    }
    if(!clusters.length){
      list.innerHTML=emptyState("No procedure clusters found at min "+d.min_sessions+" sessions.","🔄");
      return;
    }
    list.innerHTML=clusters.map((c,i)=>{
      const sr=Math.round(c.success_rate*100);
      const sBar=`<div style="flex:1;height:6px;background:var(--panel3);border-radius:999px;overflow:hidden;max-width:120px">
        <div style="height:100%;width:${sr}%;background:var(--green);border-radius:999px"></div>
      </div>`;
      const goalsHTML=c.example_goals.map(g=>
        `<span style="color:var(--muted);font-style:italic">${esc(g.slice(0,80))}</span>`
      ).join(", ");
      const stepsHTML=(c.example_steps||[]).slice(0,3).map(s=>
        `<div style="color:var(--muted);padding:1px 0">· ${esc(s.slice(0,90))}</div>`
      ).join("");
      const scBadge=c.scope_id
        ?`<span style="font-size:11px;color:var(--muted);background:var(--panel2);padding:2px 6px;border-radius:4px">📦 ${esc(c.scope_id)}</span>`:"";
      const cohBadge=`<span style="font-size:11px;color:var(--muted);background:var(--panel2);padding:2px 6px;border-radius:4px">🎯 coherence ${c.goal_coherence.toFixed(2)}</span>`;
      const antiBadge=c.anti_pattern
        ?`<span style="font-size:11px;color:var(--red);background:color-mix(in srgb,var(--red) 15%,transparent);padding:2px 6px;border-radius:4px">⚠️ anti-pattern</span>`:"";
      const compBadge=(c.competes_with&&c.competes_with.length)
        ?`<span style="font-size:11px;color:var(--amber);background:color-mix(in srgb,var(--amber) 15%,transparent);padding:2px 6px;border-radius:4px">⇄ competes with #${c.competes_with.join(", #")}</span>`:"";
      const scColor=sr>=70?'var(--green)':sr>=40?'var(--amber)':'var(--red)';
      return `<div class="proc-card" style="margin-bottom:12px">
        <div class="proc-header">
          <div class="proc-goal">🔄 Procedure #${i+1}</div>
          <div style="font-size:13px;color:var(--muted);display:flex;align-items:center;gap:4px;flex-shrink:0">
            <b style="color:var(--text)">${c.session_count}</b> sessions ·
            <span style="color:${scColor}">${sr}% success</span>
          </div>
        </div>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:12px">
          ${sBar} <span style="color:var(--muted);white-space:nowrap">${c.successes}/${c.session_count}</span>
        </div>
        <div style="font-size:12px;margin-bottom:6px;display:flex;gap:8px;flex-wrap:wrap;align-items:center">
          🎯 ${goalsHTML}</div>
        ${stepsHTML?`<div style="font-size:11px;margin-bottom:8px">${stepsHTML}</div>`:""}
        <div style="font-size:12px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:8px">
          ${scBadge}${cohBadge}${antiBadge}${compBadge}
        </div>
        <button class="btn" onclick="toggleProcSessions(this,'proc-sessions-${i}')" style="font-size:11px">
          ▸ ${c.total_session_ids} sessions
        </button>
        <div id="proc-sessions-${i}" style="display:none;margin-top:8px;padding:8px 10px;background:var(--panel2);border-radius:6px;font-size:11px;max-height:200px;overflow-y:auto">
          ${c.session_ids.map(sid=>`<div style="padding:3px 0;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:8px">
            <span onclick="loadSessionTimeline('${esc(sid)}')" style="cursor:pointer;color:var(--blue);white-space:nowrap" title="Open session timeline">${esc(sid)}</span>
          </div>`).join("")}
          ${c.total_session_ids>c.session_ids.length?`<div style="color:var(--muted);padding-top:4px">… and ${c.total_session_ids-c.session_ids.length} more</div>`:""}
        </div>
      </div>`;
    }).join("");
  }catch(e){console.error("procedures",e);list.innerHTML=`<div style="color:var(--red);font-size:12px">${esc(String(e))}</div>`;}
  finally{ld.style.display="none";}
}

function toggleProcSessions(btn,id){
  const el=document.getElementById(id);
  if(!el)return;
  const open=el.style.display==="block";
  el.style.display=open?"none":"block";
  btn.textContent=(open?"▸":"▾")+" "+btn.textContent.slice(2);
}

// ── INIT ──
window.loadSessionTimeline = loadSessionTimeline;

loadStatus();
renderPulse();
renderHistogram();
setInterval(loadStatus,REFRESH_MS);
setInterval(renderPulse,REFRESH_MS);
"""
