import { useRef, useState, type MouseEvent } from "react";
import { type Json, structuralUrl } from "./api";
import { EmptyState, Link } from "./components";
import { bucketDescription, bucketIndex, bucketLabel, bucketPeriod, countAxis, ratePath, rollingRate, rollingWindow, useRate } from "./activityChart";

type Series = { key: string; label: string };
type View = "usage" | "creation";
const LEFT = 56, RIGHT = 56, WIDTH = 900, TOP = 26, BASE = 236, HEIGHT = 268;

/**
 * Two views of the same period.
 * Usage: Retrieved bars with Used nested inside (Used is a subset of Retrieved), plus the Used/Retrieved rate on the right axis.
 * Creation: Created bars on their own count axis.
 */
export function ActivityChartPanel({ data, series, title }: {
  data?: Json; series: readonly Series[]; title: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [view, setView] = useState<View>("usage");
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [hovered, setHovered] = useState<number | null>(null);
  const [pointer, setPointer] = useState<{ x: number; y: number } | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const channels = data?.channels || {};
  const created = series.find(s => s.label === "Created");
  const retrieved = series.find(s => s.label === "Retrieved");
  const used = series.find(s => s.label === "Used");
  const kind = series.some(s => s.key.includes("procedure")) ? "procedure" : "memory";
  const assessedKey = kind === "memory" ? "assessed_memories" : "assessed_procedures";
  const retrievalKey = `${kind}_retrievals`;
  const usedKey = kind === "memory" ? "used_memories" : "used_procedures";
  const timestamps = [...new Set<number>(series.flatMap(({ key }) =>
    (channels[key] || []).map((b: any) => Number(b.ts))))].sort((a, b) => a - b);
  const values = Object.fromEntries(Object.entries(channels).map(([key, rows]) =>
    [key, new Map((rows as any[]).map(b => [Number(b.ts), Number(b.n)]))]));
  const n = (key: string, i: number) => values[key]?.get(timestamps[i]) || 0;
  const total = (key: string) => timestamps.reduce((sum, _, i) => sum + n(key, i), 0);
  const current: View = view === "creation" && created ? "creation" : "usage";
  const showRetrieved = !!retrieved && !hidden.has(retrieved.key);
  const showUsed = !!used && !hidden.has(used.key);
  const showRate = !hidden.has("use_rate");
  const rates = timestamps.map((_, i) => useRate(n(usedKey, i), n(retrievalKey, i)));
  const windowSize = rollingWindow(Number(data?.bucket_minutes || 1440));
  const trend = rollingRate(timestamps.map((_, i) => n(usedKey, i)), timestamps.map((_, i) => n(retrievalKey, i)), windowSize);
  const formatRate = (rate: number | null) => rate === null ? "—" : `${rate.toLocaleString(undefined, { maximumFractionDigits: 1 })}%`;
  const toggle = (key: string) => setHidden(prev => { const next = new Set(prev); next.has(key) ? next.delete(key) : next.add(key); return next; });
  const axisMax = current === "creation"
    ? Math.max(1, ...timestamps.map((_, i) => created ? n(created.key, i) : 0))
    : Math.max(1, ...timestamps.map((_, i) => Math.max(showRetrieved ? n(retrievalKey, i) : 0, showUsed ? n(usedKey, i) : 0)));
  const axis = countAxis(axisMax);
  const slot = (WIDTH - LEFT - RIGHT) / Math.max(1, timestamps.length);
  const barW = Math.max(2, Math.min(24, slot * 0.7));
  const x = (i: number) => LEFT + (i + 0.5) * slot;
  const y = (value: number) => BASE - value / axis.max * (BASE - TOP);
  const rateY = (value: number) => BASE - value / 100 * (BASE - TOP);
  const minutes = Number(data?.bucket_minutes || 1440);
  const selectedIndex = selected === null ? null : timestamps.indexOf(selected);
  const active = hovered ?? (selectedIndex !== null && selectedIndex >= 0 ? selectedIndex : null);
  const activeIndex = active == null ? null : Math.min(active, timestamps.length - 1);
  const tickCount = Math.min(6, timestamps.length);
  const ticks = [...new Set(Array.from({ length: tickCount }, (_, i) => Math.round(i * (timestamps.length - 1) / Math.max(1, tickCount - 1))))];
  const dateLabel = (ts: number) => new Date(ts * 1000).toLocaleString([], {
    timeZone: "UTC", month: "short", day: "numeric",
    ...(minutes < 1440 ? { hour: "2-digit" as const, minute: "2-digit" as const, hourCycle: "h23" as const } : {}),
  });
  const indexAt = (event: MouseEvent<SVGRectElement>) => {
    const bounds = event.currentTarget.ownerSVGElement!.getBoundingClientRect();
    return bucketIndex(event.clientX, bounds.left, bounds.width, timestamps.length, RIGHT);
  };
  const partial = (i: number) => timestamps[i] < Number(data?.window_start) || timestamps[i] + minutes * 60 > Number(data?.now_ts);
  const period = (i: number) => bucketPeriod(timestamps[i], minutes, Number(data?.window_start || 0), Number(data?.now_ts || Infinity));
  const coverage = (i: number) => {
    const r = n(retrievalKey, i), a = n(assessedKey, i);
    return r ? `${Math.round(a / r * 100)}% · ${a.toLocaleString()} / ${r.toLocaleString()} assessed` : "No deliveries";
  };
  const links = (i: number) => {
    const { from, to } = period(i);
    const scope = new URLSearchParams(window.location.search).get("scope") || undefined;
    return {
      retrieval: structuralUrl("/retrieval", { scope, hours: "all", from, to, contains: kind }),
      created: structuralUrl(kind === "memory" ? "/memory" : "/procedures", kind === "memory"
        ? { scope, hours: "all", saved_from: from, saved_to: to }
        : { scope, hours: "all", from, to }),
    };
  };
  if (!series.some(s => total(s.key) > 0)) return <EmptyState title={`No ${title.toLowerCase()} activity yet`}>Activity will appear here once Slowave records it in the selected period.</EmptyState>;

  const hasRate = current === "usage" && showRate && !!retrieved && !!used;
  const tooltipRows = current === "creation"
    ? created ? [{ key: created.key, label: created.label, value: n(created.key, activeIndex ?? 0).toLocaleString() }] : []
    : [retrieved, used].filter((s): s is Series => !!s).map(s => ({ key: s.key, label: s.label, value: n(s.key, activeIndex ?? 0).toLocaleString() }));

  return <div className="activity-chart-panel activity-chart-lanes">
    {created && <div className="activity-view-tabs" role="group" aria-label="Activity view">
      {(["usage", "creation"] as const).map(v => <button key={v} type="button" aria-pressed={current === v}
        onClick={() => { setView(v); setHovered(null); setPointer(null); }}>{v === "usage" ? "Usage" : "Creation"}</button>)}
    </div>}
    <div className="formation-chart" ref={ref}>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="group" aria-label={`${title} ${current}: ${current === "usage" ? "retrieved and used delivery occasions with used rate" : "created records"}`}>
        {axis.ticks.map(value => <g key={value}>
          <line className="formation-gridline" x1={LEFT} x2={WIDTH - RIGHT} y1={y(value)} y2={y(value)}/>
          <text className="formation-axis-label" x={LEFT - 10} y={y(value) + 3} textAnchor="end">{value.toLocaleString()}</text>
        </g>)}
        {current === "creation" && created && timestamps.map((_, i) =>
          <rect key={timestamps[i]} className={`activity-creation-bar${activeIndex === i ? " is-active" : ""}`}
            x={x(i) - barW / 2} y={y(n(created.key, i))} width={barW} height={BASE - y(n(created.key, i))} rx="1.5"/>)}
        {current === "usage" && <g className="activity-usage-bars">
          {showRetrieved && timestamps.map((_, i) =>
            <rect key={`r${timestamps[i]}`} className={`activity-bar-retrieved${activeIndex === i ? " is-active" : ""}`}
              x={x(i) - barW / 2} y={y(n(retrievalKey, i))} width={barW} height={BASE - y(n(retrievalKey, i))} rx="1.5"/>)}
          {showUsed && timestamps.map((_, i) =>
            <rect key={`u${timestamps[i]}`} className={`activity-bar-used${activeIndex === i ? " is-active" : ""}`}
              x={x(i) - barW * (showRetrieved ? 0.3 : 0.5)} y={y(n(usedKey, i))} width={barW * (showRetrieved ? 0.6 : 1)} height={BASE - y(n(usedKey, i))} rx="1"/>)}
        </g>}
        {hasRate && <g className="activity-rate-series" aria-label="Reported used divided by retrieved; right axis">
          {[0, 50, 100].map(value => <text key={value} className="activity-rate-axis" x={WIDTH - RIGHT + 10} y={rateY(value) + 3}>{value}%</text>)}
          <line className="formation-gridline activity-rate-mid" x1={LEFT} x2={WIDTH - RIGHT} y1={rateY(50)} y2={rateY(50)}/>
          <path className="formation-line activity-rate-line" d={ratePath(trend.map(r => r ?? 0), x, rateY)}/>
          {timestamps.length > 0 && <g className="activity-rate-end">
            <circle className="activity-rate-dot" cx={x(timestamps.length - 1)} cy={rateY(trend[timestamps.length - 1] ?? 0)} r="3.5"/>
          </g>}
          {activeIndex !== null && <circle className="activity-rate-dot" cx={x(activeIndex)} cy={rateY(trend[activeIndex] ?? 0)} r="4"/>}
        </g>}
        {activeIndex !== null && <line className="formation-crosshair" x1={x(activeIndex)} x2={x(activeIndex)} y1={TOP} y2={BASE}/>}
        {ticks.map(i => <text key={i} className="formation-axis-label" x={x(i)} y={BASE + 21} textAnchor={i === 0 ? "start" : i === timestamps.length - 1 ? "end" : "middle"}>{dateLabel(timestamps[i])}</text>)}
        <rect className="formation-hit-area" x={LEFT} y={TOP} width={WIDTH - LEFT - RIGHT} height={BASE - TOP} tabIndex={0} role="slider"
          aria-label="Activity time bucket; arrow keys inspect, Enter selects a day" aria-valuemin={0} aria-valuemax={timestamps.length - 1} aria-valuenow={activeIndex ?? 0}
          aria-valuetext={`${bucketLabel(timestamps[activeIndex ?? 0], minutes)} UTC; ${n(retrievalKey, activeIndex ?? 0)} retrieved; ${n(usedKey, activeIndex ?? 0)} used; ${formatRate(rates[activeIndex ?? 0])} used divided by retrieved`}
          onMouseMove={event => {
            setHovered(indexAt(event));
            const bounds = ref.current?.getBoundingClientRect();
            if (bounds) setPointer({ x: Math.max(8, Math.min(event.clientX - bounds.left + 16, bounds.width - 190)), y: Math.max(8, Math.min(event.clientY - bounds.top - 12, bounds.height - 150)) });
          }} onMouseLeave={() => { setHovered(null); setPointer(null); }}
          onClick={event => { const i = indexAt(event); setSelected(selected === timestamps[i] ? null : timestamps[i]); }}
          onFocus={() => { setPointer(null); setHovered(selectedIndex !== null && selectedIndex >= 0 ? selectedIndex : 0); }} onBlur={() => setHovered(null)}
          onKeyDown={event => {
            if (event.key === "Escape") { setSelected(null); setHovered(null); return; }
            if (["Enter", " "].includes(event.key)) { event.preventDefault(); setSelected(timestamps[activeIndex ?? 0]); return; }
            if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
            event.preventDefault();
            setHovered(event.key === "Home" ? 0 : event.key === "End" ? timestamps.length - 1 : Math.max(0, Math.min(timestamps.length - 1, (activeIndex ?? 0) + (event.key === "ArrowRight" ? 1 : -1))));
          }}/>
      </svg>
      {hovered !== null && activeIndex !== null && <div className="formation-tooltip activity-lane-tooltip" style={pointer ? { left: pointer.x, top: pointer.y } : { left: 68, top: 30 }}>
        <strong>{bucketLabel(timestamps[activeIndex], minutes)} UTC</strong>
        <small>{partial(activeIndex) ? "Partial period · " : ""}{selectedIndex === activeIndex ? "Selected" : "Click to inspect"}</small>
        {tooltipRows.map(row => <span key={row.key} className={`formation-tooltip-${row.key}`}><i/><span>{row.label}</span><b>{row.value}</b></span>)}
        {current === "usage" && <span className="formation-tooltip-use-rate"><i/><span>Used / retrieved</span><b>{formatRate(rates[activeIndex])}</b></span>}
        {current === "usage" && <div className="activity-tooltip-coverage">{coverage(activeIndex)}</div>}
      </div>}
    </div>
    <div className="formation-footer">
      <p className="formation-caption" title="Used is a subset of Retrieved. The trend line pools reported used ÷ all retrieved occasions over a trailing 7-day window (the tooltip also shows each bucket's own rate), including deliveries without feedback. Periods without retrievals are drawn at zero; their rate is unavailable. Legend rate = total used ÷ total retrieved for the selected period. Used is attributed to the original retrieval date.">{bucketDescription(minutes)} · UTC{current === "usage" ? " · Used ⊂ Retrieved" : ""}</p>
      <div className="formation-legend" aria-label="Visible series">
        {current === "creation" && created && <span className={`formation-legend-item formation-legend-${created.key} activity-bar-legend`}><i/>{created.label}<b>{total(created.key).toLocaleString()}</b></span>}
        {current === "usage" && [retrieved, used].filter((s): s is Series => !!s).map(s =>
          <button key={s.key} type="button" className={`formation-legend-item formation-legend-${s.key} activity-bar-legend`} aria-pressed={!hidden.has(s.key)} onClick={() => toggle(s.key)}><i/>{s.label}<b>{total(s.key).toLocaleString()}</b></button>)}
        {current === "usage" && <button type="button" className="formation-legend-item activity-rate-legend" aria-pressed={showRate} onClick={() => toggle("use_rate")} title="Total reported used ÷ total retrieved for the selected period; the line shows each bucket's rate on the right axis"><i/>Used / retrieved<b>{formatRate(useRate(total(usedKey), total(retrievalKey)))}</b></button>}
      </div>
    </div>
    {selectedIndex !== null && selectedIndex >= 0 && <div className="activity-day-detail" aria-label="Selected activity period">
      <div><strong>{bucketLabel(timestamps[selectedIndex], minutes)} UTC</strong><span>{current === "usage" ? `Reported use ${formatRate(rates[selectedIndex])} · ${coverage(selectedIndex)}` : `${n(created!.key, selectedIndex).toLocaleString()} created`}{partial(selectedIndex) ? " · Partial period" : ""}</span></div>
      <div className="activity-day-links">
        {current === "creation" ? <Link to={links(selectedIndex).created}>Inspect created {kind === "memory" ? "memories" : "procedures"} →</Link> : <Link to={links(selectedIndex).retrieval}>Inspect retrieval requests →</Link>}
        <button type="button" onClick={() => setSelected(null)} aria-label="Close selected period">×</button></div>
    </div>}
  </div>;
}
