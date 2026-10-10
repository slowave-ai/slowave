import {
  Fragment,
  lazy,
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import { createPortal } from "react-dom";
import {
  api,
  navigate,
  structuralUrl,
  useApi,
  type Json,
  type LocationState,
} from "./api";
import {
  allowActions,
  ClampedText,
  ColumnsControl,
  CopyValue,
  DefinitionTooltip,
  EmptyState,
  ErrorState,
  formatDate,
  formatDuration,
  formatRate,
  getRatePercent,
  glossary,
  Icon,
  InlineError,
  Inspector,
  Link,
  LoadingRows,
  MetricCard,
  MetricCardsSkeleton,
  PageHeader,
  Pagination,
  refreshMs,
  relativeDate,
  ScopeSelect,
  RateMetricCard,
  Section,
  SortButton,
  StatusBadge,
  TableFrame,
  truncate,
} from "./components";

type PageProps = { location: LocationState };
const GraphExplorer = lazy(() => import("./GraphExplorer"));
const sharedColumnHelp = {
  scope: "The scope (memory boundary) this record belongs to; it keeps related data isolated.",
  outcome: "The recorded task outcome: success, partial, or failure. It describes what happened, not a causal quality score.",
  verification: "How strongly the recorded outcome was checked: verified, partially verified, or unverified.",
  retrieved: "How many distinct retrieval events returned this record.",
  used: "How many retrieval events explicitly assessed this record as used.",
  effect: "The reported downstream effect of using this procedure: Helpful, No effect, Harmful, or Unknown. Unknown means no effect assessment was recorded.",
};
const formatBytes = (value: unknown) => {
  const bytes = Math.max(0, Number(value) || 0);
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let size = bytes;
  let unit = -1;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(1)} ${units[unit]}`;
};
const param = (location: LocationState, key: string, fallback = "") =>
  location.search.get(key) || fallback;
const pageNumber = (location: LocationState) =>
  Math.max(1, Number(param(location, "page", "1")) || 1);
function updateParams(
  path: string,
  location: LocationState,
  changes: Record<string, string | number | undefined>,
) {
  const query = new URLSearchParams(location.search);
  Object.entries(changes).forEach(([key, value]) =>
    value === undefined || value === "" || value === 0
      ? query.delete(key)
      : query.set(key, String(value)),
  );
  navigate(`${path}${query.size ? `?${query}` : ""}`);
}
function openDetail(to: string, location: LocationState) {
  navigate(to, {
    state: {
      listUrl: `${location.path}${location.search.size ? `?${location.search}` : ""}`,
    },
  });
}
function closeDetail(parent: string, location: LocationState) {
  navigate(location.state?.listUrl || parent);
}
function rowKeys(event: KeyboardEvent, action: () => void) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    action();
  }
}

function Availability({ home }: { home: Json }) {
  const status = home.status || {};
  const database = home.database || {};
  const latest = home.workers?.runs?.[0];
  const cells = [
    {
      label: "MCP daemon",
      state: status.daemon?.running ? "Available" : "Needs attention",
      observed: home.observed_at,
      source: "Process probe",
      detail: status.daemon?.running
        ? `Version ${status.daemon?.version || status.slowave_version || "unknown"}`
        : "No reachable process observed",
      href: "/diagnostics#services",
    },
    {
      label: "Database",
      state: !status.db_exists
        ? "Needs attention"
        : database.integrity_status === "ok"
          ? "Available"
          : database.integrity_status === "needs_attention"
            ? "Needs attention"
            : database.integrity_status === "unavailable" || database.integrity_status === "unknown"
              ? "Not checked"
              : "Unknown",
      observed: database.checked_at || home.observed_at,
      source: "Integrity check",
      detail: !status.db_exists
        ? `Cannot open ${status.db_path}`
        : formatBytes(status.db_size_bytes || database.db_size_bytes),
      href: "/diagnostics#database",
    },
    {
      label: "Maintenance",
      state: !latest
        ? "Unknown"
        : latest.error_text
          ? "Needs attention"
          : latest.ended_ts
            ? "Available"
            : "Unknown",
      observed: latest?.ended_ts || latest?.started_ts,
      source: "Last recorded run",
      detail: !latest
        ? "No maintenance pass recorded"
        : latest.error_text
          ? truncate(latest.error_text, 90)
          : `${latest.schemas_created || 0} formed · ${latest.schemas_reinforced || 0} reinforced`,
      href: "/diagnostics#maintenance",
    },
  ];
  return (
    <Section title="Availability">
      <div className="availability-strip">
        {cells.map((cell) => (
          <Link to={cell.href} className={`availability-cell availability-state-${String(cell.state).toLowerCase().replaceAll(" ", "_")}`} key={cell.label}>
            <div>
              <strong>{cell.label}</strong>
              <StatusBadge value={cell.state} />
            </div>
            <p>{cell.detail}</p>
            <small>
              {cell.source} ·{" "}
              {cell.observed ? relativeDate(cell.observed) : "not observed"}
            </small>
          </Link>
        ))}
      </div>
    </Section>
  );
}

const memoryActivitySeries = [
  { key: "schemas", label: "Created" },
  { key: "memory_retrievals", label: "Retrieved" },
  { key: "used_memories", label: "Used" },
] as const;

const procedureActivitySeries = [
  { key: "procedures", label: "Created" },
  { key: "procedure_retrievals", label: "Retrieved" },
  { key: "used_procedures", label: "Used" },
] as const;

const clampNumber = (value: number, min: number, max: number) =>
  Math.min(Math.max(value, min), max);

/** Smooth cubic path through the points, clamped so it never overshoots. */
function smoothFormationPath(points: { x: number; y: number }[]): string {
  if (points.length === 0) return "";
  if (points.length === 1) return `M ${points[0].x} ${points[0].y}`;
  let path = `M ${points[0].x} ${points[0].y}`;
  for (let index = 0; index < points.length - 1; index += 1) {
    const previous = points[index - 1] || points[index];
    const current = points[index];
    const next = points[index + 1];
    const after = points[index + 2] || next;
    const tension = 0.18;
    const controlOneX = current.x + (next.x - previous.x) * tension;
    const controlOneY = clampNumber(
      current.y + (next.y - previous.y) * tension,
      Math.min(current.y, next.y),
      Math.max(current.y, next.y),
    );
    const controlTwoX = next.x - (after.x - current.x) * tension;
    const controlTwoY = clampNumber(
      next.y - (after.y - current.y) * tension,
      Math.min(current.y, next.y),
      Math.max(current.y, next.y),
    );
    path += ` C ${controlOneX.toFixed(2)} ${controlOneY.toFixed(2)} ${controlTwoX.toFixed(2)} ${controlTwoY.toFixed(2)} ${next.x.toFixed(2)} ${next.y.toFixed(2)}`;
  }
  return path;
}

function ActivityChartPanel({
  data,
  series,
  title,
}: {
  data?: Json;
  series: readonly { key: string; label: string }[];
  title: string;
}) {
  const [visible, setVisible] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(series.map(({ key }) => [key, true])),
  );
  const [hovered, setHovered] = useState<{
    x: number;
    y: number;
    index: number;
  } | null>(null);
  const chartRef = useRef<HTMLDivElement>(null);
  const channels = data?.channels || {};
  const timestamps: string[] = Array.from(
    new Set(
      series.flatMap(({ key }) =>
        (channels[key] || []).map((bucket: any) => String(bucket.ts)),
      ),
    ),
  ).sort((a, b) => Number(a) - Number(b));
  const seriesValues = series.map(({ key }) =>
    timestamps.map((ts) =>
      Number(
        (channels[key] || []).find((bucket: any) => String(bucket.ts) === ts)?.n ||
          0,
      ),
    ),
  );
  const seriesTotals = seriesValues.map((values) =>
    values.reduce((sum, value) => sum + value, 0),
  );
  const shownSeries = series
    .map((definition, index) => ({ ...definition, index }))
    .filter((definition) => visible[definition.key]);
  const hasActivity = seriesValues.some((values) =>
    values.some((value) => value > 0),
  );
  const width = 900,
    chartHeight = 208,
    // Reserve enough room for grouped counts (for example, "10,000") so
    // right-aligned Y-axis labels remain inside the SVG viewport.
    left = 56,
    right = 28,
    top = 18,
    baseline = 176,
    plotWidth = width - left - right,
    maxBarHeight = baseline - top,
    maxValue = Math.max(1, ...shownSeries.flatMap(({ index }) => seriesValues[index]));
  const rawTickStep = maxValue / 4;
  const tickMagnitude = rawTickStep > 0 ? 10 ** Math.floor(Math.log10(rawTickStep)) : 1;
  const tickFraction = rawTickStep / tickMagnitude;
  const tickStep =
    (tickFraction <= 1 ? 1 : tickFraction <= 2 ? 2 : tickFraction <= 5 ? 5 : 10) *
    tickMagnitude;
  const axisMax = Math.max(tickStep, Math.ceil(maxValue / tickStep) * tickStep);
  const axisTicks = Array.from({ length: 5 }, (_, index) => axisMax - index * tickStep).filter(
    (value) => value >= 0,
  );
  const slot = plotWidth / Math.max(1, timestamps.length);
  const xForIndex = (index: number) => left + index * slot + slot / 2;
  const yForValue = (value: number) => baseline - (value / axisMax) * maxBarHeight;
  const xTickIndices = Array.from(
    new Set(
      Array.from({ length: Math.min(6, timestamps.length) }, (_, index) =>
        timestamps.length > 1
          ? Math.round((index * (timestamps.length - 1)) / (Math.min(6, timestamps.length) - 1))
          : 0,
      ),
    ),
  );
  const formatAxisTime = (ts: string) => {
    const date = new Date(Number(ts) * 1000);
    return Number(data?.window_hours || 24) <= 24
      ? date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      : date.toLocaleDateString([], { month: "short", day: "numeric" });
  };
  const trackHover = (event: React.MouseEvent<SVGRectElement>) => {
    const bounds = chartRef.current?.getBoundingClientRect();
    if (!bounds || timestamps.length === 0) return;
    const svgX = ((event.clientX - bounds.left) / Math.max(1, bounds.width)) * width;
    const index = clampNumber(
      Math.floor((svgX - left) / Math.max(1, slot)),
      0,
      timestamps.length - 1,
    );
    const tooltipWidth = Math.min(220, Math.max(0, bounds.width - 24));
    setHovered({
      x: Math.max(
        12,
        Math.min(event.clientX - bounds.left + 14, bounds.width - tooltipWidth - 12),
      ),
      y: Math.max(8, event.clientY - bounds.top - 10),
      index,
    });
  };
  return (
    <div className="activity-chart-panel">
      <div className="formation-chart" ref={chartRef}>
        {!hasActivity ? (
          <EmptyState title={`No ${title.toLowerCase()} activity yet`}>
            Creation, retrieval, and reported use will appear here once Slowave
            records activity in the selected period.
          </EmptyState>
        ) : (
          <>
            <svg
              viewBox={`0 0 ${width} ${chartHeight}`}
              role="img"
              aria-label={`${title} created, retrieved, and reported used over the selected period`}
            >
              <defs>
                {series.map(({ key }) => (
                  <linearGradient
                    key={key}
                    id={`formation-fill-${key}`}
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop className={`formation-stop-top formation-stop-${key}`} offset="0%" />
                    <stop className={`formation-stop-bottom formation-stop-${key}`} offset="100%" />
                  </linearGradient>
                ))}
              </defs>
              {axisTicks.map((value) => {
                const y = yForValue(value);
                return (
                  <g key={value}>
                    <line className="formation-gridline" x1={left} x2={width - right} y1={y} y2={y} />
                    <text className="formation-axis-label" x={left - 8} y={y + 4} textAnchor="end">
                      {value.toLocaleString()}
                    </text>
                  </g>
                );
              })}
              <line
                className="formation-baseline"
                x1={left}
                x2={width - right}
                y1={baseline}
                y2={baseline}
              />
              {xTickIndices.map((index) => {
                const x = xForIndex(index);
                const isFirst = index === xTickIndices[0];
                const isLast = index === xTickIndices[xTickIndices.length - 1];
                return (
                  <g key={`x-${timestamps[index]}`}>
                    <line className="formation-tick" x1={x} x2={x} y1={baseline} y2={baseline + 5} />
                    <text
                      className="formation-axis-label"
                      x={isFirst ? left : isLast ? width - right : x}
                      y={baseline + 19}
                      textAnchor={isFirst ? "start" : isLast ? "end" : "middle"}
                    >
                      {formatAxisTime(timestamps[index])}
                    </text>
                  </g>
                );
              })}
              {hovered && timestamps[hovered.index] !== undefined && (
                <line
                  className="formation-crosshair"
                  x1={xForIndex(hovered.index)}
                  x2={xForIndex(hovered.index)}
                  y1={top}
                  y2={baseline}
                />
              )}
              {shownSeries.map(({ key, index }) => {
                const points = seriesValues[index].map((value, pointIndex) => ({
                  x: xForIndex(pointIndex),
                  y: yForValue(value),
                }));
                const line = smoothFormationPath(points);
                const area = `${line} L ${points[points.length - 1].x} ${baseline} L ${points[0].x} ${baseline} Z`;
                return (
                  <g key={key} className={`formation-series formation-series-${key}`}>
                    <path
                      className="formation-area"
                      d={area}
                      fill={`url(#formation-fill-${key})`}
                    />
                    <path className="formation-line" d={line} pathLength={1} />
                    {timestamps.length <= 48
                      ? points.map((point, pointIndex) => (
                          <circle
                            className={`formation-dot ${hovered?.index === pointIndex ? "formation-dot-active" : ""}`}
                            key={`${key}-${pointIndex}`}
                            cx={point.x}
                            cy={point.y}
                            r={hovered?.index === pointIndex ? 4 : 2.4}
                          />
                        ))
                      : null}
                  </g>
                );
              })}
              <rect
                className="formation-hit-area"
                x={left}
                y={top}
                width={plotWidth}
                height={maxBarHeight}
                onMouseMove={trackHover}
                onMouseLeave={() => setHovered(null)}
              />
            </svg>
            {hovered && (
              <div
                className="formation-tooltip"
                style={{ left: hovered.x, top: hovered.y }}
              >
                <strong>{formatDate(timestamps[hovered.index])}</strong>
                {shownSeries.map(({ key, label, index }) => (
                  <span className={`formation-tooltip-${key}`} key={key}>
                    <i aria-hidden="true" />
                    {label}: {seriesValues[index][hovered.index].toLocaleString()}
                  </span>
                ))}
              </div>
            )}
          </>
        )}
      </div>
      <div className="formation-legend" aria-label="Visible series">
        {series.map(({ key, label }, index) => (
          <button
            type="button"
            key={key}
            className={`formation-legend-item formation-legend-${key}`}
            aria-pressed={visible[key]}
            onClick={() =>
              setVisible((current) => ({ ...current, [key]: !current[key] }))
            }
          >
            <i aria-hidden="true" />
            {label}
            <b>{seriesTotals[index].toLocaleString()}</b>
          </button>
        ))}
      </div>
    </div>
  );
}

function MemoryFormationChart({ data }: { data?: Json }) {
  const [activeChart, setActiveChart] = useState<"memories" | "procedures">("memories");
  const isMemories = activeChart === "memories";
  return (
    <Section
      title={
        <>
          Memory activity
          <DefinitionTooltip label="Memory activity definition">
            Memories and procedures created, retrieved, and reported used in the
            selected period. Retrieved counts are delivered item occasions;
            reported use counts only items with accepted use feedback.
          </DefinitionTooltip>
        </>
      }
    >
      <div className="knowledge-activity-tabs" role="group" aria-label="Knowledge type">
        {(["memories", "procedures"] as const).map((tab) => (
          <button
            type="button"
            key={tab}
            aria-pressed={activeChart === tab}
            className="knowledge-activity-tab"
            onClick={() => setActiveChart(tab)}
          >
            {tab === "memories" ? "Memories" : "Procedures"}
          </button>
        ))}
      </div>
      <div>
        <ActivityChartPanel
          key={activeChart}
          data={data}
          series={isMemories ? memoryActivitySeries : procedureActivitySeries}
          title={isMemories ? "Memories" : "Procedures"}
        />
      </div>
    </Section>
  );
}
const stageNames = ["Scoped", "Portable", "Contextual", "Global"];
const stageTooltip =
  "How broadly this memory is eligible for retrieval beyond its original context. Stages are assigned from recorded evidence across contexts and tasks. A higher stage means broader eligibility, not greater accuracy or usefulness.";
const stageDefinitions = [
  "Ordinarily restricted to origin context.",
  "Eligible in other contexts of the same kind when sufficiently relevant.",
  "Eligible across context kinds with score discount and relevance floor.",
  "Broad cross-context eligibility; relevance still determines selection.",
];
const feedbackTooltip =
  "Latest supported assessments divided by actually delivered item/request occasions. Repeated requests count; page retries do not. Missing feedback is unknown. Later accepted corrections may revise past periods.";
function metricHref(path: string, data: Json) {
  return `${path}?scope=${encodeURIComponent(data.scope || "")}&from=${data.window?.from || ""}&to=${data.window?.to || ""}`;
}
function LibraryCards({
  data,
  kind,
  compact = false,
}: {
  data: Json;
  kind: "memory" | "procedure";
  compact?: boolean;
}) {
  const memory = kind === "memory";
  const name = memory ? "Memory" : "Procedure";
  const path = memory ? "/memory" : "/procedures";
  return (
    <>
      <MetricCard
        title={memory ? "Active memories" : "Saved procedures"}
        value={
          data.available
            ? Number(data[`${kind}_total`]).toLocaleString()
            : "No data"
        }
        tooltip="Current canonical inventory; period selection applies to activity, not the inventory snapshot."
        href={metricHref(path, data)}
      />
      <RateMetricCard
        compact={compact}
        title={`${name} retrieval coverage`}
        numerator={data[`${kind}_exposed`]}
        denominator={data[`${kind}_total`]}
        tooltip="Distinct current inventory records delivered during the selected period divided by the current inventory. A record counts once even if delivered repeatedly. Retrieval does not establish use or quality."
        href={metricHref(path, data)}
      />
      <RateMetricCard
        compact={compact}
        title={`${name} usage rate`}
        numerator={data[`${kind}_used`]}
        denominator={data[`${kind}_assessed`]}
        secondary={`Among assessed retrieved ${memory ? "memories" : "procedures"} · ${data[`${kind}_unknown`] ?? 0} ${memory ? "unassessed" : "unknown use"}`}
        tooltip={`Distinct delivered records reported used divided by distinct delivered records with supported use assessments. Each record counts once even if delivered repeatedly. ${data[`${kind}_unknown`] ?? 0} delivered records have unknown use.`}
        href={metricHref(path, data)}
      />
      {!compact && (
        <RateMetricCard
          title="Feedback coverage · occasions"
          numerator={data[`${kind}_feedback`]}
          denominator={data[`${kind}_deliveries`]}
          secondary={`${Number(data[`${kind}_deliveries`] || 0) - Number(data[`${kind}_feedback`] || 0)} unassessed delivery occasions`}
          tooltip={`${feedbackTooltip} The same record can count again when delivered in a separate request.`}
        />
      )}
    </>
  );
}
function StageCard({ data }: { data: Json }) {
  const total = Number(data.memory_total || 0);
  return (
    <MetricCard
      className="distribution-card"
      title="Generalization stages"
      tooltip={stageTooltip}
      value={
        <div className="stage-distribution">
          <div className="distribution-bar">
            {(data.stages || [0, 0, 0, 0]).map((n: number, k: number) => (
              <span
                key={k}
                className={`stage-${k}`}
                style={{ width: `${total ? (n / total) * 100 : 0}%` }}
                title={`${k} · ${stageNames[k]}: ${n}. ${stageDefinitions[k]}`}
              />
            ))}
            {data.unknown_stage > 0 && (
              <span
                className="stage-unknown"
                style={{ width: `${(data.unknown_stage / total) * 100}%` }}
              />
            )}
          </div>
          <div className="distribution-legend stage-legend">
            {stageNames.map((name, k) => (
              <span
                key={k}
                className={`stage-label-${k}`}
                title={`${k} · ${name}: ${data.stages?.[k] || 0} memories. ${stageDefinitions[k]}`}
              >
                {k} · {name}{" "}
                {total
                  ? `${Math.round(((data.stages?.[k] || 0) / total) * 100)}%`
                  : "No data"}
              </span>
            ))}
          </div>
          {data.unknown_stage > 0 && (
            <small>{data.unknown_stage} missing or unrecognized stage</small>
          )}
        </div>
      }
    />
  );
}
const outcomeCategories = [
  { key: "successful_closed", label: "Success", color: "success" },
  { key: "partial_closed", label: "Partial", color: "partial" },
  { key: "failure_closed", label: "Failure", color: "failure" },
  { key: "unknown_outcome_closed", label: "Not reported", color: "unknown" },
];

function ReportedOutcomesCard({ summary }: { summary: Json }) {
  const total = Number(summary.closed || 0);
  return (
    <MetricCard
      className="distribution-card"
      title="Reported outcomes"
      tooltip="Distribution over ended activities, including not reported outcomes. Missing closure is not failure."
      value={
        <div className="stage-distribution">
          <div className="distribution-bar">
            {outcomeCategories.map(({ key, label, color }) => (
              <span
                key={key}
                className={`outcome-${color}`}
                style={{
                  width: `${total ? (Number(summary[key] || 0) / total) * 100 : 0}%`,
                }}
                title={`${label}: ${summary[key] || 0}`}
              />
            ))}
          </div>
          <div className="distribution-legend outcome-legend">
            {outcomeCategories.map(({ key, label, color }) => (
              <span key={key} className={`outcome-label-${color}`}>
                {label} · {summary[key] || 0}
              </span>
            ))}
          </div>
        </div>
      }
    />
  );
}

function FocusedChart({
  data,
  kind,
  tabs = false,
}: {
  data?: Json;
  kind: "memory" | "procedure";
  tabs?: boolean;
}) {
  const [selected, setSelected] = useState(kind);
  const series =
    selected === "memory" ? memoryActivitySeries : procedureActivitySeries;
  return (
    <Section title="Activity">
      {tabs && (
        <div
          className="knowledge-activity-tabs"
          role="group"
          aria-label="Knowledge type"
        >
          {(["memory", "procedure"] as const).map((k) => (
            <button
              type="button"
              className="knowledge-activity-tab"
              key={k}
              aria-pressed={selected === k}
              onClick={() => setSelected(k)}
            >
              {k === "memory" ? "Memories" : "Procedures"}
            </button>
          ))}
        </div>
      )}
      <ActivityChartPanel
        key={selected}
        data={data}
        series={tabs ? series.filter((s) => s.label !== "Created") : series}
        title={selected === "memory" ? "Memories" : "Procedures"}
      />
    </Section>
  );
}
function estimateResponseTokens(chars: unknown): number | null {
  if (chars == null || chars === "") return null;
  const value = Number(chars);
  return Number.isFinite(value) && value >= 0 ? Math.ceil(value / 4) : null;
}

function formatTokenEstimate(tokens: number | null): string {
  if (tokens == null) return "No data";

  const magnitude = Math.abs(tokens);
  const units = [
    { threshold: 1_000_000_000_000, divisor: 1_000_000_000_000, suffix: "T" },
    { threshold: 1_000_000_000, divisor: 1_000_000_000, suffix: "B" },
    { threshold: 1_000_000, divisor: 1_000_000, suffix: "M" },
    { threshold: 1_000, divisor: 1_000, suffix: "K" },
  ];
  const unit = units.find(({ threshold }) => magnitude >= threshold);
  if (!unit) return `~${tokens.toLocaleString()} tokens`;

  const amount = Math.floor((magnitude / unit.divisor) * 10) / 10;
  const formatted = Number.isInteger(amount) ? amount.toString() : amount.toFixed(1);
  return `~${tokens < 0 ? "-" : ""}${formatted}${unit.suffix} tokens`;
}

function ContextVolumeCards({ data }: { data: Json }) {
  const estimate = (kind: "Activate" | "Recall") => {
    const sample = data.context_size?.[kind];
    const medianTokens = estimateResponseTokens(sample?.median_chars);
    const samples = Number(sample?.samples);
    if (medianTokens == null || !Number.isFinite(samples) || samples <= 0)
      return null;
    return { samples, medianTokens, total: medianTokens * samples };
  };
  const activate = estimate("Activate");
  const recall = estimate("Recall");
  const combined = activate && recall ? activate.total + recall.total : null;
  return (
    <>
      <MetricCard
        title="Activate volume estimate"
        value={formatTokenEstimate(activate?.total ?? null)}
        tooltip={
          activate
            ? `Median-based proxy: ~${activate.medianTokens.toLocaleString()} tokens × ${activate.samples.toLocaleString()} Activate responses. This is not the sum of actual per-response tokens or billed model usage.`
            : "No Activate context-size samples are available."
        }
      />
      <MetricCard
        title="Recall volume estimate"
        value={formatTokenEstimate(recall?.total ?? null)}
        tooltip={
          recall
            ? `Median-based proxy: ~${recall.medianTokens.toLocaleString()} tokens × ${recall.samples.toLocaleString()} Recall responses. This is not the sum of actual per-response tokens or billed model usage.`
            : "No Recall context-size samples are available."
        }
      />
      <MetricCard
        title="Combined volume estimate"
        value={formatTokenEstimate(combined)}
        tooltip={
          activate && recall
            ? `Sum of the Activate and Recall median-based proxies (${activate.total.toLocaleString()} + ${recall.total.toLocaleString()} estimated tokens). This is not the sum of actual per-response tokens or billed model usage.`
            : "Both Activate and Recall context-size samples are needed for this estimate."
        }
      />
    </>
  );
}
function MemoryEffectiveness({ data }: { data: Json }) {
  return (
    <Section title="Overview">
      <div className="metric-card-grid home-metric-card-grid">
        <LibraryCards data={data} kind="memory" compact />
        <LibraryCards data={data} kind="procedure" compact />
        <RateMetricCard
          compact
          title="Retrieval match rate"
          numerator={
            Number(data.retrievals_total || 0) -
            Number(data.retrievals_no_match || 0)
          }
          denominator={data.retrievals_total}
          tooltip="Requests delivering supported context divided by recorded requests. Empty can be appropriate; a match does not establish relevance."
          href={metricHref("/retrieval", data)}
        />
        <RateMetricCard
          compact
          title="Feedback coverage"
          numerator={data.feedback_assessed}
          denominator={data.deliveries}
          secondary={`${data.feedback_unknown || 0} unassessed delivery occasions`}
          tooltip={feedbackTooltip}
          href={metricHref("/retrieval", data)}
        />
        <MetricCard
          title="Active scopes"
          value={
            data.available
              ? Number(data.active_scopes).toLocaleString()
              : "No data"
          }
          tooltip={glossary.active_scopes}
          href={metricHref("/memory", data)}
        />
        <ContextVolumeCards data={data} />
      </div>
    </Section>
  );
}
function PeriodFilter({
  location,
  path,
}: {
  location: PageProps["location"];
  path: string;
}) {
  return (
    <label>
      Activity period
      <select
        value={param(location, "hours", "all")}
        onChange={(e) =>
          updateParams(path, location, {
            hours: e.target.value,
            from: undefined,
            to: undefined,
            page: 1,
          })
        }
      >
        <option value="all">All times</option>
        <option value="24">Last day</option>
        <option value="168">Last week</option>
        <option value="720">Last month</option>
      </select>
    </label>
  );
}

export function HomePage({ location }: PageProps) {
  const hours = param(location, "hours", "all");
  const request = useApi<Json>(
    `/api/home?hours=${encodeURIComponent(hours)}`,
    { pollMs: refreshMs },
  );
  const home = request.data;
  return (
    <div className="page">
      <PageHeader
        title="Home"
        description="Readiness, actionable exceptions, and observed changes in the selected period."
        updatedAt={request.updatedAt}
        refreshing={request.refreshing}
        onRefresh={request.reload}
        controls={
          <div className="home-controls">
            <label className="compact-control">
              Time range
              <select
                value={hours}
                onChange={(e) =>
                  updateParams("/", location, { hours: e.target.value })
                }
              >
                <option value="3">Last 3 hours</option>
                <option value="12">Last 12 hours</option>
                <option value="24">Last 24 hours</option>
                <option value="168">Last week</option>
                <option value="720">Last month</option>
                <option value="all">All times</option>
              </select>
            </label>
          </div>
        }
      />
      <InlineError
        error={request.error}
        retained={Boolean(home)}
        retry={request.reload}
      />
      {request.loading && !home ? (
        <LoadingRows rows={9} />
      ) : request.error && !home ? (
        <ErrorState title="Home data unavailable" error={request.error} retry={request.reload} />
      ) : (
        home && (
          <>
            <Availability home={home} />
            <MemoryFormationChart data={home.activity} />
            {home.effectiveness ? (
              <MemoryEffectiveness data={home.effectiveness} />
            ) : (
              <Section title="Memory health">
                <EmptyState title="No memory metrics available">Memory-health metrics will appear when the dashboard receives memory and retrieval data.</EmptyState>
              </Section>
            )}
            <MemoryByScope breakdown={home.scope_breakdown} />
            {home.recent_changes?.length ? (
              <Section
                title={
                  <>
                    Recent activity{" "}
                    <span className="section-qualifier">selected period</span>
                  </>
                }
                actions={
                  <Link
                    to={structuralUrl("/activity", {
                      from: home.window?.from,
                      to: home.window?.to,
                    })}
                  >
                    View all activity
                  </Link>
                }
              >
                <div className="change-feed">
                  {home.recent_changes.map((item: any) => (
                    <Link
                      to={item.href}
                      className="change-row"
                      key={`${item.kind}-${item.id}`}
                    >
                      <time title={formatDate(item.observed_at)}>
                        {relativeDate(item.observed_at)}
                      </time>
                      <StatusBadge value={item.kind} />
                      <span className="change-preview">
                        {truncate(item.preview, 150)}
                      </span>
                      <span className="scope-text">
                        {truncate(item.scope || "No scope", 28)}
                      </span>
                      <StatusBadge value={item.state} />
                    </Link>
                  ))}
                </div>
              </Section>
            ) : (
              <EmptyState
                title="Slowave is ready; no memory activity yet"
                firstRun
              >
                Normal agent use will create activity here. Activity captured →
                durable memory formed → retrieval recorded → feedback observed.
              </EmptyState>
            )}
          </>
        )
      )}
    </div>
  );
}

const memoryStateOptions = [
  "active",
  "needs_review",
  "stale",
  "archived",
];

function DeleteConfirmation({
  previewUrl,
  onCancel,
  onConfirm,
}: {
  previewUrl: string;
  onCancel: () => void;
  onConfirm: () => Promise<void>;
}) {
  const preview = useApi<Json>(previewUrl);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState("");
  const affected = preview.data?.affected || [];
  const summary = preview.data?.summary || {};
  const confirm = async () => {
    setDeleting(true);
    setError("");
    try {
      await onConfirm();
    } catch (cause) {
      setError((cause as Error).message);
      setDeleting(false);
    }
  };
  return createPortal(
    <div className="confirmation" role="dialog" aria-modal="true" aria-label="Delete permanently">
      <div className="delete-confirmation">
        <h3>Delete permanently?</h3>
        <p>This cannot be undone. The selected record and the affected references below will be permanently removed or scrubbed.</p>
        {preview.loading && !preview.data ? <LoadingRows rows={3} /> : preview.error && !preview.data ? (
          <InlineError error={preview.error} retry={preview.reload} />
        ) : (
          <>
            <p><strong>Selected {preview.data?.entity?.kind}:</strong> {preview.data?.entity?.preview || preview.data?.entity?.id}</p>
            <p className="neutral"><strong>{affected.length}</strong> affected record{affected.length === 1 ? "" : "s"} · {Object.entries(summary).map(([action, count]) => `${count} ${action}`).join(" · ") || "selected record only"}</p>
            {affected.length ? <div className="delete-preview" aria-label="Affected records">
              {affected.map((item: any, index: number) => (
                <div className="delete-preview-row" key={`${item.kind}-${item.id}-${index}`}>
                  <strong>{item.kind}</strong>
                  <span>{item.action}</span>
                  <p>{item.preview || item.id}</p>
                </div>
              ))}
            </div> : <p className="neutral">No dependent records were found.</p>}
          </>
        )}
        {error && <InlineError error={error} />}
        <div className="confirmation-actions">
          <button className="button secondary" disabled={deleting} onClick={onCancel}>Cancel</button>
          <button className="button danger" disabled={!preview.data || deleting} onClick={confirm}>
            {deleting ? "Deleting…" : "Delete permanently"}
          </button>
        </div>
      </div>
    </div>
    , document.body,
  );
}

function MemoryUseRateBar({ used, assessed }: { used: unknown; assessed: unknown }) {
  const percent = getRatePercent(used, assessed);
  const available = percent !== null;
  return (
    <div
      className={`use-rate-bar${available ? "" : " is-disabled"}`}
      role="progressbar"
      aria-label={available ? `Use rate ${Math.round(percent)}%` : "Use rate unavailable"}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={available ? Math.round(percent) : undefined}
      aria-disabled={!available}
    >
      <span className="use-rate-bar-track" aria-hidden="true">
        <span className="use-rate-bar-fill" style={{ width: `${percent ?? 0}%` }} />
      </span>
      <span className="use-rate-bar-value">
        {available ? `${Math.round(percent)}%` : "—"}
      </span>
    </div>
  );
}

function SalienceBar({ salience }: { salience: unknown }) {
  const value = Number(salience);
  const percent = Number.isFinite(value)
    ? Math.min(100, Math.max(0, (value / 20) * 100))
    : null;
  const available = percent !== null;
  return (
    <div
      className={`use-rate-bar${available ? "" : " is-disabled"}`}
      role="progressbar"
      aria-label={available ? `Salience ${Math.round(percent)}%` : "Salience unavailable"}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={available ? Math.round(percent) : undefined}
      aria-disabled={!available}
    >
      <span className="use-rate-bar-track" aria-hidden="true">
        <span className="use-rate-bar-fill" style={{ width: `${percent ?? 0}%` }} />
      </span>
      <span className="use-rate-bar-value">
        {available ? `${Math.round(percent)}%` : "—"}
      </span>
    </div>
  );
}

function MemoryByScope({ breakdown }: { breakdown: Json }) {
  const rows: any[] = breakdown?.rows || [];
  const total = Number(breakdown?.total || 0);
  const other = breakdown?.other;
  if (!rows.length && !other) {
    return (
      <Section title="Memory by scope">
        <EmptyState title="No memories yet">
          Memory scopes will appear after your agent stores its first durable memory.
        </EmptyState>
      </Section>
    );
  }
  const renderRow = (
    scope: string,
    memories: number,
    neverRetrieved: number,
    key: string,
  ) => {
    const label = scope === "(no scope)" ? "No scope" : scope;
    const share = total > 0 ? (memories / total) * 100 : 0;
    const retrieved = memories - neverRetrieved;
    const coverage = memories > 0 ? (retrieved / memories) * 100 : 0;
    return (
      <tr key={key}>
        <td className="scope-text" title={label}>
          {truncate(label, 40)}
        </td>
        <td className="numeric">{memories.toLocaleString()}</td>
        <td className="numeric">
          <span
            className="use-rate-bar"
            role="progressbar"
            aria-label={`${Math.round(share)}% of memories`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(share)}
          >
            <span className="use-rate-bar-track" aria-hidden="true">
              <span className="use-rate-bar-fill" style={{ width: `${share}%` }} />
            </span>
            <span className="use-rate-bar-value">{Math.round(share)}%</span>
          </span>
        </td>
        <td className="numeric">
          <span
            className="use-rate-bar"
            role="progressbar"
            aria-label={`${Math.round(coverage)}% retrieval coverage`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(coverage)}
            title={`${retrieved.toLocaleString()} of ${memories.toLocaleString()} retrieved`}
          >
            <span className="use-rate-bar-track" aria-hidden="true">
              <span className="use-rate-bar-fill" style={{ width: `${coverage}%` }} />
            </span>
            <span className="use-rate-bar-value">{Math.round(coverage)}%</span>
          </span>
        </td>
      </tr>
    );
  };
  return (
    <Section title="Memory by scope">
      <TableFrame label="Memory by scope">
        <table>
          <thead>
            <tr>
              <th>Scope</th>
              <th className="numeric">Memories</th>
              <th className="numeric">Share</th>
              <th className="numeric">
                Retrieval coverage{" "}
                <DefinitionTooltip label="Retrieval coverage definition">
                  Share of memories in this scope that have been surfaced during
                  a retrieval at least once.
                </DefinitionTooltip>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r: any) => {
              const scope = String(r.scope);
              return renderRow(
                scope,
                Number(r.memories),
                Number(r.never_retrieved),
                scope,
              );
            })}
            {other
              ? renderRow(
                  `${other.scope} (${other.scope_count})`,
                  Number(other.memories),
                  Number(other.never_retrieved),
                  "__other__",
                )
              : null}
          </tbody>
        </table>
      </TableFrame>
    </Section>
  );
}

export function MemoryPage({ location }: PageProps) {
  const detailId = location.path.startsWith("/memory/")
    ? decodeURIComponent(location.path.split("/")[2])
    : "";
  const scope = param(location, "scope");
  const states = param(location, "states");
  const sort = param(location, "sort", "changed");
  const dir = param(location, "dir", "desc") as "asc" | "desc";
  const page = pageNumber(location);
  const [search, setSearch] = useState("");
  const [debounced, setDebounced] = useState("");
  const [visibleColumns] = useState<string[]>(["memory", "state", "salience", "scope", "changed", "use_rate", "last_used"]);
  const visible = (id: string) => visibleColumns.includes(id);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(search), 250);
    return () => clearTimeout(timer);
  }, [search]);
  const endpoint = `/api/schemas?states=${encodeURIComponent(states)}&scope=${encodeURIComponent(scope)}&sort=${sort}&dir=${dir}&page=${page}&per_page=50&q=${encodeURIComponent(debounced)}&from=${param(location, "from")}&to=${param(location, "to")}&hours=${param(location, "hours", "all")}&saved_from=${param(location, "saved_from")}`;
  const request = useApi<Json>(endpoint);
  const rows = request.data?.schemas || [];
  const pagination = request.data?.pagination || {
    page,
    per_page: 50,
    total: 0,
  };
  const counts = request.data?.status_counts || {};
  const changeSort = (column: string) =>
    updateParams("/memory", location, {
      sort: column,
      dir: sort === column && dir === "desc" ? "asc" : "desc",
      page: 1,
    });
  return (
    <div className="page">
      <PageHeader
        title="Memories"
        description="Durable context Slowave can use in future retrieval."
        updatedAt={request.updatedAt}
        refreshing={request.refreshing}
        onRefresh={request.reload}
      />
      <div className="filter-bar">
        <PeriodFilter location={location} path="/memory" />
        <label>
          Search<span className="sr-only"> memory text</span>
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && setDebounced(search)}
            placeholder="Search memory text…"
          />
        </label>
        <label>
          State
          <select
            value={states}
            onChange={(e) =>
              updateParams("/memory", location, {
                states: e.target.value,
                page: 1,
              })
            }
          >
            <option value="">All</option>
            {memoryStateOptions.map((state) => (
              <option key={state} value={state}>
                {state.replaceAll("_", " ")}
              </option>
            ))}
          </select>
        </label>
        <ScopeSelect
          value={scope}
          onChange={(next) =>
            updateParams("/memory", location, { scope: next, page: 1 })
          }
        />
        <label>
          Saved since
          <input
            type="date"
            value={
              param(location, "saved_from")
                ? new Date(Number(param(location, "saved_from")) * 1000)
                    .toISOString()
                    .slice(0, 10)
                : ""
            }
            onChange={(e) =>
              updateParams("/memory", location, {
                saved_from: e.target.value
                  ? Math.floor(new Date(e.target.value).getTime() / 1000)
                  : undefined,
                page: 1,
              })
            }
          />
        </label>
      </div>
      {states && (
        <div className="filter-chips">
          <button
            onClick={() =>
              updateParams("/memory", location, {
                states: undefined,
                page: 1,
              })
            }
          >
            State: {states || "all"} ×
          </button>
        </div>
      )}
      {request.loading && !request.data ? (
        <MetricCardsSkeleton count={4} />
      ) : request.error && !request.data ? (
        <ErrorState
          title="Memory summary unavailable"
          error={request.error}
          retry={request.reload}
        />
      ) : request.data ? (
        <>
          <div
            className="metric-card-grid knowledge-summary-grid"
            aria-label="Memory summary"
          >
            <LibraryCards data={request.data.metrics || {}} kind="memory" />
          </div>
          <div
            className="metric-card-grid distribution-card-grid generalization-card-grid"
            aria-label="Memory generalization stages"
          >
            <StageCard data={request.data.metrics || {}} />
          </div>
          <FocusedChart data={request.data.metrics?.chart} kind="memory" />
        </>
      ) : (
        <EmptyState title="No memory summary available">
          Summary metrics will appear when the memory service returns a result.
        </EmptyState>
      )}
      <InlineError
        error={request.error}
        retained={Boolean(request.data)}
        retry={request.reload}
      />
      {request.loading && !request.data ? (
        <LoadingRows />
      ) : request.error && !request.data ? (
        <ErrorState
          title="Memory results unavailable"
          error={request.error}
          retry={request.reload}
        />
      ) : rows.length ? (
        <>
          <TableFrame label="Memory results">
            <table>
              <thead>
                <tr>
                  {visible("memory") && <th
                    aria-sort={
                      sort === "content"
                        ? dir === "asc"
                          ? "ascending"
                          : "descending"
                        : "none"
                    }
                  >
                    <SortButton
                      label="Memory text"
                      active={sort === "content"}
                      direction={dir}
                      onClick={() => changeSort("content")}
                    />
                  </th>}
                  {visible("state") && <th><SortButton label="State" active={sort === "status"} direction={dir} onClick={() => changeSort("status")} /><DefinitionTooltip label="State definition">The memory's lifecycle status: active, needs review, or stale.</DefinitionTooltip></th>}
                  {visible("salience") && <th className="numeric"><SortButton label="Salience" active={sort === "salience"} direction={dir} onClick={() => changeSort("salience")} /><DefinitionTooltip label="Salience definition">A measure of how strongly this memory is favored for retrieval. Higher means it is more likely to be surfaced.</DefinitionTooltip></th>}
                  {visible("scope") && <th><SortButton label="Scope" active={sort === "scope"} direction={dir} onClick={() => changeSort("scope")} /><DefinitionTooltip label="Scope definition">{sharedColumnHelp.scope}</DefinitionTooltip></th>}
                  {visible("created") && <th
                    aria-sort={
                      sort === "formed"
                        ? dir === "asc"
                          ? "ascending"
                          : "descending"
                        : "none"
                    }
                  >
                    <SortButton
                      label="Created"
                      active={sort === "formed"}
                      direction={dir}
                      onClick={() => changeSort("formed")}
                    />
                  </th>}
                  {visible("changed") && <th
                    aria-sort={
                      sort === "changed"
                        ? dir === "asc"
                          ? "ascending"
                          : "descending"
                        : "none"
                    }
                  >
                    <SortButton
                      label="Changed"
                      active={sort === "changed"}
                      direction={dir}
                      onClick={() => changeSort("changed")}
                    />
                  </th>}
                  {visible("evidence") && <th className="numeric">
                    <SortButton label="Supporting evidence" active={sort === "evidence"} direction={dir} onClick={() => changeSort("evidence")} />{" "}
                    <DefinitionTooltip label="Evidence count definition">
                      Number of recorded provenance links. This is evidence
                      volume, not evidence quality.
                    </DefinitionTooltip>
                  </th>}
                  {visible("retrieved") && <th className="numeric">
                    <SortButton
                      label="Retrieved"
                      active={sort === "exposed"}
                      direction={dir}
                      onClick={() => changeSort("exposed")}
                    />{" "}
                    <DefinitionTooltip label="Times exposed definition">
                      {glossary.exposed}
                    </DefinitionTooltip>
                  </th>}
                  {visible("used") && <th className="numeric">
                    <SortButton
                      label="Used"
                      active={sort === "used"}
                      direction={dir}
                      onClick={() => changeSort("used")}
                    />{" "}
                    <DefinitionTooltip label="Times used definition">
                      {glossary.used}
                    </DefinitionTooltip>
                  </th>}
                  {visible("use_rate") && (
                    <th className="numeric">
                      <SortButton
                        label="Use rate among assessed %"
                        active={sort === "use_rate"}
                        direction={dir}
                        onClick={() => changeSort("use_rate")}
                      />
                      <DefinitionTooltip label="Use rate definition">
                        Delivered occasions reported used divided by delivered
                        occasions with a supported assessment; only use assessed
                        items in the denominator.
                      </DefinitionTooltip>
                    </th>
                  )}
                  {visible("irrelevant") && <th className="numeric">
                    <SortButton
                      label="Irrelevant"
                      active={sort === "irrelevant"}
                      direction={dir}
                      onClick={() => changeSort("irrelevant")}
                    />{" "}
                    <DefinitionTooltip label="Times irrelevant definition">
                      {glossary.irrelevant}
                    </DefinitionTooltip>
                  </th>}
                  {visible("stale") && (
                    <th className="numeric">
                      <SortButton
                        label="Stale feedback"
                        active={sort === "stale"}
                        direction={dir}
                        onClick={() => changeSort("stale")}
                      />
                    </th>
                  )}
                  {visible("wrong") && (
                    <th className="numeric">
                      <SortButton
                        label="Wrong feedback"
                        active={sort === "wrong"}
                        direction={dir}
                        onClick={() => changeSort("wrong")}
                      />
                    </th>
                  )}
                  {visible("related") && (
                    <th className="numeric">
                      <SortButton
                        label="Related memories"
                        active={sort === "related"}
                        direction={dir}
                        onClick={() => changeSort("related")}
                      />
                    </th>
                  )}
                  {visible("source_activity") && <th className="numeric"><SortButton label="Source activity count" active={sort === "source_activity"} direction={dir} onClick={() => changeSort("source_activity")} /></th>}
                  {visible("last_used") && <th>
                    <SortButton
                      label="Last used"
                      active={sort === "last_used"}
                      direction={dir}
                      onClick={() => changeSort("last_used")}
                    />
                  </th>}
                </tr>
              </thead>
              <tbody>
                {rows.map((memory: any) => {
                  const href = `/memory/${encodeURIComponent(memory.id)}${location.search.size ? `?${location.search}` : ""}`;
                  return (
                    <tr
                      tabIndex={0}
                      key={memory.id}
                      onClick={() => openDetail(href, location)}
                      onKeyDown={(e) =>
                        rowKeys(e, () => openDetail(href, location))
                      }
                    >
                      {visible("memory") && (
                        <td className="primary-cell">
                          <ClampedText text={memory.content} />
                        </td>
                      )}
                      {visible("state") && (
                        <td>
                          <StatusBadge value={memory.status} />
                        </td>
                      )}
                      {visible("salience") && (
                        <td className="numeric">
                          <SalienceBar salience={memory.salience} />
                        </td>
                      )}
                      {visible("scope") && (
                        <td
                          className="scope-text"
                          title={memory.scope || undefined}
                        >
                          {memory.scope
                            ? truncate(memory.scope, 30)
                            : "No scope"}
                        </td>
                      )}
                      {visible("created") && (
                        <td title={formatDate(memory.first_formed_ts)}>
                          {relativeDate(memory.first_formed_ts)}
                        </td>
                      )}
                      {visible("changed") && (
                        <td title={formatDate(memory.last_updated_ts)}>
                          {relativeDate(memory.last_updated_ts)}
                        </td>
                      )}
                      {visible("evidence") && (
                        <td className="numeric">{memory.evidence_count}</td>
                      )}
                      {visible("retrieved") && (
                        <td className="numeric">{memory.times_exposed ?? 0}</td>
                      )}
                      {visible("used") && (
                        <td className="numeric">{memory.times_used ?? 0}</td>
                      )}
                      {visible("use_rate") && (
                        <td className="numeric">
                          <MemoryUseRateBar
                            used={memory.times_used}
                            assessed={memory.times_assessed}
                          />
                        </td>
                      )}
                      {visible("irrelevant") && (
                        <td className="numeric">
                          {memory.times_irrelevant ?? 0}
                        </td>
                      )}
                      {visible("stale") && (
                        <td className="numeric">{memory.times_stale ?? 0}</td>
                      )}
                      {visible("wrong") && (
                        <td className="numeric">{memory.times_wrong ?? 0}</td>
                      )}
                      {visible("related") && (
                        <td className="numeric">{memory.related_count ?? 0}</td>
                      )}
                      {visible("source_activity") && (
                        <td className="numeric">
                          {memory.source_activity_count ?? 0}
                        </td>
                      )}
                      {visible("last_used") && <td title={formatDate(memory.last_used_ts)}>
                        {memory.last_used_ts
                          ? relativeDate(memory.last_used_ts)
                          : "—"}
                      </td>}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </TableFrame>
          <Pagination
            page={pagination.page}
            perPage={pagination.per_page}
            total={pagination.total}
            onPage={(next) => updateParams("/memory", location, { page: next })}
          />
        </>
      ) : (
        <EmptyState
          title={
            search || states !== "active,needs_review"
              ? "No memories match these filters"
              : "No durable memories yet"
          }
        >
          {search || states !== "active,needs_review"
            ? "Clear or broaden the filters to see other recorded memories."
            : "A durable memory appears after normal Slowave activity forms one."}
        </EmptyState>
      )}
      {detailId && (
        <MemoryDetail
          id={detailId}
          onClose={() => closeDetail("/memory", location)}
          onDeleted={request.reload}
        />
      )}
    </div>
  );
}

function MemoryDetail({ id, onClose, onDeleted }: {
  id: string;
  onClose: () => void;
  onDeleted?: () => Promise<void>;
}) {
  const request = useApi<Json>(`/api/schemas/${encodeURIComponent(id)}`);
  const [confirm, setConfirm] = useState(false);
  const [actionError, setActionError] = useState("");
  const data = request.data;
  const memory = data?.schema;
  const remove = async () => {
    if (!memory) return;
    setActionError("");
    try {
      await api(`/api/schemas/${memory.schema_id}/delete`, { method: "POST" });
      await onDeleted?.();
      onClose();
    } catch (error) {
      setActionError((error as Error).message);
      throw error;
    }
  };
  return (
    <Inspector
      title="Memory"
      id={id}
      state={memory && <StatusBadge value={memory.status} />}
      onClose={onClose}
    >
      <InlineError
        error={request.error || actionError}
        retained={Boolean(data)}
        retry={request.reload}
      />
      {request.loading && !data ? (
        <LoadingRows rows={5} />
      ) : request.error && !data ? (
        <ErrorState title="Memory details unavailable" error={request.error} retry={request.reload} />
      ) : memory ? (
          <>
            <h2 className="detail-title">{memory.content}</h2>
            <dl className="key-values">
              <dt>Saved</dt>
              <dd>{formatDate(memory.first_formed_ts)}</dd>
              <dt>Source event</dt>
              <dd>{formatDate(data.evidence?.[0]?.event_occurred_at || data.evidence?.[0]?.event_ts)}</dd>
              <dt>Last updated</dt>
              <dd>{formatDate(memory.last_updated_ts)}</dd>
            </dl>
            <Section
              title="Summary"
              actions={
                allowActions ? (
                  <button className="button danger-secondary" onClick={() => setConfirm(true)}>Delete</button>
                ) : (
                  <p className="notice compact">
                    This dashboard was launched read-only. Delete is unavailable.
                  </p>
                )
              }
            >
            <dl className="key-values">
              <dt>Effect on retrieval</dt>
              <dd>
                {memory.status === "stale"
                    ? "Retained as historical context but not treated as current."
                    : "Eligible for future retrieval when scope and relevance rules admit it."}
              </dd>
              <dt>State reason</dt>
              <dd>
                {memory.stale_reason || "No specific state reason recorded"}
              </dd>
              <dt>Scope</dt>
              <dd>{memory.scope || "No scope recorded"}</dd>
            </dl>
            </Section>
            <Section title="Value signals">
              {(() => {
                const totals = memory.use_history_totals || {};
                const count = (assessment: string) => Number(totals[assessment] || 0);
                const retrieved = Number(totals.retrieved || 0);
                const used = count("used");
                return <dl className="key-values wide"><dt>Retrieved</dt><dd>{retrieved}</dd><dt>Used</dt><dd>{used}</dd><dt>Use rate</dt><dd>{formatRate(used, memory.use_history_totals?.assessed)}</dd><dt>Irrelevant / stale / wrong</dt><dd>{count("irrelevant")} / {count("stale")} / {count("wrong")}</dd><dt>Last retrieved</dt><dd>{formatDate(memory.use_history_totals?.last_retrieved)}</dd></dl>;
              })()}
            </Section>
            <Section title="Related records">
              {data.evidence?.length ? (
                <div className="detail-list">
                  {data.evidence.map((item: any, index: number) => (
                    <div key={index}>
                      <div>
                        <span>Episode {item.episode_id || "—"}</span>
                        <span>
                          {item.event_ts || item.occurred_at || item.recorded_at || item.created_at
                            ? formatDate(item.event_ts || item.occurred_at || item.recorded_at || item.created_at)
                            : "Time unavailable"}
                        </span>
                      </div>
                      <p>
                        {item.quote ||
                          item.event_content ||
                          "Evidence text unavailable"}
                      </p>
                      {item.episode_session && (
                        <Link
                          to={`/activity/${encodeURIComponent(item.episode_session)}`}
                        >
                          Source activity
                        </Link>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <EmptyState title="No linked evidence">
                  No provenance rows are available for this memory.
                </EmptyState>
              )}
            </Section>
            <Section title="Related retrievals">
              {data.retrievals?.length ? (
                <div className="detail-list compact">
                  {data.retrievals.map((item: any) => (
                    <Link
                      key={item.retrieval_id}
                      to={`/retrieval/${encodeURIComponent(item.retrieval_id)}`}
                    >
                      <span>
                        {formatDate(item.created_at)} ·{" "}
                        {item.pathway || "unknown pathway"}
                      </span>
                      <Icon name="chevron" size={14} />
                    </Link>
                  ))}
                </div>
              ) : (
                <p className="neutral">
                  Not exposed in a recorded retrieval yet.
                </p>
              )}
            </Section>
            <Section title="Lifecycle and feedback">
              {data.feedback?.length ? (
                <div className="activity-stream">
                  {[
                    ...(data.feedback || []).map((item: any) => ({
                      ...item,
                      ts: item.created_at,
                      label: `Feedback: ${item.assessment || item.status}`,
                    })),
                  ]
                    .sort((a, b) => b.ts - a.ts)
                    .map((item: any, index: number) => (
                      <div key={index}>
                        <time>{formatDate(item.ts)}</time>
                        <strong>{item.label}</strong>
                        <p>
                          {item.reason ||
                            item.stale_reason ||
                            "No reason recorded"}
                        </p>
                        {item.replacement_target_id && (
                          <Link to={`/memory/${item.replacement_target_id}`}>
                            Replacement {item.replacement_target_id}
                          </Link>
                        )}
                      </div>
                    ))}
                </div>
              ) : (
                <p className="neutral">
                  No recorded feedback entries.
                </p>
              )}
            </Section>
            <Section title="Related memories">
              {data.outgoing?.length || data.incoming?.length ? (
                <div className="detail-list compact">
                  {(data.outgoing || []).map((relation: any) => (
                    <Link
                      key={`out-${relation.dst_schema_id}-${relation.relation}`}
                      to={`/memory/sch_${relation.dst_schema_id}`}
                    >
                      <span>
                        {relation.relation.replaceAll("_", " ")} · sch_
                        {relation.dst_schema_id}
                      </span>
                      <Icon name="chevron" size={14} />
                    </Link>
                  ))}
                  {(data.incoming || []).map((relation: any) => (
                    <Link
                      key={`in-${relation.src_schema_id}-${relation.relation}`}
                      to={`/memory/sch_${relation.src_schema_id}`}
                    >
                      <span>
                        {relation.relation.replaceAll("_", " ")} from sch_
                        {relation.src_schema_id}
                      </span>
                      <Icon name="chevron" size={14} />
                    </Link>
                  ))}
                </div>
              ) : (
                <p className="neutral">No recorded content relations.</p>
              )}
            </Section>
            <details className="advanced" open>
              <summary>Advanced</summary>
              <dl className="key-values">
                <dt>Importance (salience)</dt>
                <dd>{memory.salience == null ? "—" : Number(memory.salience).toFixed(1)}</dd>
                <dt>Confidence</dt>
                <dd>{memory.confidence}</dd>
                <dt>Generalization stage <DefinitionTooltip label="Generalization stage definition">{stageTooltip} Global/user scopes have explicit cross-context exceptions; stage is not an authorization guarantee.</DefinitionTooltip></dt>
                <dd>{memory.generalization_stage == null || !stageNames[memory.generalization_stage] ? "Missing / unrecognized" : `${memory.generalization_stage} · ${stageNames[memory.generalization_stage]}`}<p className="neutral">{stageDefinitions[memory.generalization_stage]}</p></dd>
                <dt>Tags</dt>
                <dd>{memory.tags?.join(", ") || "None"}</dd>
              </dl>
              <dl className="key-values wide">
                <dt>Facets</dt><dd>{memory.facets?.length ? memory.facets.join(", ") : "None"}</dd>
                <dt>Outgoing relations</dt><dd>{data.outgoing?.length || 0}</dd>
                <dt>Incoming relations</dt><dd>{data.incoming?.length || 0}</dd>
              </dl>
            </details>
            {confirm && <DeleteConfirmation previewUrl={`/api/schemas/${memory.schema_id}/delete-preview`} onCancel={() => setConfirm(false)} onConfirm={remove} />}
          </>
        ) : <EmptyState title="Memory not found">This memory no longer exists or is not available in the current database.</EmptyState>
      }
    </Inspector>
  );
}

function RetrievalUsage({ row }: { row: any }) {
  if (!row.exposed_count) return <span className="neutral">-</span>;
  return <div className="retrieval-usage-stack">{(["memory", "procedure"] as const).map((kind) => {
    const total = Number(row[kind === "memory" ? "memory_count" : "procedure_count"] || 0);
    if (!total) return null;
    const usage = row.usage_by_kind?.[kind];
    const used = Number(usage?.used || 0);
    const pending = !usage?.reported;
    const label = kind === "memory" ? "Memories" : "Procedures";
    const description = pending
      ? `${total} retrieved ${kind === "memory" ? "memories" : "procedures"}; usage feedback pending`
      : `${used} of ${total} retrieved ${kind === "memory" ? "memories" : "procedures"} reported used. Grey means not reported used.`;
    return <span key={kind} className={`retrieval-usage-row retrieval-usage-${kind}`} title={description} aria-label={description}>
      <span className="retrieval-usage-label">{label}</span>
      <span className="retrieval-usage-track" aria-hidden="true"><span style={{ width: `${Math.min(100, used / total * 100)}%` }} /></span>
      <span className={`retrieval-usage-value${pending ? " neutral" : ""}`}>{pending ? `—/${total}` : `${used}/${total}`}</span>
    </span>;
  })}</div>;
}

export function RetrievalPage({ location }: PageProps) {
  const retrievalSignalColumns = [
    ["used", "Context used"],
    ["not_used", "Not used"],
    ["irrelevant", "Irrelevant"],
    ["stale", "Stale"],
    ["wrong", "Wrong"],
    ["helped", "Helpful"],
    ["no_effect", "No effect"],
    ["harmed", "Harmful"],
    ["unknown", "Unknown"],
  ] as const;
  const retrievalColumnHelp: Record<string, string> = {
    when: "When this retrieval was recorded.",
    task: "The task, goal, or query that prompted the retrieval.",
    type: "Activation retrieves context at the start of a task; Recall is an explicit later lookup.",
    scope: sharedColumnHelp.scope,
    retrieved: "Admitted items returned by this retrieval, split into memories and procedures.",
    memories_retrieved: "Number of admitted memory items returned.",
    procedures_retrieved: "Number of admitted procedure items returned.",
    used: "Reported used / retrieved, split into memories and procedures. Bars show the proportion reported used; grey includes items without usage feedback. A dash in the fraction means usage feedback is pending for that category.",
    not_used: "Count of returned procedures explicitly assessed as not used.",
    irrelevant: "Count of returned memories explicitly assessed as irrelevant.",
    stale: "Count of returned memories explicitly assessed as stale.",
    wrong: "Count of returned memories explicitly assessed as wrong.",
    helped: "Count of returned procedures reported to have helped.",
    no_effect: "Count of returned procedures reported to have had no effect.",
    harmed: "Count of returned procedures reported to have harmed the task.",
    unknown: "Count of retrievals without an explicit accepted assessment or effect.",
    effect: "Reported downstream effect of a retrieved procedure: Helpful, No effect, Harmful, or Unknown. Unknown means no impact assessment was reported; it is not a negative result.",
    feedback: "Whether accepted feedback completely covered the items returned by this retrieval.",
    session: "The session that recorded this retrieval, or Standalone when there was none.",
  };
  const ColumnHelp = ({ id, label }: { id: string; label: string }) => (
    <DefinitionTooltip label={`${label} definition`}>
      {retrievalColumnHelp[id]}
    </DefinitionTooltip>
  );
  const [visibleColumns] = useState<string[]>(["when", "task", "type", "used", "feedback"]);
  const visible = (id: string) => visibleColumns.includes(id);
  const detailId = location.path.startsWith("/retrieval/")
    ? decodeURIComponent(location.path.split("/")[2])
    : "";
  const scope = param(location, "scope");
  const type = param(location, "type");
  const feedback = param(location, "feedback");
  const noMatch = param(location, "no_match");
  const contains = param(location, "contains");
  const includeInternal = param(location, "include_internal", "true");
  const sort = param(location, "sort", "when");
  const dir = param(location, "dir", "desc") as "asc" | "desc";
  const page = pageNumber(location);
  const [search, setSearch] = useState("");
  const [debounced, setDebounced] = useState("");
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(search), 250);
    return () => clearTimeout(timer);
  }, [search]);
  const request = useApi<Json>(
    `/api/retrievals?scope=${encodeURIComponent(scope)}&type=${type}&feedback=${feedback}&no_match=${noMatch}&contains=${contains}&include_internal=${includeInternal}&sort=${sort}&dir=${dir}&page=${page}&per_page=50&q=${encodeURIComponent(debounced)}&from=${param(location, "from")}&to=${param(location, "to")}&hours=${param(location, "hours", "all")}`,
  );
  const rows = request.data?.retrievals || [];
  const summary = request.data?.summary || {};
  const pagination = request.data?.pagination || {
    page,
    per_page: 50,
    total: 0,
  };
  const changeSort = (column: string) =>
    updateParams("/retrieval", location, {
      sort: column,
      dir: sort === column && dir === "desc" ? "asc" : "desc",
      page: 1,
    });
  return (
    <div className="page">
      <PageHeader
        title="Retrieval"
        description="Context Slowave exposed for a task. Exposure is not proof it was used."
        updatedAt={request.updatedAt}
        refreshing={request.refreshing}
        onRefresh={request.reload}
      />
      <div className="filter-bar filter-bar-nowrap">
        <PeriodFilter location={location} path="/retrieval" />
        <label>
          Search
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search task/query…"
          />
        </label>
        <label>
          Type
          <select
            value={type}
            onChange={(e) =>
              updateParams("/retrieval", location, {
                type: e.target.value,
                page: 1,
              })
            }
          >
            <option value="">All types</option>
            <option value="context">Activation</option>
            <option value="recall">Recall</option>
          </select>
        </label>
        <label>
          Feedback
          <select
            value={feedback}
            onChange={(e) =>
              updateParams("/retrieval", location, {
                feedback: e.target.value,
                page: 1,
              })
            }
          >
            <option value="">Any feedback state</option>
            <option value="complete">Complete</option>
            <option value="incomplete">Incomplete / unknown</option>
          </select>
        </label>
        <ScopeSelect
          value={scope}
          onChange={(next) =>
            updateParams("/retrieval", location, { scope: next, page: 1 })
          }
        />
        <label>
          Contains
          <select
            value={contains}
            onChange={(e) =>
              updateParams("/retrieval", location, {
                contains: e.target.value,
                page: 1,
              })
            }
          >
            <option value="">Any item</option>
            <option value="memory">Memory</option>
            <option value="procedure">Procedure</option>
          </select>
        </label>
        <label>
          Observed since
          <input
            type="date"
            value={
              param(location, "from")
                ? new Date(Number(param(location, "from")) * 1000)
                    .toISOString()
                    .slice(0, 10)
                : ""
            }
            onChange={(event) =>
              updateParams("/retrieval", location, {
                from: event.target.value
                  ? Math.floor(
                      new Date(event.target.value).getTime() / 1000,
                    )
                  : undefined,
                page: 1,
              })
            }
          />
        </label>
      </div>
      {request.loading && !request.data ? (
        <MetricCardsSkeleton count={4} />
      ) : request.error && !request.data ? (
        <ErrorState
          title="Retrieval summary unavailable"
          error={request.error}
          retry={request.reload}
        />
      ) : request.data ? (
        <>
          <div className="metric-card-grid" aria-label="Retrieval summary">
            <MetricCard
              title="Retrievals"
              value={Number(summary.retrievals || 0).toLocaleString()}
              secondary={`${request.data.metrics?.task_starts || 0} Task starts · ${request.data.metrics?.additional_requests || 0} Additional requests`}
              tooltip="Distinct recorded requests; continuations and retries retain the request identity."
            />
            <RateMetricCard
              title="Retrieval match rate"
              numerator={
                Number(summary.retrievals || 0) - Number(summary.no_match || 0)
              }
              denominator={summary.retrievals}
              secondary={`${summary.no_match || 0} empty`}
              tooltip="Requests delivering supported context divided by recorded requests. Context returned does not establish relevance."
            />
            <RateMetricCard
              title="Retrieval usage rate"
              numerator={summary.demonstrated_value}
              denominator={
                Number(summary.retrievals || 0) - Number(summary.no_match || 0)
              }
              secondary="Among requests returning context"
              tooltip="Requests with actually delivered context reported used divided by requests returning supported context. Missing feedback remains in the denominator; use does not require request finalization."
            />
            <RateMetricCard
              title="Feedback coverage"
              numerator={summary.assessed}
              denominator={summary.exposed}
              secondary={`${summary.unknown || 0} unassessed occasions`}
              tooltip={feedbackTooltip}
            />
            <RateMetricCard
              title="Irrelevant context"
              numerator={request.data.metrics?.irrelevant}
              denominator={request.data.metrics?.assessed_memory_deliveries}
              secondary={`${request.data.metrics?.missing_memory_feedback || 0} missing memory feedback`}
              tooltip="Memory deliveries the assistant reported as not relevant to that task, among assessed memory deliveries. Missing feedback is shown separately; this does not establish independent retrieval accuracy."
            />
            {(["Activate", "Recall"] as const).map((kind) => {
              const size = request.data?.metrics?.context_size?.[kind];
              const medianTokens = estimateResponseTokens(size?.median_chars);
              const p95Tokens = formatTokenEstimate(
                estimateResponseTokens(size?.p95_chars),
              );
              return (
                <MetricCard
                  key={kind}
                  title={`${kind} median context size`}
                  value={formatTokenEstimate(medianTokens)}
                  tooltip={`Median approximate serialized response size for ${kind}, including returned context and metadata; this is not an average. A stored sample may reflect a continuation page rather than all pages combined. Estimated at one token per four response characters. ${size?.samples || 0} samples; p95 ${p95Tokens}.`}
                />
              );
            })}
          </div>
          <FocusedChart data={request.data.metrics?.chart} kind="memory" tabs />
        </>
      ) : (
        <EmptyState title="No retrieval summary available">
          Summary metrics will appear when the retrieval service returns a
          result.
        </EmptyState>
      )}
      <InlineError
        error={request.error}
        retained={Boolean(request.data)}
        retry={request.reload}
      />
      {request.loading && !request.data ? (
        <LoadingRows />
      ) : request.error && !request.data ? (
        <ErrorState title="Retrieval results unavailable" error={request.error} retry={request.reload} />
      ) : rows.length ? (
        <>
          <TableFrame label="Retrieval results">
            <table className="retrieval-table">
              <colgroup><col className="retrieval-col-when" /><col /><col className="retrieval-col-type" /><col className="retrieval-col-usage" /><col className="retrieval-col-feedback" /></colgroup>
              <thead>
                <tr>
                  {visible("when") && <th aria-sort={sort === "when" ? (dir === "asc" ? "ascending" : "descending") : "none"}>
                    <SortButton label="When" active={sort === "when"} direction={dir} onClick={() => changeSort("when")} />
                    <ColumnHelp id="when" label="When" />
                  </th>}
                  {visible("task") && <th aria-sort={sort === "task" ? (dir === "asc" ? "ascending" : "descending") : "none"}>
                    <SortButton label="Task / query" active={sort === "task"} direction={dir} onClick={() => changeSort("task")} />
                    <ColumnHelp id="task" label="Task / query" />
                  </th>}
                  {visible("type") && <th><SortButton label="Type" active={sort === "type"} direction={dir} onClick={() => changeSort("type")} /><ColumnHelp id="type" label="Type" /></th>}
                  {visible("scope") && <th aria-sort={sort === "scope" ? (dir === "asc" ? "ascending" : "descending") : "none"}>
                    <SortButton label="Scope" active={sort === "scope"} direction={dir} onClick={() => changeSort("scope")} />
                    <ColumnHelp id="scope" label="Scope" />
                  </th>}
                  {visible("retrieved") && <th aria-sort={sort === "exposed" ? (dir === "asc" ? "ascending" : "descending") : "none"}>
                    <SortButton label="Retrieved" active={sort === "exposed"} direction={dir} onClick={() => changeSort("exposed")} /><ColumnHelp id="retrieved" label="Retrieved" />
                  </th>}
                  {visible("memories_retrieved") && <th className="numeric"><SortButton label="Memories retrieved" active={sort === "memories_retrieved"} direction={dir} onClick={() => changeSort("memories_retrieved")} /><ColumnHelp id="memories_retrieved" label="Memories retrieved" /></th>}
                  {visible("procedures_retrieved") && <th className="numeric"><SortButton label="Procedures retrieved" active={sort === "procedures_retrieved"} direction={dir} onClick={() => changeSort("procedures_retrieved")} /><ColumnHelp id="procedures_retrieved" label="Procedures retrieved" /></th>}
                  {retrievalSignalColumns.filter(([key]) => visible(key)).map(([key, label]) => (
                    <th className="numeric" key={key} aria-sort={sort === key ? (dir === "asc" ? "ascending" : "descending") : "none"}>
                      <SortButton label={label} active={sort === key} direction={dir} onClick={() => changeSort(key)} />
                      <ColumnHelp id={key} label={label} />
                      {key === "used" && <small className="retrieval-usage-caption">used / retrieved</small>}
                    </th>
                  ))}
                  {visible("feedback") && <th aria-sort={sort === "feedback" ? (dir === "asc" ? "ascending" : "descending") : "none"}>
                    <SortButton label="Feedback" active={sort === "feedback"} direction={dir} onClick={() => changeSort("feedback")} />
                    <ColumnHelp id="feedback" label="Feedback" />
                  </th>}
                  {visible("session") && <th><SortButton label="Session ID" active={sort === "activity"} direction={dir} onClick={() => changeSort("activity")} /><ColumnHelp id="session" label="Session ID" /></th>}
                </tr>
              </thead>
              <tbody>
                {rows.map((row: any) => {
                  const href = `/retrieval/${encodeURIComponent(row.context_id)}${location.search.size ? `?${location.search}` : ""}`;
                  return (
                    <tr
                      tabIndex={0}
                      key={row.context_id}
                      onClick={() => openDetail(href, location)}
                      onKeyDown={(e) =>
                        rowKeys(e, () => openDetail(href, location))
                      }
                    >
                      {visible("when") && <td data-label="When" title={formatDate(row.created_at)}>
                        {relativeDate(row.created_at)}
                      </td>}
                      {visible("task") && <td className="primary-cell retrieval-task-cell">
                        <ClampedText text={row.task_preview} />
                        <small className="retrieval-task-scope" title={row.scope_id || undefined}>{row.scope_id || "No scope"}</small>
                      </td>}
                      {visible("type") && <td data-label="Type"><StatusBadge value={row.retrieval_type === "context" ? "Activation" : "Recall"} /></td>}
                      {visible("scope") && <td className="scope-text" title={row.scope_id || undefined}>
                        {row.scope_id ? truncate(row.scope_id, 30) : "No scope"}
                      </td>}
                      {visible("retrieved") && <td className="badge-stack">{row.exposed_count ? <><StatusBadge value="memory" count={row.memory_count ?? 0} /><StatusBadge value="procedure" count={row.procedure_count ?? 0} /></> : <StatusBadge value="none" />}</td>}
                      {visible("memories_retrieved") && <td className="numeric">{row.memory_count ?? 0}</td>}
                      {visible("procedures_retrieved") && <td className="numeric">{row.procedure_count ?? 0}</td>}
                      {retrievalSignalColumns.filter(([key]) => visible(key)).map(([key]) => (
                        <td data-label={key === "used" ? "Context used · used / retrieved" : key} className={key === "used" ? "retrieval-usage-cell" : "numeric"} key={key}>
                          {key === "used" ? <RetrievalUsage row={row} /> : Number(row.signal_counts?.[key] || 0).toLocaleString()}
                        </td>
                      ))}
                      {visible("feedback") && <td data-label="Feedback"><StatusBadge value={row.feedback_state} /></td>}
                      {visible("session") && <td>{row.session_id ? truncate(row.session_id, 18) : "Standalone"}</td>}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </TableFrame>
          <Pagination
            page={pagination.page}
            perPage={pagination.per_page}
            total={pagination.total}
            onPage={(next) =>
              updateParams("/retrieval", location, { page: next })
            }
          />
        </>
      ) : (
        <EmptyState
          title={
            search ||
            type ||
            feedback ||
            noMatch ||
            contains ||
            scope ||
            param(location, "from")
              ? "No retrievals match these filters"
              : "No retrievals yet"
          }
        >
          {search ||
          type ||
          feedback ||
          noMatch ||
          contains ||
          scope ||
          param(location, "from")
            ? "Clear or broaden the filters to inspect other recorded exposures."
            : "Slowave records a retrieval when an agent activates or explicitly recalls memory. Empty retrievals are valid recorded rows, not errors."}
        </EmptyState>
      )}
      {detailId && (
        <RetrievalDetail
          id={detailId}
          onClose={() => closeDetail("/retrieval", location)}
        />
      )}
    </div>
  );
}

function RetrievalDetail({ id, onClose }: { id: string; onClose: () => void }) {
  const request = useApi<Json>(`/api/retrievals/${encodeURIComponent(id)}`);
  const data = request.data;
  const retrieval = data?.retrieval;
  const grouped = useMemo(
    () =>
      (data?.items || []).reduce((acc: Json, item: any) => {
        (acc[item.pathway_group] ||= []).push(item);
        return acc;
      }, {}),
    [data],
  );
  return (
    <Inspector
      title="Retrieval"
      id={id}
      state={
        retrieval && (
          <StatusBadge
            value={
              data.feedback?.some(
                (item: any) =>
                  item.status === "accepted" && item.coverage === "complete",
              )
                ? "complete"
                : "unknown"
            }
          />
        )
      }
      onClose={onClose}
    >
      <InlineError
        error={request.error}
        retained={Boolean(data)}
        retry={request.reload}
      />
      {request.loading && !data ? (
        <LoadingRows />
      ) : request.error && !data ? (
        <ErrorState title="Retrieval details unavailable" error={request.error} retry={request.reload} />
      ) : retrieval ? (
          <>
            <h2 className="detail-title">
              {truncate(
                retrieval.query || retrieval.goal || "Context exposure",
                220,
              )}
            </h2>
            <div className="private-notice">
              <Icon name="info" />
              <span>
                Local private content. Additional stored context is collapsed by
                default.
              </span>
            </div>
            <dl className="key-values">
              <dt>Source</dt>
              <dd>
                {retrieval.retrieval_type === "context"
                  ? "Activation"
                  : "Recall"}
              </dd>
              <dt>Observed</dt>
              <dd>{formatDate(retrieval.created_at)}</dd>
              <dt>Scope</dt>
              <dd>{retrieval.scope_id || "No scope"}</dd>
              <dt>Response size</dt>
              <dd>
                {retrieval.response_chars
                  ? `${retrieval.response_chars} characters`
                  : "Not recorded"}
              </dd>
              {retrieval.retrieval_type === "context" && (
                <>
                  <dt>
                    Memory page limit{" "}
                    <DefinitionTooltip label="Memory page limit definition">
                      Maximum number of memories requested for each activation page. The
                      actual returned count can be lower when fewer memories qualify or
                      the response payload limit is reached.
                    </DefinitionTooltip>
                  </dt>
                  <dd>
                    {retrieval.requested_page_size == null
                      ? "Not recorded"
                      : `${retrieval.requested_page_size} memories`}
                  </dd>
                </>
              )}
              <dt>Activity</dt>
              <dd>
                {retrieval.session_id ? (
                  <Link
                    to={`/activity/${encodeURIComponent(retrieval.session_id)}`}
                  >
                    {retrieval.session_id}
                  </Link>
                ) : (
                  "Standalone retrieval"
                )}
              </dd>
              <dt>
                Task outcome{" "}
                <DefinitionTooltip label="Task outcome definition">
                  The outcome the agent reported for the session this retrieval
                  belonged to. It is not necessarily caused by Slowave's
                  context.
                </DefinitionTooltip>
              </dt>
              <dd>
                {data.session?.outcome ? (
                  <StatusBadge value={data.session.outcome} />
                ) : (
                  "Not recorded"
                )}
              </dd>
            </dl>
            <Section title="Context used">
              <p className="neutral">Reported used / retrieved. Grey includes items not reported used; a dash means feedback is pending.</p>
              <div className="retrieval-detail-usage"><RetrievalUsage row={retrieval} /></div>
            </Section>
            <Section title="Exposed context">
              {Object.entries(grouped).length ? (
                Object.entries(grouped).map(([group, items]: [string, any]) => (
                  <div className="exposure-group" key={group}>
                    <h3>{group}</h3>
                    {items.map((item: any) => (
                      <div className="exposure-row" key={item.memory_id}>
                        <div>
                          <strong
                            className="schema-preview"
                            title={item.content_text || item.memory_id}
                          >
                            {truncate(item.content_text || item.memory_id, 220)}
                          </strong>
                          <span>
                            {item.memory_type} ·{" "}
                            {item.status || "State not recorded"}
                          </span>
                          <small>{item.pathway_explanation}</small>
                          <span className="assessment-line">
                            Assessment:{" "}
                            <StatusBadge value={item.assessment || "unknown"} />
                            {item.effect && (
                              <>
                                {" "}
                                Effect: <StatusBadge value={item.effect} />
                              </>
                            )}
                          </span>
                        </div>
                        <div>
                          <StatusBadge value={item.status} />
                          {item.memory_type === "procedure" ||
                          item.memory_type === "procedural_memory" ? (
                            <Link
                              to={`/procedures/${encodeURIComponent(item.memory_id)}`}
                            >
                              Open
                            </Link>
                          ) : (
                            <Link
                              to={`/memory/${encodeURIComponent(item.memory_id)}`}
                            >
                              Open
                            </Link>
                          )}
                        </div>
                        <details>
                          <summary>Stored pathway metadata</summary>
                          <code>{item.pathway}</code>
                          {item.reason && <p>{item.reason}</p>}
                        </details>
                      </div>
                    ))}
                  </div>
                ))
              ) : (
                <EmptyState title="Empty">
                  This retrieval returned no admitted memories or procedures.
                </EmptyState>
              )}
            </Section>
            <Section title="Observed feedback">
              {data.feedback?.length ? (
                <div className="detail-list">
                  {data.feedback.map((item: any) => (
                    <div key={item.event_id}>
                      <div>
                        <strong>
                          {item.target_kind} · {item.target_id}
                        </strong>
                        <StatusBadge value={item.status} />
                      </div>
                      <p>
                        {[item.assessment, item.effect]
                          .filter(Boolean)
                          .join(" · ")
                          .replaceAll("_", " ") || "Coverage observation only"}
                      </p>
                      <small>
                        {formatDate(item.created_at)} · {item.coverage} coverage
                      </small>
                      {item.reason && <p>{item.reason}</p>}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="neutral">
                  Unknown / not assessed. Missing feedback is not negative
                  evidence.
                </p>
              )}
            </Section>
            <details className="advanced" open>
              <summary>Advanced diagnostics</summary>
              <p className="private-notice">
                Stored context below may contain local/private task information.
              </p>
              <dl className="key-values wide">
                <dt>Situation</dt>
                <dd>
                  {retrieval.situation &&
                  Object.keys(retrieval.situation).length
                    ? JSON.stringify(retrieval.situation)
                    : "Not recorded"}
                </dd>
                <dt>Requirements</dt>
                <dd>
                  {Array.isArray(retrieval.requirements) &&
                  retrieval.requirements.length
                    ? retrieval.requirements.join(", ")
                    : "None"}
                </dd>
                <dt>Topics</dt>
                <dd>
                  {Array.isArray(retrieval.topics) && retrieval.topics.length
                    ? retrieval.topics.join(", ")
                    : "None"}
                </dd>
                <dt>Entities</dt>
                <dd>
                  {Array.isArray(retrieval.entities) && retrieval.entities.length
                    ? retrieval.entities.join(", ")
                    : "None"}
                </dd>
              </dl>
            </details>
          </>
        ) : <EmptyState title="Retrieval not found">This retrieval no longer exists or is not available in the current database.</EmptyState>
      }
    </Inspector>
  );
}

export function ProceduresPage({ location }: PageProps) {
  const detailId = location.path.startsWith("/procedures/")
    ? decodeURIComponent(location.path.split("/")[2])
    : "";
  const scope = param(location, "scope");
  const outcome = param(location, "outcome");
  const verification = param(location, "verification");
  const retrieved = param(location, "retrieved");
  const sort = param(location, "sort", "recent");
  const dir = param(location, "dir", "desc") as "asc" | "desc";
  const page = pageNumber(location);
  const [visibleColumns] = useState<string[]>(["procedure", "scope", "outcome", "verification", "created", "retrieved", "used", "effect", "last_used"]);
  const visible = (id: string) => visibleColumns.includes(id);
  const request = useApi<Json>(
    `/api/procedural-memory?cohort=all&scope=${encodeURIComponent(scope)}&outcome=${outcome}&verification=${verification}&retrieved=${retrieved}&sort=${sort}&dir=${dir}&page=${page}&per_page=50&from=${param(location, "from")}&to=${param(location, "to")}&hours=${param(location, "hours", "all")}&saved_from=${param(location, "saved_from")}`,
  );
  const rows = request.data?.procedures || [];
  const pagination = request.data?.pagination || {
    page,
    per_page: 50,
    total: request.data?.structured_attempts || 0,
  };
  const changeSort = (column: string) =>
    updateParams("/procedures", location, {
      sort: column,
      dir: sort === column && dir === "desc" ? "asc" : "desc",
      page: 1,
    });
  return (
    <div className="page">
      <PageHeader
        title="Procedures"
        description="Structured, execution-backed procedure records created when a session closes; a procedure created here is not necessarily a proven general playbook."
        updatedAt={request.updatedAt}
        refreshing={request.refreshing}
        onRefresh={request.reload}
      />
      <div className="filter-bar">
        <PeriodFilter location={location} path="/procedures" />
        <label>View
          <select value={sort} onChange={(e) => updateParams("/procedures", location, { sort: e.target.value, dir: "desc", page: 1 })}>
            <option value="recent">Recently created</option><option value="retrieved">Most retrieved</option><option value="used">Most used</option><option value="helped">Most helpful</option><option value="harmed">Most harmful</option>
          </select>
        </label>
        <label>
          Outcome
          <select
            value={outcome}
            onChange={(e) =>
              updateParams("/procedures", location, {
                outcome: e.target.value,
                page: 1,
              })
            }
          >
            <option value="">Any outcome</option>
            <option value="success">Success</option>
            <option value="partial">Partial</option>
            <option value="failure">Failure</option>
          </select>
        </label>
        <label>
          Verification
          <select
            value={verification}
            onChange={(e) =>
              updateParams("/procedures", location, {
                verification: e.target.value,
                page: 1,
              })
            }
          >
            <option value="">Any verification</option>
            <option value="verified">Verified</option>
            <option value="partially_verified">Partially verified</option>
            <option value="unverified">Unverified</option>
          </select>
        </label>
        <label>
          Exposure
          <select
            value={retrieved}
            onChange={(e) =>
              updateParams("/procedures", location, {
                retrieved: e.target.value,
                page: 1,
              })
            }
          >
            <option value="">Any exposure</option>
            <option value="yes">Retrieved</option>
            <option value="never">Never retrieved</option>
          </select>
        </label>
        <label>
          Created since
          <input
            type="date"
            onChange={(e) =>
              updateParams("/procedures", location, {
                from: e.target.value
                  ? Math.floor(new Date(e.target.value).getTime() / 1000)
                  : undefined,
                page: 1,
              })
            }
          />
        </label>
      </div>
      {request.loading && !request.data ? (
        <MetricCardsSkeleton count={4} />
      ) : request.error && !request.data ? (
        <ErrorState
          title="Procedure summary unavailable"
          error={request.error}
          retry={request.reload}
        />
      ) : request.data ? (
        <div>
          <div
            className="metric-card-grid knowledge-summary-grid"
            aria-label="Procedure summary"
          >
            <LibraryCards data={request.data.metrics || {}} kind="procedure" />
          </div>
          <div
            className="metric-card-grid"
            aria-label="Procedure usage outcomes"
          >
            <RateMetricCard
              title="Helpful rate · uses"
              numerator={request.data.metrics?.procedure_helped}
              denominator={request.data.metrics?.effect_known}
              secondary={`Among uses with reported effects · ${request.data.metrics?.effect_unknown || 0} unknown effects`}
              tooltip="Used procedure delivery occasions reported helped divided by used delivery occasions with known effects. A procedure delivered in several requests contributes several occasions; these counts can exceed the distinct-procedure usage rate."
            />
            <RateMetricCard
              title="Harmful rate · uses"
              numerator={request.data.metrics?.procedure_harmed}
              denominator={request.data.metrics?.effect_known}
              secondary={`Among uses with reported effects · ${request.data.metrics?.effect_unknown || 0} unknown effects`}
              tooltip="Used procedure delivery occasions reported harmed divided by the same known-effect use occasions. A procedure delivered in several requests contributes several occasions. No harm reports is not proof of safety."
            />
          </div>
          <FocusedChart data={request.data.metrics?.chart} kind="procedure" />
        </div>
      ) : (
        <EmptyState title="No procedure summary available">
          Summary metrics will appear when the procedure service returns a
          result.
        </EmptyState>
      )}
      <InlineError
        error={request.error}
        retained={Boolean(request.data)}
        retry={request.reload}
      />
      {request.loading && !request.data ? (
        <LoadingRows />
      ) : request.error && !request.data ? (
        <ErrorState
          title="Procedure results unavailable"
          error={request.error}
          retry={request.reload}
        />
      ) : rows.length ? (
        <>
          <TableFrame label="Procedure results">
            <table>
              <thead>
                <tr>
                  {visible("procedure") && <th aria-sort={sort === "summary" ? (dir === "asc" ? "ascending" : "descending") : "none"}>
                    <SortButton label="Procedure" active={sort === "summary"} direction={dir} onClick={() => changeSort("summary")} />
                  </th>}
                  {visible("scope") && <th><SortButton label="Scope" active={sort === "scope"} direction={dir} onClick={() => changeSort("scope")} /><DefinitionTooltip label="Scope definition">{sharedColumnHelp.scope}</DefinitionTooltip></th>}
                  {visible("outcome") && <th><SortButton label="Outcome" active={sort === "outcome"} direction={dir} onClick={() => changeSort("outcome")} /><DefinitionTooltip label="Outcome definition">{sharedColumnHelp.outcome}</DefinitionTooltip></th>}
                  {visible("verification") && <th><SortButton label="Verification" active={sort === "verification"} direction={dir} onClick={() => changeSort("verification")} /><DefinitionTooltip label="Verification definition">{sharedColumnHelp.verification}</DefinitionTooltip></th>}
                  {visible("created") && <th aria-sort={sort === "recent" ? (dir === "asc" ? "ascending" : "descending") : "none"}>
                    <SortButton label="Created" active={sort === "recent"} direction={dir} onClick={() => changeSort("recent")} />
                  </th>}
                  {visible("retrieved") && <th className="numeric"><SortButton label="Retrieved" active={sort === "retrieved"} direction={dir} onClick={() => changeSort("retrieved")} /><DefinitionTooltip label="Retrieved definition">{sharedColumnHelp.retrieved}</DefinitionTooltip></th>}
                  {visible("used") && <th className="numeric"><SortButton label="Used" active={sort === "used"} direction={dir} onClick={() => changeSort("used")} /><DefinitionTooltip label="Used definition">{sharedColumnHelp.used}</DefinitionTooltip></th>}
                  {visible("effect") && (
                    <th>
                      Effect{" "}
                      <DefinitionTooltip label="Effect definition">
                        {sharedColumnHelp.effect}
                      </DefinitionTooltip>
                    </th>
                  )}
                  {visible("last_used") && <th>
                    <SortButton
                      label="Last used"
                      active={sort === "last_used"}
                      direction={dir}
                      onClick={() => changeSort("last_used")}
                    />
                  </th>}
                  {visible("use_rate") && (
                    <th className="numeric">
                      <SortButton
                        label="Use rate"
                        active={sort === "use_rate"}
                        direction={dir}
                        onClick={() => changeSort("use_rate")}
                      />
                    </th>
                  )}
                  {visible("helped") && (
                    <th className="numeric">
                      <SortButton
                        label="Helpful count"
                        active={sort === "helped"}
                        direction={dir}
                        onClick={() => changeSort("helped")}
                      />
                    </th>
                  )}
                  {visible("no_effect") && (
                    <th className="numeric">
                      <SortButton
                        label="No-effect count"
                        active={sort === "no_effect"}
                        direction={dir}
                        onClick={() => changeSort("no_effect")}
                      />
                    </th>
                  )}
                  {visible("harmed") && (
                    <th className="numeric">
                      <SortButton
                        label="Harmful count"
                        active={sort === "harmed"}
                        direction={dir}
                        onClick={() => changeSort("harmed")}
                      />
                    </th>
                  )}
                  {visible("unknown") && (
                    <th className="numeric">
                      <SortButton
                        label="Unknown count"
                        active={sort === "unknown"}
                        direction={dir}
                        onClick={() => changeSort("unknown")}
                      />
                    </th>
                  )}
                  {visible("feedback_coverage") && <th className="numeric"><SortButton label="Feedback coverage" active={sort === "feedback_coverage"} direction={dir} onClick={() => changeSort("feedback_coverage")} /></th>}
                  {visible("last_retrieved") && (
                    <th>
                      <SortButton
                        label="Last retrieved"
                        active={sort === "last_retrieved"}
                        direction={dir}
                        onClick={() => changeSort("last_retrieved")}
                      />
                    </th>
                  )}
                  {visible("source_activity") && (
                    <th>
                      <SortButton
                        label="Source activity / session"
                        active={sort === "source_activity"}
                        direction={dir}
                        onClick={() => changeSort("source_activity")}
                      />
                    </th>
                  )}
                </tr>
              </thead>
              <tbody>
                {rows.map((row: any) => {
                  const href = `/procedures/${encodeURIComponent(row.id)}${location.search.size ? `?${location.search}` : ""}`;
                  const evidence = row.evidence || {};
                  const assessed =
                    Number(evidence.used || 0) + Number(evidence.not_used || 0);
                  return (
                    <tr
                      key={row.id}
                      tabIndex={0}
                      onClick={() => openDetail(href, location)}
                      onKeyDown={(e) =>
                        rowKeys(e, () => openDetail(href, location))
                      }
                    >
                      {visible("procedure") && <td className="primary-cell">
                        <ClampedText text={row.summary || row.goal} />
                      </td>}
                      {visible("scope") && <td className="scope-text" title={row.scope_id || undefined}>
                        {row.scope_id ? truncate(row.scope_id, 30) : "No scope"}
                      </td>}
                      {visible("outcome") && <td>
                        <StatusBadge value={row.outcome} />
                      </td>}
                      {visible("verification") && <td>
                        <StatusBadge value={row.verification?.status || "unknown"} />
                      </td>}
                      {visible("created") && <td title={formatDate(row.created_at)}>
                        {relativeDate(row.created_at)}
                      </td>}
                      {visible("retrieved") && <td className="numeric">{evidence.retrieved || 0}</td>}
                      {visible("used") && <td className="numeric">{evidence.used || 0}</td>}
                      {visible("effect") && <td className="badge-stack">{assessed ? <StatusBadge value={Number(evidence.harmed || 0) ? "harmed" : Number(evidence.no_effect || 0) ? "no effect" : Number(evidence.helped || 0) ? "helped" : "unknown"} /> : <StatusBadge value="unknown" />}</td>}
                      {visible("last_used") && <td title={formatDate(row.last_used_ts)}>{row.last_used_ts ? relativeDate(row.last_used_ts) : "—"}</td>}
                      {visible("use_rate") && <td className="numeric">{formatRate(evidence.used, evidence.retrieved)}</td>}
                      {visible("helped") && <td className="numeric">{evidence.helped || 0}</td>}
                      {visible("no_effect") && <td className="numeric">{evidence.no_effect || 0}</td>}
                      {visible("harmed") && <td className="numeric">{evidence.harmed || 0}</td>}
                      {visible("unknown") && <td className="numeric">{evidence.unknown || 0}</td>}
                      {visible("feedback_coverage") && <td className="numeric">{row.feedback_complete || 0}</td>}
                      {visible("last_retrieved") && <td title={formatDate(row.last_retrieved_ts)}>{row.last_retrieved_ts ? relativeDate(row.last_retrieved_ts) : "—"}</td>}
                      {visible("source_activity") && <td>{row.source_activity_id ? truncate(row.source_activity_id, 18) : "—"}</td>}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </TableFrame>
          <Pagination
            page={pagination.page}
            perPage={pagination.per_page}
            total={pagination.total}
            onPage={(next) =>
              updateParams("/procedures", location, { page: next })
            }
          />
        </>
      ) : (
        <EmptyState
          title={
            outcome ||
            verification ||
            retrieved ||
            scope ||
            param(location, "from")
              ? "No procedures match these filters"
              : "No procedures created yet"
          }
        >
          {outcome ||
          verification ||
          retrieved ||
          scope ||
          param(location, "from")
            ? "Clear or broaden the structural filters to inspect other procedure records."
            : "Normal sessions may close without producing a reusable procedure."}
        </EmptyState>
      )}
      {detailId && (
        <ProcedureDetail
          id={detailId}
          onClose={() => closeDetail("/procedures", location)}
          onDeleted={request.reload}
        />
      )}
    </div>
  );
}

function ProcedureDetail({ id, onClose, onDeleted }: {
  id: string;
  onClose: () => void;
  onDeleted?: () => Promise<void>;
}) {
  const request = useApi<Json>(`/api/procedures/${encodeURIComponent(id)}`);
  const [confirm, setConfirm] = useState(false);
  const [actionError, setActionError] = useState("");
  const p = request.data?.procedure;
  const source = p?.source_session;
  const remove = async () => {
    if (!p) return;
    setActionError("");
    try {
      await api(`/api/procedures/${encodeURIComponent(p.id)}/delete`, { method: "POST" });
      await onDeleted?.();
      onClose();
    } catch (error) {
      setActionError((error as Error).message);
      throw error;
    }
  };
  return (
    <Inspector
      title="Procedure"
      id={id}
      state={p && <StatusBadge value={p.outcome} />}
      onClose={onClose}
    >
      <InlineError
        error={request.error || actionError}
        retained={Boolean(p)}
        retry={request.reload}
      />
      {request.loading && !p ? (
        <LoadingRows />
      ) : request.error && !p ? (
        <ErrorState title="Procedure details unavailable" error={request.error} retry={request.reload} />
      ) : p ? (
          <>
            <h2 className="detail-title">{p.summary}</h2>
            <p className="neutral">
              Created from execution evidence; not an automatically validated
              general playbook.
            </p>
            {allowActions ? (
              <div className="detail-actions"><button className="button danger-secondary" onClick={() => setConfirm(true)}>Delete</button></div>
            ) : <p className="notice compact">This dashboard was launched read-only. Delete is unavailable.</p>}
            <dl className="key-values">
              <dt>Created</dt>
              <dd>{formatDate(p.created_at)}</dd>
              <dt>Verification</dt>
              <dd>
                <StatusBadge
                  value={source?.verification?.status || "unknown"}
                />{" "}
                {source?.verification?.summary ||
                  "No verification summary recorded"}
              </dd>
              <dt>Outcome</dt>
              <dd>
                <StatusBadge value={p.outcome} />{" "}
                {p.outcome_summary || "No outcome summary recorded"}
              </dd>
              <dt>Scope</dt>
              <dd>{p.scope_id || "No scope"}</dd>
              <dt>Source activity</dt>
              <dd>
                {source?.id ? (
                  <Link to={`/activity/${encodeURIComponent(source.id)}`}>
                    {source.id}
                  </Link>
                ) : (
                  "Unavailable"
                )}
              </dd>
            </dl>
            <Section title="Value signals">
              {(() => {
                const evidence = p.evidence || {};
                const retrieved = Number(p.use_history_totals?.retrieved || 0);
                const used = Number(p.use_history_totals?.used || 0);
                return (
                  <dl className="key-values wide">
                    <dt>Retrieved</dt><dd>{retrieved}</dd>
                    <dt>Used</dt><dd>{used}</dd>
                    <dt>Use rate among assessed</dt><dd>{formatRate(used, p.use_history_totals?.assessed)}</dd>
                    <dt>Effect</dt>
                    <dd className="badge-stack">
                      <StatusBadge value="helped" count={Number(evidence.helped || 0)} />
                      <StatusBadge value="no effect" count={Number(evidence.no_effect || 0)} />
                      <StatusBadge value="harmed" count={Number(evidence.harmed || 0)} />
                      <StatusBadge value="unknown" count={Number(evidence.unknown_effect || 0)} />
                    </dd>
                    <dt>Last retrieved</dt><dd>{formatDate(p.use_history_totals?.last_retrieved)}</dd>
                    <dt>Last used</dt><dd>{formatDate(p.use_history_totals?.last_used)}</dd>
                  </dl>
                );
              })()}
            </Section>
            <Section title="Applicability and context">
              {Object.keys(p.context || {}).length ? (
                <dl className="key-values">
                  {Object.entries(p.context).map(([key, value]) => (
                    <Fragment key={key}>
                      <dt>{key.replaceAll("_", " ")}</dt>
                      <dd>{String(value)}</dd>
                    </Fragment>
                  ))}
                </dl>
              ) : (
                <p className="neutral">No applicability context recorded.</p>
              )}
            </Section>
            <Section title="Goal">
              <p>{p.goal || "No goal recorded"}</p>
            </Section>
            <Section title="Steps">
              {p.steps?.length ? (
                <ol className="procedure-steps">
                  {p.steps.map((step: any, index: number) => (
                    <li key={index}>{step.summary || step}</li>
                  ))}
                </ol>
              ) : (
                <p className="neutral">No steps recorded.</p>
              )}
            </Section>
            <Section title="Caveats">
              {p.caveats?.length ? (
                <ul>
                  {p.caveats.map((item: string) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              ) : (
                <p className="neutral">No caveats recorded.</p>
              )}
            </Section>
            <Section title="Retrieval exposures">
              {p.retrievals?.length ? (
                <div className="detail-list compact">
                  {p.retrievals.map((item: any) => (
                    <Link
                      key={item.retrieval_id}
                      to={`/retrieval/${encodeURIComponent(item.retrieval_id)}`}
                    >
                      <span>
                        {formatDate(item.created_at)} · {item.retrieval_type}
                      </span>
                      <Icon name="chevron" size={14} />
                    </Link>
                  ))}
                </div>
              ) : (
                <p className="neutral">
                  Not retrieved yet. This is neutral; it is not evidence against
                  the procedure.
                </p>
              )}
            </Section>
            <Section title="Explicit use and effect feedback">
              {p.feedback?.length ? (
                <div className="detail-list">
                  {p.feedback.map((item: any) => (
                    <div key={item.event_id}>
                      <div>
                        <strong>
                          {item.assessment || "Assessment unavailable"} ·{" "}
                          {item.effect || "unknown effect"}
                        </strong>
                        <StatusBadge value={item.status} />
                      </div>
                      <p>
                        {item.contribution ||
                          item.reason ||
                          "No contribution note recorded"}
                      </p>
                      <Link
                        to={`/retrieval/${encodeURIComponent(item.retrieval_id)}`}
                      >
                        Open retrieval
                      </Link>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="neutral">
                  No explicit transfer feedback recorded.
                </p>
              )}
            </Section>
            <details className="advanced" open>
              <summary>Advanced</summary>
              <dl className="key-values wide">{Object.entries(p.evidence || {}).map(([key, value]) => <Fragment key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{String(value)}</dd></Fragment>)}</dl>
            </details>
            {confirm && <DeleteConfirmation previewUrl={`/api/procedures/${encodeURIComponent(p.id)}/delete-preview`} onCancel={() => setConfirm(false)} onConfirm={remove} />}
          </>
        ) : <EmptyState title="Procedure not found">This procedure no longer exists or is not available in the current database.</EmptyState>
      }
    </Inspector>
  );
}

export function ActivityPage({ location }: PageProps) {
  const detailId = location.path.startsWith("/activity/")
    ? decodeURIComponent(location.path.split("/")[2])
    : "";
  const page = pageNumber(location);
  const sort = param(location, "sort", "started");
  const dir = param(location, "dir", "desc") as "asc" | "desc";
  const [search, setSearch] = useState("");
  const [debounced, setDebounced] = useState("");
  const activityColumns = [
    { id: "started", label: "Started" }, { id: "task", label: "Task / goal" }, { id: "scope", label: "Scope" },
    { id: "outcome", label: "Outcome" }, { id: "closure", label: "Closure", description: "Whether the session's feedback loop closed (complete, incomplete, or pending)." },
    { id: "duration", label: "Duration", description: "Elapsed time between the session start and close." },
    { id: "retrievals", label: "Retrievals", description: "Distinct retrieval events recorded in this session." },
    { id: "memories_touched", label: "Memories touched", description: "Distinct memory records with evidence linked to this session." },
    { id: "procedure", label: "Procedure record", description: "Whether closing this session created a reusable procedure." },
    { id: "retrieved_memories", label: "Retrieved memories" }, { id: "retrieved_procedures", label: "Retrieved procedures" },
    { id: "feedback_coverage", label: "Feedback coverage" }, { id: "episodes", label: "Episodes formed" },
    { id: "timeline_events", label: "Timeline event count" }, { id: "last_event", label: "Last event" },
    { id: "verification", label: "Verification" },
  ];
  const [visibleColumns, setVisibleColumns] = useState<string[]>([
    "started", "task", "scope", "outcome", "closure", "duration", "retrievals", "memories_touched", "procedure",
  ]);
  const visible = (id: string) => visibleColumns.includes(id);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(search), 250);
    return () => clearTimeout(timer);
  }, [search]);
  const activityFilters = `scope=${encodeURIComponent(param(location, "scope"))}&outcome=${param(location, "outcome")}&feedback=${param(location, "feedback")}&agent=${encodeURIComponent(param(location, "agent"))}&continuity=${encodeURIComponent(param(location, "continuity"))}&lane=${param(location, "lane")}&from=${param(location, "from")}&to=${param(location, "to")}&q=${encodeURIComponent(debounced)}`;
  const rowsRequest = useApi<Json>(`/api/activity?${activityFilters}&sort=${sort}&dir=${dir}&page=${page}&per_page=50&include_summary=false`);
  const summaryRequest = useApi<Json>(`/api/activity?${activityFilters}&summary_only=true`);
  const reloadActivity = () => {
    void rowsRequest.reload();
    void summaryRequest.reload();
  };
  const rows = rowsRequest.data?.activities || [];
  const pagination = rowsRequest.data?.pagination || {
    page,
    per_page: 50,
    total: 0,
  };
  const changeSort = (column: string) =>
    updateParams("/activity", location, {
      sort: column,
      dir: sort === column && dir === "desc" ? "asc" : "desc",
      page: 1,
    });
  return (
    <div className="page">
      <PageHeader
        title="Activity"
        description="Recorded task segments and the consequential links Slowave observed."
        updatedAt={rowsRequest.updatedAt || summaryRequest.updatedAt}
        refreshing={rowsRequest.refreshing || summaryRequest.refreshing}
        onRefresh={reloadActivity}
      />
      <div className="filter-bar">
        <label>
          Search
          <input
            placeholder="Search task or goal…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </label>
        <label>
          Outcome
          <select
            value={param(location, "outcome")}
            onChange={(e) =>
              updateParams("/activity", location, {
                outcome: e.target.value,
                page: 1,
              })
            }
          >
            <option value="">Any outcome</option>
            <option value="success">Success</option>
            <option value="partial">Partial</option>
            <option value="failure">Failure</option>
          </select>
        </label>
        <label>
          Closure
          <select
            value={param(location, "feedback")}
            onChange={(e) =>
              updateParams("/activity", location, {
                feedback: e.target.value,
                page: 1,
              })
            }
          >
            <option value="">Any closure</option>
            <option value="complete">Complete</option>
            <option value="incomplete">Incomplete</option>
            <option value="pending">Pending</option>
          </select>
        </label>
        <label>
          Started since
          <input
            type="date"
            value={
              param(location, "from")
                ? new Date(Number(param(location, "from")) * 1000)
                    .toISOString()
                    .slice(0, 10)
                : ""
            }
            onChange={(event) =>
              updateParams("/activity", location, {
                from: event.target.value
                  ? Math.floor(new Date(event.target.value).getTime() / 1000)
                  : undefined,
                page: 1,
              })
            }
          />
        </label>
        <details className="filter-menu">
          <summary>More filters</summary>
          <div>
            <ColumnsControl columns={activityColumns} visible={visibleColumns} onChange={setVisibleColumns} />
          </div>
        </details>
      </div>
      {param(location, "lane") && (
        <div className="filter-chips">
          <button
            onClick={() =>
              updateParams("/activity", location, { lane: undefined, page: 1 })
            }
          >
            Activity lane: {param(location, "lane").replaceAll("_", " ")} ×
          </button>
        </div>
      )}
      {summaryRequest.loading && !summaryRequest.data ? (
        <MetricCardsSkeleton count={3} />
      ) : summaryRequest.error && !summaryRequest.data ? (
        <ErrorState
          title="Activity summary unavailable"
          error={summaryRequest.error}
          retry={reloadActivity}
        />
      ) : summaryRequest.data ? (
        <>
          <div
            className="metric-card-grid activity-metric-card-grid"
            aria-label="Activity summary"
          >
            <MetricCard
              title="Activities"
              value={Number(
                summaryRequest.data?.summary?.eligible || 0,
              ).toLocaleString()}
              secondary={`${summaryRequest.data?.summary?.closed || 0} Ended · ${summaryRequest.data?.summary?.pending || 0} Ongoing`}
              tooltip="Activity records started in period. An activity is a work segment; several may represent one human task."
            />
            <RateMetricCard
              title="Feedback completion"
              numerator={summaryRequest.data?.summary?.complete}
              denominator={summaryRequest.data?.summary?.closure_eligible}
              secondary={`${summaryRequest.data?.summary?.closure_unclassified || 0} older untracked ended records excluded`}
              tooltip="Ended activities with finalized item accountability divided by ended activities with supported feedback tracking. Ongoing activities are excluded."
            />
            <RateMetricCard
              title="Context usage rate"
              numerator={summaryRequest.data?.summary?.context_use}
              denominator={summaryRequest.data?.summary?.context_denominator}
              secondary="Among activities receiving supported context"
              tooltip="Activities with a delivered item reported used divided by activities receiving supported context. Each activity counts once."
            />
          </div>
          <div
            className="metric-card-grid distribution-card-grid outcome-card-grid"
            aria-label="Activity reported outcomes"
          >
            <ReportedOutcomesCard
              summary={summaryRequest.data?.summary || {}}
            />
          </div>
        </>
      ) : (
        <EmptyState title="No activity summary available">
          Summary metrics will appear when the activity service returns a
          result.
        </EmptyState>
      )}
      <InlineError
        error={rowsRequest.error || summaryRequest.error}
        retained={Boolean(rowsRequest.data || summaryRequest.data)}
        retry={reloadActivity}
      />
      {rowsRequest.loading && !rowsRequest.data ? (
        <LoadingRows />
      ) : rowsRequest.error && !rowsRequest.data ? (
        <ErrorState title="Activity results unavailable" error={rowsRequest.error} retry={reloadActivity} />
      ) : rows.length ? (
        <>
          <TableFrame label="Activity results">
            <table>
              <thead>
                <tr>
                  {visible("started") && <th aria-sort={sort === "started" ? (dir === "asc" ? "ascending" : "descending") : "none"}>
                    <SortButton label="Started" active={sort === "started"} direction={dir} onClick={() => changeSort("started")} />
                  </th>}
                  {visible("task") && <th aria-sort={sort === "task" ? (dir === "asc" ? "ascending" : "descending") : "none"}>
                    <SortButton label="Task / goal" active={sort === "task"} direction={dir} onClick={() => changeSort("task")} />
                  </th>}
                  {visible("scope") && <th><SortButton label="Scope" active={sort === "scope"} direction={dir} onClick={() => changeSort("scope")} /><DefinitionTooltip label="Scope definition">{sharedColumnHelp.scope}</DefinitionTooltip></th>}
                  {visible("outcome") && <th><SortButton label="Outcome" active={sort === "outcome"} direction={dir} onClick={() => changeSort("outcome")} /></th>}
                  {visible("closure") && <th><SortButton label="Closure" active={sort === "closure"} direction={dir} onClick={() => changeSort("closure")} /><DefinitionTooltip label="Closure definition">Whether the session's feedback loop closed. Pending means the session is still open.</DefinitionTooltip></th>}
                  {visible("duration") && <th className="numeric"><SortButton label="Duration" active={sort === "duration"} direction={dir} onClick={() => changeSort("duration")} /></th>}
                  {visible("retrievals") && <th className="numeric"><SortButton label="Retrievals" active={sort === "retrievals"} direction={dir} onClick={() => changeSort("retrievals")} /><DefinitionTooltip label="Retrievals definition">Distinct retrieval events recorded in this session.</DefinitionTooltip></th>}
                  {visible("memories_touched") && <th className="numeric"><SortButton label="Memories touched" active={sort === "memories_touched"} direction={dir} onClick={() => changeSort("memories_touched")} /><DefinitionTooltip label="Memories touched definition">Distinct memory records with evidence linked to this session.</DefinitionTooltip></th>}
                  {visible("procedure") && <th><SortButton label="Procedure record" active={sort === "procedure"} direction={dir} onClick={() => changeSort("procedure")} /><DefinitionTooltip label="Procedure record definition">Whether closing this session created a reusable procedure. Creation does not automatically mean it is a proven playbook.</DefinitionTooltip></th>}
                  {visible("retrieved_memories") && <th className="numeric"><SortButton label="Retrieved memories" active={sort === "retrieved_memories"} direction={dir} onClick={() => changeSort("retrieved_memories")} /></th>}
                  {visible("retrieved_procedures") && <th className="numeric"><SortButton label="Retrieved procedures" active={sort === "retrieved_procedures"} direction={dir} onClick={() => changeSort("retrieved_procedures")} /></th>}
                  {visible("feedback_coverage") && <th><SortButton label="Feedback coverage" active={sort === "feedback_coverage"} direction={dir} onClick={() => changeSort("feedback_coverage")} /></th>}
                  {visible("episodes") && <th className="numeric"><SortButton label="Episodes formed" active={sort === "episodes"} direction={dir} onClick={() => changeSort("episodes")} /></th>}
                  {visible("timeline_events") && <th className="numeric"><SortButton label="Timeline event count" active={sort === "timeline_events"} direction={dir} onClick={() => changeSort("timeline_events")} /></th>}
                  {visible("last_event") && <th><SortButton label="Last event" active={sort === "last_event"} direction={dir} onClick={() => changeSort("last_event")} /></th>}
                  {visible("verification") && <th><SortButton label="Verification" active={sort === "verification"} direction={dir} onClick={() => changeSort("verification")} /></th>}
                </tr>
              </thead>
              <tbody>
                {rows.map((row: any) => {
                  const href = `/activity/${encodeURIComponent(row.id)}${location.search.size ? `?${location.search}` : ""}`;
                  return (
                    <tr
                      key={row.id}
                      tabIndex={0}
                      onClick={() => openDetail(href, location)}
                      onKeyDown={(e) =>
                        rowKeys(e, () => openDetail(href, location))
                      }
                    >
                      {visible("started") && <td title={formatDate(row.started_ts)}>
                        {relativeDate(row.started_ts)}
                      </td>}
                      {visible("task") && <td className="primary-cell">
                        <ClampedText
                          text={row.goal_preview || "Untitled activity"}
                        />
                      </td>}
                      {visible("scope") && <td className="scope-text" title={row.scope_id || undefined}>
                        {row.scope_id ? truncate(row.scope_id, 30) : "No scope"}
                      </td>}
                      {visible("outcome") && <td>
                        <StatusBadge value={row.outcome} />
                      </td>}
                      {visible("closure") && <td>
                        <StatusBadge value={row.feedback_status} />
                      </td>}
                      {visible("duration") && <td className="numeric">
                        {row.duration_s != null ? formatDuration(row.duration_s) : "—"}
                      </td>}
                      {visible("retrievals") && <td className="numeric">{row.retrieval_count || 0}</td>}
                      {visible("memories_touched") && <td className="numeric">{row.memory_count || 0}</td>}
                      {visible("procedure") && <td><StatusBadge value={row.procedure_state || "none"} /></td>}
                      {visible("retrieved_memories") && <td className="numeric">{row.retrieved_memory_count || 0}</td>}
                      {visible("retrieved_procedures") && <td className="numeric">{row.retrieved_procedure_count || 0}</td>}
                      {visible("feedback_coverage") && <td><StatusBadge value={row.feedback_status} /></td>}
                      {visible("episodes") && <td className="numeric">{row.episode_count || 0}</td>}
                      {visible("timeline_events") && <td className="numeric">{row.timeline_event_count || 0}</td>}
                      {visible("last_event") && <td title={formatDate(row.last_event_ts)}>{row.last_event_ts ? relativeDate(row.last_event_ts) : "—"}</td>}
                      {visible("verification") && <td><StatusBadge value={row.verification_status || "unknown"} /></td>}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </TableFrame>
          <Pagination
            page={pagination.page}
            perPage={pagination.per_page}
            total={pagination.total}
            onPage={(next) =>
              updateParams("/activity", location, { page: next })
            }
          />
        </>
      ) : (
        <EmptyState
          title={
            search || location.search.size
              ? "No activity matches these filters"
              : "No activity yet"
          }
        >
          {search || location.search.size
            ? "Clear or broaden the filters to inspect other recorded task segments."
            : "An activity appears when an agent uses Slowave. A zero count is normal on a new installation."}
        </EmptyState>
      )}
      {detailId && (
        <ActivityDetail
          id={detailId}
          onClose={() => closeDetail("/activity", location)}
        />
      )}
    </div>
  );
}

function ActivityDetail({ id, onClose }: { id: string; onClose: () => void }) {
  const request = useApi<Json>(`/api/activity/${encodeURIComponent(id)}`);
  const data = request.data;
  const session = data?.session;
  const stream = useMemo(() => {
    if (!data) return [];
    return [
      ...(data.events || []).map((item: any) => ({
        ...item,
        ts: item.ts,
        kind: "Event",
        private: true,
        summary: item.type,
      })),
      ...(data.episodes || []).map((item: any) => ({
        ...item,
        ts: item.ts,
        kind: "Episode",
        summary: `Episode ${item.id} formed`,
      })),
      ...(data.retrievals || []).map((item: any) => ({
        ...item,
        ts: item.created_at,
        kind: "Retrieval",
        summary: item.query || item.goal || "Context exposure",
      })),
      ...(data.feedback || []).map((item: any) => ({
        ...item,
        ts: item.created_at,
        kind: "Feedback",
        summary: `${item.target_kind} ${item.assessment || item.coverage || "observed"}`,
      })),
      ...(data.memories || []).map((item: any) => ({
        ...item,
        ts: item.first_formed_ts,
        kind: "Memory",
        summary: item.content_text,
      })),
    ].sort((a, b) => Number(a.ts || 0) - Number(b.ts || 0));
  }, [data]);
  return (
    <Inspector
      title="Activity"
      id={id}
      state={session && <StatusBadge value={session.outcome} />}
      onClose={onClose}
    >
      <InlineError
        error={request.error}
        retained={Boolean(data)}
        retry={request.reload}
      />
      {request.loading && !data ? (
        <LoadingRows />
      ) : request.error && !data ? (
        <ErrorState title="Activity details unavailable" error={request.error} retry={request.reload} />
      ) : session ? (
          <>
            <h2 className="detail-title">
              {session.final_goal ||
                session.initial_goal ||
                session.goal ||
                "Untitled activity"}
            </h2>
            <dl className="key-values">
              <dt>Initial goal</dt>
              <dd>{session.initial_goal || session.goal || "Not recorded"}</dd>
              <dt>Final goal</dt>
              <dd>{session.final_goal || "Not recorded"}</dd>
              <dt>Scope</dt>
              <dd>{session.scope_id || "No scope"}</dd>
              <dt>Agent / integration</dt>
              <dd>{session.agent || "Unknown"}</dd>
              <dt>Outcome</dt>
              <dd>
                <StatusBadge value={session.outcome} />{" "}
                {session.outcome_summary || "No outcome summary"}
              </dd>
              <dt>Verification</dt>
              <dd>
                <StatusBadge
                  value={session.verification?.status || "unknown"}
                />{" "}
                {session.verification?.summary || "No verification summary"}
              </dd>
              <dt>Feedback closure</dt>
              <dd>
                <StatusBadge value={session.feedback_status} />
              </dd>
            </dl>
            {session.continuity_id && (
              <div className="notice info">
                <Icon name="info" />
                <div>
                  <strong>Related sessions</strong>
                  <span>
                    Correlation by continuity ID; this is not a complete or
                    independently validated work-attempt model.
                  </span>
                </div>
              </div>
            )}
            <Section title="Chronological record">
              <div className="activity-stream">
                {stream.map((item: any, index: number) => (
                  <div
                    key={`${item.kind}-${item.id || item.context_id || item.event_id || index}`}
                  >
                    <time>{formatDate(item.ts)}</time>
                    <StatusBadge value={item.kind.toLowerCase()} />
                    <p>{truncate(item.summary, 260)}</p>
                    {item.kind === "Retrieval" && (
                      <Link
                        to={`/retrieval/${encodeURIComponent(item.context_id)}`}
                      >
                        Open retrieval
                      </Link>
                    )}
                    {item.kind === "Memory" && (
                      <Link to={`/memory/sch_${item.schema_id}`}>
                        Open memory
                      </Link>
                    )}
                    {item.private && (
                      <details>
                        <summary>Local/private event prose</summary>
                        <pre>{item.content}</pre>
                      </details>
                    )}
                  </div>
                ))}
              </div>
            </Section>
            {data.procedure && (
              <Section title="Created procedure">
                <Link
                  to={`/procedures/${encodeURIComponent(data.procedure.id)}`}
                >
                  {data.procedure.summary}
                </Link>
              </Section>
            )}
            {data.related_sessions?.length > 1 && (
              <Section title="Related sessions">
                <div className="detail-list compact">
                  {data.related_sessions.map((item: any) => (
                    <Link
                      key={item.id}
                      to={`/activity/${encodeURIComponent(item.id)}`}
                    >
                      <span>
                        {formatDate(item.started_ts)} ·{" "}
                        {truncate(item.goal, 100)}
                      </span>
                      <Icon name="chevron" size={14} />
                    </Link>
                  ))}
                </div>
              </Section>
            )}
            <details className="advanced" open>
              <summary>Advanced context</summary>
              <dl className="key-values wide">{Object.entries(session.task_context || {}).map(([key, value]) => <><dt key={`${key}-label`}>{key.replaceAll("_", " ")}</dt><dd key={key}>{typeof value === "object" ? Array.isArray(value) ? value.join(", ") : Object.entries(value as Record<string, unknown>).map(([k, v]) => `${k}: ${String(v)}`).join(" · ") : String(value)}</dd></>)}</dl>
            </details>
          </>
        ) : <EmptyState title="Activity not found">This activity no longer exists or is not available in the current database.</EmptyState>
      }
    </Inspector>
  );
}

export function DiagnosticsPage({}: PageProps) {
  const daemon = useApi<Json>("/api/daemon", { pollMs: refreshMs });
  const database = useApi<Json>("/api/db/health", { pollMs: refreshMs });
  const [runSort, setRunSort] = useState("started");
  const [runDir, setRunDir] = useState<"asc" | "desc">("desc");
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [maintenanceExpanded, setMaintenanceExpanded] = useState(false);
  const runColumns = [
    { id: "started", label: "Started" }, { id: "duration", label: "Duration" }, { id: "result", label: "Result" },
    { id: "episodes", label: "Episodes processed" }, { id: "formed", label: "Formed" }, { id: "reinforced", label: "Reinforced" },
    { id: "retired", label: "Retired" }, { id: "errors", label: "Errors" },
    { id: "skipped", label: "Skipped" }, { id: "pass", label: "Pass number" }, { id: "error_category", label: "Error category" },
  ];
  const [visibleRunColumns, setVisibleRunColumns] = useState<string[]>([
    "started", "duration", "result", "episodes", "formed", "reinforced", "retired", "errors",
  ]);
  const runVisible = (id: string) => visibleRunColumns.includes(id);
  const workers = useApi<Json>(`/api/worker/runs?limit=${maintenanceExpanded ? 50 : 10}&range=1m&sort=${runSort}&dir=${runDir}`, {
    pollMs: refreshMs,
  });
  const status = useApi<Json>("/api/status", { pollMs: refreshMs });
  const updated = [daemon.updatedAt, database.updatedAt, workers.updatedAt]
    .filter(Boolean)
    .sort()
    .at(-1);
  const refreshAll = () => {
    void daemon.reload();
    void database.reload();
    void workers.reload();
    void status.reload();
  };
  const runs = workers.data?.runs || [];
  const selectedRun = runs.find((run: any) => run.id === selectedRunId) || null;
  const changeRunSort = (column: string) => {
    if (runSort === column) {
      setRunDir((d) => (d === "desc" ? "asc" : "desc"));
    } else {
      setRunSort(column);
      setRunDir("desc");
    }
  };
  const databaseHealthValue = (value: unknown) =>
    database.data ? String(value || "Unavailable") : database.loading ? "Checking…" : "Unavailable";
  const databaseHealthSecondary = database.data && database.error
    ? <span className="diagnostic-stale">Stale · refresh failed</span>
    : !database.data && database.error
      ? <span className="diagnostic-stale">Health check failed</span>
      : undefined;
  const storageValue = (value: unknown) =>
    value != null ? `${Number(value).toLocaleString()} bytes` : !status.data && status.loading ? "Checking…" : "Unavailable";
  const foreignKeyValue = database.data
    ? Array.isArray(database.data.foreign_key_check)
      ? database.data.foreign_key_check.length ? `${database.data.foreign_key_check.length} issues` : "No issues observed"
      : "Unavailable"
    : database.loading ? "Checking…" : "Unavailable";
  const recentRuns = workers.data?.summary?.recent_7d;
  const duplicateRows = Number(status.data?.schema_health?.active_exact_duplicate_rows || 0);
  const attentionItems = [
    !daemon.data?.running && {
      title: "MCP daemon is unavailable",
      detail: "Start the local daemon to make Slowave available to connected clients.",
      href: "/diagnostics#services",
    },
    (!status.data?.db_exists || database.data?.integrity_status === "needs_attention") && {
      title: "Database integrity needs attention",
      detail: "Inspect database health before relying on this installation.",
      href: "/diagnostics#database",
    },
    Number(recentRuns?.failed || 0) > 0 && {
      title: `${Number(recentRuns.failed).toLocaleString()} maintenance run${Number(recentRuns.failed) === 1 ? "" : "s"} failed in the last 7 days`,
      detail: "Open maintenance history to inspect the failure details.",
      href: "/diagnostics#maintenance",
    },
    Number(recentRuns?.incomplete || 0) > 0 && {
      title: `${Number(recentRuns.incomplete).toLocaleString()} maintenance run${Number(recentRuns.incomplete) === 1 ? " is" : "s are"} incomplete`,
      detail: "An incomplete run may still be in progress; investigate if it remains unchanged.",
      href: "/diagnostics#maintenance",
    },
    duplicateRows > 0 && {
      title: `${duplicateRows.toLocaleString()} exact duplicate memory row${duplicateRows === 1 ? "" : "s"} detected`,
      detail: "Duplicates are an operational signal to investigate, not an automatic data-loss condition.",
      href: "/memory",
    },
  ].filter(Boolean) as { title: string; detail: string; href: string }[];
  return (
    <div className="page">
      <PageHeader
        title="Diagnostics"
        description="Operational health, maintenance, and investigation for this local Slowave installation."
        updatedAt={updated}
        refreshing={
          daemon.refreshing || database.refreshing || workers.refreshing
        }
        onRefresh={refreshAll}
      />
      <InlineError
        error={daemon.error || database.error || workers.error}
        retained={Boolean(daemon.data || database.data || workers.data)}
        retry={refreshAll}
      />
      <Section title="System health" id="services">
        {!status.data || !workers.data || !database.data ? status.error || workers.error || database.error ? <ErrorState title="Operational health unavailable" error={status.error || workers.error || database.error} retry={refreshAll} /> : <MetricCardsSkeleton count={4} /> : <>
          <div className={`diagnostics-health-banner ${attentionItems.length ? "needs-attention" : "healthy"}`} role="status">
            <div>
              <strong>{attentionItems.length ? `${attentionItems.length} item${attentionItems.length === 1 ? "" : "s"} need attention` : "No active issues detected"}</strong>
              <span>{attentionItems.length ? "Start with the items below; detailed operational evidence remains available on this page." : "Daemon, database, and recently observed maintenance signals are healthy."}</span>
            </div>
          </div>
          {attentionItems.length > 0 && <div className="diagnostics-attention-list" aria-label="Items needing attention">
            {attentionItems.map((item) => <Link className="diagnostics-attention-item" to={item.href} key={item.title}>
              <strong>{item.title}</strong>
              <span>{item.detail}</span>
              <small>Inspect</small>
            </Link>)}
          </div>}
          <div className="metric-card-grid" aria-label="Operational health summary">
            <MetricCard title="MCP daemon" value={daemon.data ? daemon.data.running ? "Responding" : "Not reachable" : "Not checked"} secondary={`Local HTTP probe · ${daemon.updatedAt ? formatDate(new Date(daemon.updatedAt).getTime() / 1000) : "Not checked"}`} tooltip="Whether the local HTTP MCP daemon responded to a health probe." className={daemon.data?.running ? "metric-active" : "metric-warning"} />
            <MetricCard title="Database integrity" value={database.data?.integrity_status === "ok" ? "Passed" : database.data?.integrity_status === "needs_attention" ? "Failed" : databaseHealthValue(database.data?.integrity_status)} secondary={databaseHealthSecondary} tooltip="The latest database integrity check status. This is an operational check, not a memory-quality measure." className={database.data?.integrity_status === "ok" ? "metric-active" : database.data?.integrity_status === "needs_attention" ? "metric-warning" : "metric-subtle"} />
            <MetricCard title="Last maintenance run" value={workers.data?.summary?.last_completed_ts ? formatDate(workers.data.summary.last_completed_ts) : "No data"} tooltip="Most recent completed maintenance run; recorded errors remain available in run history." className="metric-subtle" />
          </div>
        </>}
      </Section>
      <Section title="Pipeline behavior">
        <p className="neutral">Recent maintenance behavior helps explain whether the local pipeline is keeping up. These are operational signals, not memory-value scores.</p>
        {!workers.data ? workers.error ? <ErrorState title="Pipeline behavior unavailable" error={workers.error} retry={workers.reload} /> : <MetricCardsSkeleton count={3} /> : <div className="metric-card-grid pipeline-metric-grid">
          <RateMetricCard title="Run reliability" numerator={workers.data.summary?.finished_without_error} denominator={workers.data.summary?.finished} secondary={`${workers.data.summary?.incomplete_passes || 0} running / interrupted excluded`} tooltip="Finished runs without recorded error divided by finished runs." />
          <MetricCard title="Median run duration" value={workers.data.summary?.median_duration_ms == null ? "No data" : `${Number(workers.data.summary.median_duration_ms).toLocaleString()} ms`} secondary={`${workers.data.summary?.duration_samples || 0} samples · ${workers.data.summary?.missing_durations || 0} missing · ${workers.data.summary?.zero_durations || 0} recorded zero`} tooltip="Median of finished runs with recorded duration, including failures. Durations use whole-second timestamps × 1000: recorded zero does not establish zero-time execution." />
        </div>}
        <details className="advanced diagnostics-inventory">
          <summary>Operational inventory</summary><p className="neutral">Worker process: {workers.data?.worker?.running ? "Running" : "Not observed"} · observed {formatDate(workers.data?.worker?.observed_at)}. Process presence does not establish successful processing.</p>
          <p className="neutral">These retained record totals are useful for diagnostics, but they do not measure memory quality.</p>
          <div className="lifetime-totals">{[["Sessions", status.data?.stats?.sessions], ["Raw events", status.data?.stats?.raw_events], ["Episodes", status.data?.stats?.episodes], ["Memories", status.data?.stats?.schemas]].map(([label, value]) => <span key={String(label)}><strong>{status.data ? value == null ? "Unavailable" : Number(value).toLocaleString() : status.loading ? "Checking…" : "Unavailable"}</strong>{label}</span>)}</div>
        </details>
      </Section>
      <Section title="Consolidation history" id="maintenance" actions={<button className="button secondary" onClick={() => setMaintenanceExpanded((expanded) => !expanded)}>{maintenanceExpanded ? "Show recent 10" : "Show 50 runs"}</button>}>
        <p className="neutral">Showing the {maintenanceExpanded ? "50 most recent" : "10 most recent"} recorded passes. Open a row to inspect counts, configuration, and any error details.</p>
        <div className="filter-bar">
          <details className="filter-menu">
            <summary>Columns</summary>
            <div>
              <ColumnsControl columns={runColumns} visible={visibleRunColumns} onChange={setVisibleRunColumns} />
            </div>
          </details>
        </div>
        {workers.loading && !workers.data ? (
          <LoadingRows />
        ) : workers.error && !workers.data ? (
          <ErrorState title="Consolidation runs unavailable" error={workers.error} retry={workers.reload} />
        ) : runs.length ? (
          <TableFrame label="Maintenance runs">
            <table>
              <thead>
                <tr>
                  {runVisible("started") && <th aria-sort={runSort === "started" ? (runDir === "asc" ? "ascending" : "descending") : "none"}>
                    <SortButton label="Started" active={runSort === "started"} direction={runDir} onClick={() => changeRunSort("started")} />
                  </th>}
                  {runVisible("duration") && <th className="numeric"><SortButton label="Duration" active={runSort === "duration"} direction={runDir} onClick={() => changeRunSort("duration")} /></th>}
                  {runVisible("result") && <th><SortButton label="Result" active={runSort === "result"} direction={runDir} onClick={() => changeRunSort("result")} /></th>}
                  {runVisible("episodes") && <th className="numeric"><SortButton label="Episodes processed" active={runSort === "episodes"} direction={runDir} onClick={() => changeRunSort("episodes")} /></th>}
                  {runVisible("formed") && <th className="numeric"><SortButton label="Formed" active={runSort === "formed"} direction={runDir} onClick={() => changeRunSort("formed")} /></th>}
                  {runVisible("reinforced") && <th className="numeric"><SortButton label="Reinforced" active={runSort === "reinforced"} direction={runDir} onClick={() => changeRunSort("reinforced")} /></th>}
                  {runVisible("retired") && <th className="numeric"><SortButton label="Retired" active={runSort === "retired"} direction={runDir} onClick={() => changeRunSort("retired")} /></th>}
                  {runVisible("errors") && <th className="numeric"><SortButton label="Errors" active={runSort === "errors"} direction={runDir} onClick={() => changeRunSort("errors")} /></th>}
                  {runVisible("skipped") && <th className="numeric"><SortButton label="Skipped" active={runSort === "skipped"} direction={runDir} onClick={() => changeRunSort("skipped")} /></th>}
                  {runVisible("pass") && <th className="numeric"><SortButton label="Pass number" active={runSort === "pass"} direction={runDir} onClick={() => changeRunSort("pass")} /></th>}
                  {runVisible("error_category") && <th><SortButton label="Error category" active={runSort === "error_category"} direction={runDir} onClick={() => changeRunSort("error_category")} /></th>}
                </tr>
              </thead>
              <tbody>
                {runs.map((run: any) => (
                  <tr
                    key={run.id}
                    tabIndex={0}
                    onClick={() => setSelectedRunId(run.id)}
                    onKeyDown={(e) => rowKeys(e, () => setSelectedRunId(run.id))}
                  >
                    {runVisible("started") && <td title={formatDate(run.started_ts)}>{relativeDate(run.started_ts)}</td>}
                    {runVisible("duration") && <td className="numeric">{run.duration_ms != null ? formatDuration(run.duration_ms / 1000) : "—"}</td>}
                    {runVisible("result") && <td><StatusBadge value={run.error_text ? "failure" : run.ended_ts ? "success" : "incomplete"} /></td>}
                    {runVisible("episodes") && <td className="numeric">{run.episodes_processed || 0}</td>}
                    {runVisible("formed") && <td className="numeric">{run.schemas_created || 0}</td>}
                    {runVisible("reinforced") && <td className="numeric">{run.schemas_reinforced || 0}</td>}
                    {runVisible("retired") && <td className="numeric">{run.schemas_decayed || 0}</td>}
                    {runVisible("errors") && <td className="numeric">{run.error_text ? 1 : 0}</td>}
                    {runVisible("skipped") && <td className="numeric">{run.schemas_skipped || 0}</td>}
                    {runVisible("pass") && <td className="numeric">{run.id}</td>}
                    {runVisible("error_category") && <td className="long-text-cell">{truncate(run.error_text, 60) || "—"}</td>}
                  </tr>
                ))}
              </tbody>
            </table>
          </TableFrame>
        ) : (
          <EmptyState title="No maintenance passes recorded">
            An idle installation without recorded passes is not automatically
            unhealthy.
          </EmptyState>
        )}
      </Section>
      <Section title="Storage" id="database">
        <dl className="key-values wide">
          <dt>Database path</dt>
          <dd>
            <CopyValue
              value={
                database.data?.db_path || status.data?.db_path || (database.loading || status.loading ? "Checking…" : "Unavailable")
              }
            />
          </dd>
          <dt>Database size</dt>
          <dd>
            {status.data?.db_size_bytes != null ? formatBytes(status.data.db_size_bytes) : storageValue(status.data?.db_size_bytes)}
          </dd>
          <dt>WAL size</dt>
          <dd>
            {status.data?.wal_size_bytes != null ? formatBytes(status.data.wal_size_bytes) : storageValue(status.data?.wal_size_bytes)}
          </dd>
          <dt>Integrity</dt>
          <dd>{database.data?.integrity_status === "ok" ? "ok" : database.data?.integrity_status === "needs_attention" ? "needs attention" : databaseHealthValue(database.data?.integrity_status)}</dd>
          <dt>Foreign keys</dt>
          <dd>{foreignKeyValue}</dd>
        </dl>
      </Section>
      {selectedRun && (
        <WorkerRunDetail run={selectedRun} onClose={() => setSelectedRunId(null)} />
      )}
    </div>
  );
}
function WorkerRunDetail({ run, onClose }: { run: any; onClose: () => void }) {
  const result = run.error_text
    ? "failure"
    : run.ended_ts
      ? "success"
      : "incomplete";
  return (
    <Inspector
      title="Consolidation run"
      id={`run-${run.id}`}
      state={<StatusBadge value={result} />}
      onClose={onClose}
    >
      <Section title="Summary">
        <dl className="key-values">
          <dt>Result</dt>
          <dd>
            {run.error_text
              ? "The pass ended with an error."
              : run.ended_ts
                ? "The pass completed successfully."
                : "The pass has not ended; it may still be running."}
          </dd>
          <dt>Started</dt>
          <dd>{formatDate(run.started_ts)}</dd>
          <dt>Ended</dt>
          <dd>{run.ended_ts ? formatDate(run.ended_ts) : "—"}</dd>
          <dt>Duration</dt>
          <dd>{run.duration_ms != null ? formatDuration(run.duration_ms / 1000) : "—"}</dd>
        </dl>
      </Section>
      <Section title="Item counts">
        <dl className="key-values wide">
          <dt>Prototypes processed</dt><dd>{run.prototypes_processed || 0}</dd>
          <dt>Episodes processed</dt><dd>{run.episodes_processed || 0}</dd>
          <dt>Memories formed</dt><dd>{run.schemas_created || 0}</dd>
          <dt>Memories reinforced</dt><dd>{run.schemas_reinforced || 0}</dd>
          <dt>Memories skipped</dt><dd>{run.schemas_skipped || 0}</dd>
          <dt>Memories retired</dt><dd>{run.schemas_decayed || 0}</dd>
          <dt>Procedures promoted</dt><dd>{run.procedures_promoted || 0}</dd>
          <dt>Procedures generalized</dt><dd>{run.procedures_generalized || 0}</dd>
        </dl>
      </Section>
      {run.error_text && (
        <Section title="Errors">
          <pre className="detail-pre">{run.error_text}</pre>
        </Section>
      )}
      <Section title="Configuration">
        <dl className="key-values wide">
          <dt>Trigger</dt>
          <dd>{String(run.triggered_by || "worker").replaceAll("_", " ")}</dd>
          <dt>Pass number</dt>
          <dd>{run.id}</dd>
        </dl>
      </Section>
      <p className="neutral">
        Affected memory records are not retained per run; this summary reports
        aggregate counts only.
      </p>
    </Inspector>
  );
}
export function GraphPage() {
  const [selectedSchema, setSelectedSchema] = useState("");
  const [graphScope, setGraphScope] = useState("");
  const [graphStatuses, setGraphStatuses] = useState(
    "active,needs_review,stale",
  );
  const [graphRelations, setGraphRelations] = useState("relates_to,coactivated_with");
  const graph = useApi<Json>(
    `/api/graph/schemas?limit=all&relations=${encodeURIComponent(graphRelations)}&scope=${encodeURIComponent(graphScope)}&statuses=${encodeURIComponent(graphStatuses)}`,
  );
  const graphData = graph.data;
  return (
    <div className="page">
      <PageHeader
        title="Memory graph"
        description="Explore connected memory clusters. Use search for exact lookup; use the graph for browsing."
        updatedAt={graph.updatedAt}
        refreshing={graph.refreshing}
        onRefresh={graph.reload}
      />
      <InlineError
        error={graph.error}
        retained={Boolean(graphData)}
        retry={graph.reload}
      />
      <Section title="Explore your memory network">
          <p className="neutral">Each dot is a memory. Vertex color shows lifecycle state (green active, yellow review, red stale); blue connections are related memories and violet connections are co-activations. Select a dot to inspect it.</p>
          <div className="filter-bar graph-controls">
            <label>Scope<select value={graphScope} onChange={(event) => setGraphScope(event.target.value)}><option value="">All scopes</option>{[...new Set((graphData?.nodes || []).map((node: any) => node.scope).filter(Boolean))].map((scope: any) => <option key={scope} value={scope}>{scope}</option>)}</select></label>
            <div className="toggle-group" role="group" aria-label="Memory states"><span className="toggle-label">States</span>{[["active","Active"],["needs_review","Review"],["stale","Stale"]].map(([value,label]) => <button type="button" key={value} className={`toggle-badge toggle-state-${value} ${graphStatuses.split(",").includes(value) ? "selected" : ""}`} aria-pressed={graphStatuses.split(",").includes(value)} onClick={() => setGraphStatuses((old) => { const values = old.split(",").filter(Boolean); const next = values.includes(value) ? values.filter((item) => item !== value) : [...values, value]; return next.join(","); })}>{label}</button>)}</div>
            <div className="toggle-group" role="group" aria-label="Edge types"><span className="toggle-label">Edges</span>{[["relates_to","Related to","related"],["coactivated_with","Co-activated","coactivated"]].map(([value,label,kind]) => <button type="button" key={value} className={`toggle-badge toggle-${kind} ${graphRelations.split(",").includes(value) ? "selected" : ""}`} aria-pressed={graphRelations.split(",").includes(value)} onClick={() => setGraphRelations((old) => { const values = old.split(",").filter(Boolean); const next = values.includes(value) ? values.filter((item) => item !== value) : [...values, value]; return next.join(","); })}>{label}</button>)}</div>
          </div>
          {graphData && Number(graphData.nodes?.length || 0) >= Number(graphData.limit || 0) && <p className="neutral">Display limit reached ({graphData.limit}); additional memories may be omitted.</p>}
          {graph.loading && !graphData ? <LoadingRows /> : graph.error && !graphData ? <ErrorState title="Graph unavailable" error={graph.error} retry={graph.reload} /> : graphData?.nodes?.length ? <Suspense fallback={<LoadingRows rows={3} />}><GraphExplorer data={graphData} onSelect={setSelectedSchema} /></Suspense> : <EmptyState title="No connected memories">At least two related memories are needed to draw a network.</EmptyState>}
      </Section>
      {selectedSchema && <MemoryDetail id={selectedSchema} onClose={() => setSelectedSchema("")} />}
    </div>
  );
}
