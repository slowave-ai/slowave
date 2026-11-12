import { ActivityChartPanel } from "./ActivityTimeline";
import {
  Fragment,
  lazy,
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
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
  ActiveFilters,
  allowActions,
  ClampedText,
  ClientBadge,
  ClientStack,
  CodeBlock,
  ColumnsControl,
  CopyValue,
  DefinitionTooltip,
  DetailSection,
  EmptyState,
  EMPTY,
  ErrorState,
  formatCount,
  formatDate,
  formatDuration,
  formatRate,
  formatRatePercent,
  getRatePercent,
  glossary,
  Icon,
  IdChip,
  InlineError,
  Inspector,
  KeyValueList,
  Link,
  LoadingRows,
  MetricCard,
  MetricCardsSkeleton,
  PageHeader,
  Pagination,
  refreshMs,
  relativeDate,
  RowLink,
  ScopeChip,
  ScopeSelect,
  RateMetricCard,
  Section,
  SortableTh,
  SortButton,
  StatusBadge,
  TableFrame,
  TimeCell,
  truncate,
  TypeBadge,
  useColumns,
} from "./components";
import { statusLabel as statusLabelOf } from "./tones";
import {
  fallbackLabel,
  formatBytes,
  formatClock,
  formatDay,
  formatMs,
  formatPercentOf,
  formatTokens,
  fromDateInput,
  policyLabel,
  shortScope,
  toDateInput,
} from "./format";
import { isLibraryPath } from "./timeRange";

type PageProps = { location: LocationState };
const GraphExplorer = lazy(() => import("./GraphExplorer"));
const sharedColumnHelp = {
  scope: "The scope (memory boundary) this record belongs to; it keeps related data isolated.",
  outcome: "The recorded task outcome: success, partial, or failure. It describes what happened, not a causal quality score.",
  verification: "How strongly the recorded outcome was checked: verified, partially verified, or unverified.",
  retrieved: "How many distinct retrieval events returned this record.",
  used: "How many retrieval events explicitly assessed this record as used.",
  usage: "Used / retrieved delivery occasions for this record. The bar shows the share reported used; grey includes occasions without usage feedback, and a dash means feedback is pending.",
  effect: "The reported downstream effect of using this procedure: Helpful, No effect, Harmful, or Unknown. Unknown means no effect assessment was recorded.",
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
  const entityId = decodeURIComponent(to.match(/^\/(?:memory|retrieval|procedures|activity|sessions)\/([^/?]+)/)?.[1] || "");
  navigate(to, {
    state: {
      listUrl: `${location.path}${location.search.size ? `?${location.search}` : ""}`,
      listScrollY: window.scrollY,
      returnFocusId: entityId,
    },
  });
}
function closeDetail(parent: string, location: LocationState) {
  const returningToInspector = Boolean(location.state?.returnUrl);
  const returnScroll = returningToInspector
    ? Number(location.state?.returnScrollTop || 0)
    : Number(location.state?.listScrollY || 0);
  navigate(location.state?.returnUrl || location.state?.listUrl || parent, {
    state: location.state?.returnState || null,
    preserveScroll: true,
  });
  let attempts = 0;
  const restoreOrigin = () => {
    attempts += 1;
    if (returningToInspector) {
      const content = document.querySelector<HTMLElement>(".inspector-content");
      const focusHref = location.state?.returnFocusHref;
      const target = focusHref && Array.from(document.querySelectorAll<HTMLAnchorElement>(".inspector a[href]")).find(
          (link) => link.getAttribute("href") === focusHref,
        );
      if (content && target) {
        content.scrollTop = returnScroll;
        target.focus();
        return;
      }
    } else {
      const focusHref = location.state?.returnFocusHref;
      const focusId = location.state?.returnFocusId;
      const target = focusHref
        ? Array.from(document.querySelectorAll<HTMLAnchorElement>("a[href]")).find((link) => link.getAttribute("href") === focusHref)
        : focusId
          ? Array.from(document.querySelectorAll<HTMLElement>("[data-detail-id]")).find((row) => row.dataset.detailId === focusId)
          : null;
      const focusTarget = (target as HTMLElement | null | undefined)?.querySelector<HTMLElement>("a.row-link") || target;
      if (target) {
        window.scrollTo({ top: returnScroll, behavior: "auto" });
        (focusTarget as HTMLElement).focus();
        return;
      }
    }
    if (attempts < 40) window.setTimeout(restoreOrigin, 50);
  };
  window.requestAnimationFrame(() => window.requestAnimationFrame(restoreOrigin));
}
function rowKeys(event: KeyboardEvent, action: () => void) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    action();
  }
}

const dash = <span className="muted">{EMPTY}</span>;

/** Attributes shared by every clickable table row: identity, open-row state and mouse activation. */
function rowAttrs(id: string, openId: string, open: () => void) {
  return {
    "data-detail-id": id,
    "aria-current": openId === id ? ("true" as const) : undefined,
    onClick: open,
  };
}

/** Controlled local-time date filter (stored as epoch seconds in the URL). */
function DateFilter({
  label,
  seconds,
  endOfDay = false,
  onChange,
}: {
  label: string;
  seconds: string;
  endOfDay?: boolean;
  onChange: (seconds: number | undefined) => void;
}) {
  return (
    <label>
      {label}
      <input
        type="date"
        value={toDateInput(seconds)}
        onChange={(event) => onChange(fromDateInput(event.target.value, endOfDay))}
      />
    </label>
  );
}

/** An out-of-range page (for example after filtering) redirects to the last real page instead of an empty state. */
function useClampPage(
  path: string,
  location: LocationState,
  pagination: { page: number; per_page: number; total: number } | undefined,
  rowCount: number,
) {
  const requested = pageNumber(location);
  const last = pagination ? Math.max(1, Math.ceil(pagination.total / Math.max(1, pagination.per_page))) : 1;
  useEffect(() => {
    if (pagination && pagination.total > 0 && rowCount === 0 && requested > last) {
      const query = new URLSearchParams(location.search);
      if (last <= 1) query.delete("page");
      else query.set("page", String(last));
      navigate(`${path}${query.size ? `?${query}` : ""}`, { replace: true, preserveScroll: true });
    }
  }, [requested, last, pagination?.total, rowCount]);
}

function periodChip(location: LocationState, clear: () => void) {
  const from = param(location, "from");
  const to = param(location, "to");
  const hours = param(location, "hours");
  if (from || to) {
    return { key: "period", label: `Period: ${from ? formatDay(from) : "…"} – ${to ? formatDay(to) : "now"}`, onRemove: clear };
  }
  if (hours && hours !== "all") return { key: "period", label: `Period: last ${hours === "168" ? "week" : hours === "720" ? "month" : `${hours} hours`}`, onRemove: clear };
  return null;
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
  const [expanded, setExpanded] = useState(false);
  const problems = cells.filter((c) => c.state !== "Available");
  const healthy = problems.length === 0;
  const tone = healthy ? "ok" : problems.some((c) => c.state === "Needs attention") ? "bad" : "warn";
  const headline = healthy
    ? "All systems operational"
    : `${problems.length} of ${cells.length} need attention`;
  return (
    <Section title="Availability">
      <div className={`availability-summary availability-tone-${tone}`}>
        <Link to="/diagnostics" className="availability-headline">
          <i aria-hidden="true" />
          {headline}
        </Link>
        <div className="availability-facts">
          {cells.map((cell) => (
            <span key={cell.label} title={`${cell.source} · ${cell.observed ? relativeDate(cell.observed) : "not observed"}`}>
              <em>{cell.label}</em> {cell.detail}
            </span>
          ))}
        </div>
        {healthy && (
          <button
            type="button"
            className="availability-toggle"
            aria-expanded={expanded}
            onClick={() => setExpanded((open) => !open)}
          >
            {expanded ? "Hide details" : "Details"}
          </button>
        )}
      </div>
      {(expanded || !healthy) && (
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
      )}
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
  const current = new URLSearchParams(window.location.search);
  if (isLibraryPath(path)) return structuralUrl(path, { scope: data.scope || undefined });
  return structuralUrl(path, {
    scope: data.scope || undefined,
    hours: current.get("hours") || "all",
    from: current.get("from") || undefined,
    to: current.get("to") || undefined,
  });
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
        className={memory ? "metric-memory" : "metric-procedure"}
        title={memory ? "Active memories" : "Saved procedures"}
        value={
          data.available
            ? Number(data[`${kind}_total`]).toLocaleString()
            : EMPTY
        }
        tooltip="Current canonical inventory; date filters narrow the list, not this total."
        href={metricHref(path, data)}
      />
      <RateMetricCard
        compact={compact}
        className="metric-retrieval"
        title={`${name} retrieval coverage`}
        numerator={data[`${kind}_exposed`]}
        denominator={data[`${kind}_universe`] ?? data[`${kind}_total`]}
        tooltip="Distinct records delivered during the selected period divided by the current inventory plus any retired records delivered in the period. A record counts once even if delivered repeatedly. Retrieval does not establish use or quality."
        href={metricHref(path, data)}
      />
      <RateMetricCard
        compact={compact}
        className="metric-used"
        title={`${name} usage rate`}
        numerator={data[`${kind}_used`]}
        denominator={data[`${kind}_assessed`]}
        secondary={`Among assessed retrieved ${memory ? "memories" : "procedures"} · ${data[`${kind}_unknown`] ?? 0} ${memory ? "unassessed" : "unknown use"}`}
        tooltip={`Distinct delivered records reported used divided by distinct delivered records with supported use assessments. Each record counts once even if delivered repeatedly. ${data[`${kind}_unknown`] ?? 0} delivered records have unknown use.`}
        href={metricHref(path, data)}
      />
      {!compact && (
        <RateMetricCard
          title="Feedback coverage" className="metric-rate"
          numerator={data[`${kind}_feedback`]}
          denominator={data[`${kind}_deliveries`]}
          secondary={`${Number(data[`${kind}_deliveries`] || 0) - Number(data[`${kind}_feedback`] || 0)} unassessed delivery occasions`}
          tooltip={`${feedbackTooltip} The same record can count again when delivered in a separate request.`}
        />
      )}
    </>
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
      tooltip="Distribution over ended sessions, including not reported outcomes. Missing closure is not failure."
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
    <Section title="Activity" className="focused-activity">
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
function formatTokenEstimate(tokens: number | null): string {
  if (tokens == null) return EMPTY;

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

function TokenConsumptionCard({ data }: { data: Json }) {
  const total = data.context_size?.total;
  const breakdown = data.context_size?.breakdown || {};
  const totalTokens = total?.total_tokens ?? 0;

  const endpointLabels: Record<string, string> = {
    activate: "Activate",
    recall: "Recall",
    remember: "Remember",
    feedback: "Feedback",
    commit: "Commit",
  };

  // Professional color palette (OKLCH-based, accessible)
  const endpointColors: Record<string, string> = {
    activate: "var(--chart-retrieved)",
    recall: "var(--chart-used)",
    remember: "var(--chart-created)",
    feedback: "var(--chart-rate)",
    commit: "var(--accent-procedure)",
  };

  // Merge historical activation records into the current endpoint label.
  const normalized: Record<string, any> = {};
  for (const [kind, val] of Object.entries(breakdown) as [string, any][]) {
    const key = kind === "context" ? "activate" : kind;
    const previous = normalized[key] || { total_tokens: 0, samples: 0 };
    normalized[key] = {
      total_tokens: previous.total_tokens + val.total_tokens,
      samples: previous.samples + val.samples,
    };
  }
  const rawItems = Object.entries(normalized)
    .map(([key, val]: [string, any]) => ({
      key,
      label: endpointLabels[key] || key,
      tokens: val.total_tokens,
      samples: val.samples,
      color: endpointColors[key] || "var(--accent-neutral)",
    }))
    .filter((d) => d.tokens > 0)
    .sort((a, b) => b.tokens - a.tokens);

  // Keep bar widths proportional without rounding away small endpoints.
  const visibleTotal = rawItems.reduce((sum, d) => sum + d.tokens, 0);
  const breakdownItems = rawItems.map((d) => ({
    ...d,
    pct: visibleTotal > 0 ? (d.tokens / visibleTotal) * 100 : 0,
  }));

  return (
    <MetricCard
        className="distribution-card token-consumption-breakdown"
        title="Token consumption"
        tooltip="Slowave adds memories and procedures to your agent’s context window so it can reuse relevant knowledge. This is an approximate endpoint-by-endpoint breakdown of the tokens added by MCP responses, including metadata, in the selected period (estimated at one token per four characters)."
        value={
          <div className="stage-distribution">
            <strong className="token-total">{formatTokenEstimate(totalTokens)}</strong>
            <div className="distribution-bar" style={{ width: "100%" }}>
              {breakdownItems.map((d) => (
                <span
                  key={d.key}
                  style={{
                    width: `${d.pct}%`,
                    backgroundColor: d.color,
                  }}
                  title={`${d.label}: ${formatTokenEstimate(d.tokens)} (${Math.round(d.pct)}%) — ${d.samples.toLocaleString()} calls`}
                />
              ))}
              {breakdownItems.length === 0 && (
                <span className="stage-unknown" style={{ width: "100%" }} />
              )}
            </div>
            <div className="distribution-legend stage-legend">
              {breakdownItems.map((d) => (
                <span
                  key={d.key}
                  style={{ borderLeftColor: d.color }}
                  title={`${d.label}: ${formatTokenEstimate(d.tokens)} (${Math.round(d.pct)}%) — ${d.samples.toLocaleString()} calls`}
                >
                  <span style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
                    <span
                      style={{
                        width: "12px",
                        height: "12px",
                        borderRadius: "2px",
                        backgroundColor: d.color,
                        flexShrink: 0,
                      }}
                    />
                    {d.label}
                  </span>
                </span>
              ))}
            </div>
          </div>
        }
      />
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
          className="metric-retrieval"
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
          className="metric-rate"
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
              : EMPTY
          }
          tooltip={glossary.active_scopes}
          href={metricHref("/memory", data)}
        />
        <TokenConsumptionCard data={data} />
      </div>
    </Section>
  );
}
function PeriodFilter({
  location,
  path,
  hideLabel = false,
}: {
  location: PageProps["location"];
  path: string;
  hideLabel?: boolean;
}) {
  return (
    <label>
      <span className={hideLabel ? "sr-only" : undefined}>Time range</span>
      <select
        value={param(location, "from") || param(location, "to") ? "custom" : param(location, "hours", "all")}
        onChange={(e) =>
          updateParams(path, location, {
            hours: e.target.value,
            from: undefined,
            to: undefined,
            saved_from: undefined,
            saved_to: undefined,
            page: 1,
          })
        }
      >
        {(param(location, "from") || param(location, "to")) && <option value="custom" disabled>Custom period</option>}
        <option value="3">Last 3 hours</option>
        <option value="12">Last 12 hours</option>
        <option value="24">Last 24 hours</option>
        <option value="168">Last week</option>
        <option value="720">Last month</option>
        <option value="all">All times</option>
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
        className="home-header"
        hideRefreshLabel
        description="Readiness, actionable exceptions, and observed changes in the selected period."
        updatedAt={request.updatedAt}
        refreshing={request.refreshing}
        onRefresh={request.reload}
        controls={
          <div className="home-controls">
            <PeriodFilter location={location} path="/" hideLabel />
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
                    to={structuralUrl("/sessions", {
                      from: home.window?.from,
                      to: home.window?.to,
                    })}
                  >
                    View all sessions
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
      className={`use-rate-bar bar-neutral${available ? "" : " is-disabled"}`}
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
        <td className="numeric">{formatCount(memories)}</td>
        <td className="numeric">
          <span
            className="use-rate-bar bar-neutral"
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
            className="use-rate-bar bar-retrieved"
            role="progressbar"
            aria-label={`${Math.round(coverage)}% retrieval coverage`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(coverage)}
            title={`${retrieved.toLocaleString()} of ${formatCount(memories)} retrieved`}
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

const memoryColumns = [
  { id: "memory", label: "Memory", locked: true },
  { id: "state", label: "State" },
  { id: "scope", label: "Scope" },
  { id: "usage", label: "Used", description: "Used / retrieved occasions. The bar shows the share reported used." },
  { id: "last_used", label: "Last used" },
  { id: "changed", label: "Updated" },
  { id: "client", label: "Client" },
  { id: "salience", label: "Salience (0–20)", description: "How strongly this memory is favoured for retrieval. Higher is more likely to be surfaced." },
  { id: "created", label: "Saved" },
  { id: "evidence", label: "Supporting evidence" },
  { id: "retrieved", label: "Times retrieved" },
  { id: "used_count", label: "Times used" },
  { id: "irrelevant", label: "Irrelevant feedback" },
  { id: "stale", label: "Stale feedback" },
  { id: "wrong", label: "Wrong feedback" },
  { id: "related", label: "Related memories" },
  { id: "source_activity", label: "Source sessions" },
] as const;
const memoryDefaultColumns = ["memory", "state", "scope", "usage", "last_used", "changed", "client"];
const DEFAULT_MEMORY_STATES = "active,needs_review";

export function MemoryPage({ location }: PageProps) {
  const detailId = location.path.startsWith("/memory/")
    ? decodeURIComponent(location.path.split("/")[2])
    : "";
  const scope = param(location, "scope");
  const statesParam = param(location, "states");
  // "" = default (current), "all" = every state, otherwise an explicit comma list.
  const states = statesParam === "all" ? "" : statesParam || DEFAULT_MEMORY_STATES;
  const sort = param(location, "sort", "changed");
  const dir = param(location, "dir", "desc") as "asc" | "desc";
  const page = pageNumber(location);
  const [search, setSearch] = useState("");
  const [debounced, setDebounced] = useState("");
  const [visibleColumns, setVisibleColumns, resetColumns] = useColumns("memory", memoryDefaultColumns);
  const visible = (id: string) => visibleColumns.includes(id);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(search), 250);
    return () => clearTimeout(timer);
  }, [search]);
  const endpoint = `/api/schemas?states=${encodeURIComponent(states)}&scope=${encodeURIComponent(scope)}&sort=${sort}&dir=${dir}&page=${page}&per_page=50&q=${encodeURIComponent(debounced)}&hours=all&saved_from=${param(location, "saved_from")}&saved_to=${param(location, "saved_to")}`;
  const request = useApi<Json>(endpoint);
  const rows = request.data?.schemas || [];
  const pagination = request.data?.pagination || {
    page,
    per_page: 50,
    total: 0,
  };
  useClampPage("/memory", location, request.data?.pagination, rows.length);
  const counts: Record<string, number> = request.data?.status_counts || {};
  const countOf = (...keys: string[]) => keys.reduce((total, key) => total + Number(counts[key] || 0), 0);
  const stateChips = [
    { id: "", label: "Current", value: DEFAULT_MEMORY_STATES, count: countOf("active", "needs_review"), hint: "Active and needs-review memories" },
    { id: "needs_review", label: "Needs review", value: "needs_review", count: countOf("needs_review"), hint: "Memories flagged for confirmation or correction" },
    { id: "stale", label: "Stale", value: "stale", count: countOf("stale"), hint: "Memories that may be out of date" },
    ...(countOf("archived") ? [{ id: "archived", label: "Archived", value: "archived", count: countOf("archived"), hint: "Retained for history only" }] : []),
    { id: "all", label: "All", value: "", count: countOf("active", "needs_review", "stale", "archived"), hint: "Every state" },
  ];
  const changeSort = (column: string) =>
    updateParams("/memory", location, {
      sort: column,
      dir: sort === column && dir === "desc" ? "asc" : "desc",
      page: 1,
    });
  const clearAll = () => {
    setSearch("");
    setDebounced("");
    navigate("/memory");
  };
  const savedFrom = param(location, "saved_from");
  const savedTo = param(location, "saved_to");
  const chips = [
    debounced && { key: "q", label: `Search: “${truncate(debounced, 30)}”`, onRemove: () => { setSearch(""); setDebounced(""); } },
    statesParam && { key: "states", label: `State: ${statesParam === "all" ? "all" : statesParam.replaceAll("_", " ").replaceAll(",", ", ")}`, onRemove: () => updateParams("/memory", location, { states: undefined, page: 1 }) },
    scope && { key: "scope", label: `Scope: ${shortScope(scope)}`, onRemove: () => updateParams("/memory", location, { scope: undefined, page: 1 }) },
    (savedFrom || savedTo) && { key: "saved", label: savedTo ? `Created: ${savedFrom ? formatDay(savedFrom) : "…"} – ${formatDay(savedTo)}` : `Created since ${formatDay(savedFrom)}`, onRemove: () => updateParams("/memory", location, { saved_from: undefined, saved_to: undefined, page: 1 }) },
  ].filter(Boolean) as { key: string; label: string; onRemove: () => void }[];
  const filtered = chips.length > 0;
  const th = (label: string, column: string, extra: { numeric?: boolean; help?: string } = {}) => (
    <SortableTh label={label} column={column} sort={sort} dir={dir} onSort={changeSort} {...extra} />
  );
  return (
    <div className="page">
      <PageHeader
        title="Memories"
        description="Durable context Slowave can use in future retrieval. The list shows memories in the library; use Created since to narrow by date. Inventory totals describe the current library."
        updatedAt={request.updatedAt}
        refreshing={request.refreshing}
        onRefresh={request.reload}
      />
      <div className="filter-bar">
        <label>
          Search<span className="sr-only"> memory text</span>
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && setDebounced(search)}
            placeholder="Search memory text…"
          />
        </label>
        <ScopeSelect
          value={scope}
          onChange={(next) =>
            updateParams("/memory", location, { scope: next, page: 1 })
          }
        />
        <DateFilter
          label="Created since"
          seconds={savedFrom}
          onChange={(value) => updateParams("/memory", location, { saved_from: value, saved_to: undefined, page: 1 })}
        />
      </div>
      <div className="state-chips" role="group" aria-label="Memory state">
        {stateChips.map((chip) => {
          const pressed = (chip.id === "" ? !statesParam : chip.id === "all" ? statesParam === "all" : statesParam === chip.value);
          return (
            <button
              key={chip.label}
              type="button"
              className="state-chip"
              aria-pressed={pressed}
              title={chip.hint}
              onClick={() => updateParams("/memory", location, { states: chip.id === "" ? undefined : chip.id === "all" ? "all" : chip.value, page: 1 })}
            >
              {chip.label}<b>{request.data ? formatCount(chip.count) : EMPTY}</b>
            </button>
          );
        })}
      </div>
      <ActiveFilters chips={chips} onClear={clearAll} />
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
          <div className="table-toolbar">
            <span>{formatCount(pagination.total)} {pagination.total === 1 ? "memory" : "memories"}</span>
            <ColumnsControl columns={memoryColumns} visible={visibleColumns} onChange={setVisibleColumns} onReset={resetColumns} />
          </div>
          <TableFrame label="Memory results">
            <table className="library-table">
              <thead>
                <tr>
                  {visible("memory") && th("Memory", "content")}
                  {visible("state") && th("State", "status", { help: "The memory's lifecycle status: active, needs review, or stale." })}
                  {visible("scope") && th("Scope", "scope", { help: sharedColumnHelp.scope })}
                  {visible("usage") && th("Used", "used", { help: "Used / retrieved occasions for this memory. The bar shows the share reported used; grey includes occasions without usage feedback." })}
                  {visible("last_used") && th("Last used", "last_used")}
                  {visible("changed") && th("Updated", "changed")}
                  {visible("client") && <SortableTh label="Client" />}
                  {visible("salience") && th("Salience", "salience", { numeric: true, help: "Retrieval priority on a 0–20 scale. Higher means the memory is more likely to be surfaced." })}
                  {visible("created") && th("Saved", "formed")}
                  {visible("evidence") && th("Evidence", "evidence", { numeric: true, help: "Number of recorded provenance links. This is evidence volume, not evidence quality." })}
                  {visible("retrieved") && th("Retrieved", "exposed", { numeric: true, help: glossary.exposed })}
                  {visible("used_count") && th("Used (count)", "used", { numeric: true, help: glossary.used })}
                  {visible("irrelevant") && th("Irrelevant", "irrelevant", { numeric: true, help: glossary.irrelevant })}
                  {visible("stale") && th("Stale fb.", "stale", { numeric: true })}
                  {visible("wrong") && th("Wrong fb.", "wrong", { numeric: true })}
                  {visible("related") && th("Related", "related", { numeric: true })}
                  {visible("source_activity") && th("Sessions", "source_activity", { numeric: true })}
                </tr>
              </thead>
              <tbody>
                {rows.map((memory: any) => {
                  const href = `/memory/${encodeURIComponent(memory.id)}${location.search.size ? `?${location.search}` : ""}`;
                  const open = () => openDetail(href, location);
                  return (
                    <tr key={memory.id} {...rowAttrs(memory.id, detailId, open)}>
                      {visible("memory") && (
                        <td className="primary-cell">
                          <RowLink href={href} onOpen={open}><ClampedText text={memory.content} /></RowLink>
                          <span className="meta">{memory.id} · saved <TimeCell ts={memory.first_formed_ts} /></span>
                        </td>
                      )}
                      {visible("state") && <td><StatusBadge value={memory.status} /></td>}
                      {visible("scope") && <td><ScopeChip scope={memory.scope} /></td>}
                      {visible("usage") && (
                        <td>
                          <UsageCell used={memory.times_used} retrieved={memory.times_exposed} assessed={memory.times_assessed} />
                        </td>
                      )}
                      {visible("last_used") && <td><TimeCell ts={memory.last_used_ts} /></td>}
                      {visible("changed") && <td><TimeCell ts={memory.last_updated_ts} /></td>}
                      {visible("client") && <td><ClientStack clients={memory.clients} /></td>}
                      {visible("salience") && (
                        <td className="numeric salience-value">{memory.salience == null ? EMPTY : Number(memory.salience).toFixed(1)}</td>
                      )}
                      {visible("created") && <td><TimeCell ts={memory.first_formed_ts} /></td>}
                      {visible("evidence") && <td className="numeric">{formatCount(memory.evidence_count ?? 0)}</td>}
                      {visible("retrieved") && <td className="numeric">{formatCount(memory.times_exposed ?? 0)}</td>}
                      {visible("used_count") && <td className="numeric">{formatCount(memory.times_used ?? 0)}</td>}
                      {visible("irrelevant") && <td className="numeric">{formatCount(memory.times_irrelevant ?? 0)}</td>}
                      {visible("stale") && <td className="numeric">{formatCount(memory.times_stale ?? 0)}</td>}
                      {visible("wrong") && <td className="numeric">{formatCount(memory.times_wrong ?? 0)}</td>}
                      {visible("related") && <td className="numeric">{formatCount(memory.related_count ?? 0)}</td>}
                      {visible("source_activity") && <td className="numeric">{formatCount(memory.source_activity_count ?? 0)}</td>}
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
          title={filtered ? "No memories match these filters" : "No durable memories yet"}
          action={filtered ? <button type="button" className="button secondary" onClick={clearAll}>Clear filters</button> : undefined}
        >
          {filtered
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

const relationLabel = (value: unknown) => String(value || "related").replaceAll("_", " ");
const memoryHref = (id: unknown) => `/memory/${String(id).startsWith("sch_") ? id : `sch_${id}`}`;

function ConnectionItem({ id, preview, status, caption }: { id: string; preview?: string; status?: string; caption: string }) {
  return (
    <li>
      <Link to={memoryHref(id)}>
        <span className="item-main">
          <span>{truncate(preview || `Memory sch_${String(id).replace(/^sch_/, "")}`, 140)}</span>
          <small>{caption}</small>
        </span>
        <span className="item-end">{status && <StatusBadge value={status} />}<Icon name="chevron" size={14} /></span>
      </Link>
    </li>
  );
}

function MemoryDetail({ id, onClose, onDeleted }: {
  id: string;
  onClose: () => void;
  onDeleted?: () => Promise<void>;
}) {
  const request = useApi<Json>(`/api/schemas/${encodeURIComponent(id)}`, { clearOnUrlChange: true });
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
  const totals = memory?.use_history_totals || {};
  const retrieved = Number(totals.retrieved || 0);
  const assessed = Number(totals.assessed || 0);
  const used = Number(totals.used || 0);
  const replacementId = memory?.replacement_memory_id || data?.replacement_memory_id;
  const staleReason = memory?.stale_reason || data?.stale_reason;
  const evidence: any[] = data?.evidence || [];
  const evidenceTotal = Number(data?.evidence_total ?? evidence.length);
  const hasText = (item: any) => item.has_text ?? Boolean(item.quote || item.event_content);
  const quotes = evidence.filter(hasText);
  const textless = evidence.length - quotes.length;
  const sessions: any[] = [...(data?.sessions || [])].sort((a, b) => Number(Boolean(b.is_origin)) - Number(Boolean(a.is_origin)));
  const connectionCount = (data?.outgoing?.length || 0) + (data?.incoming?.length || 0) + (data?.coact_outgoing?.length || 0) + (data?.coact_incoming?.length || 0);
  return (
    <Inspector
      title="Memory"
      id={id}
      label={memory ? truncate(memory.content, 140) : undefined}
      hideLabel
      state={memory && <StatusBadge value={memory.status} />}
      sections={[
        { id: "memory-detail", label: "Memory" },
        { id: "memory-why", label: "Why it exists" },
        { id: "memory-history", label: "Use history" },
        { id: "memory-connections", label: "Connections" },
        { id: "memory-technical", label: "Technical" },
      ]}
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
            <DetailSection id="memory-detail" title="Memory">
              <h2 className="detail-title">{memory.content}</h2>
              <div className="summary-strip">
                <span className="strip-state">
                  <StatusBadge value={memory.status} />
                  <DefinitionTooltip label="What this state means">
                    {memory.status === "stale"
                      ? "Retained as historical context but not treated as current."
                      : "Eligible for future retrieval when scope and relevance rules admit it."}
                  </DefinitionTooltip>
                </span>
                <ScopeChip scope={memory.scope} />
                <ClientStack clients={memory.clients} />
                <span>saved <TimeCell ts={memory.first_formed_ts} absolute /></span>
                <span>updated <TimeCell ts={memory.last_updated_ts} absolute /></span>
                <span>{assessed ? <>used <strong>{used} of {assessed}</strong> assessed ({formatRatePercent(used, assessed)})</> : "no usage feedback yet"}</span>
                <span><strong>{formatCount(retrieved)}</strong> retrieved</span>
              </div>
              {(memory.status === "stale" || memory.status === "needs_review") && (
                <div className={`banner${memory.status === "stale" ? "" : " info"}`} role="note">
                  <Icon name="warning" />
                  <div>
                    <strong>{memory.status === "stale" ? "This memory is marked stale" : "This memory needs review"}</strong>
                    <span>{staleReason || "No reason was recorded."}</span>
                    {replacementId && <span>Replaced by <Link to={memoryHref(replacementId)}>{String(replacementId)}</Link></span>}
                  </div>
                </div>
              )}
            </DetailSection>
            <DetailSection id="memory-why" title="Why this memory exists">
              <p className="neutral">
                {evidenceTotal
                  ? `Built from ${formatCount(evidenceTotal)} evidence ${evidenceTotal === 1 ? "record" : "records"}${sessions.length ? ` across ${formatCount(data?.sessions_total ?? sessions.length)} ${(data?.sessions_total ?? sessions.length) === 1 ? "session" : "sessions"}` : ""}.`
                  : "No provenance rows are available for this memory."}
                {evidenceTotal > evidence.length ? ` Showing ${formatCount(evidence.length)} of ${formatCount(evidenceTotal)}.` : ""}
              </p>
              {sessions.length > 0 && (
                <ul className="link-list">
                  {sessions.map((item: any) => (
                    <li key={item.session_id}>
                      <Link to={`/sessions/${encodeURIComponent(item.session_id)}`}>
                        <span className="item-main">
                          <span>{truncate(item.goal || "Untitled session", 130)}</span>
                          <small><TimeCell ts={item.started_ts} absolute /> · {item.is_origin ? "Source evidence" : "Retrieved there"}{item.retrieval_count ? ` · retrieved ${item.retrieval_count}×` : ""}</small>
                        </span>
                        <span className="item-end"><StatusBadge value={item.outcome || "not_reported"} /></span>
                      </Link>
                    </li>
                  ))}
                </ul>
              )}
              {data?.sessions_total > sessions.length && <p className="neutral">Showing the latest {sessions.length} of {data.sessions_total} sessions.</p>}
              {quotes.length > 0 && <h3>Source quotes</h3>}
              {quotes.map((item: any, index: number) => (
                <blockquote className="quote-card" key={index}>
                  {item.quote || item.event_content}
                  <small>
                    <TimeCell ts={item.event_ts || item.occurred_at || item.recorded_at || item.created_at} absolute />
                    {item.episode_session && <> · <Link to={`/sessions/${encodeURIComponent(item.episode_session)}`}>{item.session_goal ? truncate(item.session_goal, 80) : "Source session"}</Link></>}
                  </small>
                </blockquote>
              ))}
              {textless > 0 && <p className="neutral">{formatCount(textless)} evidence {textless === 1 ? "record has" : "records have"} no stored text.</p>}
            </DetailSection>
            <DetailSection id="memory-history" title="Use history">
              <p className="neutral">
                {retrieved
                  ? `Retrieved ${formatCount(retrieved)}×; ${formatCount(assessed)} assessed (${formatCount(Number(totals.unknown || 0))} unassessed). ${formatCount(Number(totals.irrelevant || 0))} irrelevant · ${formatCount(Number(totals.stale || 0))} stale · ${formatCount(Number(totals.wrong || 0))} wrong. Use and effect are agent-reported.`
                  : "Not exposed in a recorded retrieval yet."}
              </p>
              {data.retrievals?.length ? <ul className="link-list">
                {data.retrievals.map((item: any) => (
                  <li key={item.retrieval_id}>
                    <Link to={`/retrieval/${encodeURIComponent(item.retrieval_id)}`}>
                      <span className="item-main">
                        <span>{truncate(item.task_preview || "No query recorded", 130)}</span>
                        <small><TimeCell ts={item.created_at} absolute /> · {item.retrieval_type === "context" ? "Activation" : "Recall"}{item.feedback?.reason ? ` · “${truncate(item.feedback.reason, 100)}”` : ""}</small>
                      </span>
                      <span className="item-end"><StatusBadge value={item.feedback?.assessment || "unknown"} />{item.feedback?.effect && <StatusBadge value={item.feedback.effect} />}</span>
                    </Link>
                  </li>
                ))}
              </ul> : null}
              {data.retrievals?.length >= 50 && <p className="neutral">Showing the latest 50 retrieval occasions.</p>}
            </DetailSection>
            <DetailSection id="memory-connections" title={`Connections${connectionCount ? ` (${connectionCount})` : ""}`}>
              {connectionCount ? (
                <>
                  {(data.outgoing?.length || data.incoming?.length) ? <>
                    <h3>Related memories</h3>
                    <ul className="link-list">
                      {(data.outgoing || []).map((relation: any) => (
                        <ConnectionItem key={`out-${relation.dst_schema_id}-${relation.relation}`} id={relation.dst_schema_id} preview={relation.preview} status={relation.status} caption={`This memory ${relationLabel(relation.relation)} →`} />
                      ))}
                      {(data.incoming || []).map((relation: any) => (
                        <ConnectionItem key={`in-${relation.src_schema_id}-${relation.relation}`} id={relation.src_schema_id} preview={relation.preview} status={relation.status} caption={`← ${relationLabel(relation.relation)} this memory`} />
                      ))}
                    </ul></> : null}
                  {(data.coact_outgoing?.length || data.coact_incoming?.length) ? <>
                    <h3>Retrieved together</h3>
                    <ul className="link-list">
                      {[...(data.coact_outgoing || []), ...(data.coact_incoming || [])].map((relation: any) => {
                        const other = relation.src_schema_id === memory.schema_id ? relation.dst_schema_id : relation.src_schema_id;
                        return <ConnectionItem key={`coact-${relation.src_schema_id}-${relation.dst_schema_id}`} id={other} preview={relation.preview} status={relation.status} caption="Often returned in the same retrieval" />;
                      })}
                    </ul></> : null}
                </>
              ) : (
                <p className="neutral">No recorded connections.</p>
              )}
            </DetailSection>
            <DetailSection id="memory-technical" title="Technical details" collapsed>
              <KeyValueList items={[
                ["Importance (salience)", memory.salience == null ? EMPTY : `${Number(memory.salience).toFixed(1)} of 20`],
                ["Confidence", memory.confidence ?? EMPTY],
                [<>Generalization stage <DefinitionTooltip label="Generalization stage definition">{stageTooltip} Global/user scopes have explicit cross-context exceptions; stage is not an authorization guarantee.</DefinitionTooltip></>,
                  memory.generalization_stage == null || !stageNames[memory.generalization_stage] ? "Missing / unrecognized" : `${memory.generalization_stage} · ${stageNames[memory.generalization_stage]}`],
                ["Tags", memory.tags?.join(", ") || "None"],
                ["Facets", memory.facets?.length ? memory.facets.join(", ") : "None"],
                ["Relations", `${data.outgoing?.length || 0} outgoing · ${data.incoming?.length || 0} incoming`],
              ]} wide />
            </DetailSection>
            <div className="detail-footer">
              {allowActions ? (
                <button className="button danger-secondary" onClick={() => setConfirm(true)}>Delete memory…</button>
              ) : (
                <p className="notice compact">This dashboard was launched read-only. Delete is unavailable.</p>
              )}
            </div>
            {confirm && <DeleteConfirmation previewUrl={`/api/schemas/${memory.schema_id}/delete-preview`} onCancel={() => setConfirm(false)} onConfirm={remove} />}
          </>
        ) : <EmptyState title="Memory not found">This memory no longer exists or is not available in the current database.</EmptyState>
      }
    </Inspector>
  );
}

/** One record's reported use: used / retrieved delivery occasions, shown as a bar plus numbers. */
function UsageCell({ used, retrieved, assessed }: { used: unknown; retrieved: unknown; assessed?: unknown }) {
  const total = Number(retrieved || 0);
  if (!total) return dash;
  const usedCount = Number(used || 0);
  const pending = assessed !== undefined && Number(assessed || 0) === 0;
  const description = pending
    ? `${total} retrieved; usage feedback pending`
    : `${usedCount} of ${total} retrieved occasions reported used${assessed !== undefined ? ` · ${Number(assessed || 0)} assessed` : ""}`;
  return (
    <span className="usage-cell" title={description} aria-label={description}>
      <span className="retrieval-usage-track" aria-hidden="true"><span style={{ width: `${Math.min(100, usedCount / total * 100)}%` }} /></span>
      <span className={`usage-numbers${pending ? " muted" : ""}`}>{pending ? `${EMPTY}/${formatCount(total)}` : `${formatCount(usedCount)}/${formatCount(total)}`}</span>
    </span>
  );
}

function RetrievalUsage({ row }: { row: any }) {
  if (!row.exposed_count) return dash;
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
      <span className={`retrieval-usage-value${pending ? " muted" : ""}`}>{pending ? `${EMPTY}/${formatCount(total)}` : `${formatCount(used)}/${formatCount(total)}`}</span>
    </span>;
  })}</div>;
}

/** "3 mem · 1 proc" or a muted "Empty" when nothing was returned. */
function returnedSummary(row: any) {
  const memories = Number(row.memory_count || 0);
  const procedures = Number(row.procedure_count || 0);
  const total = Number(row.returned_count ?? row.exposed_count ?? memories + procedures);
  if (!total) return <span className="muted">Empty</span>;
  const parts = [memories ? `${formatCount(memories)} mem` : "", procedures ? `${formatCount(procedures)} proc` : ""].filter(Boolean);
  return <span title={`${total} returned`}>{parts.join(" · ") || formatCount(total)}</span>;
}

const feedbackStateOf = (row: any) => {
  const raw = String(row.feedback_state || "");
  if (raw === "complete") return "complete";
  if (raw === "partial" || raw === "incomplete") return "partial";
  return "none";
};

export function RetrievalPage({ location }: PageProps) {
  const retrievalSignalColumns = [
    ["used", "Usage"],
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
  const retrievalColumns = [
    { id: "when", label: "When", locked: true },
    { id: "task", label: "Task / query" },
    { id: "client", label: "Client" },
    { id: "type", label: "Type" },
    { id: "returned", label: "Returned" },
    { id: "used", label: "Usage" },
    { id: "feedback", label: "Feedback" },
    { id: "session", label: "Session" },
    { id: "scope", label: "Scope" },
    { id: "memories_retrieved", label: "Memories returned" },
    { id: "procedures_retrieved", label: "Procedures returned" },
    ...retrievalSignalColumns.filter(([key]) => key !== "used").map(([id, label]) => ({ id, label })),
  ];
  const [visibleColumns, setVisibleColumns, resetColumns] = useColumns("retrieval", ["when", "task", "client", "type", "returned", "used", "feedback", "session"]);
  const visible = (id: string) => visibleColumns.includes(id);
  const detailId = location.path.startsWith("/retrieval/")
    ? decodeURIComponent(location.path.split("/")[2])
    : "";
  const scope = param(location, "scope");
  const type = param(location, "type");
  const feedback = param(location, "feedback");
  const noMatch = param(location, "no_match");
  const returned = param(location, "returned") || (noMatch === "true" ? "none" : "");
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
    `/api/retrievals?scope=${encodeURIComponent(scope)}&type=${type}&feedback=${feedback}&no_match=${noMatch}&returned=${returned}&contains=${contains}&include_internal=${includeInternal}&sort=${sort}&dir=${dir}&page=${page}&per_page=50&q=${encodeURIComponent(debounced)}&from=${param(location, "from")}&to=${param(location, "to")}&hours=${param(location, "hours", "all")}`,
  );
  const rows = request.data?.retrievals || [];
  const summary = request.data?.summary || {};
  const pagination = request.data?.pagination || {
    page,
    per_page: 50,
    total: 0,
  };
  useClampPage("/retrieval", location, request.data?.pagination, rows.length);
  const clearAll = () => {
    setSearch("");
    setDebounced("");
    navigate("/retrieval");
  };
  const fromParam = param(location, "from");
  const retrievalChips = [
    debounced && { key: "q", label: `Search: “${truncate(debounced, 30)}”`, onRemove: () => { setSearch(""); setDebounced(""); } },
    type && { key: "type", label: `Type: ${type === "context" ? "Activation" : "Recall"}`, onRemove: () => updateParams("/retrieval", location, { type: undefined, page: 1 }) },
    returned && { key: "returned", label: returned === "none" ? "Returned: nothing" : "Returned: something", onRemove: () => updateParams("/retrieval", location, { returned: undefined, no_match: undefined, page: 1 }) },
    feedback && { key: "feedback", label: `Feedback: ${feedback === "incomplete" ? "partial" : feedback}`, onRemove: () => updateParams("/retrieval", location, { feedback: undefined, page: 1 }) },
    scope && { key: "scope", label: `Scope: ${shortScope(scope)}`, onRemove: () => updateParams("/retrieval", location, { scope: undefined, page: 1 }) },
    contains && { key: "contains", label: `Contains: ${contains}`, onRemove: () => updateParams("/retrieval", location, { contains: undefined, page: 1 }) },
    fromParam && { key: "from", label: `Observed since ${formatDay(fromParam)}`, onRemove: () => updateParams("/retrieval", location, { from: undefined, page: 1 }) },
    periodChip({ ...location, search: new URLSearchParams([...location.search].filter(([key]) => key !== "from")) }, () => updateParams("/retrieval", location, { hours: undefined, to: undefined, page: 1 })),
  ].filter(Boolean) as { key: string; label: string; onRemove: () => void }[];
  const retrievalFiltered = retrievalChips.length > 0;
  const th = (label: string, column?: string, extra: { numeric?: boolean; help?: string } = {}) => (
    <SortableTh label={label} column={column} sort={sort} dir={dir} onSort={changeSort} {...extra} />
  );
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
        controls={
          <div className="home-controls">
            <PeriodFilter location={location} path="/retrieval" hideLabel />
          </div>
        }
      />
      <div className="filter-bar">
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
          <select value={type} onChange={(e) => updateParams("/retrieval", location, { type: e.target.value, page: 1 })}>
            <option value="">All types</option>
            <option value="context">Activation</option>
            <option value="recall">Recall</option>
          </select>
        </label>
        <label>
          Returned
          <select value={returned} onChange={(e) => updateParams("/retrieval", location, { returned: e.target.value, no_match: undefined, page: 1 })}>
            <option value="">Any</option>
            <option value="any">Returned something</option>
            <option value="none">Returned nothing</option>
          </select>
        </label>
        <label>
          Feedback
          <select value={feedback === "incomplete" ? "partial" : feedback} onChange={(e) => updateParams("/retrieval", location, { feedback: e.target.value, page: 1 })}>
            <option value="">Any feedback state</option>
            <option value="complete">Complete</option>
            <option value="partial">Partial</option>
            <option value="none">None</option>
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
          <select value={contains} onChange={(e) => updateParams("/retrieval", location, { contains: e.target.value, page: 1 })}>
            <option value="">Any item</option>
            <option value="memory">Memory</option>
            <option value="procedure">Procedure</option>
          </select>
        </label>
        <DateFilter
          label="Observed since"
          seconds={fromParam}
          onChange={(value) => updateParams("/retrieval", location, { from: value, page: 1 })}
        />
      </div>
      <ActiveFilters chips={retrievalChips} onClear={clearAll} />
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
              value={formatCount(summary.retrievals || 0)}
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
          <div className="table-toolbar">
            <span>{formatCount(pagination.total)} {pagination.total === 1 ? "retrieval" : "retrievals"}</span>
            <ColumnsControl columns={retrievalColumns} visible={visibleColumns} onChange={setVisibleColumns} onReset={resetColumns} />
          </div>
          <TableFrame label="Retrieval results">
            <table className="retrieval-table">
              <thead>
                <tr>
                  {visible("when") && th("When", "when", { help: retrievalColumnHelp.when })}
                  {visible("task") && th("Task / query", "task", { help: retrievalColumnHelp.task })}
                  {visible("client") && th("Client")}
                  {visible("type") && th("Type", "type", { help: retrievalColumnHelp.type })}
                  {visible("returned") && th("Returned", "exposed", { help: retrievalColumnHelp.retrieved })}
                  {visible("used") && th("Usage", undefined, { help: retrievalColumnHelp.used })}
                  {visible("feedback") && th("Feedback", "feedback", { help: retrievalColumnHelp.feedback })}
                  {visible("session") && th("Session", "activity", { help: retrievalColumnHelp.session })}
                  {visible("scope") && th("Scope", "scope", { help: retrievalColumnHelp.scope })}
                  {visible("memories_retrieved") && th("Memories", "memories_retrieved", { numeric: true, help: retrievalColumnHelp.memories_retrieved })}
                  {visible("procedures_retrieved") && th("Procedures", "procedures_retrieved", { numeric: true, help: retrievalColumnHelp.procedures_retrieved })}
                  {retrievalSignalColumns.filter(([key]) => key !== "used" && visible(key)).map(([key, label]) => th(label, key, { numeric: true, help: retrievalColumnHelp[key] }))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row: any) => {
                  const href = `/retrieval/${encodeURIComponent(row.context_id)}${location.search.size ? `?${location.search}` : ""}`;
                  const open = () => openDetail(href, location);
                  const task = row.task_preview || "No query recorded";
                  const typeLabel = row.type_label || (row.retrieval_type === "context" ? "Activation" : "Recall");
                  return (
                    <tr key={row.context_id} {...rowAttrs(row.context_id, detailId, open)}>
                      {visible("when") && <td className="nowrap"><RowLink href={href} onOpen={open}><TimeCell ts={row.created_at} /></RowLink></td>}
                      {visible("task") && <td className="primary-cell" title={task}>
                        <ClampedText text={task} />
                      </td>}
                      {visible("client") && <td><ClientBadge client={row.client} /></td>}
                      {visible("type") && <td><TypeBadge type={String(typeLabel).toLowerCase()} label={typeLabel} /></td>}
                      {visible("returned") && <td className="nowrap">{returnedSummary(row)}</td>}
                      {visible("used") && <td className="retrieval-usage-cell"><RetrievalUsage row={row} /></td>}
                      {visible("feedback") && <td><StatusBadge value={feedbackStateOf(row)} /></td>}
                      {visible("session") && <td className="session-cell" onClick={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()}>
                        {row.session_id
                          ? <Link to={`/sessions/${encodeURIComponent(row.session_id)}#retrieval-${encodeURIComponent(row.context_id)}`}><ClampedText text={row.session_goal || truncate(row.session_id, 18)} lines={1} /></Link>
                          : <span className="muted">Standalone</span>}
                      </td>}
                      {visible("scope") && <td><ScopeChip scope={row.scope_id} /></td>}
                      {visible("memories_retrieved") && <td className="numeric">{formatCount(row.memory_count ?? 0)}</td>}
                      {visible("procedures_retrieved") && <td className="numeric">{formatCount(row.procedure_count ?? 0)}</td>}
                      {retrievalSignalColumns.filter(([key]) => key !== "used" && visible(key)).map(([key]) => (
                        <td className="numeric" key={key}>{formatCount(row.signal_counts?.[key] || 0)}</td>
                      ))}
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
          title={retrievalFiltered ? "No retrievals match these filters" : "No retrievals yet"}
          action={retrievalFiltered ? <button type="button" className="button secondary" onClick={clearAll}>Clear filters</button> : undefined}
        >
          {retrievalFiltered
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
  const request = useApi<Json>(`/api/retrievals/${encodeURIComponent(id)}`, { clearOnUrlChange: true });
  const data = request.data;
  const retrieval = data?.retrieval;
  const session = data?.session || retrieval?.session;
  const items: any[] = data?.items || [];
  const grouped = useMemo(
    () =>
      items.reduce((acc: Json, item: any) => {
        (acc[item.pathway_group || "Returned"] ||= []).push(item);
        return acc;
      }, {}),
    [data],
  );
  const accepted: any[] = (data?.feedback || []).filter((item: any) => item.status === "accepted");
  const rejected: any[] = (data?.feedback || []).filter((item: any) => item.status !== "accepted");
  const reasonFor = (item: any) => accepted.find((fb) => fb.target_id === item.memory_id && fb.reason)?.reason;
  const feedbackValue = retrieval ? (retrieval.feedback_complete ? "complete" : retrieval.feedback_observed ? "partial" : "none") : "none";
  const memories = items.filter((item) => item.memory_type !== "procedure" && item.memory_type !== "procedural_memory").length;
  const procedures = items.length - memories;
  const sessionId = retrieval?.session_id || session?.id;
  const title = retrieval ? retrieval.query || retrieval.goal || "Context exposure" : "";
  return (
    <Inspector
      title="Retrieval"
      id={id}
      label={retrieval ? truncate(title, 140) : undefined}
      hideLabel
      state={retrieval && <StatusBadge value={feedbackValue === "complete" ? "feedback_complete" : feedbackValue === "partial" ? "feedback_incomplete" : "feedback_unknown"} />}
      sections={[
        { id: "retrieval-request", label: "Request" },
        { id: "retrieval-context", label: "Returned context" },
        { id: "retrieval-technical", label: "Technical" },
      ]}
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
            <DetailSection id="retrieval-request" title="Request">
              <h2 className="detail-title">{title}</h2>
              <div className="summary-strip">
                <TypeBadge type={retrieval.retrieval_type === "context" ? "activation" : "recall"} label={retrieval.type_label} />
                <TimeCell ts={retrieval.created_at} absolute />
                <ClientBadge client={session?.client || data.session?.client} />
                <ScopeChip scope={retrieval.scope_id} />
                <span>{items.length ? <><strong>{memories}</strong> mem{procedures ? <> · <strong>{procedures}</strong> proc</> : null} returned</> : "Nothing returned"}</span>
                <span><DefinitionTooltip label="Usage definition">Reported used / returned. Grey includes items without a reported use; a dash means feedback is pending.</DefinitionTooltip> usage</span>
              </div>
              <div className="retrieval-detail-usage"><RetrievalUsage row={retrieval} /></div>
              {sessionId && (
                <KeyValueList items={[
                  ["Session", <Link to={`/sessions/${encodeURIComponent(sessionId)}#retrieval-${encodeURIComponent(retrieval.context_id || retrieval.retrieval_id || "")}`}>{session?.goal ? truncate(session.goal, 140) : "Open session"}</Link>],
                  [<>Task outcome <DefinitionTooltip label="Task outcome definition">The outcome the agent reported for the session this retrieval belonged to. It is not necessarily caused by Slowave's context.</DefinitionTooltip></>,
                    session?.outcome ? <StatusBadge value={session.outcome} /> : <StatusBadge value="not_reported" />],
                ]} />
              )}
              {!sessionId && <p className="neutral">Standalone retrieval, not part of a session.</p>}
            </DetailSection>
            <DetailSection id="retrieval-context" title="Returned context">
              {Object.entries(grouped).length ? (
                Object.entries(grouped).map(([group, groupItems]: [string, any]) => (
                  <div className="exposure-group" key={group}>
                    <h3>{group}</h3>
                    <ol className="ranked-list">
                      {groupItems.map((item: any, index: number) => {
                        const isProcedure = item.memory_type === "procedure" || item.memory_type === "procedural_memory";
                        const text = item.title || item.content_text || item.memory_id;
                        const href = isProcedure ? `/procedures/${encodeURIComponent(item.memory_id)}` : `/memory/${encodeURIComponent(item.memory_id)}`;
                        const why = item.pathway_explanation || item.reason;
                        const reason = reasonFor(item);
                        return (
                          <li className="ranked-item" key={item.memory_id}>
                            <span className="rank" aria-label={`Rank ${item.rank ?? index + 1}`}>{item.rank ?? index + 1}</span>
                            <div className="ranked-body">
                              <Link to={href} className="ranked-title">{truncate(text, 200)}</Link>
                              <div className="badge-stack">
                                <TypeBadge type={isProcedure ? "procedure" : "memory"} />
                                {item.status && <StatusBadge value={item.status} />}
                                <StatusBadge value={item.assessment || "unknown"} label={`Assessment: ${statusLabelOf(item.assessment || "unknown")}`} />
                                {item.effect && <StatusBadge value={item.effect} label={`Effect: ${statusLabelOf(item.effect)}`} />}
                              </div>
                              {why && <small className="why">Why: {why}</small>}
                              {reason && <small className="why">Agent note: “{reason}”</small>}
                              <details className="item-details">
                                <summary>Details</summary>
                                <KeyValueList wide items={[
                                  ["Shown as", item.content_source === "stored snapshot" ? "Text as delivered at the time" : "Current library text (the delivered text was not kept)"],
                                  item.status_source && item.status_source !== item.content_source ? ["State taken from", item.status_source] : null,
                                  item.current_status && item.current_status !== item.status ? ["State today", statusLabelOf(item.current_status)] : null,
                                  item.phase ? ["Phase", item.phase] : null,
                                  item.pathway ? ["Pathway", <code>{item.pathway}</code>] : null,
                                  ["ID", <IdChip value={item.memory_id} />],
                                ]} />
                              </details>
                            </div>
                          </li>
                        );
                      })}
                    </ol>
                  </div>
                ))
              ) : (
                <EmptyState title="Empty">
                  This retrieval returned no admitted memories or procedures.
                </EmptyState>
              )}
              {!accepted.length && items.length > 0 && <p className="neutral">No assessments recorded. Missing feedback is not negative evidence.</p>}
              {rejected.length > 0 && (
                <details className="advanced">
                  <summary>Rejected submissions ({rejected.length})</summary>
                  <ul className="link-list">
                    {rejected.map((item: any) => (
                      <li key={item.event_id}>
                        <div>
                          <span className="item-main">
                            <span>{truncate(item.target_title || item.target_id, 120)}</span>
                            <small>{item.rejection_reason || item.reason || "No rejection reason recorded."} · <TimeCell ts={item.created_at} absolute /></small>
                          </span>
                          <StatusBadge value={item.status} />
                        </div>
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </DetailSection>
            <DetailSection id="retrieval-technical" title="Technical details" collapsed>
              <KeyValueList wide items={[
                ["Retrieval ID", <IdChip value={id} head={12} tail={6} />],
                ["Response size", retrieval.response_chars ? `${formatCount(retrieval.response_chars)} characters` : "Not recorded"],
                ["Scope", retrieval.scope_id || "No scope"],
                sessionId ? ["Session ID", <IdChip value={String(sessionId)} head={12} tail={6} />] : null,
                retrieval.retrieval_type === "context" ? [<>Memory page limit <DefinitionTooltip label="Memory page limit definition">Maximum number of memories requested for each activation page. The actual returned count can be lower when fewer memories qualify or the response payload limit is reached.</DefinitionTooltip></>, retrieval.requested_page_size == null ? "Not recorded" : `${retrieval.requested_page_size} memories`] : null,
                ["Policy version", policyLabel(retrieval.retrieval_policy_version)],
                retrieval.policy_fallback
                  ? [<>Ranking fallback <DefinitionTooltip label="Ranking fallback definition">This retrieval was not ranked by the calibrated admission, so results come from the fallback rule. The reason is recorded with the retrieval.</DefinitionTooltip></>, fallbackLabel(retrieval.policy_fallback)]
                  : null,
                ["Lifecycle version", retrieval.lifecycle_version || "Not recorded"],
                ["Situation", retrieval.situation && Object.keys(retrieval.situation).length ? <StructuredValue value={retrieval.situation} /> : "Not recorded"],
                ["Requirements", <StructuredValue value={retrieval.requirements} />],
                ["Topics", <StructuredValue value={retrieval.topics} />],
                ["Entities", <StructuredValue value={retrieval.entities} />],
              ]} />
              <p className="neutral">The task text and returned memory text are stored locally and shown here.</p>
            </DetailSection>
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
  const procedureColumns = [
    { id: "procedure", label: "Procedure", locked: true },
    { id: "outcome", label: "Outcome" },
    { id: "verified", label: "Verified" },
    { id: "used", label: "Used" },
    { id: "effect", label: "Effect" },
    { id: "scope", label: "Scope" },
    { id: "created", label: "Created" },
    { id: "last_used", label: "Last used" },
    { id: "client", label: "Client" },
    { id: "source_activity", label: "Source task" },
    { id: "use_rate", label: "Use rate" },
    { id: "helped", label: "Helpful count" },
    { id: "no_effect", label: "No-effect count" },
    { id: "harmed", label: "Harmful count" },
    { id: "unknown", label: "Unknown count" },
    { id: "feedback_coverage", label: "Feedback coverage" },
    { id: "last_retrieved", label: "Last retrieved" },
  ];
  const [visibleColumns, setVisibleColumns, resetColumns] = useColumns("procedures", ["procedure", "outcome", "verified", "used", "effect", "scope", "created", "last_used"]);
  const visible = (id: string) => visibleColumns.includes(id);
  const request = useApi<Json>(
    `/api/procedural-memory?cohort=all&scope=${encodeURIComponent(scope)}&outcome=${outcome}&verification=${verification}&retrieved=${retrieved}&sort=${sort}&dir=${dir}&page=${page}&per_page=50&hours=all&saved_from=${param(location, "saved_from")}&saved_to=${param(location, "saved_to")}`,
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
  useClampPage("/procedures", location, request.data?.pagination, rows.length);
  const clearAll = () => navigate("/procedures");
  const savedFrom = param(location, "saved_from");
  const savedTo = param(location, "saved_to");
  const procedureChips = [
    outcome && { key: "outcome", label: `Outcome: ${outcome}`, onRemove: () => updateParams("/procedures", location, { outcome: undefined, page: 1 }) },
    verification && { key: "verification", label: `Verified: ${verification.replaceAll("_", " ")}`, onRemove: () => updateParams("/procedures", location, { verification: undefined, page: 1 }) },
    retrieved && { key: "retrieved", label: retrieved === "never" ? "Never retrieved" : "Retrieved", onRemove: () => updateParams("/procedures", location, { retrieved: undefined, page: 1 }) },
    scope && { key: "scope", label: `Scope: ${shortScope(scope)}`, onRemove: () => updateParams("/procedures", location, { scope: undefined, page: 1 }) },
    (savedFrom || savedTo) && { key: "saved", label: savedTo ? `Created: ${savedFrom ? formatDay(savedFrom) : "…"} – ${formatDay(savedTo)}` : `Created since ${formatDay(savedFrom)}`, onRemove: () => updateParams("/procedures", location, { saved_from: undefined, saved_to: undefined, page: 1 }) },
  ].filter(Boolean) as { key: string; label: string; onRemove: () => void }[];
  const procedureFiltered = procedureChips.length > 0;
  const th = (label: string, column?: string, extra: { numeric?: boolean; help?: string } = {}) => (
    <SortableTh label={label} column={column} sort={sort} dir={dir} onSort={changeSort} {...extra} />
  );
  return (
    <div className="page">
      <PageHeader
        title="Procedures"
        description="Execution-backed procedure records in the library; use Created since to narrow by date. Inventory totals describe the current library; a saved procedure is not necessarily a proven general playbook."
        updatedAt={request.updatedAt}
        refreshing={request.refreshing}
        onRefresh={request.reload}
      />
      <div className="filter-bar">
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
        <ScopeSelect value={scope} onChange={(next) => updateParams("/procedures", location, { scope: next, page: 1 })} />
        <DateFilter
          label="Created since"
          seconds={savedFrom}
          onChange={(value) => updateParams("/procedures", location, { saved_from: value, saved_to: undefined, page: 1 })}
        />
      </div>
      <ActiveFilters chips={procedureChips} onClear={clearAll} />
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
              title="Helpful rate · uses" className="metric-helpful"
              numerator={request.data.metrics?.procedure_helped}
              denominator={request.data.metrics?.effect_known}
              secondary={`Among uses with reported effects · ${request.data.metrics?.effect_unknown || 0} unknown effects`}
              tooltip="Used procedure delivery occasions reported helped divided by used delivery occasions with known effects. A procedure delivered in several requests contributes several occasions; these counts can exceed the distinct-procedure usage rate."
            />
            <RateMetricCard
              title="Harmful rate · uses" className="metric-harmful"
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
          <div className="table-toolbar">
            <span>{formatCount(pagination.total)} {pagination.total === 1 ? "procedure" : "procedures"}</span>
            <ColumnsControl columns={procedureColumns} visible={visibleColumns} onChange={setVisibleColumns} onReset={resetColumns} />
          </div>
          <TableFrame label="Procedure results">
            <table className="library-table">
              <thead>
                <tr>
                  {visible("procedure") && th("Procedure", "summary")}
                  {visible("outcome") && th("Outcome", undefined, { help: sharedColumnHelp.outcome })}
                  {visible("verified") && th("Verified", undefined, { help: sharedColumnHelp.verification })}
                  {visible("used") && th("Used", "retrieved", { help: sharedColumnHelp.usage })}
                  {visible("effect") && th("Effect", "helped", { help: sharedColumnHelp.effect })}
                  {visible("scope") && th("Scope", "scope", { help: sharedColumnHelp.scope })}
                  {visible("created") && th("Created", "recent")}
                  {visible("last_used") && th("Last used", "last_used")}
                  {visible("client") && th("Client")}
                  {visible("source_activity") && th("Source task", "source_activity")}
                  {visible("use_rate") && th("Use rate", "use_rate", { numeric: true })}
                  {visible("helped") && th("Helpful", "helped", { numeric: true })}
                  {visible("no_effect") && th("No effect", "no_effect", { numeric: true })}
                  {visible("harmed") && th("Harmful", "harmed", { numeric: true })}
                  {visible("unknown") && th("Unknown", "unknown", { numeric: true })}
                  {visible("feedback_coverage") && th("Feedback complete", "feedback_coverage", { numeric: true })}
                  {visible("last_retrieved") && th("Last retrieved", "last_retrieved")}
                </tr>
              </thead>
              <tbody>
                {rows.map((row: any) => {
                  const href = `/procedures/${encodeURIComponent(row.id)}${location.search.size ? `?${location.search}` : ""}`;
                  const open = () => openDetail(href, location);
                  const evidence = row.evidence || {};
                  const used = Number(evidence.used || 0);
                  const assessed = used + Number(evidence.not_used || 0);
                  const helped = Number(row.helped_count ?? evidence.helped ?? 0);
                  const noEffect = Number(row.no_effect_count ?? evidence.no_effect ?? 0);
                  const harmed = Number(row.harmed_count ?? evidence.harmed ?? 0);
                  const steps = Number(row.step_count ?? 0);
                  const outcomeValue = row.outcome;
                  const verifiedValue = row.verification_status;
                  return (
                    <tr key={row.id} {...rowAttrs(row.id, detailId, open)}>
                      {visible("procedure") && <td className="primary-cell">
                        <RowLink href={href} onOpen={open}><ClampedText text={row.summary || row.goal || "Untitled procedure"} /></RowLink>
                        {steps > 0 && <span className="meta">{formatCount(steps)} {steps === 1 ? "step" : "steps"}</span>}
                      </td>}
                      {visible("outcome") && <td>{outcomeValue ? <StatusBadge value={outcomeValue} /> : dash}</td>}
                      {visible("verified") && <td>{verifiedValue ? <StatusBadge value={verifiedValue} /> : dash}</td>}
                      {visible("used") && <td><UsageCell used={evidence.used} retrieved={evidence.retrieved} assessed={assessed} /></td>}
                      {visible("effect") && <td>
                        {!assessed
                          ? dash
                          : helped + noEffect + harmed === 0
                            ? <span className="muted">Unknown</span>
                            : <span className="effect-cell" title={`${helped} helped · ${noEffect} no effect · ${harmed} harmed`}>
                                <span className={helped ? "eff-helped" : "eff-none"}>{formatCount(helped)} helped</span>
                                <span className="eff-none">{formatCount(noEffect)} no effect</span>
                                <span className={harmed ? "eff-harmed" : "eff-none"}>{formatCount(harmed)} harmed</span>
                              </span>}
                      </td>}
                      {visible("scope") && <td><ScopeChip scope={row.scope_id} /></td>}
                      {visible("created") && <td><TimeCell ts={row.created_at} /></td>}
                      {visible("last_used") && <td><TimeCell ts={row.last_used_ts} /></td>}
                      {visible("client") && <td><ClientBadge client={row.client} /></td>}
                      {visible("source_activity") && <td className="session-cell" onClick={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()}>
                        {row.source_activity_id
                          ? <Link to={`/sessions/${encodeURIComponent(row.source_activity_id)}`}><ClampedText text={row.source_goal || truncate(row.source_activity_id, 18)} lines={1} /></Link>
                          : dash}
                      </td>}
                      {visible("use_rate") && <td className="numeric">{formatRate(evidence.used, evidence.retrieved)}</td>}
                      {visible("helped") && <td className="numeric">{formatCount(helped)}</td>}
                      {visible("no_effect") && <td className="numeric">{formatCount(noEffect)}</td>}
                      {visible("harmed") && <td className="numeric">{formatCount(harmed)}</td>}
                      {visible("unknown") && <td className="numeric">{formatCount(evidence.unknown || 0)}</td>}
                      {visible("feedback_coverage") && <td className="numeric">{formatCount(row.feedback_complete || 0)}</td>}
                      {visible("last_retrieved") && <td><TimeCell ts={row.last_retrieved_ts} /></td>}
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
          title={procedureFiltered ? "No procedures match these filters" : "No procedures created yet"}
          action={procedureFiltered ? <button type="button" className="button secondary" onClick={clearAll}>Clear filters</button> : undefined}
        >
          {procedureFiltered
            ? "Clear or broaden the filters to inspect other procedure records."
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

/** Structured values (applicability context, historical evidence) as chips and nested lists; never raw JSON. */
function StructuredValue({ value }: { value: unknown }): ReactNode {
  if (value === null || value === undefined || value === "") return <span className="muted">{EMPTY}</span>;
  if (Array.isArray(value)) {
    if (!value.length) return <span className="muted">None</span>;
    if (value.every((item) => item === null || typeof item !== "object")) {
      return <ul className="chip-list">{value.map((item, index) => <li key={index}>{String(item)}</li>)}</ul>;
    }
    return <ul className="nested-list">{value.map((item, index) => <li key={index}><StructuredValue value={item} /></li>)}</ul>;
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (!entries.length) return <span className="muted">None</span>;
    return (
      <dl className="key-values nested">
        {entries.map(([key, item]) => (
          <Fragment key={key}>
            <dt>{key.replaceAll("_", " ")}</dt>
            <dd><StructuredValue value={item} /></dd>
          </Fragment>
        ))}
      </dl>
    );
  }
  if (typeof value === "boolean") return <>{value ? "Yes" : "No"}</>;
  return <>{String(value)}</>;
}

function ProcedureDetail({ id, onClose, onDeleted }: {
  id: string;
  onClose: () => void;
  onDeleted?: () => Promise<void>;
}) {
  const request = useApi<Json>(`/api/procedures/${encodeURIComponent(id)}`, { clearOnUrlChange: true });
  const [confirm, setConfirm] = useState(false);
  const [actionError, setActionError] = useState("");
  const p = request.data?.procedure;
  const source = p?.source_session;
  const feedbackByRetrieval = new Map<string, any>();
  for (const item of p?.feedback || []) {
    if (item.status === "accepted" && !feedbackByRetrieval.has(item.retrieval_id)) {
      feedbackByRetrieval.set(item.retrieval_id, item);
    }
  }
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
  const evidence = p?.evidence || {};
  const totals = p?.use_history_totals || {};
  const retrieved = Number(totals.retrieved || 0);
  const used = Number(totals.used || 0);
  const assessed = Number(totals.assessed || 0);
  const helped = Number(evidence.helped || 0);
  const noEffect = Number(evidence.no_effect || 0);
  const harmed = Number(evidence.harmed || 0);
  const verification = source?.verification;
  const rejected = (p?.feedback || []).filter((item: any) => item.status !== "accepted");
  const [showRaw, setShowRaw] = useState(false);
  return (
    <Inspector
      title="Procedure"
      id={id}
      label={p ? truncate(p.summary, 140) : undefined}
      hideLabel
      state={p && <StatusBadge value={p.outcome} label={`Source task: ${statusLabelOf(p.outcome)}${verification?.status ? ` · ${statusLabelOf(verification.status)}` : ""}`} />}
      sections={[
        { id: "procedure-method", label: "Steps" },
        { id: "procedure-history", label: "Use history" },
        { id: "procedure-source", label: "Source task" },
        { id: "procedure-technical", label: "Technical" },
      ]}
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
            <DetailSection id="procedure-method" title="Procedure">
              <h2 className="detail-title">{p.summary}</h2>
              <div className="summary-strip">
                <ScopeChip scope={p.scope_id} />
                <ClientBadge client={p.client} />
                <span>created <TimeCell ts={p.created_at} absolute /></span>
                <span>retrieved <strong>{formatCount(retrieved)}×</strong></span>
                {verification?.summary && <span title={verification.summary}>verification: {truncate(verification.summary, 120)}</span>}
              </div>
              {p.outcome !== "success" && (
                <div className="banner" role="note">
                  <Icon name="warning" />
                  <div>
                    <strong>{p.outcome === "failure" ? "The source task reported failure" : "The source task reported a partial outcome"}</strong>
                    <span>Review the caveats and check applicability before applying these steps.</span>
                  </div>
                </div>
              )}
              <h3>Steps</h3>
              {p.steps?.length ? <ol className="procedure-steps">{p.steps.map((step: any, index: number) => <li key={index}>{typeof step === "string" ? step : step?.summary || <StructuredValue value={step} />}</li>)}</ol> : <p className="neutral">No steps recorded.</p>}
              <h3>Caveats</h3>
              {p.caveats?.length ? <ul className="nested-list">{p.caveats.map((item: string) => <li key={item}>{item}</li>)}</ul> : <p className="neutral">No caveats were recorded with this procedure.</p>}
              <h3>When this applies</h3>
              {Object.keys(p.context || {}).length ? <StructuredValue value={p.context} /> : <p className="neutral">No applicability context recorded.</p>}
            </DetailSection>
            <DetailSection id="procedure-history" title="Use history">
              <p className="neutral">
                {retrieved
                  ? `Retrieved ${formatCount(retrieved)}×; ${formatCount(assessed)} assessed, ${formatCount(used)} reported used${assessed ? ` (${formatRatePercent(used, assessed)})` : ""}. Last retrieved ${formatDate(totals.last_retrieved)}. Use and effect are agent-reported.`
                  : "Not retrieved yet. This is neutral; it is not evidence against the procedure."}
              </p>
              <div className="tile-row" aria-label="Reported effect among uses">
                <div className="tile ok"><strong>{formatCount(helped)}</strong><span>Helped</span></div>
                <div className="tile"><strong>{formatCount(noEffect)}</strong><span>No effect</span></div>
                <div className={`tile${harmed ? " bad" : ""}`}><strong>{formatCount(harmed)}</strong><span>Harmed</span></div>
              </div>
              {Number(evidence.unknown_effect || 0) > 0 && <p className="neutral">{formatCount(evidence.unknown_effect)} uses had no effect assessment.</p>}
              {p.retrievals?.length ? <ul className="link-list">{p.retrievals.map((item: any) => {
                const feedback = feedbackByRetrieval.get(item.retrieval_id);
                return (
                  <li key={item.retrieval_id}>
                    <Link to={`/retrieval/${encodeURIComponent(item.retrieval_id)}`}>
                      <span className="item-main">
                        <span>{truncate(item.task_preview || item.query || "No query recorded", 130)}</span>
                        <small><TimeCell ts={item.created_at} absolute /> · {item.retrieval_type === "context" ? "Activation" : "Recall"}{feedback?.reason ? ` · “${truncate(feedback.reason, 100)}”` : ""}</small>
                      </span>
                      <span className="item-end"><StatusBadge value={feedback?.assessment || "unknown"} />{feedback?.effect && <StatusBadge value={feedback.effect} />}</span>
                    </Link>
                  </li>
                );
              })}</ul> : null}
              {rejected.length > 0 && <details className="advanced"><summary>Rejected submissions ({rejected.length})</summary><ul className="link-list">{rejected.map((item: any) => <li key={item.event_id}><div><span className="item-main"><span>{item.assessment || "Submission"} · {item.status}</span><small>{item.rejection_reason || item.reason || "No rejection reason recorded."}</small></span><Link to={`/retrieval/${encodeURIComponent(item.retrieval_id)}`}>Open retrieval</Link></div></li>)}</ul></details>}
            </DetailSection>
            <DetailSection id="procedure-source" title="Source task">
              <KeyValueList items={[
                ["Goal", p.goal || "No goal recorded"],
                ["Session", source?.id ? <Link to={`/sessions/${encodeURIComponent(source.id)}`}>{source.goal ? truncate(source.goal, 140) : "Open source session"}</Link> : "Unavailable"],
                ["Outcome", <><StatusBadge value={p.outcome} />{p.outcome_summary && <p className="inline-prose">{p.outcome_summary}</p>}</>],
                ["Verification", <><StatusBadge value={verification?.status || "unknown"} />{verification?.summary && <p className="inline-prose">{verification.summary}</p>}</>],
              ]} />
            </DetailSection>
            <DetailSection id="procedure-technical" title="Technical details" collapsed>
              <KeyValueList wide items={[
                ["Procedure ID", <IdChip value={p.id} head={12} tail={6} />],
                ["Lifecycle version", source?.lifecycle_version || "Not recorded"],
                ...Object.entries(p.historical_evidence || {}).map(([key, value]) => [key.replaceAll("_", " "), <StructuredValue value={value} />] as [ReactNode, ReactNode]),
              ]} />
              <button type="button" className="link-button" onClick={() => setShowRaw((v) => !v)}>{showRaw ? "Hide raw" : "View raw"}</button>
              {showRaw && <CodeBlock label="Raw procedure record">{JSON.stringify(p, null, 2)}</CodeBlock>}
            </DetailSection>
            <div className="detail-footer">
              {allowActions ? (
                <button className="button danger-secondary" onClick={() => setConfirm(true)}>Delete procedure…</button>
              ) : (
                <p className="notice compact">This dashboard was launched read-only. Delete is unavailable.</p>
              )}
            </div>
            {confirm && <DeleteConfirmation previewUrl={`/api/procedures/${encodeURIComponent(p.id)}/delete-preview`} onCancel={() => setConfirm(false)} onConfirm={remove} />}
          </>
        ) : <EmptyState title="Procedure not found">This procedure no longer exists or is not available in the current database.</EmptyState>
      }
    </Inspector>
  );
}

export function ActivityPage({ location }: PageProps) {
  const detailId = location.path.startsWith("/sessions/") || location.path.startsWith("/activity/")
    ? decodeURIComponent(location.path.split("/")[2])
    : "";
  const page = pageNumber(location);
  const sort = param(location, "sort", "started");
  const dir = param(location, "dir", "desc") as "asc" | "desc";
  const [search, setSearch] = useState("");
  const [debounced, setDebounced] = useState("");
  const activityColumns = [
    { id: "started", label: "Started", locked: true },
    { id: "task", label: "Goal" },
    { id: "client", label: "Client", description: "The coding agent client that ran this session." },
    { id: "outcome", label: "Outcome" },
    { id: "status", label: "Status", description: "Running, ended, or idle-closed (closed by the idle reaper without a commit)." },
    { id: "feedback", label: "Feedback", description: "Whether every returned item received an explicit assessment: complete, partial, none, or still open." },
    { id: "duration", label: "Duration", description: "Elapsed time between the session start and close." },
    { id: "retrievals", label: "Retrievals", description: "Distinct retrieval events recorded in this session." },
    { id: "saved", label: "Saved", description: "Memories saved with remember during this session." },
    { id: "used", label: "Used", description: "Returned items the agent reported using." },
    { id: "scope", label: "Scope (separate column)" },
    { id: "memories_touched", label: "Memories touched", description: "Distinct memory records with evidence linked to this session." },
    { id: "procedure", label: "Procedure record", description: "Whether closing this session created a reusable procedure." },
    { id: "retrieved_memories", label: "Retrieved memories" },
    { id: "retrieved_procedures", label: "Retrieved procedures" },
    { id: "episodes", label: "Episodes formed" },
    { id: "timeline_events", label: "Timeline event count" },
    { id: "last_event", label: "Last event" },
    { id: "verification", label: "Verification" },
  ];
  const [visibleColumns, setVisibleColumns, resetColumns] = useColumns("sessions", ["started", "task", "client", "outcome", "status", "feedback", "duration", "retrievals", "saved", "used"]);
  const visible = (id: string) => visibleColumns.includes(id);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(search), 250);
    return () => clearTimeout(timer);
  }, [search]);
  const preset = param(location, "preset");
  const activityFilters = `scope=${encodeURIComponent(param(location, "scope"))}&outcome=${param(location, "outcome")}&feedback=${param(location, "feedback")}&preset=${preset}&agent=${encodeURIComponent(param(location, "agent"))}&continuity=${encodeURIComponent(param(location, "continuity"))}&lane=${param(location, "lane")}&hours=${param(location, "hours", "all")}&from=${param(location, "from")}&to=${param(location, "to")}&q=${encodeURIComponent(debounced)}`;
  const rowsRequest = useApi<Json>(`/api/sessions?${activityFilters}&sort=${sort}&dir=${dir}&page=${page}&per_page=50&include_summary=false`);
  const summaryRequest = useApi<Json>(`/api/sessions?${activityFilters}&summary_only=true`);
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
    updateParams("/sessions", location, {
      sort: column,
      dir: sort === column && dir === "desc" ? "asc" : "desc",
      page: 1,
    });
  useClampPage("/sessions", location, rowsRequest.data?.pagination, rows.length);
  const clearAll = () => {
    setSearch("");
    setDebounced("");
    navigate("/sessions");
  };
  const fromParam = param(location, "from");
  const agentParam = param(location, "agent") || param(location, "client");
  const [knownClients, setKnownClients] = useState<Record<string, string>>({});
  useEffect(() => {
    if (!rows.length) return;
    setKnownClients((old) => {
      const next = { ...old };
      for (const row of rows) if (row.client?.key) next[row.client.key] = row.client.name || row.client.key;
      return next;
    });
  }, [rows]);
  const sessionChips = [
    debounced && { key: "q", label: `Search: “${truncate(debounced, 30)}”`, onRemove: () => { setSearch(""); setDebounced(""); } },
    param(location, "outcome") && { key: "outcome", label: `Outcome: ${param(location, "outcome") === "unreported" ? "not reported" : param(location, "outcome")}`, onRemove: () => updateParams("/sessions", location, { outcome: undefined, page: 1 }) },
    agentParam && { key: "agent", label: `Client: ${knownClients[agentParam] || agentParam}`, onRemove: () => updateParams("/sessions", location, { agent: undefined, client: undefined, page: 1 }) },
    param(location, "feedback") && { key: "feedback", label: `Feedback: ${param(location, "feedback")}`, onRemove: () => updateParams("/sessions", location, { feedback: undefined, page: 1 }) },
    param(location, "scope") && { key: "scope", label: `Scope: ${shortScope(param(location, "scope"))}`, onRemove: () => updateParams("/sessions", location, { scope: undefined, page: 1 }) },
    param(location, "lane") && { key: "lane", label: `Session lane: ${param(location, "lane").replaceAll("_", " ")}`, onRemove: () => updateParams("/sessions", location, { lane: undefined, page: 1 }) },
    param(location, "continuity") && { key: "continuity", label: "Same conversation", onRemove: () => updateParams("/sessions", location, { continuity: undefined, page: 1 }) },
    fromParam && { key: "from", label: `Started since ${formatDay(fromParam)}`, onRemove: () => updateParams("/sessions", location, { from: undefined, page: 1 }) },
    periodChip({ ...location, search: new URLSearchParams([...location.search].filter(([key]) => key !== "from")) }, () => updateParams("/sessions", location, { hours: undefined, to: undefined, page: 1 })),
  ].filter(Boolean) as { key: string; label: string; onRemove: () => void }[];
  const sessionFiltered = sessionChips.length > 0 || Boolean(preset);
  const th = (label: string, column?: string, extra: { numeric?: boolean; help?: string } = {}) => (
    <SortableTh label={label} column={column} sort={sort} dir={dir} onSort={changeSort} {...extra} />
  );
  const presets = [
    ["", "All"],
    ["problems", "Failed or partial"],
    ["no_feedback", "No feedback"],
    ["idle_closed", "Idle-closed"],
  ];
  return (
    <div className="page">
      <PageHeader
        title="Sessions"
        description="Every Slowave session with its ordered calls, and the memories and procedures linked to it."
        updatedAt={rowsRequest.updatedAt || summaryRequest.updatedAt}
        refreshing={rowsRequest.refreshing || summaryRequest.refreshing}
        onRefresh={reloadActivity}
        hideAutoRefresh
        controls={
          <div className="home-controls">
            <PeriodFilter location={location} path="/sessions" hideLabel />
          </div>
        }
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
          <select value={param(location, "outcome")} onChange={(e) => updateParams("/sessions", location, { outcome: e.target.value, page: 1 })}>
            <option value="">Any outcome</option>
            <option value="success">Success</option>
            <option value="partial">Partial</option>
            <option value="failure">Failure</option>
            <option value="unreported">Not reported</option>
          </select>
        </label>
        <label>
          Client
          <select value={agentParam} onChange={(e) => updateParams("/sessions", location, { agent: e.target.value, client: undefined, page: 1 })}>
            <option value="">Any client</option>
            {agentParam && !knownClients[agentParam] && <option value={agentParam}>{agentParam}</option>}
            {Object.entries(knownClients).map(([key, name]) => <option key={key} value={key}>{name}</option>)}
          </select>
        </label>
        <DateFilter
          label="Started since"
          seconds={fromParam}
          onChange={(value) => updateParams("/sessions", location, { from: value, page: 1 })}
        />
      </div>
      <div className="state-chips" role="group" aria-label="Session presets">
        {presets.map(([value, label]) => (
          <button key={label} type="button" className="state-chip" aria-pressed={preset === value} onClick={() => updateParams("/sessions", location, { preset: value || undefined, page: 1 })}>{label}</button>
        ))}
      </div>
      <ActiveFilters chips={sessionChips} onClear={clearAll} />
      {summaryRequest.loading && !summaryRequest.data ? (
        <MetricCardsSkeleton count={3} />
      ) : summaryRequest.error && !summaryRequest.data ? (
        <ErrorState
          title="Session summary unavailable"
          error={summaryRequest.error}
          retry={reloadActivity}
        />
      ) : summaryRequest.data ? (
        <>
          <div
            className="metric-card-grid activity-metric-card-grid"
            aria-label="Session summary"
          >
            <MetricCard
              title="Sessions"
              value={Number(
                summaryRequest.data?.summary?.eligible || 0,
              ).toLocaleString()}
              secondary={`${summaryRequest.data?.summary?.closed || 0} Ended · ${summaryRequest.data?.summary?.pending || 0} Ongoing`}
              tooltip="Sessions started in the period. A session is one task from activate to commit; several may belong to one conversation."
            />
            <RateMetricCard
              title="Feedback completion"
              numerator={summaryRequest.data?.summary?.complete}
              denominator={summaryRequest.data?.summary?.closure_eligible}
              secondary={`${summaryRequest.data?.summary?.closure_unclassified || 0} older untracked ended records excluded`}
              tooltip="Ended sessions with finalized item accountability divided by ended sessions with supported feedback tracking. Ongoing sessions are excluded."
            />
            <RateMetricCard
              title="Context usage rate"
              numerator={summaryRequest.data?.summary?.context_use}
              denominator={summaryRequest.data?.summary?.context_denominator}
              secondary="Among sessions receiving supported context"
              tooltip="Sessions with a delivered item reported used divided by sessions receiving supported context. Each session counts once."
            />
          </div>
          <div
            className="metric-card-grid distribution-card-grid outcome-card-grid"
            aria-label="Session reported outcomes"
          >
            <ReportedOutcomesCard
              summary={summaryRequest.data?.summary || {}}
            />
          </div>
        </>
      ) : (
        <EmptyState title="No session summary available">
          Summary metrics will appear when the sessions service returns a
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
        <ErrorState title="Session results unavailable" error={rowsRequest.error} retry={reloadActivity} />
      ) : rows.length ? (
        <>
          <div className="table-toolbar">
            <span>{formatCount(pagination.total)} {pagination.total === 1 ? "session" : "sessions"}</span>
            <ColumnsControl columns={activityColumns} visible={visibleColumns} onChange={setVisibleColumns} onReset={resetColumns} />
          </div>
          <TableFrame label="Session results">
            <table className="activity-table session-accent">
              <thead>
                <tr>
                  {visible("started") && th("Started", "started")}
                  {visible("task") && th("Goal", "task")}
                  {visible("client") && th("Client")}
                  {visible("outcome") && th("Outcome", "outcome", { help: sharedColumnHelp.outcome })}
                  {visible("status") && th("Status", undefined, { help: "Running: no end recorded yet. Ended: closed with a commit. Idle-closed: closed after inactivity without a commit." })}
                  {visible("feedback") && th("Feedback", "closure", { help: "Whether every returned item received an explicit assessment: complete, partial, none, or still open." })}
                  {visible("duration") && th("Duration", "duration", { numeric: true })}
                  {visible("retrievals") && th("Retrievals", "retrievals", { numeric: true, help: "Distinct retrieval events recorded in this session." })}
                  {visible("saved") && th("Saved", undefined, { numeric: true, help: "Memories saved with remember during this session." })}
                  {visible("used") && th("Used", undefined, { numeric: true, help: "Returned items the agent reported using." })}
                  {visible("scope") && th("Scope", "scope", { help: sharedColumnHelp.scope })}
                  {visible("memories_touched") && th("Memories touched", "memories_touched", { numeric: true })}
                  {visible("procedure") && th("Procedure", "procedure")}
                  {visible("retrieved_memories") && th("Retrieved memories", "retrieved_memories", { numeric: true })}
                  {visible("retrieved_procedures") && th("Retrieved procedures", "retrieved_procedures", { numeric: true })}
                  {visible("episodes") && th("Episodes", "episodes", { numeric: true })}
                  {visible("timeline_events") && th("Timeline events", "timeline_events", { numeric: true })}
                  {visible("last_event") && th("Last event", "last_event")}
                  {visible("verification") && th("Verification", "verification")}
                </tr>
              </thead>
              <tbody>
                {rows.map((row: any) => {
                  const href = `/sessions/${encodeURIComponent(row.id)}${location.search.size ? `?${location.search}` : ""}`;
                  const open = () => openDetail(href, location);
                  const status = row.status || (row.ended_ts ? "ended" : null);
                  const feedbackState = row.feedback_state || (row.feedback_status === "complete" ? "complete" : row.feedback_status === "incomplete" ? "partial" : row.feedback_status === "pending" ? "pending_open" : "none");
                  const outcomeKnown = ["success", "partial", "failure"].includes(String(row.outcome));
                  return (
                    <tr key={row.id} {...rowAttrs(row.id, detailId, open)}>
                      {visible("started") && <td className="nowrap"><TimeCell ts={row.started_ts} /></td>}
                      {visible("task") && <td className="primary-cell">
                        <RowLink href={href} onOpen={open}><ClampedText text={row.goal_preview || "Untitled session"} /></RowLink>
                        {row.scope_id && !visible("scope") && <span className="meta"><ScopeChip scope={row.scope_id} /></span>}
                      </td>}
                      {visible("client") && <td><ClientBadge client={row.client} /></td>}
                      {visible("outcome") && <td>{outcomeKnown ? <StatusBadge value={row.outcome} /> : status === "running" ? dash : <StatusBadge value="not_reported" />}</td>}
                      {visible("status") && <td>{status ? <StatusBadge value={status} /> : dash}</td>}
                      {visible("feedback") && <td><StatusBadge value={feedbackState} /></td>}
                      {visible("duration") && <td className="numeric">{row.duration_s != null ? formatDuration(row.duration_s) : dash}</td>}
                      {visible("retrievals") && <td className="numeric">{formatCount(row.retrieval_count || 0)}</td>}
                      {visible("saved") && <td className="numeric">{formatCount(row.saved_count ?? 0)}</td>}
                      {visible("used") && <td className="numeric" title={row.assessed_count != null ? `${row.used_count ?? 0} used of ${row.assessed_count} assessed · ${row.returned_count ?? 0} returned` : undefined}>{formatCount(row.used_count ?? 0)}</td>}
                      {visible("scope") && <td><ScopeChip scope={row.scope_id} /></td>}
                      {visible("memories_touched") && <td className="numeric">{formatCount(row.memory_count || 0)}</td>}
                      {visible("procedure") && <td><StatusBadge value={row.procedure_state || "none"} /></td>}
                      {visible("retrieved_memories") && <td className="numeric">{formatCount(row.retrieved_memory_count || 0)}</td>}
                      {visible("retrieved_procedures") && <td className="numeric">{formatCount(row.retrieved_procedure_count || 0)}</td>}
                      {visible("episodes") && <td className="numeric">{formatCount(row.episode_count || 0)}</td>}
                      {visible("timeline_events") && <td className="numeric">{formatCount(row.timeline_event_count || 0)}</td>}
                      {visible("last_event") && <td><TimeCell ts={row.last_event_ts} /></td>}
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
              updateParams("/sessions", location, { page: next })
            }
          />
        </>
      ) : (
        <EmptyState
          title={sessionFiltered ? "No session matches these filters" : "No session yet"}
          action={sessionFiltered ? <button type="button" className="button secondary" onClick={clearAll}>Clear filters</button> : undefined}
        >
          {sessionFiltered
            ? "Clear or broaden the filters to inspect other recorded task segments."
            : "A session appears when an agent uses Slowave. A zero count is normal on a new installation."}
        </EmptyState>
      )}
      {detailId && (
        <ActivityDetail
          id={detailId}
          onClose={() => closeDetail("/sessions", location)}
        />
      )}
    </div>
  );
}

const CALL_LABELS: Record<string, string> = {
  activate: "Activate", recall: "Recall", recall_page: "Recall page", remember: "Remember",
  feedback: "Feedback", commit: "Commit", session_reaped: "Idle close", session_start: "Start",
};
const CALL_ORDER = ["activate", "recall", "recall_page", "remember", "feedback", "commit", "session_reaped", "session_start"];
const EXPOSURE_KINDS = ["activate", "recall", "recall_page"];
const CALL_PAGE = 200;
const GAP_SECONDS = 300;

function itemHref(item: any) {
  return item.kind === "procedure"
    ? `/procedures/${encodeURIComponent(item.id)}`
    : `/memory/${encodeURIComponent(item.id)}`;
}

const itemText = (item: any) => String(item?.text ?? item?.title ?? item?.content_text ?? item?.preview ?? "").trim();
const plural = (count: number, one: string, many = `${one}s`) => `${formatCount(count)} ${count === 1 ? one : many}`;

function formatOffset(seconds: number) {
  return `+${formatDuration(seconds)}`;
}

function capitalize(value: string) {
  const text = String(value || "").replaceAll("_", " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function SessionCallLog({ sessionId, data }: { sessionId: string; data: Json }) {
  const [extra, setExtra] = useState<any[]>([]);
  const [kind, setKind] = useState("");
  const [filtered, setFiltered] = useState<{ kind: string; events: any[]; total: number } | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [moreError, setMoreError] = useState("");
  const [highlight, setHighlight] = useState("");
  const [copied, setCopied] = useState("");
  const all: any[] = [...(data.call_log || []), ...extra];
  const total: number = data.call_log_total || 0;
  const kindCounts: Record<string, number> = data.call_log_kinds || {};
  const shown: any[] = kind && filtered?.kind === kind ? filtered.events : kind ? [] : all;
  const shownTotal = kind ? (filtered?.kind === kind ? filtered.total : kindCounts[kind] || 0) : total;
  const start = Number(all[0]?.ts || 0);
  const sessionOutcome = data.session?.outcome;
  const idleClosed = data.session?.status === "idle_closed";

  const fetchPage = async (afterId: number, kinds: string) => {
    const query = `after_id=${afterId}&limit=${CALL_PAGE}${kinds ? `&kinds=${encodeURIComponent(kinds)}` : ""}`;
    return api<Json>(`/api/sessions/${encodeURIComponent(sessionId)}/events?${query}`);
  };

  // Filtering happens on the server so chip counts and results cover the whole session.
  useEffect(() => {
    if (!kind) return;
    let cancelled = false;
    fetchPage(0, kind)
      .then((page) => { if (!cancelled) setFiltered({ kind, events: page.events || [], total: page.filtered_total || 0 }); })
      .catch((error) => { if (!cancelled) setMoreError(error instanceof Error ? error.message : "Could not filter calls"); });
    return () => { cancelled = true; };
  }, [kind, sessionId]);

  const exposureByRetrieval = useMemo(() => {
    const map = new Map<string, any>();
    for (const item of all) if (EXPOSURE_KINDS.includes(item.kind) && item.retrieval_id) map.set(item.retrieval_id, item);
    return map;
  }, [all.length]);

  const focusCall = (id: number) => {
    setHighlight(`call-${id}`);
    window.history.replaceState(window.history.state, "", `${window.location.pathname}${window.location.search}#call-${id}`);
    window.requestAnimationFrame(() => document.getElementById(`call-${id}`)?.scrollIntoView({ block: "center", behavior: "smooth" }));
  };

  // Deep links (#call-<event id> / #retrieval-<retrieval id>): keep loading pages until the target is found.
  useEffect(() => {
    const hash = decodeURIComponent(window.location.hash.replace(/^#/, ""));
    if (!hash || kind) return;
    const find = (list: any[]) =>
      hash.startsWith("call-")
        ? list.find((item) => `call-${item.id}` === hash)
        : hash.startsWith("retrieval-")
          ? list.find((item) => item.retrieval_id === hash.slice("retrieval-".length) && EXPOSURE_KINDS.includes(item.kind))
          : undefined;
    let cancelled = false;
    (async () => {
      let list = all;
      let target = find(list);
      let guard = 0;
      while (!target && list.length < total && guard++ < 25 && !cancelled) {
        const page = await fetchPage(list[list.length - 1].id, "");
        const more: any[] = page.events || [];
        if (!more.length) break;
        list = [...list, ...more];
        target = find(list);
        if (!cancelled) setExtra((items) => [...items, ...more.filter((m) => !items.some((i) => i.id === m.id))]);
      }
      if (!target || cancelled) return;
      setHighlight(`call-${target.id}`);
      window.requestAnimationFrame(() => document.getElementById(`call-${target.id}`)?.scrollIntoView({ block: "center" }));
    })().catch(() => undefined);
    return () => { cancelled = true; };
  }, [sessionId, all.length === 0, window.location.hash]);

  const loadMore = async () => {
    const last = shown[shown.length - 1];
    if (!last) return;
    setLoadingMore(true);
    setMoreError("");
    try {
      const page = await fetchPage(last.id, kind);
      if (kind) setFiltered((current) => current && { ...current, events: [...current.events, ...(page.events || [])] });
      else setExtra((items) => [...items, ...(page.events || [])]);
    } catch (error) {
      setMoreError(error instanceof Error ? error.message : "Could not load more calls");
    } finally {
      setLoadingMore(false);
    }
  };

  const copyLink = (rowId: string) => {
    try {
      void navigator.clipboard?.writeText(`${window.location.origin}/sessions/${encodeURIComponent(sessionId)}#${rowId}`);
      setCopied(rowId);
    } catch {
      setCopied("");
    }
    window.setTimeout(() => setCopied((current) => (current === rowId ? "" : current)), 1500);
  };

  const chipKinds = Object.entries(kindCounts)
    .filter(([, count]) => Number(count) > 0)
    .sort(([a], [b]) => (CALL_ORDER.indexOf(a) + 1 || 99) - (CALL_ORDER.indexOf(b) + 1 || 99));
  const rows: ReactNode[] = [];
  shown.forEach((item: any, position: number) => {
    const previous = shown[position - 1];
    if (previous && Number(item.ts) - Number(previous.ts) > GAP_SECONDS) {
      rows.push(<li key={`gap-${item.id}`} className="call-gap" aria-label={`Pause of ${formatDuration(Number(item.ts) - Number(previous.ts))}`}><span>Paused {formatDuration(Number(item.ts) - Number(previous.ts))}</span></li>);
    }
    const rowId = `call-${item.id}`;
    const exposure = EXPOSURE_KINDS.includes(item.kind);
    const failed = item.status === "error" || Boolean(item.error_code);
    const label = CALL_LABELS[item.kind] || capitalize(item.kind);
    const assessed = item.kind === "feedback" ? exposureByRetrieval.get(item.retrieval_id) : undefined;
    const textById = new Map<string, string>((assessed?.items || []).map((x: any) => [x.id, itemText(x)]));
    const offset = start && item.ts ? formatOffset(Math.max(0, Number(item.ts) - start)) : "";
    const returned: any[] = item.items || [];
    const memoryCount = returned.filter((x) => x.kind !== "procedure").length;
    const procedureCount = returned.length - memoryCount;
    const returnedTotal = Number(item.count_n ?? returned.length ?? 0);
    const saved: any[] = item.saved_memories?.length
      ? item.saved_memories
      : item.memory_id ? [{ id: item.memory_id, text: item.summary }] : [];
    let title: ReactNode;
    let subtitle: ReactNode = null;
    if (item.kind === "activate") {
      title = `Activated · ${plural(memoryCount || (returnedTotal - procedureCount), "memory", "memories")}, ${plural(procedureCount, "procedure")}`;
      if (item.summary) subtitle = <span className="call-sub clamp-text" title={item.summary}>{item.summary}</span>;
    } else if (item.kind === "recall" || item.kind === "recall_page") {
      title = item.kind === "recall_page" ? "Next page of results" : `Recalled · ${plural(returnedTotal, "item")}`;
      if (item.summary && item.kind === "recall") subtitle = <span className="call-sub clamp-text" title={item.summary}>{item.summary}</span>;
    } else if (item.kind === "feedback") {
      title = assessed ? `Assessed the result of call ${assessed.ordinal}` : "Assessed a retrieval";
    } else if (item.kind === "commit") {
      title = <>Closed the task {sessionOutcome && <StatusBadge value={sessionOutcome} />}</>;
      if (item.summary) subtitle = <span className="call-sub clamp-text" title={item.summary}>{item.summary}</span>;
    } else if (item.kind === "session_reaped") {
      title = "No commit received; closed after inactivity";
    } else if (item.kind === "remember") {
      title = saved.length > 1 ? `Saved ${plural(saved.length, "memory", "memories")}` : "Saved a memory";
      const preview = itemText(saved[0]) || item.summary;
      if (preview) subtitle = <span className="call-sub clamp-text" title={preview}>{preview}</span>;
    } else {
      title = truncate(item.summary || label, 160);
    }
    rows.push(
      <li key={item.id} id={rowId} className={`call-row kind-${item.kind}${highlight === rowId ? " highlighted" : ""}${failed ? " failed" : ""}`}>
        <span className="call-ordinal" aria-hidden="true">{item.ordinal}</span>
        <div className="call-main">
          <div className="call-head">
            <span className="call-kind">{label}</span>
            {failed && <span className="call-failed">failed{item.error_code ? ` · ${item.error_code}` : ""}</span>}
            <span className="call-title">{title}</span>
            <time className="call-time" title={formatDate(item.ts)}>{formatClock(item.ts)}{offset ? ` · ${offset}` : ""}</time>
          </div>
          {subtitle}
          {(item.duration_ms != null || item.estimated_tokens != null || item.reconstructed) && (
            <div className="call-meta muted">
              {[
                item.duration_ms != null ? formatMs(item.duration_ms) : null,
                item.estimated_tokens != null ? `~${formatTokens(item.estimated_tokens)} tokens` : null,
                item.reconstructed ? "reconstructed from stored records" : null,
              ].filter(Boolean).join(" · ")}
            </div>
          )}
          {exposure && (
            <div className="call-meta">
              {formatCount(returnedTotal)} returned
              {item.feedback_count ? ` · ${formatCount(item.feedback_count)} assessed` : ""}
              {item.retrieval_id && <> · <Link to={`/retrieval/${encodeURIComponent(item.retrieval_id)}`}>Open retrieval</Link></>}
            </div>
          )}
          {exposure && returned.length > 0 && (
            <details className="call-items-details">
              <summary>{plural(returned.length, "item")} returned</summary>
              <ul className="call-items">
                {returned.map((x: any) => (
                  <li key={x.id}><TypeBadge type={x.kind === "procedure" ? "procedure" : "memory"} /><Link to={itemHref(x)}>{truncate(itemText(x) || x.id, 120)}</Link></li>
                ))}
              </ul>
            </details>
          )}
          {item.kind === "remember" && (
            saved.length > 0 && saved[0].id && !item.pending_consolidation ? (
              <ul className="call-items">
                {saved.map((x: any) => <li key={x.id}><TypeBadge type="memory" /><Link to={`/memory/${encodeURIComponent(x.id)}`}>{itemText(x) ? truncate(itemText(x), 120) : "Open saved memory"}</Link></li>)}
              </ul>
            ) : (
              <div className="call-meta"><StatusBadge value="pending" label="Pending consolidation" /> <span className="muted">The memory appears once Slowave consolidates it.</span></div>
            )
          )}
          {item.kind === "feedback" && (
            <>
              <div className="call-meta">
                {assessed
                  ? <>On <button type="button" className="link-button" onClick={() => focusCall(assessed.id)}>call {assessed.ordinal} ({CALL_LABELS[assessed.kind]})</button></>
                  : item.retrieval_id ? <Link to={`/retrieval/${encodeURIComponent(item.retrieval_id)}`}>Open assessed retrieval</Link> : null}
              </div>
              {(item.assessments || []).length > 0 ? (
                <ul className="call-items">
                  {item.assessments.map((x: any, i: number) => (
                    <li key={`${x.target_id}-${i}`}>
                      <StatusBadge value={x.assessment || "unknown"} />
                      {x.effect && <StatusBadge value={x.effect} />}
                      <Link to={itemHref({ id: x.target_id, kind: x.target_kind === "procedure" ? "procedure" : "memory" })}>{truncate(textById.get(x.target_id) || x.target_title || x.target_id, 120)}</Link>
                    </li>
                  ))}
                </ul>
              ) : <div className="call-meta muted">No accepted assessments</div>}
            </>
          )}
        </div>
        <button type="button" className="call-link" aria-label={`Copy link to call ${item.ordinal}`} onClick={() => copyLink(rowId)}>{copied === rowId ? "Copied" : "Copy link"}</button>
      </li>,
    );
  });
  return (
    <>
      {chipKinds.length > 1 && (
        <div className="chip-row" role="group" aria-label="Filter calls by kind">
          <button type="button" className={`chip${kind ? "" : " active"}`} aria-pressed={!kind} onClick={() => setKind("")}>All {formatCount(total)}</button>
          {chipKinds.map(([name, count]) => (
            <button key={name} type="button" className={`chip${kind === name ? " active" : ""}`} aria-pressed={kind === name} onClick={() => setKind(kind === name ? "" : name)}>{CALL_LABELS[name] || name} {formatCount(count)}</button>
          ))}
        </div>
      )}
      {shown.length ? (
        <ol className="call-log">{rows}</ol>
      ) : (
        <p className="neutral">{kind ? (filtered?.kind === kind ? "No calls of this kind." : "Loading…") : "No call log was recorded for this session."}</p>
      )}
      {idleClosed && !kind && shown.length > 0 && !shown.some((item) => item.kind === "session_reaped") && shown.length >= total && (
        <p className="neutral">No commit received; closed after inactivity.</p>
      )}
      {shown.length > 0 && shown.length < shownTotal && (
        <p className="neutral">
          Showing {formatCount(shown.length)} of {formatCount(shownTotal)} calls.{" "}
          <button type="button" className="chip" onClick={loadMore} disabled={loadingMore}>{loadingMore ? "Loading…" : "Load more"}</button>
        </p>
      )}
      {moreError && <p className="neutral">{moreError}</p>}
    </>
  );
}

function SessionRecordList({ items, empty }: { items: any[]; empty: string }) {
  if (!items.length) return <p className="neutral">{empty}</p>;
  return (
    <ul className="link-list">
      {items.map((item: any, index: number) => {
        const id = String(item.id ?? item.memory_id ?? item.schema_id ?? "");
        const isProcedure = item.kind === "procedure" || item.memory_type === "procedure";
        const text = itemText(item) || id;
        const href = isProcedure ? `/procedures/${encodeURIComponent(id)}` : `/memory/${id.startsWith("sch_") ? id : `sch_${id}`}`;
        return (
          <li key={`${id}-${index}`}>
            <Link to={href}>
              <span className="item-main">
                <span>{truncate(text, 160)}</span>
                {(item.reason || item.note) && <small>{truncate(String(item.reason || item.note), 140)}</small>}
              </span>
              <span className="item-end">
                <TypeBadge type={isProcedure ? "procedure" : "memory"} />
                {item.status && <StatusBadge value={item.status} />}
                {item.assessment && <StatusBadge value={item.assessment} />}
                {item.effect && <StatusBadge value={item.effect} />}
              </span>
            </Link>
          </li>
        );
      })}
    </ul>
  );
}

function ActivityDetail({ id, onClose }: { id: string; onClose: () => void }) {
  const request = useApi<Json>(`/api/sessions/${encodeURIComponent(id)}`, { clearOnUrlChange: true });
  const data = request.data;
  const session = data?.session;
  const stream = useMemo(() => {
    if (!data) return [];
    return [
      ...(data.events || []).map((item: any) => ({ ...item, kind: "Event", private: true, summary: item.type })),
      ...(data.episodes || []).map((item: any) => ({ ...item, kind: "Episode", summary: `Episode ${item.id} formed` })),
    ].sort((a: any, b: any) => Number(a.ts || 0) - Number(b.ts || 0));
  }, [data]);
  const goal = session ? session.final_goal || session.initial_goal || session.goal || "Untitled session" : "";
  const retrievedItems: any[] = data?.retrieved_items || [];
  const savedItems: any[] = data?.saved_items?.length ? data.saved_items : (data?.memories || []).map((item: any) => ({ id: `sch_${item.schema_id}`, text: item.content_text, status: item.status, kind: "memory" }));
  const usedItems = retrievedItems.filter((item) => item.assessment === "used" || item.used === true);
  const kinds: Record<string, number> = data?.call_log_kinds || {};
  const recalls = Number(kinds.activate || 0) + Number(kinds.recall || 0) + Number(kinds.recall_page || 0) || Number(data?.retrievals?.length || 0);
  const returnedCount = Number(session?.returned_count ?? retrievedItems.length);
  const usedCount = Number(session?.used_count ?? usedItems.length);
  const savedCount = Number(session?.saved_count ?? savedItems.length);
  const status = session?.status || (session?.ended_ts ? "ended" : null);
  const feedbackState = session?.feedback_state || (session?.feedback_status === "complete" ? "complete" : session?.feedback_status === "incomplete" ? "partial" : session?.feedback_status === "pending" ? "pending_open" : "none");
  const chain: any[] = [...(data?.related_sessions || [])].sort((x, y) => Number(x.started_ts) - Number(y.started_ts));
  const chainAt = chain.findIndex((item) => item.id === session?.id);
  const duration = session?.ended_ts && session?.started_ts ? formatDuration(session.ended_ts - session.started_ts) : null;
  const metaLine = session ? [
    session.client?.name,
    session.scope_id ? shortScope(session.scope_id) : "No scope",
    session.started_ts ? `${formatDate(session.started_ts)} → ${session.ended_ts ? formatDate(session.ended_ts) : "still open"}${duration ? ` (${duration})` : ""}` : null,
    chain.length > 1 && chainAt >= 0 ? `${chainAt + 1} of ${chain.length} in this conversation` : null,
  ].filter(Boolean).join(" · ") : "";
  return (
    <Inspector
      title="Session"
      id={id}
      label={session ? truncate(goal, 140) : undefined}
      hideLabel
      state={session && <>{status && <StatusBadge value={status} />}</>}
      sections={[
        { id: "activity-task", label: "Summary" },
        { id: "call-log", label: "Call log" },
        { id: "activity-records", label: "Records" },
        { id: "activity-technical", label: "Technical" },
      ]}
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
        <ErrorState title="Session details unavailable" error={request.error} retry={request.reload} />
      ) : session ? (
          <>
            <DetailSection id="activity-task" title="Summary">
              <h2 className="detail-title">{goal}</h2>
              <div className="badge-stack">
                {["success", "partial", "failure"].includes(String(session.outcome)) ? <StatusBadge value={session.outcome} /> : status === "running" ? null : <StatusBadge value="not_reported" />}
                {status && <StatusBadge value={status} />}
                {session.verification?.status && <StatusBadge value={session.verification.status} />}
              </div>
              <p className="meta-line">{metaLine}</p>
              {session.initial_goal && session.final_goal && session.initial_goal !== session.final_goal && <p className="neutral">Started as: {session.initial_goal}</p>}
              {session.outcome_summary ? <p className="prose">{session.outcome_summary}</p> : status !== "running" && <p className="neutral">No outcome summary recorded.</p>}
              {session.verification?.summary && <p className="prose"><strong>Verification:</strong> {session.verification.summary}</p>}
              <div className="scorecard" aria-label="Session scorecard">
                <div className="tile"><strong>{formatCount(recalls)}</strong><span>Recalls</span></div>
                <div className="tile"><strong>{formatCount(returnedCount)}</strong><span>Returned</span></div>
                <div className="tile"><strong>{formatCount(usedCount)}</strong><span>Used</span></div>
                <div className="tile"><strong>{formatCount(savedCount)}</strong><span>Saved</span></div>
                <div className="tile"><StatusBadge value={feedbackState} label={statusLabelOf(feedbackState)} /><span>Feedback</span></div>
              </div>
              {chain.length > 1 && chainAt >= 0 && (
                <nav className="session-nav" aria-label="Sessions in this conversation">
                  {chain[chainAt - 1] ? <Link to={`/sessions/${encodeURIComponent(chain[chainAt - 1].id)}`}>← Older session</Link> : <span />}
                  <span className="neutral">{chainAt + 1} of {chain.length} in this conversation</span>
                  {chain[chainAt + 1] ? <Link to={`/sessions/${encodeURIComponent(chain[chainAt + 1].id)}`}>Newer session →</Link> : <span />}
                </nav>
              )}
            </DetailSection>
            <DetailSection
              id="call-log"
              title="Call log"
              actions={
                <DefinitionTooltip label="About the call log">
                  Slowave calls in the order they were received. Slowave stores the task text and the text of saved memories and shows both here; replies from the host client are not stored.{data.call_log_approximate ? " Older calls were reconstructed from stored records, so order within the same second is approximate." : ""} Related sessions are correlated by continuity ID, not a verified work chain.
                </DefinitionTooltip>
              }
            >
              <SessionCallLog sessionId={id} data={data} />
            </DetailSection>
            <DetailSection id="activity-records" title="Records">
              <h3>Saved in this session ({formatCount(savedItems.length)})</h3>
              <SessionRecordList items={savedItems} empty="No memories were saved in this session." />
              <h3>Used in this session ({formatCount(usedItems.length)})</h3>
              <SessionRecordList items={usedItems} empty={retrievedItems.length ? "No returned item was reported used." : "No retrieved items recorded for this session."} />
              {data.procedure && (
                <>
                  <h3>Procedure created</h3>
                  <ul className="link-list">
                    <li><Link to={`/procedures/${encodeURIComponent(data.procedure.id)}`}><span className="item-main"><span>{truncate(data.procedure.summary, 160)}</span></span><Icon name="chevron" size={14} /></Link></li>
                  </ul>
                </>
              )}
              {data.feedback?.some((item: any) => item.status !== "accepted") && (
                <details className="advanced">
                  <summary>Rejected submissions ({data.feedback.filter((item: any) => item.status !== "accepted").length})</summary>
                  <ul className="link-list">
                    {data.feedback.filter((item: any) => item.status !== "accepted").map((item: any) => (
                      <li key={item.event_id}><div><span className="item-main"><span>{truncate(item.target_title || item.target_id, 120)}</span><small>{item.rejection_reason || item.reason || "No rejection reason recorded."}</small></span><StatusBadge value={item.status} /></div></li>
                    ))}
                  </ul>
                </details>
              )}
            </DetailSection>
            <DetailSection id="activity-technical" title="Technical details" collapsed>
              <KeyValueList wide items={[
                ["Session ID", <IdChip value={id} head={12} tail={6} />],
                ["Agent / integration", session.agent || "Not recorded"],
                session.continuity_id ? ["Continuity ID", <IdChip value={String(session.continuity_id)} head={12} tail={6} />] : null,
              ]} />
              <h3>Raw recorded events</h3>
              {stream.length ? (
                <div className="activity-stream">
                  {stream.map((item: any, index: number) => (
                    <div key={`${item.kind}-${item.id || index}`}>
                      <time>{formatDate(item.ts)}</time>
                      <StatusBadge value={item.kind.toLowerCase()} />
                      <p>{truncate(item.summary, 260)}</p>
                      {item.private && item.content && (
                        <details>
                          <summary>Event content</summary>
                          <pre>{item.content}</pre>
                        </details>
                      )}
                    </div>
                  ))}
                </div>
              ) : <p className="neutral">No raw events recorded.</p>}
              <h3>Structured task context</h3>
              {Object.keys(session.task_context || {}).length ? <StructuredValue value={session.task_context} /> : <p className="neutral">None recorded.</p>}
            </DetailSection>
          </>
        ) : <EmptyState title="Session not found">This session no longer exists or is not available in the current database.</EmptyState>
      }
    </Inspector>
  );
}

export function DiagnosticsPage({ location }: PageProps) {
  const daemon = useApi<Json>("/api/daemon", { pollMs: refreshMs });
  const database = useApi<Json>("/api/db/health", { pollMs: refreshMs });
  const runSort = param(location, "rsort", "started");
  const runDir = param(location, "rdir", "desc") as "asc" | "desc";
  const runResultFilter = param(location, "result");
  const hideEmptyRuns = param(location, "hide_empty") === "1";
  const selectedRunId = Number(param(location, "run")) || null;
  const [maintenanceExpanded, setMaintenanceExpanded] = useState(false);
  const runColumns = [
    { id: "started", label: "Started", locked: true },
    { id: "duration", label: "Duration" },
    { id: "result", label: "Result" },
    { id: "episodes", label: "Episodes processed" },
    { id: "formed", label: "Memories created" },
    { id: "reinforced", label: "Strengthened" },
    { id: "priority_decay", label: "Deprioritized" },
    { id: "skipped", label: "Skipped" },
    { id: "pass", label: "Run number" },
    { id: "error_category", label: "Error" },
  ];
  const [visibleRunColumns, setVisibleRunColumns, resetRunColumns] = useColumns("diagnostics-runs", ["started", "duration", "result", "episodes", "formed", "reinforced", "priority_decay"]);
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
  const selectRun = (runId?: number) => updateParams("/diagnostics", location, { run: runId });
  const changeRunSort = (column: string) =>
    updateParams("/diagnostics", location, {
      rsort: column === "started" && runSort !== column ? undefined : column,
      rdir: runSort === column ? (runDir === "desc" ? "asc" : "desc") : undefined,
    });
  const runOutcome = (run: any): "failed" | "running" | "changed" | "unchanged" =>
    run.error_text ? "failed"
      : !run.ended_ts ? "running"
        : Number(run.schemas_created || 0) + Number(run.schemas_reinforced || 0) + Number(run.schemas_decayed || 0) > 0 ? "changed" : "unchanged";
  const allRuns: any[] = workers.data?.runs || [];
  const visibleRuns = allRuns.filter((run) => (!runResultFilter || runOutcome(run) === runResultFilter) && (!hideEmptyRuns || runOutcome(run) !== "unchanged"));
  const databaseHealthValue = (value: unknown) =>
    database.data ? String(value || "Unavailable") : database.loading ? "Checking…" : "Unavailable";
  const databaseHealthSecondary = database.data && database.error
    ? <span className="diagnostic-stale">Stale · refresh failed</span>
    : !database.data && database.error
      ? <span className="diagnostic-stale">Health check failed</span>
      : undefined;
  const storageValue = (value: unknown) =>
    value != null ? `${formatCount(value)} bytes` : !status.data && status.loading ? "Checking…" : "Unavailable";
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
      title: `${formatCount(recentRuns.failed)} maintenance run${Number(recentRuns.failed) === 1 ? "" : "s"} failed in the last 7 days`,
      detail: "Open maintenance history to inspect the failure details.",
      href: "/diagnostics#maintenance",
    },
    Number(recentRuns?.incomplete || 0) > 0 && {
      title: `${formatCount(recentRuns.incomplete)} maintenance run${Number(recentRuns.incomplete) === 1 ? " is" : "s are"} incomplete`,
      detail: "An incomplete run may still be in progress; investigate if it remains unchanged.",
      href: "/diagnostics#maintenance",
    },
    duplicateRows > 0 && {
      title: `${formatCount(duplicateRows)} exact duplicate memory row${duplicateRows === 1 ? "" : "s"} detected`,
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
            <MetricCard title="Last maintenance run" value={workers.data?.summary?.last_completed_ts ? formatDate(workers.data.summary.last_completed_ts) : EMPTY} tooltip="Most recent completed maintenance run; recorded errors remain available in run history." className="metric-subtle" />
          </div>
        </>}
      </Section>
      <Section title="Pipeline behavior">
        <p className="neutral">Recent maintenance behavior helps explain whether the local pipeline is keeping up. These are operational signals, not memory-value scores.</p>
        {!workers.data ? workers.error ? <ErrorState title="Pipeline behavior unavailable" error={workers.error} retry={workers.reload} /> : <MetricCardsSkeleton count={3} /> : <div className="metric-card-grid pipeline-metric-grid">
          <RateMetricCard title="Run reliability" numerator={workers.data.summary?.finished_without_error} denominator={workers.data.summary?.finished} secondary={`${workers.data.summary?.incomplete_passes || 0} running / interrupted excluded`} tooltip="Finished runs without recorded error divided by finished runs." />
          <MetricCard title="Median run duration" value={workers.data.summary?.median_duration_ms == null ? EMPTY : formatMs(workers.data.summary.median_duration_ms)} secondary={`${workers.data.summary?.duration_samples || 0} samples · ${workers.data.summary?.missing_durations || 0} missing · ${workers.data.summary?.zero_durations || 0} under 1 second`} tooltip="Median duration for finished runs with recorded timing, including failures. Times come from whole-second timestamps, so values under one second are grouped as under one second." />
        </div>}
        <details className="advanced diagnostics-inventory">
          <summary>Operational inventory</summary><p className="neutral">Worker process: {workers.data?.worker?.running ? "Running" : "Not observed"} · observed {formatDate(workers.data?.worker?.observed_at)}. Process presence does not establish successful processing.</p>
          <p className="neutral">These retained record totals are useful for diagnostics, but they do not measure memory quality.</p>
          <div className="lifetime-totals">{[["Sessions", status.data?.stats?.sessions], ["Raw events", status.data?.stats?.raw_events], ["Episodes", status.data?.stats?.episodes], ["Memories", status.data?.stats?.schemas]].map(([label, value]) => <span key={String(label)}><strong>{status.data ? value == null ? "Unavailable" : Number(value).toLocaleString() : status.loading ? "Checking…" : "Unavailable"}</strong>{label}</span>)}</div>
        </details>
      </Section>
      <Section title="Consolidation history" id="maintenance" actions={<button className="button secondary" onClick={() => setMaintenanceExpanded((expanded) => !expanded)}>{maintenanceExpanded ? "Show recent 10" : "Show 50 runs"}</button>}>
        <p className="formation-caption">Last 30 days</p>
        <p className="neutral">Showing the {maintenanceExpanded ? "50 most recent" : "10 most recent"} recorded passes. Open a row to inspect counts, configuration, and any error details.</p>
        <div className="filter-bar">
          <label>
            Result
            <select value={runResultFilter} onChange={(e) => updateParams("/diagnostics", location, { result: e.target.value })}>
              <option value="">Any result</option>
              <option value="failed">Failed</option>
              <option value="running">Running / incomplete</option>
              <option value="changed">Changed memories</option>
              <option value="unchanged">No changes</option>
            </select>
          </label>
          <label className="check-label">
            <span>Empty runs</span>
            <span className="check-row"><input type="checkbox" checked={hideEmptyRuns} onChange={(e) => updateParams("/diagnostics", location, { hide_empty: e.target.checked ? "1" : undefined })} /> Hide runs that changed nothing</span>
          </label>
          <ColumnsControl columns={runColumns} visible={visibleRunColumns} onChange={setVisibleRunColumns} onReset={resetRunColumns} />
        </div>
        {workers.loading && !workers.data ? (
          <LoadingRows />
        ) : workers.error && !workers.data ? (
          <ErrorState title="Consolidation runs unavailable" error={workers.error} retry={workers.reload} />
        ) : visibleRuns.length ? (
          <TableFrame label="Maintenance runs">
            <table>
              <thead>
                <tr>
                  {runVisible("started") && <SortableTh label="Started" column="started" sort={runSort} dir={runDir} onSort={changeRunSort} />}
                  {runVisible("duration") && <SortableTh label="Duration" column="duration" sort={runSort} dir={runDir} onSort={changeRunSort} numeric />}
                  {runVisible("result") && <SortableTh label="Result" column="result" sort={runSort} dir={runDir} onSort={changeRunSort} help="Failed: the pass recorded an error. Running: not ended yet. Changed: created, strengthened or deprioritized memories. No changes: completed without changing any memory." />}
                  {runVisible("episodes") && <SortableTh label="Episodes processed" column="episodes" sort={runSort} dir={runDir} onSort={changeRunSort} numeric />}
                  {runVisible("formed") && <SortableTh label="Memories created" column="formed" sort={runSort} dir={runDir} onSort={changeRunSort} numeric />}
                  {runVisible("reinforced") && <SortableTh label="Strengthened" column="reinforced" sort={runSort} dir={runDir} onSort={changeRunSort} numeric help="Memories whose retrieval priority was raised during this pass." />}
                  {runVisible("priority_decay") && <SortableTh label="Deprioritized" column="decayed" sort={runSort} dir={runDir} onSort={changeRunSort} numeric help="Memories whose retrieval priority was reduced during this pass. This does not mean the memory was retired or removed." />}
                  {runVisible("skipped") && <SortableTh label="Skipped" column="skipped" sort={runSort} dir={runDir} onSort={changeRunSort} numeric />}
                  {runVisible("pass") && <SortableTh label="Run number" column="pass" sort={runSort} dir={runDir} onSort={changeRunSort} numeric />}
                  {runVisible("error_category") && <SortableTh label="Error" column="error_category" sort={runSort} dir={runDir} onSort={changeRunSort} />}
                </tr>
              </thead>
              <tbody>
                {visibleRuns.map((run: any) => {
                  const outcome = runOutcome(run);
                  const open = () => selectRun(run.id);
                  return (
                    <tr key={run.id} {...rowAttrs(`run-${run.id}`, selectedRunId ? `run-${selectedRunId}` : "", open)}>
                      {runVisible("started") && <td className="primary-cell">
                        <RowLink href={`/diagnostics?run=${run.id}`} onOpen={open}><TimeCell ts={run.started_ts} /></RowLink>
                        <span className="meta">{String(run.triggered_by || "worker").replaceAll("_", " ")}</span>
                      </td>}
                      {runVisible("duration") && <td className="numeric">{run.duration_ms != null ? formatMs(run.duration_ms) : dash}</td>}
                      {runVisible("result") && <td>
                        {outcome === "failed" ? <StatusBadge value="failed" />
                          : outcome === "running" ? <StatusBadge value="running" />
                            : outcome === "changed" ? <StatusBadge value="success" label="Changed memories" />
                              : <StatusBadge value="none" label="No changes" />}
                      </td>}
                      {runVisible("episodes") && <td className="numeric">{formatCount(run.episodes_processed || 0)}</td>}
                      {runVisible("formed") && <td className="numeric">{formatCount(run.schemas_created || 0)}</td>}
                      {runVisible("reinforced") && <td className="numeric">{formatCount(run.schemas_reinforced || 0)}</td>}
                      {runVisible("priority_decay") && <td className="numeric">{formatCount(run.schemas_decayed || 0)}</td>}
                      {runVisible("skipped") && <td className="numeric">{formatCount(run.schemas_skipped || 0)}</td>}
                      {runVisible("pass") && <td className="numeric">{run.id}</td>}
                      {runVisible("error_category") && <td className="long-text-cell" title={run.error_text || undefined}>{run.error_text ? truncate(run.error_text, 60) : dash}</td>}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </TableFrame>
        ) : (
          allRuns.length ? (
            <EmptyState
              title="No runs match these filters"
              action={<button type="button" className="button secondary" onClick={() => updateParams("/diagnostics", location, { result: undefined, hide_empty: undefined })}>Clear filters</button>}
            >
              {allRuns.length} recorded {allRuns.length === 1 ? "run is" : "runs are"} hidden by the current filters.
            </EmptyState>
          ) : (
          <EmptyState title="No maintenance passes recorded">
            An idle installation without recorded passes is not automatically
            unhealthy.
          </EmptyState>
          )
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
        <WorkerRunDetail run={selectedRun} onClose={() => selectRun()} />
      )}
    </div>
  );
}
function WorkerRunDetail({ run, onClose }: { run: any; onClose: () => void }) {
  const failed = Boolean(run.error_text);
  const result = failed ? "failed" : run.ended_ts ? "success" : "running";
  const changed = Number(run.schemas_created || 0) + Number(run.schemas_reinforced || 0) + Number(run.schemas_decayed || 0) > 0;
  return (
    <Inspector
      title="Consolidation run"
      id={`run-${run.id}`}
      label={`Maintenance run ${run.id}`}
      hideLabel
      state={<StatusBadge value={result} label={failed ? "Failed" : run.ended_ts ? (changed ? "Changed memories" : "No changes") : "Running"} />}
      sections={[
        { id: "run-summary", label: "Run" },
        { id: "run-processing", label: "What it did" },
        { id: "run-diagnostics", label: "Details" },
      ]}
      onClose={onClose}
    >
      {failed && (
        <div className="banner bad" role="alert">
          <Icon name="warning" />
          <div>
            <strong>This run ended with an error</strong>
            <CodeBlock label="Recorded error">{run.error_text}</CodeBlock>
          </div>
        </div>
      )}
      <DetailSection id="run-summary" title={`Maintenance run ${run.id}`}>
        <KeyValueList items={[
          ["Result", failed ? "Ended with an error." : run.ended_ts ? "Completed successfully." : "Not ended yet; it may still be running."],
          ["Started", <TimeCell ts={run.started_ts} absolute />],
          ["Ended", run.ended_ts ? <TimeCell ts={run.ended_ts} absolute /> : EMPTY],
          ["Duration", run.duration_ms == null ? "Not recorded" : formatMs(run.duration_ms)],
          ["Triggered by", String(run.triggered_by || "worker").replaceAll("_", " ")],
        ]} />
      </DetailSection>
      <DetailSection id="run-processing" title="What it did">
        <KeyValueList wide items={[
          ["Episodes processed", formatCount(run.episodes_processed || 0)],
          ["Prototypes processed", formatCount(run.prototypes_processed || 0)],
          ["Memories created", formatCount(run.schemas_created || 0)],
          ["Memories strengthened", formatCount(run.schemas_reinforced || 0)],
          ["Memories deprioritized", <>{formatCount(run.schemas_decayed || 0)} <DefinitionTooltip label="What deprioritized means">Retrieval priority was reduced. The memory is not retired or removed. Affected memory records are not retained per run; these are aggregate counts only.</DefinitionTooltip></>],
          ["Memories skipped", formatCount(run.schemas_skipped || 0)],
        ]} />
      </DetailSection>
      <DetailSection id="run-diagnostics" title="Details" collapsed>
        <KeyValueList wide items={[
          ["Run number", String(run.id)],
          !failed ? ["Error", "None recorded"] : null,
        ]} />
      </DetailSection>
    </Inspector>
  );
}
export function GraphPage({ location }: PageProps) {
  const selectedSchema = param(location, "memory");
  const setSelectedSchema = (id: string) => updateParams("/graph", location, { memory: id || undefined });
  const [nodeSearch, setNodeSearch] = useState("");
  const graphScope = param(location, "scope");
  const graphStatuses = param(location, "statuses", "active,needs_review,stale");
  const graphRelations = param(location, "relations", "relates_to,coactivated_with");
  const setGraphFilter = (key: string, value: string) => updateParams("/graph", location, { [key]: value });
  const graph = useApi<Json>(
    `/api/graph/schemas?limit=all&hours=${param(location, "hours", "all")}&from=${param(location, "from")}&to=${param(location, "to")}&relations=${encodeURIComponent(graphRelations)}&scope=${encodeURIComponent(graphScope)}&statuses=${encodeURIComponent(graphStatuses)}`,
  );
  const graphData = graph.data;
  const graphNodes = graphData?.nodes || [];
  const matchingGraphNodes = graphNodes.filter((node: any) => `${node.id} ${node.content || node.label}`.toLowerCase().includes(nodeSearch.toLowerCase()));
  return (
    <div className="page">
      <PageHeader
        title="Memory graph"
        description="Explore connections between memories updated in the selected period. Use search for exact lookup; use the graph for browsing."
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
            <PeriodFilter location={location} path="/graph" />
            <label>Scope<select value={graphScope} onChange={(event) => setGraphFilter("scope", event.target.value)}><option value="">All scopes</option>{[...new Set((graphData?.nodes || []).map((node: any) => node.scope).filter(Boolean))].map((scope: any) => <option key={scope} value={scope}>{scope}</option>)}</select></label>
            <div className="toggle-group" role="group" aria-label="Memory states"><span className="toggle-label">States</span>{[["active","Active"],["needs_review","Review"],["stale","Stale"]].map(([value,label]) => <button type="button" key={value} className={`toggle-badge toggle-state-${value} ${graphStatuses.split(",").includes(value) ? "selected" : ""}`} aria-pressed={graphStatuses.split(",").includes(value)} onClick={() => { const values = graphStatuses.split(",").filter(Boolean); setGraphFilter("statuses", values.includes(value) ? values.filter((item) => item !== value).join(",") : [...values, value].join(",")); }}>{label}</button>)}</div>
            <div className="toggle-group" role="group" aria-label="Edge types"><span className="toggle-label">Edges</span>{[["relates_to","Related to","related"],["coactivated_with","Co-activated","coactivated"]].map(([value,label,kind]) => <button type="button" key={value} className={`toggle-badge toggle-${kind} ${graphRelations.split(",").includes(value) ? "selected" : ""}`} aria-pressed={graphRelations.split(",").includes(value)} onClick={() => { const values = graphRelations.split(",").filter(Boolean); setGraphFilter("relations", values.includes(value) ? values.filter((item) => item !== value).join(",") : [...values, value].join(",")); }}>{label}</button>)}</div>
          </div>
          {graphData && Number(graphData.nodes?.length || 0) >= Number(graphData.limit || 0) && <p className="neutral">Display limit reached ({graphData.limit}); additional memories may be omitted.</p>}
          {graph.loading && !graphData ? <LoadingRows /> : graph.error && !graphData ? <ErrorState title="Graph unavailable" error={graph.error} retry={graph.reload} /> : graphData?.nodes?.length ? <Suspense fallback={<LoadingRows rows={3} />}><GraphExplorer data={graphData} onSelect={setSelectedSchema} /></Suspense> : <EmptyState title="No connected memories">At least two related memories are needed to draw a network.</EmptyState>}
          {graphNodes.length > 0 && <Section title="Find a memory" id="graph-memory-list"><p className="neutral">The list provides keyboard access and exact text lookup. Layout distance does not measure semantic similarity.</p><input aria-label="Search graph memories" placeholder="Search memory text or ID" value={nodeSearch} onChange={(event) => setNodeSearch(event.target.value)} /><div className="detail-list graph-memory-list">{matchingGraphNodes.slice(0, 100).map((node: any) => <button key={node.id} className="graph-memory-option" onClick={() => setSelectedSchema(node.id)}><span><strong>{truncate(node.content || node.label, 150)}</strong><small>{node.id} · {node.scope || "No scope"} · {node.status || "state not recorded"}</small></span><Icon name="chevron" size={14} /></button>)}</div>{matchingGraphNodes.length > 100 && <p className="neutral">Showing first 100 matches. Refine the search to see more.</p>}</Section>}
      </Section>
      {selectedSchema && <MemoryDetail id={selectedSchema} onClose={() => setSelectedSchema("")} />}
    </div>
  );
}
