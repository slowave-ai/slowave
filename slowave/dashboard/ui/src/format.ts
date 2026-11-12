// Pure formatting helpers shared by every dashboard table and drawer.
// Keep this file free of DOM / React imports so it can be unit-tested with node --test.

/** Single placeholder for "no value" everywhere in the dashboard. */
export const EMPTY = "—";

const LOCALE = "en-GB";

const isBlank = (value: unknown) => value === null || value === undefined || value === "";

export function formatCount(value: unknown): string {
  if (isBlank(value)) return EMPTY;
  const n = Number(value);
  return Number.isFinite(n) ? n.toLocaleString(LOCALE) : EMPTY;
}

/** `value` is a percentage (0-100). Sub-1% / super-99% values keep one decimal so they never read as 0% / 100%. */
export function formatPercent(value: unknown, digits = 0): string {
  if (isBlank(value)) return EMPTY;
  const n = Number(value);
  if (!Number.isFinite(n)) return EMPTY;
  if (n === 0) return "0%";
  if (n === 100) return "100%";
  const rounded = Number(n.toFixed(digits));
  if (rounded === 0 || rounded === 100) return `${n.toFixed(1)}%`;
  return `${rounded.toLocaleString(LOCALE)}%`;
}

export function formatPercentOf(numerator: unknown, denominator: unknown, digits = 0): string {
  if (isBlank(numerator) || isBlank(denominator)) return EMPTY;
  const d = Number(denominator);
  const n = Number(numerator);
  if (!Number.isFinite(n) || !Number.isFinite(d) || d <= 0) return EMPTY;
  return formatPercent((n / d) * 100, digits);
}

/** Seconds -> "<1s", "42s", "1m", "1m 5s", "2h 3m". Rounds once up front so 59.6s is "1m", never "60s". */
export function formatDuration(seconds: unknown): string {
  if (isBlank(seconds)) return EMPTY;
  const value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) return EMPTY;
  if (value < 1) return "<1s";
  const total = Math.round(value);
  if (total < 60) return `${total}s`;
  if (total < 3600) {
    const m = Math.floor(total / 60);
    const s = total % 60;
    return s ? `${m}m ${s}s` : `${m}m`;
  }
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  return m ? `${h}h ${m}m` : `${h}h`;
}

export function formatMs(ms: unknown): string {
  if (isBlank(ms)) return EMPTY;
  const n = Number(ms);
  if (!Number.isFinite(n) || n < 0) return EMPTY;
  if (n < 1000) return n < 1 ? "<1ms" : `${Math.round(n)}ms`;
  return formatDuration(n / 1000);
}

export function formatBytes(value: unknown): string {
  if (isBlank(value)) return EMPTY;
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
}

/** Estimated token counts: 950 -> "950", 1234 -> "1.2k", 2_500_000 -> "2.5M". */
export function formatTokens(value: unknown): string {
  if (isBlank(value)) return EMPTY;
  const n = Number(value);
  if (!Number.isFinite(n) || n < 0) return EMPTY;
  if (n < 1000) return String(Math.round(n));
  if (n < 1_000_000) return `${(n / 1000).toFixed(n < 10_000 ? 1 : 0).replace(/\.0$/, "")}k`;
  return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
}

const date = (seconds: unknown) => {
  if (isBlank(seconds)) return null;
  const n = Number(seconds);
  if (!Number.isFinite(n) || n <= 0) return null;
  const d = new Date(n * 1000);
  return Number.isNaN(d.getTime()) ? null : d;
};

/** Unambiguous local timestamp: "6 Nov 2026 14:03". */
export function formatDateTime(seconds: unknown): string {
  const d = date(seconds);
  if (!d) return EMPTY;
  const day = d.toLocaleDateString(LOCALE, { day: "numeric", month: "short", year: "numeric" });
  const time = d.toLocaleTimeString(LOCALE, { hour: "2-digit", minute: "2-digit", hour12: false });
  return `${day} ${time}`;
}

/** Unambiguous local date: "6 Nov 2026". */
export function formatDay(seconds: unknown): string {
  const d = date(seconds);
  return d ? d.toLocaleDateString(LOCALE, { day: "numeric", month: "short", year: "numeric" }) : EMPTY;
}

/** Local clock time: "14:03:07". */
export function formatClock(seconds: unknown): string {
  const d = date(seconds);
  return d ? d.toLocaleTimeString(LOCALE, { hour12: false }) : EMPTY;
}

/** Relative under 7 days ("3h ago"), otherwise "6 Nov 2026". `now` is epoch seconds (injectable for tests). */
export function formatRelative(seconds: unknown, now: number = Date.now() / 1000): string {
  const d = date(seconds);
  if (!d) return EMPTY;
  const delta = now - Number(seconds);
  if (delta < 0) return formatDay(seconds);
  if (delta < 60) return "just now";
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  if (delta < 604800) return `${Math.floor(delta / 86400)}d ago`;
  return formatDay(seconds);
}

/** Scope without the "project:" prefix; the full value is shown in a title. */
export function shortScope(scope: unknown): string {
  const text = String(scope || "");
  return text.startsWith("project:") ? text.slice("project:".length) : text;
}

/** Middle-ellipsis for identifiers: "0f3a9c1e-…-7b2d". */
export function middleEllipsis(value: unknown, head = 8, tail = 4): string {
  const text = String(value ?? "");
  if (text.length <= head + tail + 1) return text;
  return `${text.slice(0, head)}…${text.slice(-tail)}`;
}

/** Local-time yyyy-mm-dd for <input type=date> from epoch seconds. */
export function toDateInput(seconds: unknown): string {
  const d = date(seconds);
  if (!d) return "";
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

/** Epoch seconds at local midnight (or end of day) of a yyyy-mm-dd string. */
export function fromDateInput(value: string, endOfDay = false): number | undefined {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) return undefined;
  const d = new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]), endOfDay ? 23 : 0, endOfDay ? 59 : 0, endOfDay ? 59 : 0);
  return Math.floor(d.getTime() / 1000);
}

/** Display name for a recorded retrieval policy id. Unknown ids stay readable. */
const POLICY_LABELS: Record<string, string> = {
  "activation-calibrated-v2": "Calibrated admission (v2)",
  "recall-calibrated-v2": "Calibrated admission, recall (v2)",
  "activation-pool-relative-v1": "Pool-relative (v1)",
  "recall-pool-relative-v1": "Pool-relative, recall (v1)",
  "activation-complementary-v1": "Complementary (legacy)",
  "relevant-set-v2": "Relevant set (v2)",
  "shared-multilingual-v1": "Shared multilingual pipeline",
  "activate-hybrid-dogfood-v1": "Hybrid dogfood pipeline",
  "multilingual-retrieval-baseline-v1": "Multilingual baseline pipeline",
  "facet-v1": "Facet retrieval",
  "continuity-v1": "Continuity retrieval",
  "strict-v9": "Strict (legacy)",
};

export function policyLabel(id: unknown): string {
  const key = String(id ?? "").trim();
  if (!key) return "Not recorded";
  return POLICY_LABELS[key] ?? key;
}

/** Why calibrated admission did not rank a response (recorded when it fell back). */
const FALLBACK_REASONS: Record<string, string> = {
  no_model_encoder: "embedding model unavailable",
  disabled: "calibrated admission disabled in this runtime",
  unavailable: "scoring model unavailable",
};

export function fallbackLabel(reason: unknown): string {
  const key = String(reason ?? "").trim();
  return key ? FALLBACK_REASONS[key] ?? key : "";
}
