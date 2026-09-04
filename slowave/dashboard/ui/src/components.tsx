import {
  useEffect,
  useId,
  useRef,
  useState,
  type MouseEvent,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import { navigate, useApi, type Json } from "./api";

const root = document.getElementById("root")!;
export const allowActions = root.dataset.allowActions !== "false";
export const refreshMs = Math.max(
  5000,
  Number(localStorage.getItem("slowave-refresh-ms") || root.dataset.refreshMs || 15000),
);
export const version = root.dataset.version || "—";

// Shared glossary used by the effectiveness surface and the memories table so
// the same term is explained identically everywhere it appears.
export const glossary: Record<string, string> = {
  exposed:
    "Context Slowave offered to an agent at activation or recall. Exposure is not use: an item can be exposed and never acted on.",
  retrieved:
    "Unique active memory records admitted into context during the selected period. A memory retrieved multiple times counts once. Coverage is not a measure of memory quality.",
  used: "The agent reported using this item via feedback. 'Used' is an explicit claim, not an inference from exposure. A 'used' assessment reinforces the memory for future retrieval.",
  irrelevant:
    "The agent reported this item as not relevant to the task. Unassessed items are not counted as irrelevant.",
  "no match":
    "A retrieval that returned no admitted memories or procedures (count = 0). Empty does not mean Slowave failed to store memories, only that none were admitted for this task.",
  "feedback complete":
    "A retrieval whose feedback coverage was marked complete, so every exposed item has an explicit assessment. Incomplete coverage means some items are unassessed.",
  reinforced:
    "A 'used' assessment strengthens the memory's salience for future retrieval. Reinforcement is an effect of use, not proof the memory is correct.",
  helpful:
    "For procedures, the agent reported the procedure helped the task (effect = helped). Helpfulness is distinct from use and from the task outcome: Slowave's context is not necessarily the cause of a task outcome.",
  active_memories:
    "Memory records currently in the active lifecycle state. This is the current library size, not the number retrieved for the selected period.",
  changed_memories:
    "Memory records changed during the selected Home date range. Each record is counted once even if it changed multiple times.",
  active_scopes:
    "Distinct scopes containing at least one active memory record.",
  active_state:
    "Memory records currently in the active lifecycle state.",
  needs_review_state:
    "Memory records currently marked for review because they may need confirmation or correction.",
  stale_state:
    "Memory records currently marked stale because they may be out of date.",
  forgotten_state:
    "Memory records suppressed from ordinary retrieval because they were explicitly forgotten.",
  archived_state:
    "Memory records retained for history but excluded from the active library.",
  retrievals_total:
    "Retrieval operations observed in the selected period and filters. Activations and recalls are counted as retrievals.",
  retrievals_no_match:
    "Retrievals that admitted no memory or procedure. This is not proof that no stored memory was relevant.",
  retrievals_feedback_complete:
    "Retrievals whose exposed items all received explicit feedback. The numerator is complete retrievals; the denominator is all retrievals.",
  retrievals_unknown:
    "Exposed memory and procedure items without an explicit assessment in the selected retrievals. Unknown is not negative evidence.",
  retrievals_demonstrated_value:
    "Feedback-complete retrievals with at least one returned memory assessed as used or returned procedure assessed as helpful. This demonstrates recorded value, not causal impact on task success.",
  activity_context_use:
    "Activities with complete feedback and retrieval where at least one returned memory was assessed as used or procedure as helpful. This is recorded context use, not proof Slowave caused the outcome.",
};

const iconPaths: Record<string, ReactNode> = {
  home: (
    <>
      <path d="M3 10.5 8 6l5 4.5" />
      <path d="M4.5 9.5V14h7V9.5" />
    </>
  ),
  memory: (
    <>
      <path d="M4 3.5h6.5A1.5 1.5 0 0 1 12 5v8H5.5A1.5 1.5 0 0 1 4 11.5z" />
      <path d="M4 11.5A1.5 1.5 0 0 1 5.5 10H12" />
    </>
  ),
  retrieval: (
    <>
      <path d="M3 5h8" />
      <path d="m9 3 2 2-2 2" />
      <path d="M13 11H5" />
      <path d="m7 9-2 2 2 2" />
    </>
  ),
  procedure: (
    <>
      <path d="M5 3h7v10H5z" />
      <path d="M3 5v8h7" />
      <path d="M7 6h3M7 8.5h3M7 11h2" />
    </>
  ),
  activity: (
    <>
      <path d="M2.5 8h2l1.2-3 2.4 6 1.5-3H13.5" />
    </>
  ),
  graph: (
    <>
      <circle cx="4" cy="5" r="1.5" />
      <circle cx="12" cy="4" r="1.5" />
      <circle cx="8" cy="12" r="1.5" />
      <path d="m5.3 5.4 5.4-.8M4.8 6.2l2.4 4.5M11.2 5.3 8.8 10.8" />
    </>
  ),
  external: (
    <>
      <path d="M9 3h4v4" />
      <path d="m13 3-6 6" />
      <path d="M11 8.5V12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h3.5" />
    </>
  ),
  diagnostics: (
    <>
      <circle cx="8" cy="8" r="2.2" />
      <path d="M8 2.5v1.3M8 12.2v1.3M2.5 8h1.3M12.2 8h1.3M4.1 4.1l.9.9M11 11l.9.9M11 5l.9-.9M4.1 11.9l.9-.9" />
    </>
  ),
  refresh: (
    <>
      <path d="M13 5V2.5L11.5 4A5.5 5.5 0 1 0 13 9" />
    </>
  ),
  sun: (
    <>
      <circle cx="8" cy="8" r="2.5" />
      <path d="M8 1.5v1M8 13.5v1M1.5 8h1M13.5 8h1M3.4 3.4l.7.7M11.9 11.9l.7.7M11.9 4.1l.7-.7M3.4 12.6l.7-.7" />
    </>
  ),
  moon: <path d="M12.5 10.8A5.5 5.5 0 0 1 5.2 3.5 5.5 5.5 0 1 0 12.5 10.8Z" />,
  copy: (
    <>
      <rect x="5" y="5" width="8" height="8" rx="1" />
      <path d="M3 10V3h7" />
    </>
  ),
  close: <path d="m4 4 8 8M12 4l-8 8" />,
  chevron: <path d="m6 4 4 4-4 4" />,
  warning: (
    <>
      <path d="M8 2.5 14 13H2z" />
      <path d="M8 6v3M8 11.5h.01" />
    </>
  ),
  info: (
    <>
      <circle cx="8" cy="8" r="6" />
      <path d="M8 7v4M8 4.5h.01" />
    </>
  ),
};

export function Icon({ name, size = 16 }: { name: string; size?: number }) {
  return (
    <svg
      className="icon"
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {iconPaths[name]}
    </svg>
  );
}

export function Link({
  to,
  children,
  className = "",
  state,
}: {
  to: string;
  children: ReactNode;
  className?: string;
  state?: any;
}) {
  return (
    <a
      href={to}
      className={className}
      onClick={(event) => {
        if (!event.metaKey && !event.ctrlKey && event.button === 0) {
          event.preventDefault();
          navigate(to, { state });
        }
      }}
    >
      {children}
    </a>
  );
}

const navigation = [
  ["/", "Home", "home"],
  ["/memory", "Memories", "memory"],
  ["/procedures", "Procedures", "procedure"],
  ["/retrieval", "Retrieval", "retrieval"],
  ["/activity", "Activity", "activity"],
  ["/graph", "Memory graph", "graph"],
  ["/diagnostics", "Diagnostics", "diagnostics"],
] as const;

export function AppShell({
  path,
  children,
}: {
  path: string;
  children: ReactNode;
}) {
  const [theme, setTheme] = useState(
    () => localStorage.getItem("slowave-theme") || "system",
  );
  useEffect(() => {
    const dark =
      theme === "dark" ||
      (theme === "system" &&
        matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    document.documentElement.style.colorScheme = dark ? "dark" : "light";
    localStorage.setItem("slowave-theme", theme);
  }, [theme]);
  const active = (href: string) =>
    href === "/" ? path === "/" : path === href || path.startsWith(`${href}/`);
  const cycleTheme = () =>
    setTheme((old) =>
      old === "system" ? "light" : old === "light" ? "dark" : "system",
    );
  return (
    <div className="app-shell">
      <aside className="side-nav">
        <div className="brand">
          <img src="/img/slowave-logo-small.jpeg" alt="" />
          <div>
            <strong>Slowave</strong>
            <span>Dashboard</span>
          </div>
        </div>
        <nav aria-label="Primary">
          {navigation.map(([href, label, icon]) => (
            <Link key={href} to={href} className={active(href) ? "active" : ""}>
              <Icon name={icon} />
              <span>{label}</span>
            </Link>
          ))}
        </nav>
        <div className="resources-nav">
          <span className="nav-group-label">Resources</span>
          <a href="https://github.com/mrsalty/slowave" target="_blank" rel="noreferrer">
            <Icon name="external" />
            <span>GitHub</span>
            <small aria-hidden="true">↗</small>
          </a>
          <a href="/docs" target="_blank" rel="noreferrer" title="Documentation">
            <Icon name="external" />
            <span>Docs</span>
            <small aria-hidden="true">↗</small>
          </a>
        </div>
        <div className="nav-footer">
          <button
            onClick={cycleTheme}
            aria-label={`Theme: ${theme}. Change theme`}
          >
            <Icon name={theme === "dark" ? "moon" : "sun"} />
            <span>Theme: {theme}</span>
          </button>
        </div>
      </aside>
      <div className="mobile-bar">
        <div className="brand">
          <img src="/img/slowave-logo-small.jpeg" alt="" />
          <strong>Slowave</strong>
        </div>
        <button
          onClick={cycleTheme}
          aria-label={`Theme: ${theme}. Change theme`}
        >
          <Icon name={theme === "dark" ? "moon" : "sun"} />
        </button>
      </div>
      <nav className="mobile-nav" aria-label="Primary">
        {navigation.map(([href, label, icon]) => (
          <Link key={href} to={href} className={active(href) ? "active" : ""}>
            <Icon name={icon} />
            <span>{label}</span>
          </Link>
        ))}
      </nav>
      <main id="main-content">{children}</main>
    </div>
  );
}

export function PageHeader({
  title,
  description,
  updatedAt,
  refreshing,
  onRefresh,
  controls,
}: {
  title: string;
  description?: string;
  updatedAt?: Date;
  refreshing?: boolean;
  onRefresh?: () => void;
  controls?: ReactNode;
}) {
  const [interval, setInterval] = useState(String(refreshMs));
  useEffect(() => {
    document.title = `${title} · Slowave`;
  }, [title]);
  return (
    <header className="page-header">
      <div>
        <h1>{title}</h1>
        {description && <p>{description}</p>}
      </div>
      <div className="page-actions">
        {controls}
        {onRefresh && (
          <div className="refresh-status">
            <label className="refresh-interval">Auto-refresh
              <select value={interval} onChange={(e) => { localStorage.setItem("slowave-refresh-ms", e.target.value); setInterval(e.target.value); window.location.reload(); }}>
                <option value="5000">5 sec</option><option value="15000">15 sec</option><option value="30000">30 sec</option><option value="60000">1 min</option><option value="300000">5 min</option>
              </select>
            </label>
            <button
              className="button secondary"
              onClick={onRefresh}
              disabled={refreshing}
            >
              <Icon name="refresh" /> Refresh
            </button>
          </div>
        )}
      </div>
    </header>
  );
}

const labels: Record<string, string> = {
  active: "Active",
  needs_review: "Needs review",
  stale: "Stale",
  forgotten: "Suppressed",
  archived: "Archived",
  complete: "Complete",
  incomplete: "Incomplete",
  pending: "Pending",
  success: "Success",
  partial: "Partial",
  failure: "Failure",
  unknown: "Unknown",
  retrieval: "Retrieval",
  activity: "Activity",
  procedure: "Procedure",
  memory: "Memory",
  event: "Raw Events",
  episode: "Episode",
  feedback: "Feedback",
  recorded: "Recorded",
  captured: "Created",
  used: "Used",
  helped: "Helpful",
  harmed: "Harmful",
  irrelevant: "Irrelevant",
  none: "None",
  no_effect: "No effect",
  no_match: "Empty",
  verified: "Verified",
  unverified: "Unverified",
  partially_verified: "Partially verified",
  retired: "Retired",
  failed: "Failed",
  running: "Running",
};
export function StatusBadge({ value, count }: { value?: string | null; count?: number }) {
  const key = String(value || "unknown")
    .toLowerCase()
    .replaceAll(" ", "_");
  return (
    <span className={`status-badge status-${key}`}>
      <i aria-hidden="true" />
      {labels[key] || String(value || "Unknown").replaceAll("_", " ")}{count !== undefined ? ` · ${count.toLocaleString()}` : ""}
    </span>
  );
}

export function DefinitionTooltip({
  label,
  children,
  inline = false,
}: {
  label: string;
  children: ReactNode;
  inline?: boolean;
}) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState<{ top: number; left: number } | null>(null);
  const timer = useRef<number | undefined>(undefined);
  const triggerRef = useRef<HTMLElement>(null);
  const show = () => {
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      const rect = triggerRef.current?.getBoundingClientRect();
      if (!rect) return;
      const width = Math.min(360, Math.max(220, window.innerWidth - 32));
      setPosition({
        top: rect.bottom + 6,
        left: Math.min(Math.max(16, rect.left), Math.max(16, window.innerWidth - width - 16)),
      });
      setOpen(true);
    }, 120);
  };
  const hide = () => {
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setOpen(false), 120);
  };
  useEffect(() => {
    const close = (event: KeyboardEvent) => event.key === "Escape" && setOpen(false);
    const dismiss = (event: globalThis.MouseEvent) => {
      if (!triggerRef.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("keydown", close);
    document.addEventListener("mousedown", dismiss);
    return () => {
      document.removeEventListener("keydown", close);
      document.removeEventListener("mousedown", dismiss);
    };
  }, []);
  useEffect(() => {
    if (!open) return;
    const reposition = () => {
      const rect = triggerRef.current?.getBoundingClientRect();
      if (!rect) return;
      const width = Math.min(360, Math.max(220, window.innerWidth - 32));
      setPosition({
        top: rect.bottom + 6,
        left: Math.min(Math.max(16, rect.left), Math.max(16, window.innerWidth - width - 16)),
      });
    };
    reposition();
    window.addEventListener("resize", reposition);
    window.addEventListener("scroll", reposition, true);
    return () => {
      window.removeEventListener("resize", reposition);
      window.removeEventListener("scroll", reposition, true);
    };
  }, [open]);
  const toggle = (event?: MouseEvent) => {
    event?.stopPropagation();
    window.clearTimeout(timer.current);
    if (open) {
      setOpen(false);
    } else {
      const rect = triggerRef.current?.getBoundingClientRect();
      if (rect) {
        const width = Math.min(360, Math.max(220, window.innerWidth - 32));
        setPosition({
          top: rect.bottom + 6,
          left: Math.min(Math.max(16, rect.left), Math.max(16, window.innerWidth - width - 16)),
        });
      }
      setOpen(true);
    }
  };
  const tooltip = open && position ? createPortal(
    <span
      className="tooltip definition-tooltip-popover"
      role="tooltip"
      id={id}
      style={{ top: position.top, left: position.left }}
      onMouseEnter={() => window.clearTimeout(timer.current)}
      onMouseLeave={hide}
    >
      {children}
    </span>,
    document.body,
  ) : null;
  return (
    <span className="definition">
      {inline ? (
        <span
          ref={triggerRef}
          className="icon-button definition-trigger"
          role="img"
          aria-label={label}
          aria-describedby={open ? id : undefined}
          onMouseEnter={show}
          onMouseLeave={hide}
          onClick={toggle}
        >
          <Icon name="info" size={14} />
        </span>
      ) : (
        <button
          ref={(element) => {
            triggerRef.current = element;
          }}
          type="button"
          className="icon-button"
          aria-label={label}
          aria-describedby={open ? id : undefined}
          onFocus={show}
          onBlur={hide}
          onMouseEnter={show}
          onMouseLeave={hide}
          onClick={toggle}
        >
          <Icon name="info" size={14} />
        </button>
      )}
      {tooltip}
    </span>
  );
}

export function ScopeSelect({
  value,
  onChange,
}: {
  value: string;
  onChange: (scope: string) => void;
}) {
  const request = useApi<Json>("/api/scopes");
  const scopes: string[] = request.data?.scopes || [];
  return (
    <label>
      Scope
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">All scopes</option>
        {scopes.map((scope) => (
          <option key={scope} value={scope}>
            {scope}
          </option>
        ))}
      </select>
    </label>
  );
}
export function ClampedText({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <button
      type="button"
      className="clamp-toggle"
      aria-expanded={expanded}
      onClick={(event) => {
        event.stopPropagation();
        setExpanded(!expanded);
      }}
      onKeyDown={(event) => event.stopPropagation()}
    >
      <span className={expanded ? "clamp-text expanded" : "clamp-text"}>
        {text}
      </span>
      <Icon name="chevron" size={12} />
    </button>
  );
}




export function CopyValue({
  value,
  label = "Copy ID",
}: {
  value: string;
  label?: string;
}) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      className="copy-value"
      onClick={async () => {
        await navigator.clipboard.writeText(value);
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1500);
      }}
    >
      <code>{value}</code>
      <Icon name="copy" size={14} />
      <span className="sr-only" aria-live="polite">
        {copied ? "Copied" : label}
      </span>
    </button>
  );
}

export function InlineError({
  error,
  retained,
  retry,
}: {
  error?: string;
  retained?: boolean;
  retry?: () => void;
}) {
  return error ? (
    <div className="notice error" role="alert">
      <Icon name="warning" />
      <div>
        <strong>Could not update this section</strong>
        <span>
          {error}
          {retained ? " · Retaining the last successful data." : ""}
        </span>
      </div>
      {retry && (
        <button className="button secondary" onClick={retry}>
          Retry
        </button>
      )}
    </div>
  ) : null;
}
export function LoadingRows({ rows = 6 }: { rows?: number }) {
  return (
    <div className="skeleton-table" aria-label="Loading">
      <div className="skeleton skeleton-head" />
      {Array.from({ length: rows }, (_, index) => (
        <div className="skeleton skeleton-row" key={index} />
      ))}
    </div>
  );
}
export function MetricCardsSkeleton({ count = 4 }: { count?: number }) {
  return (
    <div className="metric-card-grid metric-card-grid-skeleton" aria-label="Loading summary">
      {Array.from({ length: count }, (_, index) => <div className="metric-card-skeleton" key={index} />)}
    </div>
  );
}
export function EmptyState({
  title,
  children,
  firstRun = false,
}: {
  title: string;
  children: ReactNode;
  firstRun?: boolean;
}) {
  return (
    <div className={`empty-state ${firstRun ? "first-run" : ""}`}>
      {firstRun && (
        <svg viewBox="0 0 80 48" aria-hidden="true">
          <path d="M8 36h15l7-20 10 29 9-20 7 11h16" />
        </svg>
      )}
      <strong>{title}</strong>
      <p>{children}</p>
    </div>
  );
}
export function ErrorState({
  title = "Data unavailable",
  error,
  retry,
}: {
  title?: string;
  error?: string;
  retry?: () => void;
}) {
  return (
    <EmptyState title={title}>
      {error || "The requested data could not be loaded."}
      {retry && <button className="button secondary empty-state-retry" onClick={retry}>Retry</button>}
    </EmptyState>
  );
}
export function Section({
  title,
  actions,
  children,
  id,
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  id?: string;
}) {
  return (
    <section className="section" id={id}>
      {(title || actions) && (
        <div className="section-header">
          <h2>{title}</h2>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}
export function TableFrame({
  children,
  label,
}: {
  children: ReactNode;
  label: string;
}) {
  return (
    <div className="table-frame" role="region" aria-label={label} tabIndex={0}>
      {children}
    </div>
  );
}
export function SortButton({
  label,
  active,
  direction,
  onClick,
}: {
  label: string;
  active?: boolean;
  direction?: "asc" | "desc";
  onClick: () => void;
}) {
  return (
    <button className="sort-button" onClick={onClick}>
      {label}
      {active && (
        <span aria-hidden="true">{direction === "asc" ? " ↑" : " ↓"}</span>
      )}
    </button>
  );
}

export function ColumnsControl({
  columns,
  visible,
  onChange,
}: {
  columns: readonly { id: string; label: string; description?: string }[];
  visible: readonly string[];
  onChange: (next: string[]) => void;
}) {
  return (
    <details className="columns-control">
      <summary>Columns <span>{visible.length}/{columns.length}</span></summary>
      <div>
        {columns.map((column) => {
          const checked = visible.includes(column.id);
          return (
            <label key={column.id} title={column.description}>
              <input
                type="checkbox"
                checked={checked}
                onChange={() => onChange(checked ? visible.filter((id) => id !== column.id) : [...visible, column.id])}
              />
              {column.label}
            </label>
          );
        })}
      </div>
    </details>
  );
}

export function Pagination({
  page,
  perPage,
  total,
  onPage,
}: {
  page: number;
  perPage: number;
  total: number;
  onPage: (page: number) => void;
}) {
  const pages = Math.max(1, Math.ceil(total / perPage));
  return (
    <div className="pagination">
      <span>
        {total.toLocaleString()} results · page {page} of {pages}
      </span>
      <div>
        <button
          className="button secondary"
          disabled={page <= 1}
          onClick={() => onPage(page - 1)}
        >
          Previous
        </button>
        <button
          className="button secondary"
          disabled={page >= pages}
          onClick={() => onPage(page + 1)}
        >
          Next
        </button>
      </div>
    </div>
  );
}

export function Inspector({
  title,
  id,
  state,
  onClose,
  children,
}: {
  title: string;
  id: string;
  state?: ReactNode;
  onClose: () => void;
  children: ReactNode;
}) {
  return (
    <div
      className="inspector-backdrop"
      onMouseDown={(event) => event.target === event.currentTarget && onClose()}
    >
      <aside className="inspector" aria-label={`${title} detail`}>
        <header>
          <div>
            <span className="entity-type">{title}</span>
            <div className="inspector-id">
              <CopyValue value={id} />
              {state}
            </div>
          </div>
          <button
            className="icon-button close"
            onClick={onClose}
            aria-label="Close detail"
          >
            <Icon name="close" />
          </button>
        </header>
        <div className="inspector-content">{children}</div>
      </aside>
    </div>
  );
}

const dashboardDateLocale = "en-GB";
export const formatDate = (value: any) =>
  value ? new Date(Number(value) * 1000).toLocaleString(dashboardDateLocale) : "Unknown";
export const relativeDate = (value: any) => {
  if (!value) return "Unknown";
  const delta = Date.now() / 1000 - Number(value);
  if (delta < 60) return "just now";
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  if (delta < 604800) return `${Math.floor(delta / 86400)}d ago`;
  return new Date(Number(value) * 1000).toLocaleDateString(dashboardDateLocale);
};
export const truncate = (value: any, max = 120) => {
  const text = String(value || "");
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
};

export const formatRate = (numerator: unknown, denominator: unknown) => {
  const parts = formatRateParts(numerator, denominator);
  return parts ? `${parts.ratio} ${parts.percent}` : "—";
};

const ratePercentLabel = (numerator: number, denominator: number) => {
  if (numerator === denominator) return "100%";
  if (numerator === 0) return "0%";
  const percent = (numerator / denominator) * 100;
  const rounded = Math.round(percent);
  return rounded === 0 || rounded === 100 ? `${percent.toFixed(1)}%` : `${rounded}%`;
};

export const formatRateParts = (numerator: unknown, denominator: unknown) => {
  if (
    numerator === null || numerator === undefined || numerator === "" ||
    denominator === null || denominator === undefined || denominator === ""
  ) return null;
  const used = Number(numerator);
  const retrieved = Number(denominator);
  if (!Number.isFinite(used) || !Number.isFinite(retrieved)) return null;
  return {
    ratio: `${used.toLocaleString()} / ${retrieved.toLocaleString()} ·`,
    percent: retrieved > 0 ? ratePercentLabel(used, retrieved) : "—",
  };
};

export const getRatePercent = (numerator: unknown, denominator: unknown) => {
  if (numerator === null || numerator === undefined || numerator === "" || denominator === null || denominator === undefined || denominator === "") return null;
  const used = Number(numerator);
  const retrieved = Number(denominator);
  if (!Number.isFinite(used) || !Number.isFinite(retrieved) || retrieved <= 0) return null;
  return Math.min(100, Math.max(0, (used / retrieved) * 100));
};

export const formatRatePercent = (numerator: unknown, denominator: unknown) => {
  const parts = formatRateParts(numerator, denominator);
  return parts?.percent || "—";
};

export const formatDuration = (seconds: unknown) => {
  const value = Number(seconds);
  if (!Number.isFinite(value)) return "—";
  if (value < 60) return `${Math.round(value)}s`;
  const minutes = Math.floor(value / 60);
  return value < 3600 ? `${minutes}m ${Math.round(value % 60)}s` : `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
};

export function MetricCard({
  title,
  value,
  tooltip,
  secondary,
  href,
  className = "",
}: {
  title: string;
  value: ReactNode;
  tooltip: string;
  secondary?: ReactNode;
  href?: string;
  className?: string;
}) {
  const displayValue = typeof value === "string" ? <strong>{value}</strong> : value;
  const card = (
    <div className={`metric-card effectiveness-card ${className}`}>
      <div className="effectiveness-card-title">
        <span className="effectiveness-card-title-text">{title}</span>
        <DefinitionTooltip label={`${title} definition`}>{tooltip}</DefinitionTooltip>
      </div>
      <div className="effectiveness-card-value">{displayValue}</div>
      {secondary && <div className="metric-card-secondary">{secondary}</div>}
    </div>
  );
  return href ? <Link to={href} className="metric-card-link">{card}</Link> : card;
}

export function RateMetricCard({
  title,
  numerator,
  denominator,
  tooltip,
  secondary,
  href,
  className = "",
}: {
  title: string;
  numerator: unknown;
  denominator: unknown;
  tooltip: string;
  secondary?: ReactNode;
  href?: string;
  className?: string;
}) {
  const unavailable = [numerator, denominator].some(
    (value) => value === null || value === undefined || value === "",
  );
  const denominatorNumber = Number(denominator);
  const lowSample = !unavailable && denominatorNumber > 0 && denominatorNumber < 10;
  const secondaryContent = secondary || lowSample ? (
    <>
      {secondary}
      {lowSample && <span className="metric-low-sample">Low sample · n={denominatorNumber.toLocaleString()}</span>}
    </>
  ) : undefined;
  const parts = formatRateParts(numerator, denominator);
  return (
    <MetricCard
      title={title}
      value={unavailable || !parts ? "—" : (
        <span className="metric-rate-value">
          <span className="metric-rate-ratio">{parts?.ratio}</span>
          <span className="metric-rate-percent">{parts?.percent}</span>
        </span>
      )}
      tooltip={tooltip}
      href={href}
      className={className}
      secondary={secondaryContent}
    />
  );
}
