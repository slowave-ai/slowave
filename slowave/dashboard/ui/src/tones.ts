// Semantic tone for every status-like key rendered by StatusBadge.
export type Tone = "ok" | "warn" | "bad" | "neutral" | "info";

export const toneMap: Record<string, Tone> = {
  // good
  active: "ok", current: "ok", available: "ok", complete: "ok", success: "ok", verified: "ok", helped: "ok",
  used: "ok", accepted: "ok", feedback_complete: "ok", ended: "ok", healthy: "ok", ok: "ok",
  // needs attention
  needs_review: "warn", needs_attention: "warn", partial: "warn", partially_verified: "warn", pending: "warn",
  pending_open: "warn", incomplete: "warn", stale: "warn", feedback_incomplete: "warn", feedback_partial: "warn",
  "corrected_/_out_of_date": "warn", rejected: "warn",
  // bad
  failure: "bad", failed: "bad", harmed: "bad", wrong: "bad", error: "bad",
  // no value / inert
  archived: "neutral", retired: "neutral", none: "neutral", unknown: "neutral", irrelevant: "neutral",
  no_effect: "neutral", not_used: "neutral", no_match: "neutral", idle_closed: "neutral", unverified: "neutral",
  recorded: "neutral", captured: "neutral", suppressed: "neutral", feedback_unknown: "neutral", not_reported: "neutral",
  feedback_none: "neutral",
  // informational / entity kinds
  running: "info", activation: "info", recall: "info", context: "info", event: "info", episode: "info",
  feedback: "info", memory: "info", procedure: "info", retrieval: "info", session: "info", activity: "info",
};

export const statusKey = (value: unknown) =>
  String(value || "unknown").trim().toLowerCase().replaceAll(" ", "_");

export const statusTone = (value: unknown): Tone => toneMap[statusKey(value)] ?? "neutral";

export const statusLabels: Record<string, string> = {
  active: "Active", needs_review: "Needs review", stale: "Stale", archived: "Archived",
  complete: "Complete", incomplete: "Incomplete", pending: "Pending", pending_open: "Open",
  success: "Success", partial: "Partial", failure: "Failure", failed: "Failed", unknown: "Unknown",
  retrieval: "Retrieval", activity: "Session", session: "Session", procedure: "Procedure", memory: "Memory",
  event: "Raw event", episode: "Episode", feedback: "Feedback", recorded: "Recorded", captured: "Created",
  used: "Used", helped: "Helpful", harmed: "Harmful", irrelevant: "Irrelevant", none: "None",
  no_effect: "No effect", no_match: "Empty", verified: "Verified", unverified: "Unverified",
  partially_verified: "Partially verified", retired: "Retired", running: "Running", ended: "Ended",
  idle_closed: "Idle-closed", not_reported: "Not reported", accepted: "Accepted", rejected: "Rejected",
  activation: "Activation", recall: "Recall", context: "Activation", wrong: "Wrong", not_used: "Not used",
  feedback_complete: "Feedback complete", feedback_incomplete: "Feedback incomplete", feedback_unknown: "Feedback unknown",
  feedback_partial: "Feedback partial", feedback_none: "No feedback", error: "Error",
};

export const statusLabel = (value: unknown) => {
  const key = statusKey(value);
  return statusLabels[key] || String(value || "Unknown").replaceAll("_", " ");
};
