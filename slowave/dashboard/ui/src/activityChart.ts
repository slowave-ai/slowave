/** Counts use whole-number ticks and always include the zero baseline. */
export function countAxis(maxValue: number) {
  const rawStep = Math.max(1, maxValue) / 4;
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const fraction = rawStep / magnitude;
  const step = Math.max(1, (fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10) * magnitude);
  const max = Math.max(step, Math.ceil(maxValue / step) * step);
  const ticks = Array.from({ length: Math.round(max / step) + 1 }, (_, i) => max - i * step);
  return { max, ticks };
}

/** Use the rendered SVG width, which may exceed its container on small screens. */
export function bucketIndex(clientX: number, svgLeft: number, svgWidth: number, count: number, right = 28) {
  if (count <= 1) return 0;
  const x = ((clientX - svgLeft) / svgWidth) * 900;
  return Math.max(0, Math.min(count - 1, Math.floor((x - 56) / ((900 - 56 - right) / count))));
}

export function bucketLabel(ts: number, minutes: number, locale?: string) {
  return new Date(ts * 1000).toLocaleString(locale, {
    timeZone: "UTC", year: "numeric", month: "short", day: "numeric",
    ...(minutes < 1440 ? { hour: "2-digit" as const, minute: "2-digit" as const, hourCycle: "h23" as const } : {}),
  });
}

export function bucketDescription(minutes: number) {
  return minutes < 60 ? `${minutes}-minute counts` : minutes === 60 ? "Hourly counts"
    : minutes === 1440 ? "Daily counts" : minutes === 10080 ? "Weekly counts" : `${minutes / 60}-hour counts`;
}

/** Drill-down APIs use inclusive bounds; never include the next bucket. */
export function bucketPeriod(ts: number, minutes: number, windowStart: number, now: number) {
  return { from: Math.max(ts, windowStart), to: Math.min(ts + minutes * 60 - 1, now) };
}

/** Missing deliveries are undefined rates, not zero-use days. */
export function useRate(used: number, retrieved: number): number | null {
  return retrieved > 0 ? used / retrieved * 100 : null;
}

/** Missing buckets break the line; observed zero rates remain on the baseline. */
export function ratePath(values: Array<number | null>, x: (i: number) => number, y: (n: number) => number) {
  return values.map((value, i) => value === null ? "" : `${i === 0 || values[i - 1] === null ? "M" : "L"} ${x(i)} ${y(value)}`).filter(Boolean).join(" ");
}

/** Buckets in a trailing 7-day window; coarser buckets use one bucket. */
export function rollingWindow(minutes: number, days = 7) {
  return Math.max(1, Math.round(days * 1440 / Math.max(1, minutes)));
}

/** Pooled trailing rate (sum used / sum retrieved), so busy buckets weigh more than empty ones. */
export function rollingRate(used: number[], retrieved: number[], window: number): Array<number | null> {
  return used.map((_, i) => {
    let u = 0, r = 0;
    for (let j = Math.max(0, i - window + 1); j <= i; j++) { u += used[j]; r += retrieved[j]; }
    return useRate(u, r);
  });
}
