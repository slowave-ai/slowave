/** Pages whose content is a function of the shared time range ("Period"). */
const ACTIVITY_PATHS = ["/", "/retrieval", "/sessions", "/activity", "/graph"];
/** Library pages: their only date filter is "Created since" (saved_from/saved_to). */
const LIBRARY_PATHS = ["/memory", "/procedures"];

const under = (path: string, root: string) => path === root || path.startsWith(`${root}/`);
export const isLibraryPath = (path: string) => LIBRARY_PATHS.some((root) => under(path, root));
export const isActivityPath = (path: string) =>
  path === "/" || ACTIVITY_PATHS.some((root) => root !== "/" && under(path, root));

/**
 * Carry the shared time range (rolling `hours`, or a custom `from`/`to` pair)
 * to another dashboard page. Library pages never receive it and never
 * originate it; explicit destination bounds always win.
 */
export function carryTimeRange(to: string, currentPath: string, currentSearch: string, origin: string): string {
  const destination = new URL(to, origin);
  if (destination.origin !== origin) return to;
  const explicit = ["hours", "from", "to"].some((key) => destination.searchParams.has(key));
  if (!explicit && !isLibraryPath(destination.pathname) && !isLibraryPath(currentPath)) {
    const current = new URLSearchParams(currentSearch);
    const from = current.get("from");
    const upper = current.get("to");
    if ((from || upper) && isActivityPath(destination.pathname)) {
      if (from) destination.searchParams.set("from", from);
      if (upper) destination.searchParams.set("to", upper);
    } else {
      const hours = current.get("hours");
      if (hours) destination.searchParams.set("hours", hours);
    }
  }
  return `${destination.pathname}${destination.search}${destination.hash}`;
}
