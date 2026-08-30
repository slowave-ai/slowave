import { useCallback, useEffect, useRef, useState } from "react";

export type Json = Record<string, any>;

export async function api<T = Json>(
  url: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(url, init);
  const payload = await response
    .json()
    .catch(() => ({ error: response.statusText }));
  if (!response.ok || payload?.error)
    throw new Error(
      payload?.error || `${response.status} ${response.statusText}`,
    );
  return payload as T;
}

export function useApi<T = Json>(
  url: string | null,
  options: { pollMs?: number } = {},
) {
  const [data, setData] = useState<T>();
  const [loading, setLoading] = useState(Boolean(url));
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string>();
  const [updatedAt, setUpdatedAt] = useState<Date>();
  const controller = useRef<AbortController | undefined>(undefined);
  const dataRef = useRef<T | undefined>(undefined);

  const reload = useCallback(async () => {
    if (!url) return;
    if (controller.current) return;
    const next = new AbortController();
    controller.current = next;
    setError(undefined);
    setLoading((old) => old && dataRef.current === undefined);
    setRefreshing(dataRef.current !== undefined);
    try {
      const result = await api<T>(url, { signal: next.signal });
      if (controller.current === next) {
        dataRef.current = result;
        setData(result);
        setUpdatedAt(new Date());
      }
    } catch (cause) {
      if ((cause as Error).name !== "AbortError")
        setError((cause as Error).message);
    } finally {
      if (controller.current === next) {
        controller.current = undefined;
        setLoading(false);
        setRefreshing(false);
      }
    }
  }, [url]);

  useEffect(() => {
    void reload();
    return () => {
      controller.current?.abort();
      controller.current = undefined;
    };
  }, [url]); // reload intentionally omitted: data retention must not trigger a refetch loop

  useEffect(() => {
    if (!url || !options.pollMs) return;
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") void reload();
    }, options.pollMs);
    return () => window.clearInterval(timer);
  }, [url, options.pollMs, reload]);

  return { data, loading, refreshing, error, updatedAt, reload };
}

export type LocationState = {
  path: string;
  search: URLSearchParams;
  state: any;
};

export function useLocation() {
  const current = (): LocationState => ({
    path: window.location.pathname.replace(/\/$/, "") || "/",
    search: new URLSearchParams(window.location.search),
    state: window.history.state,
  });
  const [location, setLocation] = useState(current);
  useEffect(() => {
    const update = () => setLocation(current());
    window.addEventListener("popstate", update);
    window.addEventListener("slowave:navigate", update);
    return () => {
      window.removeEventListener("popstate", update);
      window.removeEventListener("slowave:navigate", update);
    };
  }, []);
  return location;
}

export function navigate(
  to: string,
  options: { replace?: boolean; state?: any } = {},
) {
  const method = options.replace ? "replaceState" : "pushState";
  window.history[method](options.state || null, "", to);
  window.dispatchEvent(new Event("slowave:navigate"));
  window.scrollTo({ top: 0, behavior: "auto" });
}

export function structuralUrl(
  path: string,
  params: Record<string, string | number | undefined>,
) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== "" && value !== 0)
      search.set(key, String(value));
  });
  const suffix = search.toString();
  return suffix ? `${path}?${suffix}` : path;
}
