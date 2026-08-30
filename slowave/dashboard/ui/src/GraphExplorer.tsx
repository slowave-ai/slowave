import { useEffect, useRef, useState } from "react";
import type { Json } from "./api";

export default function GraphExplorer({ data, onSelect }: { data: Json; onSelect?: (id: string) => void }) {
  const container = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; title: string; detail: string } | null>(null);
  useEffect(() => {
    let disposed = false;
    let graph: import("cytoscape").Core | undefined;
    void import("cytoscape").then(({ default: cytoscape }) => {
      if (disposed || !container.current) return;
      const css = getComputedStyle(document.documentElement);
      const palette = {
        activity: css.getPropertyValue("--accent-activity").trim() || "#2563eb",
        success: css.getPropertyValue("--success").trim() || "#1a7f37",
        warning: css.getPropertyValue("--warning").trim() || "#9a6700",
        danger: css.getPropertyValue("--danger").trim() || "#cf222e",
        muted: css.getPropertyValue("--muted").trim() || "#59636e",
        procedure: css.getPropertyValue("--accent-procedure").trim() || "#7c3aed",
        link: css.getPropertyValue("--link").trim() || "#0969da",
      };
      const degree = new Map<string, number>();
      (data.edges || []).forEach((edge: any) => { degree.set(edge.source, (degree.get(edge.source) || 0) + 1); degree.set(edge.target, (degree.get(edge.target) || 0) + 1); });
      graph = cytoscape({
        container: container.current,
        elements: [
          ...(data.nodes || []).map((node: any, index: number) => {
            const total = Math.max(1, data.nodes.length);
            const theta = index * Math.PI * (3 - Math.sqrt(5));
            const y = 1 - (index / Math.max(1, total - 1)) * 2;
            const radius = Math.sqrt(1 - y * y);
            return {
              position: { x: Math.cos(theta) * radius * 500, y: y * 500 + Math.sin(theta) * radius * 95 },
              data: { id: node.id, label: node.label, content: node.content, scope: node.scope, status: node.status, salience: node.salience, confidence: node.confidence, degree: degree.get(node.id) || 0 },
            };
          }),
          ...(data.edges || []).map((edge: any) => ({
            data: {
              id: edge.id,
              source: edge.source,
              target: edge.target,
              relation: edge.relation,
              sourceLabel: (data.nodes || []).find((node: any) => node.id === edge.source)?.label || edge.source,
              targetLabel: (data.nodes || []).find((node: any) => node.id === edge.target)?.label || edge.target,
            },
          })),
        ],
        style: [
          {
            selector: "node",
            style: {
              "background-color": palette.activity,
              "background-opacity": 0.9,
              width: "mapData(degree, 0, 40, 5, 17)",
              height: "mapData(degree, 0, 40, 5, 17)",
              label: "",
              color: "#8b949e",
              "font-size": "10px",
              "font-weight": 500,
              "text-wrap": "ellipsis",
              "text-max-width": "90px",
            },
          },
          {
            selector: "edge",
            style: {
              width: 0.45,
              opacity: 0.28,
              "line-color": palette.activity,
              "curve-style": "haystack",
            },
          },
          { selector: 'edge[relation = "coactivated_with"]', style: { "line-color": palette.procedure, opacity: 0.3, width: 0.4 } },
          { selector: 'node[status = "active"]', style: { "background-color": palette.success } },
          { selector: 'node[status = "needs_review"]', style: { "background-color": palette.warning } },
          { selector: 'node[status = "stale"]', style: { "background-color": palette.danger } },
          { selector: 'node[status = "forgotten"]', style: { "background-color": palette.muted } },
          { selector: 'node[status = "archived"]', style: { "background-color": palette.muted, "background-opacity": 0.65 } },
          { selector: "node:selected", style: { label: "data(label)", "text-opacity": 1, "text-background-opacity": 0.85, "text-background-color": "#161b22", "text-background-padding": "4px", "text-margin-y": "-12px", "border-width": 3, "border-color": palette.link } as any },
          {
            selector: ":selected",
            style: { "border-width": 3, "border-color": palette.link },
          },
        ],
        layout: {
          name: "preset",
          animate: false,
          fit: true,
          padding: 24,
        },
      });
      // Keep the graph centered, but open it slightly closer to the memory
      // nodes than the default fit viewport.
      graph.zoom(Math.min(graph.maxZoom(), graph.zoom() * 1.25));
      graph.on("tap", "node", (event) => onSelect?.(event.target.id()));
      graph.on("mouseover", "node", (event) => { const n = event.target.data(); setTooltip({ x: event.renderedPosition.x, y: event.renderedPosition.y, title: n.label, detail: `${n.status} · ${n.scope || "no scope"} · salience ${Number(n.salience).toFixed(2)} · confidence ${Number(n.confidence).toFixed(2)}` }); });
      graph.on("mouseover", "edge", (event) => { const e = event.target.data(); setTooltip({ x: event.renderedPosition.x, y: event.renderedPosition.y, title: e.relation.replaceAll("_", " "), detail: `${e.sourceLabel} → ${e.targetLabel}` }); });
      graph.on("mouseout", "node, edge", () => setTooltip(null));
    });
    return () => {
      disposed = true;
      graph?.destroy();
    };
  }, [data]);
  return (
    <div className="graph-explorer-wrap"><div className="graph-explorer" ref={container} role="img" aria-label={`Memory graph with ${data.nodes?.length || 0} memories and ${data.edges?.length || 0} relations.`} />{tooltip && <div className="graph-tooltip" style={{ left: tooltip.x + 14, top: tooltip.y + 14 }}><strong>{tooltip.title}</strong><span>{tooltip.detail}</span></div>}</div>
  );
}
