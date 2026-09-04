import { useLocation } from "./api";
import { AppShell } from "./components";
import {
  ActivityPage,
  DiagnosticsPage,
  GraphPage,
  HomePage,
  MemoryPage,
  ProceduresPage,
  RetrievalPage,
} from "./pages";

export default function App() {
  const location = useLocation();
  const path = location.path;
  let page;
  if (path === "/") page = <HomePage location={location} />;
  else if (path === "/memory" || path.startsWith("/memory/"))
    page = <MemoryPage location={location} />;
  else if (path === "/retrieval" || path.startsWith("/retrieval/"))
    page = <RetrievalPage location={location} />;
  else if (path === "/procedures" || path.startsWith("/procedures/"))
    page = <ProceduresPage location={location} />;
  else if (path === "/activity" || path.startsWith("/activity/"))
    page = <ActivityPage location={location} />;
  else if (path === "/graph") page = <GraphPage />;
  else if (path === "/docs") page = <div className="page"><header className="page-header"><div><h1>Docs</h1><p>Documentation is coming soon.</p></div></header></div>;
  else page = <DiagnosticsPage location={location} />;
  return <AppShell path={path}>{page}</AppShell>;
}
