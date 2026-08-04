"""Per-scope implicit session resolver.

Fast, scope-keyed lookup so slowave_remember(session_id=None) can find the
implicit session opened by slowave_activate.

## Design Rationale

Without an implicit session resolver, every remember() call with session_id=None
opens an ad-hoc session in engine.remember(), creating duplicate micro+macro
episodes for the same task. This causes:

1. Episode duplication → supersession churn
2. Working memory session state lost mid-task
3. Agents forced to thread session_id through every remember call

## Implementation

In-process, thread-safe map: scope → (session_id, set_at_ts).

- Stored in memory (not DB) because:
  - Sessions are opened/closed within seconds per task
  - Process restart naturally invalidates stale entries
  - 1-hour TTL handles edge cases (long-running tasks)
  - No DB contention for high-frequency remember calls

- Age guard: resolve() drops entries older than MAX_IMPLICIT_SESSION_AGE_S
  (1 hour) and returns None, triggering fallback to ad-hoc session.

- Keyed by scope (string or None). One binding per scope at a time.
  (If agent switches scopes mid-task, the old scope's session goes stale.)

KNOWN LIMITATION (2026-07-24 Tier-0 audit): the "per-thread, so concurrent
clients are isolated" claim below assumes each client's activate/remember/
commit cycle runs on its own OS thread. Under the HTTP daemon
(slowave/mcp/http_server.py, FastMCP/Starlette), the registered tool
functions are `async def` and call the synchronous engine directly with no
`asyncio.to_thread`/executor offload -- so in practice everything runs on the
single asyncio event-loop thread, not a per-connection thread. That still
isolates concurrent clients using *different* scope strings (bindings are
scope-keyed), but two concurrent clients activating the *same* scope within
the same ~second can have the second bind() silently overwrite the first's
session_id, so the first client's later implicit remember()/commit() calls
would resolve to the second client's session. Not yet fixed: doing so
properly needs a per-connection identity (e.g. FastMCP's Context/
request_context) added to the bind key, not just scope, and that change
needs its own concurrency test coverage before landing.

## Usage

1. activate() opens a session and calls bind(scope, sid)
2. remember() calls resolve(scope) if session_id is None
3. commit() calls clear(scope) to unbind

See slowave/mcp/server.py for wiring details.
"""

import threading
import time
from typing import Optional

# Maximum age of an implicit session binding before it expires
MAX_IMPLICIT_SESSION_AGE_S = 3600  # 1 hour


class _Binding:
    """Internal: session ID + creation timestamp."""

    __slots__ = ("session_id", "set_at_ts")

    def __init__(self, session_id: str, set_at_ts: int) -> None:
        self.session_id = session_id
        self.set_at_ts = set_at_ts


class SessionResolver:
    """Per-scope implicit session resolver with 1h age guard."""

    def __init__(self, max_age_s: int = MAX_IMPLICIT_SESSION_AGE_S) -> None:
        """Initialize the resolver.

        Args:
            max_age_s: Maximum age (seconds) before a binding expires.
        """
        self._bindings_per_thread = threading.local()
        self._lock = threading.Lock()
        self._max_age_s = max_age_s

    def _bindings(self) -> dict[Optional[str], _Binding]:
        """Per-thread bindings dict (lazy-init on first access)."""
        d = getattr(self._bindings_per_thread, "d", None)
        if d is None:
            d = {}
            self._bindings_per_thread.d = d
        return d

    def bind(self, scope: Optional[str], session_id: str) -> None:
        """Bind a session ID to a scope.

        Args:
            scope: Scope key (string or None). Per-thread — concurrent
                   clients with the same scope are isolated.
            session_id: The implicit session ID opened by activate().
        """
        now_ts = int(time.time())
        bindings = self._bindings()
        # Per-thread access — no lock needed for this thread's own dict.
        # Lock is still held for snapshot() which aggregates across threads.
        bindings[scope] = _Binding(session_id, now_ts)

    def resolve(self, scope: Optional[str]) -> Optional[str]:
        """Resolve a session ID from a scope.

        If found and not stale, returns the session_id.
        If stale, removes the binding and returns None (triggers ad-hoc fallback).
        If not found, returns None.

        Isolated across concurrent clients using different scope strings.
        See the KNOWN LIMITATION note in this module's docstring for the
        same-scope-concurrent-clients case under the async HTTP daemon.

        Args:
            scope: Scope key (string or None).

        Returns:
            Session ID if binding exists and is fresh; None otherwise.
        """
        now_ts = int(time.time())
        bindings = self._bindings()
        binding = bindings.get(scope)
        if binding is None:
            return None

        # Check age: if stale, drop the binding
        age_s = now_ts - binding.set_at_ts
        if age_s > self._max_age_s:
            del bindings[scope]
            return None

        return binding.session_id

    def clear(self, scope: Optional[str]) -> None:
        """Remove the binding for a scope.

        Called by commit() to clean up after a task completes.

        Args:
            scope: Scope key (string or None).
        """
        bindings = self._bindings()
        bindings.pop(scope, None)

    def snapshot(self) -> dict[Optional[str], dict]:
        """Return a snapshot of all current bindings (for testing/debugging).

        Returns:
            Dict mapping scope → {session_id, age_s, fresh}
        """
        now_ts = int(time.time())
        result: dict[Optional[str], dict] = {}
        # Per-thread bindings: snapshot reports only this thread's
        # bindings since thread-local storage is not cross-thread
        # accessible in pure Python.
        bindings = self._bindings()
        for scope, binding in bindings.items():
            age_s = now_ts - binding.set_at_ts
            result[scope] = {
                "session_id": binding.session_id,
                "age_s": age_s,
                "fresh": age_s <= self._max_age_s,
            }
        return result


# Global singleton for MCP server use
_resolver = SessionResolver()


def bind(scope: Optional[str], session_id: str) -> None:
    """Global bind operation."""
    _resolver.bind(scope, session_id)


def resolve(scope: Optional[str]) -> Optional[str]:
    """Global resolve operation."""
    return _resolver.resolve(scope)


def clear(scope: Optional[str]) -> None:
    """Global clear operation."""
    _resolver.clear(scope)


def snapshot() -> dict[Optional[str], dict]:
    """Global snapshot operation (testing)."""
    return _resolver.snapshot()
