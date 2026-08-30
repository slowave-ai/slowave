#!/usr/bin/env python3
"""Generate a sanitized Slowave demo database for launch screenshots.

Produces a fresh SQLite database containing only invented, non-sensitive
content: no private prompts, local paths, identifiers, or project details.
The dataset is shaped so the beta dashboard's launch-gate surfaces are
populated and legible:

  * Home "Memory effectiveness" (cohort-correct, numerator/denominator)
  * Per-scope selection (three invented scopes)
  * Retrieval value trace (task -> exposed items -> assessment -> outcome)
  * Memories table exposure/usage columns

Usage:
    python scripts/generate_demo_data.py --db ./demo-slowave.db
    slowave dashboard --db ./demo-slowave.db

Timestamps are relative to generation time so the Home activity lanes and
"recent changes" are populated; the relative structure is deterministic.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

from slowave.core.config import SlowaveConfig
from slowave.storage.sqlite_db import SQLiteConfig, SQLiteDB

# Invented memories: (scope, content, schema_class).
MEMORIES = [
    (
        "project:acme-web",
        "The checkout flow must keep the 3-step funnel; removing a step measurably lowered conversion.",
        "decision",
    ),
    (
        "project:acme-web",
        "Prefer server-side rendering for the marketing pages to keep first paint under 1.2s.",
        "preference",
    ),
    (
        "project:acme-web",
        "The billing service idempotency key is required on every retry; omitting it double-charges.",
        "constraint",
    ),
    (
        "project:acme-web",
        "Session cookies are HttpOnly and SameSite=Lax; never set them from client JS.",
        "constraint",
    ),
    (
        "project:acme-mobile",
        "The app targets the two most recent OS majors only; older versions are out of support.",
        "decision",
    ),
    (
        "project:acme-mobile",
        "Offline sync uses a last-write-wins merge with a server timestamp tiebreak.",
        "decision",
    ),
    (
        "project:acme-mobile",
        "Push notifications are opt-in and must be re-confirmed after a reinstall.",
        "constraint",
    ),
    (
        "project:acme-infra",
        "Deploys are blue/green behind the load balancer; rollback is a single DNS flip.",
        "decision",
    ),
    (
        "project:acme-infra",
        "The staging database is reseeded from a sanitized production snapshot nightly.",
        "fact",
    ),
    (
        "project:acme-infra",
        "Alert on p95 latency above 400ms for the API tier; page only if it holds for 5 minutes.",
        "preference",
    ),
]

# Invented sessions: (id, scope, goal, outcome, feedback_status).
SESSIONS = [
    ("sess_1", "project:acme-web", "Fix the checkout conversion regression", "success", "complete"),
    (
        "sess_2",
        "project:acme-web",
        "Add idempotency to the billing retry path",
        "success",
        "complete",
    ),
    ("sess_3", "project:acme-mobile", "Implement offline sync merge", "partial", "complete"),
    ("sess_4", "project:acme-infra", "Set up blue/green deploys", "success", "complete"),
    ("sess_5", "project:acme-web", "Audit cookie flags across the app", "success", "incomplete"),
    ("sess_6", "project:acme-mobile", "Revisit push notification opt-in", "failure", "complete"),
]

# Invented retrievals: (context_id, session_id, scope, query, count_n, seconds_ago).
RETRIEVALS = [
    ("ctx_1", "sess_1", "project:acme-web", "why did checkout conversion drop", 3, 3600),
    ("ctx_2", "sess_2", "project:acme-web", "billing retry idempotency", 2, 7200),
    ("ctx_3", "sess_3", "project:acme-mobile", "offline sync conflict resolution", 2, 10800),
    ("ctx_4", "sess_4", "project:acme-infra", "deploy rollback strategy", 2, 14400),
    ("ctx_5", "sess_5", "project:acme-web", "cookie security flags", 2, 18000),
    ("ctx_6", "sess_6", "project:acme-mobile", "push notification opt-in rules", 0, 21600),
]

# Exposed items: (context_id, memory_id, memory_type, rank, pathway).
EXPOSURES = [
    ("ctx_1", "sch_1", "schema", 1, "direct"),
    ("ctx_1", "sch_2", "schema", 2, "graph"),
    ("ctx_1", "sch_3", "schema", 3, "exploration"),
    ("ctx_2", "sch_3", "schema", 1, "direct"),
    ("ctx_2", "sch_4", "schema", 2, "graph"),
    ("ctx_3", "sch_5", "schema", 1, "direct"),
    ("ctx_3", "sch_6", "schema", 2, "graph"),
    ("ctx_4", "sch_8", "schema", 1, "direct"),
    ("ctx_4", "proc_1", "procedural_memory", 2, "direct"),
    ("ctx_5", "sch_4", "schema", 1, "direct"),
    ("ctx_5", "sch_1", "schema", 2, "graph"),
]

# Feedback: (event_id, retrieval_id, target_kind, target_id, assessment, effect).
FEEDBACK = [
    ("fb_1", "ctx_1", "memory", "sch_1", "used", None),
    ("fb_2", "ctx_1", "memory", "sch_2", "irrelevant", None),
    ("fb_3", "ctx_1", "memory", "sch_3", "used", None),
    ("fb_4", "ctx_2", "memory", "sch_3", "used", None),
    ("fb_5", "ctx_2", "memory", "sch_4", "stale", None),
    ("fb_6", "ctx_3", "memory", "sch_5", "used", None),
    ("fb_7", "ctx_3", "memory", "sch_6", "wrong", None),
    ("fb_8", "ctx_4", "memory", "sch_8", "used", None),
    ("fb_9", "ctx_4", "procedure", "proc_1", "used", "helped"),
    ("fb_10", "ctx_5", "memory", "sch_4", "used", None),
    ("fb_11", "ctx_5", "memory", "sch_1", "irrelevant", None),
]

# A captured procedure (version 2) recorded as a task_complete raw event.
PROCEDURE_EVENT = {
    "procedure": {
        "version": 2,
        "summary": "Ship a blue/green deploy with a single-DNS-flip rollback.",
        "context": {"stack": "load balancer + two identical app tiers"},
        "steps": [
            {"summary": "Provision the green tier alongside blue."},
            {"summary": "Run the smoke suite against green."},
            {"summary": "Flip the load balancer to green."},
            {"summary": "Keep blue warm for one hour before teardown."},
        ],
        "caveats": ["Do not flip during a traffic spike."],
    },
    "procedure_uses": [
        {
            "procedure_id": "proc_1",
            "use": "used",
            "effect": "helped",
            "contribution": "Rollback completed in under a minute.",
        }
    ],
}


def _connect(path: Path) -> sqlite3.Connection:
    database = SQLiteDB(SQLiteConfig(path=str(path)))
    database.init_schema(SlowaveConfig.default_schema_path())
    database.close()
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def seed(connection: sqlite3.Connection) -> None:
    now = int(time.time())
    for index, (scope, content, schema_class) in enumerate(MEMORIES, 1):
        connection.execute(
            "INSERT INTO schemas (content_text, scope_id, status, first_formed_ts, "
            "last_updated_ts, facets_json) VALUES (?, ?, 'active', ?, ?, ?)",
            (
                content,
                scope,
                now - 1000 * index,
                now - 100 * index,
                json.dumps({"schema_class": schema_class}),
            ),
        )
    for session_id, scope, goal, outcome, feedback_status in SESSIONS:
        connection.execute(
            "INSERT INTO sessions (id, agent, scope_id, started_ts, ended_ts, goal, "
            "initial_goal, final_goal, outcome, outcome_summary, feedback_status, "
            "lifecycle_version) VALUES (?, 'demo-agent', ?, ?, ?, ?, ?, ?, ?, ?, ?, 'v10')",
            (
                session_id,
                scope,
                now - 3600,
                now - 3000,
                goal,
                goal,
                goal,
                outcome,
                f"Demo outcome for {goal}.",
                feedback_status,
            ),
        )
    for context_id, session_id, scope, query, count_n, seconds_ago in RETRIEVALS:
        connection.execute(
            "INSERT INTO context_recall_events (context_id, retrieval_type, session_id, "
            "scope_id, query, count_n, created_at, lifecycle_version) "
            "VALUES (?, 'context', ?, ?, ?, ?, ?, 'v10')",
            (context_id, session_id, scope, query, count_n, now - seconds_ago),
        )
    for context_id, memory_id, memory_type, rank, pathway in EXPOSURES:
        connection.execute(
            "INSERT INTO context_recall_items (context_id, memory_id, memory_type, rank, "
            "content_text, admitted, pathway, created_at) "
            "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
            (context_id, memory_id, memory_type, rank, "demo content", pathway, now),
        )
    for event_id, retrieval_id, target_kind, target_id, assessment, effect in FEEDBACK:
        connection.execute(
            "INSERT INTO feedback_events (event_id, retrieval_id, target_kind, target_id, "
            "assessment, effect, coverage, source_contract, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'complete', 'slowave_feedback:v9', ?)",
            (event_id, retrieval_id, target_kind, target_id, assessment, effect, now - 100),
        )
    connection.execute(
        "INSERT INTO raw_events (session_id, ts, type, content, metadata_json, logic_version) "
        "VALUES ('sess_4', ?, 'task_complete', 'captured procedure', ?, '0')",
        (now - 200, json.dumps(PROCEDURE_EVENT)),
    )
    connection.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="./demo-slowave.db", help="Output database path")
    args = parser.parse_args()
    path = Path(args.db).expanduser().resolve()
    if path.exists():
        os.remove(path)
    connection = _connect(path)
    try:
        seed(connection)
    finally:
        connection.close()
    print(f"Wrote sanitized demo database: {path}")
    print("Open it with:  slowave dashboard --db " + str(path))


if __name__ == "__main__":
    main()
