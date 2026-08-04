#!/usr/bin/env python3
"""Synthetic ground-truth benchmark for procedural session clustering.

The content-proxy backtest (validate_procedural_clustering_backtest.py)
showed the embedding+alignment method recovers real structure that TF-IDF
misses, but on real historical data it recovered *topical* continuity, not
*repeated-procedure* structure -- because that corpus (remember:* content)
structurally can't contain repeated action-procedures. This script isolates
the one question that corpus can't answer: given genuine repeated
procedures at varying levels of lexical similarity, does either method
actually recognize them as the same procedure?

Design goals (avoid overfitting the test to the expected answer):
  - Ground truth is fixed BEFORE looking at any clustering output.
  - Four concepts (unrelated domains) each expressed across four similarity
    tiers, written independently per tier rather than derived by lightly
    editing the canonical version -- tier D is written with a genuinely
    disjoint vocabulary, not a light paraphrase of tier A.
  - Topic decoys: same surface vocabulary as a concept, but a genuinely
    different procedure (e.g. "rollback a deploy" vs "deploy a service").
    These test whether a method clusters by *topic* (as the real-data
    backtest showed happens) instead of by *procedure*.
  - Unrelated distractors: no shared vocabulary or concept with anything.
  - Evaluated as pairwise precision/recall/F1 against ground truth, swept
    across a full threshold range for BOTH methods -- not a single
    threshold chosen after seeing results. Recall is also broken down by
    tier-pair difficulty, and decoy contamination is reported explicitly.

Usage:
    python scripts/validate_procedural_clustering_synthetic.py
    python scripts/validate_procedural_clustering_synthetic.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_procedural_signal as baseline  # noqa: E402
import validate_procedural_clustering_backtest as vb  # noqa: E402


# ---------------------------------------------------------------------------
# Ground-truth synthetic dataset
# ---------------------------------------------------------------------------
# tier: A = near-duplicate (positive control), B = paraphrase,
#       C = reordered/inserted/deleted steps, D = same concept, ~zero
#       lexical overlap with tier A of the same concept.
# concept: ground-truth label. Two sessions are a "true pair" iff they share
#       a non-null concept. decoy_of: topic overlaps a concept but is a
#       genuinely different procedure -- must NOT be a true pair with it.

SESSIONS: list[dict[str, Any]] = [
    # --- Concept: deploy_service ---------------------------------------
    dict(id="deploy_A1", concept="deploy_service", tier="A", outcome="success",
         goal="deploy auth service to production",
         steps=["Ran full regression test suite (312 passed)",
                "Built Docker image auth-service:v4.2.0",
                "Pushed image to container registry",
                "Applied rolling update to production",
                "Verified health endpoint returns 200"]),
    dict(id="deploy_A2", concept="deploy_service", tier="A", outcome="success",
         goal="deploy billing service to production",
         steps=["Ran full regression test suite (298 passed)",
                "Built Docker image billing-service:v2.1.0",
                "Pushed image to container registry",
                "Applied rolling update to production",
                "Verified health endpoint returns 200"]),
    dict(id="deploy_B1", concept="deploy_service", tier="B", outcome="success",
         goal="ship the notifications service",
         steps=["Executed the complete test battery and confirmed all green",
                "Assembled a fresh container image for the release",
                "Uploaded the image to the artifact repository",
                "Rolled the new version out gradually across production nodes",
                "Confirmed the service responds healthy after rollout"]),
    dict(id="deploy_C1", concept="deploy_service", tier="C", outcome="success",
         goal="release the search-indexer service",
         steps=["Built the container image for search-indexer v1.8",
                "Ran the automated test suite before shipping (all passed)",
                "Notified the on-call channel about the upcoming rollout",
                "Pushed the image to the registry",
                "Rolled out to production nodes in batches",
                "Checked the health endpoint post-rollout"]),
    dict(id="deploy_D1", concept="deploy_service", tier="D", outcome="success",
         goal="get the new payments module live for customers",
         steps=["Confirmed nothing was broken by running every automated check we have",
                "Packaged the latest code into a portable unit ready to run anywhere",
                "Sent that package to the shared location other machines pull from",
                "Swapped the running instances over to the new version a few at a time",
                "Made sure the service was answering correctly once switched over"]),
    dict(id="deploy_decoy", concept=None, decoy_of="deploy_service", tier="decoy",
         outcome="success", goal="rollback a bad production deploy",
         steps=["Identified the last known-good image tag from deploy history",
                "Re-pointed production traffic back to the previous image immediately",
                "Killed the pods running the broken version",
                "Filed an incident report documenting the failed deploy",
                "Scheduled a retro to discuss what went wrong"]),

    # --- Concept: debug_flaky_test --------------------------------------
    dict(id="debug_A1", concept="debug_flaky_test", tier="A", outcome="success",
         goal="fix flaky test in auth module",
         steps=["Reproduced the failure locally by re-running the test 20 times",
                "Bisected the last 10 commits to find which one introduced the flake",
                "Found the commit that removed a required sleep/wait",
                "Reverted the offending change and added a proper wait condition",
                "Re-ran the test 50 times with zero failures to confirm the fix"]),
    dict(id="debug_A2", concept="debug_flaky_test", tier="A", outcome="success",
         goal="fix flaky test in payments module",
         steps=["Reproduced the failure locally by re-running the test 20 times",
                "Bisected the last 10 commits to find which one introduced the flake",
                "Found the commit that introduced a race condition in setup",
                "Reverted the offending change and added proper synchronization",
                "Re-ran the test 50 times with zero failures to confirm the fix"]),
    dict(id="debug_B1", concept="debug_flaky_test", tier="B", outcome="success",
         goal="chase down an intermittent test failure in checkout",
         steps=["Got the test to fail on my machine by looping it many times",
                "Walked backwards through recent commits to isolate the culprit",
                "Pinpointed a change that dropped an important delay",
                "Undid that change and put in place a correct wait",
                "Looped the test dozens more times with no further failures"]),
    dict(id="debug_C1", concept="debug_flaky_test", tier="C", outcome="partial",
         goal="stabilize flaky checkout test",
         steps=["Opened a ticket to track the flaky test investigation",
                "Bisected recent commits to find the introducing change",
                "Reproduced the failure by running the test repeatedly",
                "Identified a missing synchronization point as the cause",
                "Added the fix and reran the suite to confirm stability"]),
    dict(id="debug_D1", concept="debug_flaky_test", tier="D", outcome="success",
         goal="work out why a check keeps failing sometimes",
         steps=["Made the problem show up reliably by trying it over and over",
                "Went back through what changed recently to narrow down the cause",
                "Landed on one change that took out something the timing depended on",
                "Put things back the way they should be and handled the timing properly",
                "Tried it many more times and it held up every time"]),
    dict(id="debug_decoy", concept=None, decoy_of="debug_flaky_test", tier="decoy",
         outcome="success", goal="add test coverage for the payments retry logic",
         steps=["Reviewed the payments retry code to understand untested paths",
                "Wrote three new unit tests covering retry backoff behavior",
                "Added a test for the max-retries-exhausted case",
                "Ran the new tests locally to confirm they pass",
                "Opened a PR with the new test coverage"]),

    # --- Concept: investigate_incident ----------------------------------
    dict(id="incident_A1", concept="investigate_incident", tier="A", outcome="success",
         goal="investigate checkout latency spike incident",
         steps=["Pulled error logs for the affected time window",
                "Correlated the spike with a recent config change",
                "Identified the change as the root cause",
                "Rolled back the config change",
                "Confirmed latency returned to baseline"]),
    dict(id="incident_A2", concept="investigate_incident", tier="A", outcome="success",
         goal="investigate login failure spike incident",
         steps=["Pulled error logs for the affected time window",
                "Correlated the spike with a recent config change",
                "Identified the change as the root cause",
                "Rolled back the config change",
                "Confirmed error rate returned to baseline"]),
    dict(id="incident_B1", concept="investigate_incident", tier="B", outcome="success",
         goal="figure out the cause of the API 500 spike",
         steps=["Gathered the error logs around the time things went wrong",
                "Lined up the spike against anything deployed recently",
                "Traced it to a configuration tweak that had just shipped",
                "Reverted that configuration change",
                "Watched the error rate drop back to normal"]),
    dict(id="incident_C1", concept="investigate_incident", tier="C", outcome="success",
         goal="root-cause the search outage",
         steps=["Notified stakeholders that an investigation was underway",
                "Gathered logs from the incident window",
                "Cross-referenced the timing with recent deploys and config changes",
                "Found the responsible config change and reverted it",
                "Verified metrics recovered and closed the incident"]),
    dict(id="incident_D1", concept="investigate_incident", tier="D", outcome="success",
         goal="understand why things briefly broke for users",
         steps=["Collected the records covering the rough period",
                "Lined that window up against anything that had just shipped",
                "Traced the trouble to a setting that had just been flipped",
                "Put that setting back the way it was",
                "Watched things return to how they should be"]),
    dict(id="incident_decoy", concept=None, decoy_of="investigate_incident", tier="decoy",
         outcome="success", goal="write the incident postmortem",
         steps=["Gathered the timeline of events from the incident channel",
                "Summarized root cause and impact for stakeholders",
                "Documented the remediation steps taken",
                "Listed follow-up action items with owners",
                "Shared the postmortem doc for review"]),

    # --- Concept: local_dev_setup ----------------------------------------
    dict(id="devsetup_A1", concept="local_dev_setup", tier="A", outcome="success",
         goal="set up local dev env for frontend repo",
         steps=["Cloned the frontend repository",
                "Installed dependencies via package manager",
                "Copied .env.example to .env and filled in local values",
                "Ran the dev server and confirmed it loads at localhost"]),
    dict(id="devsetup_A2", concept="local_dev_setup", tier="A", outcome="success",
         goal="set up local dev env for backend repo",
         steps=["Cloned the backend repository",
                "Installed dependencies via package manager",
                "Copied .env.example to .env and filled in local values",
                "Ran the dev server and confirmed it loads at localhost"]),
    dict(id="devsetup_B1", concept="local_dev_setup", tier="B", outcome="success",
         goal="get the mobile repo running locally",
         steps=["Grabbed a local copy of the mobile codebase",
                "Pulled in all required packages",
                "Duplicated the example environment file and populated real values",
                "Started the local server and checked it came up correctly"]),
    dict(id="devsetup_C1", concept="local_dev_setup", tier="C", outcome="success",
         goal="onboard onto the analytics repo",
         steps=["Requested repo access from the team lead",
                "Cloned the analytics repository",
                "Copied the example env file and filled in local secrets",
                "Installed dependencies",
                "Started the server and verified it was reachable locally"]),
    dict(id="devsetup_D1", concept="local_dev_setup", tier="D", outcome="success",
         goal="get a brand-new codebase working on my machine",
         steps=["Pulled down a personal copy of the project",
                "Brought in everything the project needs to run",
                "Made my own private copy of the settings template with real values",
                "Fired it up and made sure it answered on my machine"]),
    dict(id="devsetup_decoy", concept=None, decoy_of="local_dev_setup", tier="decoy",
         outcome="success", goal="upgrade the shared UI library version across repos",
         steps=["Bumped the UI library version in package.json",
                "Ran the test suite in each consuming repo",
                "Fixed two breaking prop-type changes",
                "Opened PRs in each affected repo"]),

    # --- Unrelated distractors (no concept, no decoy_of) -----------------
    dict(id="unrelated_1", concept=None, tier="unrelated", outcome="success",
         goal="update onboarding documentation",
         steps=["Reviewed the current onboarding doc for outdated screenshots",
                "Rewrote the setup section for the new CLI flow",
                "Added a troubleshooting FAQ section",
                "Asked two new hires to review it for clarity"]),
    dict(id="unrelated_2", concept=None, tier="unrelated", outcome="success",
         goal="renew SSL certificate for the docs site",
         steps=["Checked cert expiry date in the registrar dashboard",
                "Requested a new certificate via ACME",
                "Uploaded the new cert to the CDN",
                "Verified HTTPS still works from a clean browser"]),
]


def _to_engine_sessions(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": s["id"],
            "goal": s["goal"],
            "outcome": s["outcome"],
            "scope_id": "project:synthetic",
            "step_contents": s["steps"],
            "has_steps": True,
            "concept": s.get("concept"),
            "tier": s["tier"],
            "decoy_of": s.get("decoy_of"),
        }
        for s in raw
    ]


# ---------------------------------------------------------------------------
# Ground-truth transparency check: how disjoint is tier D really?
# ---------------------------------------------------------------------------


def _word_overlap(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def report_lexical_overlap_sanity_check(sessions: list[dict[str, Any]]) -> None:
    print("Sanity check: word-overlap (Jaccard-min) of each tier vs that concept's tier A,")
    print("joined step text -- confirms tier D is genuinely non-overlapping, not a light edit.\n")
    by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in sessions:
        if s["concept"]:
            by_concept[s["concept"]].append(s)
    for concept, members in by_concept.items():
        tier_a = next(m for m in members if m["tier"] == "A")
        a_text = " ".join(tier_a["step_contents"])
        print(f"  {concept}:")
        for m in sorted(members, key=lambda x: x["tier"]):
            if m["id"] == tier_a["id"]:
                continue
            overlap = _word_overlap(a_text, " ".join(m["step_contents"]))
            print(f"    tier {m['tier']} ({m['id']}): word overlap vs tier A = {overlap:.2f}")
    print()


# ---------------------------------------------------------------------------
# Ground-truth pairwise evaluation
# ---------------------------------------------------------------------------


def _true_pairs(sessions: list[dict[str, Any]]) -> set[frozenset[str]]:
    pairs = set()
    for i in range(len(sessions)):
        for j in range(i + 1, len(sessions)):
            si, sj = sessions[i], sessions[j]
            if si["concept"] and si["concept"] == sj["concept"]:
                pairs.add(frozenset((si["id"], sj["id"])))
    return pairs


def _predicted_pairs(clusters: dict[int, list[dict[str, Any]]]) -> set[frozenset[str]]:
    pairs = set()
    for members in clusters.values():
        ids = [m["id"] for m in members]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.add(frozenset((ids[i], ids[j])))
    return pairs


def _prf(true_pairs: set, pred_pairs: set) -> dict[str, float]:
    tp = len(true_pairs & pred_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _tier_pair_recall(
    sessions: list[dict[str, Any]], true_pairs: set, pred_pairs: set
) -> dict[str, tuple[int, int]]:
    id_to_tier = {s["id"]: s["tier"] for s in sessions}
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for pair in true_pairs:
        a, b = tuple(pair)
        key = "-".join(sorted((id_to_tier[a], id_to_tier[b])))
        buckets[key][1] += 1
        if pair in pred_pairs:
            buckets[key][0] += 1
    return {k: (v[0], v[1]) for k, v in sorted(buckets.items())}


def _decoy_contamination(
    sessions: list[dict[str, Any]], clusters: dict[int, list[dict[str, Any]]]
) -> list[tuple[str, str]]:
    contaminated = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        concepts_in_cluster = {m["concept"] for m in members if m["concept"]}
        for d in members:
            if d.get("decoy_of") and d["decoy_of"] in concepts_in_cluster:
                contaminated.append((d["id"], d["decoy_of"]))
    return contaminated


# ---------------------------------------------------------------------------
# Sweep runners
# ---------------------------------------------------------------------------


def sweep_old(sessions: list[dict[str, Any]], true_pairs: set, thresholds: list[float]) -> list[dict[str, Any]]:
    rows = []
    for thr in thresholds:
        clusters = baseline.cluster_by_step_content(sessions, threshold=thr)
        pred_pairs = _predicted_pairs(clusters)
        metrics = _prf(true_pairs, pred_pairs)
        contamination = _decoy_contamination(sessions, clusters)
        rows.append({"threshold": thr, **metrics, "contamination": contamination,
                      "n_clusters": sum(1 for m in clusters.values() if len(m) >= 2)})
    return rows


def sweep_new(
    sessions: list[dict[str, Any]],
    true_pairs: set,
    thresholds: list[float],
    step_cache: dict,
    goal_cache: dict,
) -> list[dict[str, Any]]:
    rows = []
    for thr in thresholds:
        clusters = vb.cluster_by_embedding_alignment(sessions, step_cache, goal_cache, thr)
        pred_pairs = _predicted_pairs(clusters)
        metrics = _prf(true_pairs, pred_pairs)
        contamination = _decoy_contamination(sessions, clusters)
        rows.append({"threshold": thr, **metrics, "contamination": contamination,
                      "n_clusters": sum(1 for m in clusters.values() if len(m) >= 2)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    sessions = _to_engine_sessions(SESSIONS)
    n_concept_members = sum(1 for s in sessions if s["concept"])
    n_decoys = sum(1 for s in sessions if s.get("decoy_of"))
    n_unrelated = sum(1 for s in sessions if s["tier"] == "unrelated")

    if not args.json:
        print(f"Synthetic dataset: {len(sessions)} sessions "
              f"({n_concept_members} true concept members across 4 concepts, "
              f"{n_decoys} topic decoys, {n_unrelated} unrelated distractors)\n")
        report_lexical_overlap_sanity_check(sessions)

    true_pairs = _true_pairs(sessions)

    # OLD: TF-IDF cosine over short paraphrased sentences runs low -- sweep wide.
    old_thresholds = [round(x * 0.05, 2) for x in range(1, 15)]  # 0.05 .. 0.70
    old_rows = sweep_old(sessions, true_pairs, old_thresholds)

    from slowave.symbolic.encoder import TextEncoder

    encoder = TextEncoder()
    step_cache, goal_cache = vb._build_embedding_cache(sessions, encoder)
    new_thresholds = [round(x * 0.05, 2) for x in range(4, 16)]  # 0.20 .. 0.75
    new_rows = sweep_new(sessions, true_pairs, new_thresholds, step_cache, goal_cache)

    best_old = max(old_rows, key=lambda r: r["f1"])
    best_new = max(new_rows, key=lambda r: r["f1"])

    if args.json:
        print(json.dumps({
            "n_true_pairs": len(true_pairs),
            "old_sweep": old_rows,
            "new_sweep": new_rows,
            "best_old": best_old,
            "best_new": best_new,
        }, indent=2, default=str))
        return

    print(f"Total true (same-concept) pairs: {len(true_pairs)}\n")

    print("=" * 78)
    print("OLD (TF-IDF + positional) threshold sweep")
    print("=" * 78)
    print(f"{'thr':>5} {'clusters':>9} {'P':>6} {'R':>6} {'F1':>6} {'tp':>4} {'fp':>4} {'fn':>4}  contamination")
    for r in old_rows:
        contam = ",".join(f"{a}->{b}" for a, b in r["contamination"]) or "-"
        print(f"{r['threshold']:>5} {r['n_clusters']:>9} {r['precision']:>6.2f} {r['recall']:>6.2f} "
              f"{r['f1']:>6.2f} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4}  {contam}")

    print(f"\nBest OLD by F1: threshold={best_old['threshold']} "
          f"P={best_old['precision']:.2f} R={best_old['recall']:.2f} F1={best_old['f1']:.2f}")

    print("\n" + "=" * 78)
    print("NEW (embedding + alignment) threshold sweep")
    print("=" * 78)
    print(f"{'thr':>5} {'clusters':>9} {'P':>6} {'R':>6} {'F1':>6} {'tp':>4} {'fp':>4} {'fn':>4}  contamination")
    for r in new_rows:
        contam = ",".join(f"{a}->{b}" for a, b in r["contamination"]) or "-"
        print(f"{r['threshold']:>5} {r['n_clusters']:>9} {r['precision']:>6.2f} {r['recall']:>6.2f} "
              f"{r['f1']:>6.2f} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4}  {contam}")

    print(f"\nBest NEW by F1: threshold={best_new['threshold']} "
          f"P={best_new['precision']:.2f} R={best_new['recall']:.2f} F1={best_new['f1']:.2f}")

    # Tier-pair recall breakdown at each method's own best-F1 threshold.
    print("\n" + "=" * 78)
    print("Recall by tier-pair difficulty (at each method's best-F1 threshold)")
    print("=" * 78)
    old_clusters_best = baseline.cluster_by_step_content(sessions, threshold=best_old["threshold"])
    new_clusters_best = vb.cluster_by_embedding_alignment(sessions, step_cache, goal_cache, best_new["threshold"])
    old_pred = _predicted_pairs(old_clusters_best)
    new_pred = _predicted_pairs(new_clusters_best)
    old_buckets = _tier_pair_recall(sessions, true_pairs, old_pred)
    new_buckets = _tier_pair_recall(sessions, true_pairs, new_pred)
    all_keys = sorted(set(old_buckets) | set(new_buckets))
    print(f"{'tier pair':>12}  {'OLD found/total':>16}  {'NEW found/total':>16}")
    for k in all_keys:
        of, ot = old_buckets.get(k, (0, 0))
        nf, nt = new_buckets.get(k, (0, 0))
        print(f"{k:>12}  {of:>7}/{ot:<7}  {nf:>7}/{nt:<7}")


if __name__ == "__main__":
    main()
