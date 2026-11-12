"""Quotas that keep the explicit ``always`` flag from monopolising the standing tier.

An ``always`` rule outranks everything, so without a limit a handful of short ``always`` writes
(by a buggy or hostile agent) evict every ordinary rule.  Limits, applied before ranking:

* at most ``ALWAYS_MAX_SLOTS`` (3) ``always`` rules - a user rarely has more than a few rules
  that must reach *every* task; three keeps the strongest ones without crowding the tier;
* at most ``ALWAYS_MAX_CHAR_FRACTION`` (40%) of the character budget, so two long ``always`` rules
  still leave 60% of the budget (about 3-4 rules) for situation-ranked ones;
* surplus ``always`` rules are demoted to ordinary rules (still eligible, ranked by situation),
  not dropped.

Separately a *floor* guarantees the best ``FLOOR_PROJECT`` project-scope and ``FLOOR_GLOBAL``
user-global ordinary rules are placed ahead of the remaining ordinary rules, so neither scope can
starve the other.  Both functions are pure and operate on ``Candidate``-like objects.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

ALWAYS_MAX_SLOTS = 3
ALWAYS_MAX_CHAR_FRACTION = 0.4
FLOOR_PROJECT = 3
FLOOR_GLOBAL = 1


def cap_always(cands: Sequence[Any], cost: Callable[[Any], int], max_chars: int) -> int:
    """Demote surplus ``always`` candidates in place; return how many were demoted."""
    flagged = sorted(
        (c for c in cands if c.always), key=lambda c: (-c.salience, -c.ts, c.memory_id)
    )
    budget = int(max_chars * ALWAYS_MAX_CHAR_FRACTION)
    kept = spent = demoted = 0
    for c in flagged:
        if kept < ALWAYS_MAX_SLOTS and spent + cost(c) <= budget:
            kept += 1
            spent += cost(c)
        else:
            c.always = False
            demoted += 1
    return demoted


def floor_order(ordered: Sequence[Any]) -> list[Any]:
    """Reorder an already ranked list: always rules, floor picks, then the rest (rank order)."""
    always = [c for c in ordered if c.always]
    rest = [c for c in ordered if not c.always]
    picked: set[int] = set()
    for want_glob, n in ((False, FLOOR_PROJECT), (True, FLOOR_GLOBAL)):
        for c in [c for c in rest if c.glob is want_glob][:n]:
            picked.add(c.memory_id)
    floor = [c for c in rest if c.memory_id in picked]
    return always + floor + [c for c in rest if c.memory_id not in picked]
