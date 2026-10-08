"""Shared precedence for current learning evidence; audit rows remain immutable."""


def effective_feedback_sql(alias: str = "f") -> str:
    """Select latest accepted active observation per retrieval and target."""
    return (
        f"{alias}.status = 'accepted' AND {alias}.mutation_mode = 'active' AND NOT EXISTS ("
        "SELECT 1 FROM feedback_events newer "
        f"WHERE newer.retrieval_id = {alias}.retrieval_id "
        f"AND newer.target_kind = {alias}.target_kind AND newer.target_id = {alias}.target_id "
        "AND newer.status = 'accepted' AND newer.mutation_mode = 'active' "
        f"AND (newer.created_at > {alias}.created_at OR "
        f"(newer.created_at = {alias}.created_at AND newer.rowid > {alias}.rowid)))"
    )
