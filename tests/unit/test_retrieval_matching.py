from slowave.core.retrieval_matching import (
    CandidateSignals,
    MatchingConfig,
    match_candidates,
    select_matches,
)


def test_rrf_combines_available_ranks_and_keeps_missing_channels_null() -> None:
    results = match_candidates(
        [
            CandidateSignals(memory_id=9, dense_cosine=0.42, dense_rank=1),
            CandidateSignals(
                memory_id=2, lexical_score=-3.0, lexical_rank=1, lexical_specific=True
            ),
            CandidateSignals(
                memory_id=5,
                dense_cosine=0.31,
                dense_rank=2,
                lexical_score=-1.0,
                lexical_rank=2,
                lexical_specific=True,
            ),
        ],
        config=MatchingConfig(rrf_k=60),
    )

    by_id = {item.memory_id: item for item in results}
    assert by_id[9].fusion_score == 1 / 61
    assert by_id[9].lexical_rank is None
    assert by_id[5].fusion_score == 2 / 62
    assert [item.memory_id for item in results] == [5, 2, 9]


def test_relevance_requires_strong_dense_or_specific_lexical_evidence() -> None:
    results = match_candidates(
        [
            CandidateSignals(memory_id=1, dense_cosine=0.19, dense_rank=1),
            CandidateSignals(
                memory_id=2, lexical_score=-2.0, lexical_rank=1, lexical_specific=False
            ),
            CandidateSignals(
                memory_id=3, lexical_score=-2.0, lexical_rank=2, lexical_specific=True
            ),
        ]
    )
    by_id = {item.memory_id: item for item in results}
    assert not by_id[1].relevance_passed
    assert not by_id[2].relevance_passed
    assert by_id[3].relevance_passed
    assert [item.memory_id for item in select_matches(results, 2)] == [3]


def test_ties_are_deterministic_and_salience_cannot_create_relevance() -> None:
    results = match_candidates(
        [
            CandidateSignals(memory_id=4, dense_cosine=0.1, dense_rank=1, salience=100),
            CandidateSignals(memory_id=2, dense_cosine=0.4, dense_rank=1),
        ]
    )
    assert [item.memory_id for item in results] == [2, 4]
    assert not results[1].relevance_passed
