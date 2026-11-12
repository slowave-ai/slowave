import sqlite3

import pytest

from slowave.storage.embedding_migration_audit import (
    audit_embedding_migration,
    cue_text_alternatives,
    require_reconstructable_sources,
)


def test_audit_detects_missing_reconstruction_sources():
    db = sqlite3.connect(":memory:")
    db.executescript("""
    CREATE TABLE raw_events(id INTEGER, embedding BLOB);
    CREATE TABLE episodic_memories(id INTEGER);
    CREATE TABLE episode_text(episode_id INTEGER, event_ids TEXT);
    CREATE TABLE schemas(embedding BLOB, n_facet_axes INTEGER, supporting_episode_ids TEXT);
    CREATE TABLE semantic_prototypes(id INTEGER, support_count INTEGER);
    CREATE TABLE episode_prototype_map(prototype_id INTEGER);
    CREATE TABLE prototype_edges(id INTEGER);
    CREATE TABLE context_recall_events(cue_embedding BLOB, cue_dim INTEGER, scope_id TEXT, task_type TEXT, query TEXT, goal TEXT, situation_json TEXT, requirements_json TEXT, topics_json TEXT, entities_json TEXT);
    CREATE TABLE retrieval_cue_prototypes(id INTEGER PRIMARY KEY, embedding BLOB, dim INTEGER, scope_id TEXT, task_type TEXT);
    CREATE TABLE schema_retrieval_evidence(id INTEGER);
    CREATE TABLE schema_passage_embeddings(id INTEGER);
    CREATE TABLE procedure_embeddings(id INTEGER);
    INSERT INTO episodic_memories VALUES (1), (2);
    INSERT INTO episode_text VALUES (1, '{"ids":[1]}');
    INSERT INTO semantic_prototypes VALUES (1, 2), (2, 0);
    INSERT INTO episode_prototype_map VALUES (1);
    INSERT INTO retrieval_cue_prototypes VALUES (1,X'0102',2,'project:a',NULL), (2,X'03',1,'project:b',NULL);
    INSERT INTO context_recall_events(cue_embedding,cue_dim,scope_id,task_type,query) VALUES (X'0102',2,'project:a',NULL,'one'), (X'0102',2,'project:a',NULL,'two');
    INSERT INTO raw_events VALUES (1, X'00');
    INSERT INTO schemas VALUES (X'00',1,'{"ids":[1,2]}');
    """)
    before = db.total_changes
    result = audit_embedding_migration(db)
    assert result["cue_prototypes_with_distinct_snapshot_fields"] == 1
    assert result["counts"]["schema_facets"] == 1
    assert result["counts"]["cue_prototypes"] == 2
    assert result["counts"]["prototype_support_differs_from_members"] == 1
    assert result["missing_sources"] == {
        "episodes_without_provenance": 1,
        "episodes_with_missing_event_vectors": 0,
        "episodes_with_invalid_provenance": 0,
        "facets_with_missing_episode_vectors": 0,
        "semantic_prototypes_without_members": 1,
        "cue_prototypes_without_snapshot": 1,
    }
    assert db.total_changes == before


def test_ambiguous_cue_alternatives_are_collected_without_choosing():
    db = sqlite3.connect(":memory:")
    db.executescript("""
    CREATE TABLE raw_events(id INTEGER, embedding BLOB);
    CREATE TABLE episodic_memories(id INTEGER);
    CREATE TABLE episode_text(episode_id INTEGER, event_ids TEXT);
    CREATE TABLE schemas(embedding BLOB, n_facet_axes INTEGER, supporting_episode_ids TEXT);
    CREATE TABLE semantic_prototypes(id INTEGER, support_count INTEGER);
    CREATE TABLE episode_prototype_map(prototype_id INTEGER);
    CREATE TABLE prototype_edges(id INTEGER);
    CREATE TABLE context_recall_events(cue_embedding BLOB, cue_dim INTEGER, scope_id TEXT, task_type TEXT, query TEXT, goal TEXT, situation_json TEXT, requirements_json TEXT, topics_json TEXT, entities_json TEXT);
    CREATE TABLE retrieval_cue_prototypes(id INTEGER, embedding BLOB, dim INTEGER, scope_id TEXT, task_type TEXT);
    CREATE TABLE schema_retrieval_evidence(id INTEGER);
    CREATE TABLE schema_passage_embeddings(id INTEGER);
    CREATE TABLE procedure_embeddings(id INTEGER);
    INSERT INTO retrieval_cue_prototypes VALUES (1,X'0102',2,'project:a',NULL);
    INSERT INTO schema_retrieval_evidence VALUES (7);
    INSERT INTO context_recall_events VALUES
      (X'0102',2,'project:a',NULL,'first',NULL,'{}','[]','[]','[]'),
      (X'0102',2,'project:a',NULL,'second',NULL,'{}','[]','[]','[]');
    """)
    before = db.total_changes
    require_reconstructable_sources(db)
    alternatives = cue_text_alternatives(db)
    assert len(alternatives[1]) == 2, "both source cues must be preserved"
    assert db.execute("SELECT count(*) FROM schema_retrieval_evidence").fetchone()[0] == 1
    assert db.total_changes == before
    db.execute("DELETE FROM context_recall_events WHERE query='second'")
    require_reconstructable_sources(db)
    assert len(cue_text_alternatives(db)[1]) == 1
