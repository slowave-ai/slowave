"""Published MCP schemas should teach scalar and batch call shapes."""

import jsonschema

from slowave.mcp.tools import FeedbackArguments, RememberArguments, _tool_schema


def test_remember_schema_has_separate_scalar_and_batch_shapes():
    schema = _tool_schema(RememberArguments.model_json_schema(), "slowave_remember")
    assert schema["examples"][0]["memories"]
    scalar = {"scope": "project:test", "session_id": "sess", "content": "Claim", "type": "fact"}
    batch = {
        "scope": "project:test",
        "session_id": "sess",
        "memories": [{"content": "Claim", "type": "fact"}],
    }
    for value in [scalar, batch]:
        jsonschema.validate(value, schema)
    for value in [{**scalar, "memories": batch["memories"]}, {**batch, "type": "fact"}]:
        try:
            jsonschema.validate(value, schema)
        except jsonschema.ValidationError:
            pass
        else:
            raise AssertionError("mixed remember mode passed published schema")


def test_feedback_schema_teaches_items_alone_and_per_item_coverage():
    schema = _tool_schema(FeedbackArguments.model_json_schema(), "slowave_feedback")
    valid = {"items": [{"retrieval_id": "ctx", "coverage": "complete"}]}
    jsonschema.validate(valid, schema)
    for value in [{**valid, "retrieval_id": "ctx"}, {**valid, "coverage": "complete"}]:
        try:
            jsonschema.validate(value, schema)
        except jsonschema.ValidationError:
            pass
        else:
            raise AssertionError("mixed feedback mode passed published schema")
