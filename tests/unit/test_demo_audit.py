"""Native client output may spill a complete MCP response out of its preview."""

from private.demos.ai_tinkerers_2026.scripts.audit import decode_payload


def test_decode_native_compact_json_spill_after_truncated_pretty_output():
    full = '{"ok":true,"data":{"retrieval_id":"rec_1","memories":[]}}'
    value = {
        "content": 'Output too large to read at once. Preview: {"ok": true, ...',
        "detailedContent": "prelude\n" + full,
    }
    assert decode_payload(value) == {"ok": True, "data": {"retrieval_id": "rec_1", "memories": []}}


def test_decode_nested_native_content_and_empty_result():
    assert decode_payload({"content": [{"type": "text", "text": '{"ok":true,"data":{}}'}]}) == {
        "ok": True,
        "data": {},
    }
    assert decode_payload("no JSON result") is None


def test_native_event_timing_parser_handles_milliseconds_and_missing_times():
    from private.demos.ai_tinkerers_2026.scripts.audit import _seconds_between

    assert _seconds_between("2026-10-13T11:08:18.528Z", "2026-10-13T11:08:22.744Z") == 4.216
    assert _seconds_between(None, "2026-10-13T11:08:22Z") is None
