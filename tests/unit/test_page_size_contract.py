"""Page-size validation must reject coercion and keep continuation shape exact."""

import pytest
from pydantic import ValidationError

from slowave.mcp.activation_catalog import DEFAULT_MEMORY_PAGE_SIZE
from slowave.mcp.tools import ActivateArguments, RecallArguments


@pytest.mark.parametrize("value", [0, 11, -1, True, False, 5.0, "5"])
def test_page_size_rejects_out_of_range_and_nonintegers(value):
    with pytest.raises(ValidationError):
        ActivateArguments(task="Check", initial_goal="Check", scope="project:test", page_size=value)
    with pytest.raises(ValidationError):
        RecallArguments(session_id="sess", scope="project:test", query="Check", page_size=value)


def test_default_and_continuation_page_size_contract():
    assert (
        ActivateArguments(task="Check", initial_goal="Check", scope="project:test").page_size
        == DEFAULT_MEMORY_PAGE_SIZE
    )
    assert ActivateArguments.model_json_schema()["properties"]["page_size"]["default"] == (
        DEFAULT_MEMORY_PAGE_SIZE
    )
    for value in [1, 5, 10]:
        assert (
            RecallArguments(
                session_id="sess", scope="project:test", query="Check", page_size=value
            ).page_size
            == value
        )
        with pytest.raises(ValidationError):
            RecallArguments(
                session_id="sess", scope="project:test", continue_from="cursor", page_size=value
            )
