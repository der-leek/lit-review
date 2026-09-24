"""Tests for query-optimizer draft normalization and sentinel handling."""

from __future__ import annotations

from lit_review.query_optimizer import _normalize_drafts


def _drafts(*drafts: dict) -> dict:
    """Wrap draft dicts in a parsed-plan dict."""
    return {"drafts": list(drafts)}


def test_min_citation_count_sentinels_normalized_to_none() -> None:
    """Apple's `-1`/`0` "no minimum" sentinels must not reach the S2 client."""
    result = _normalize_drafts(
        _drafts(
            {
                "query": "federated learning",
                "min_citation_count": -1,
                "year_range": None,
                "fields_of_study": [],
            },
            {
                "query": "medical imaging",
                "min_citation_count": 0,
                "year_range": None,
                "fields_of_study": [],
            },
            {
                "query": "privacy preserving",
                "min_citation_count": None,
                "year_range": None,
                "fields_of_study": [],
            },
        )
    )

    assert [d["min_citation_count"] for d in result] == [None, None, None]


def test_real_min_citation_count_preserved() -> None:
    """A genuine minimum is kept."""
    result = _normalize_drafts(
        _drafts(
            {
                "query": "transformers",
                "min_citation_count": 50,
                "year_range": None,
                "fields_of_study": [],
            },
            {
                "query": "attention",
                "min_citation_count": 10,
                "year_range": None,
                "fields_of_study": [],
            },
            {
                "query": "self attention",
                "min_citation_count": 100,
                "year_range": None,
                "fields_of_study": [],
            },
        )
    )

    assert [d["min_citation_count"] for d in result] == [50, 10, 100]


def test_year_range_sentinel_normalized_to_none() -> None:
    """Empty-string and string-null year ranges become None."""
    result = _normalize_drafts(
        _drafts(
            {
                "query": "neural networks",
                "year_range": "",
                "fields_of_study": [],
            },
            {
                "query": "deep learning",
                "year_range": "2020-2024",
                "fields_of_study": [],
            },
            {
                "query": "transformers",
                "year_range": "null",
                "fields_of_study": [],
            },
        )
    )

    assert [d["year_range"] for d in result] == [None, "2020-2024", None]


def test_pads_to_three_from_one() -> None:
    """Fewer than three drafts repeat the first (best) draft."""
    result = _normalize_drafts(
        _drafts(
            {
                "query": "only draft",
                "min_citation_count": -1,
                "year_range": "",
                "fields_of_study": [],
            }
        )
    )

    assert len(result) == 3
    assert [d["query"] for d in result] == ["only draft", "only draft", "only draft"]
    assert {d["min_citation_count"] for d in result} == {None}
    assert {d["year_range"] for d in result} == {None}


def test_empty_drafts_return_empty_list() -> None:
    """No usable drafts yields an empty list (pipeline aborts cleanly)."""
    assert _normalize_drafts({}) == []
    assert _normalize_drafts({"drafts": []}) == []
    assert _normalize_drafts({"drafts": [{"query": "   "}]}) == []
