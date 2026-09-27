import pytest

from auto_lora.matching import (
    LoraRow,
    RowsFormatError,
    alias_pattern,
    find_matches,
    parse_rows,
    split_aliases,
)


def compiled(alias):
    pattern = alias_pattern(alias)
    assert pattern is not None
    return pattern


def matched_loras(rows, prompt):
    matches, _ = find_matches(rows, prompt)
    return [m.row.lora for m in matches]


class TestParseRows:
    def test_empty_values(self):
        assert parse_rows(None) == []
        assert parse_rows("") == []
        assert parse_rows("   ") == []
        assert parse_rows("[]") == []

    def test_json_string(self):
        rows = parse_rows(
            '[{"lora": "a.safetensors", "triggers": "alice", "regex": false,'
            ' "strength": 0.8, "enabled": true}]'
        )
        assert rows == [LoraRow(lora="a.safetensors", triggers="alice", strength=0.8)]

    def test_defaults_for_missing_fields(self):
        assert parse_rows('[{"lora": "a.safetensors"}]') == [LoraRow(lora="a.safetensors")]

    def test_accepts_python_list(self):
        assert parse_rows([{"lora": "a.safetensors", "triggers": "x"}]) == [
            LoraRow(lora="a.safetensors", triggers="x")
        ]

    def test_accepts_python_repr_string(self):
        # ComfyUI str()-coerces a list sent through the API for a STRING input.
        value = str([{"lora": "a.safetensors", "regex": True, "enabled": False}])
        assert parse_rows(value) == [LoraRow(lora="a.safetensors", regex=True, enabled=False)]

    def test_numeric_string_strength(self):
        assert parse_rows('[{"lora": "a", "strength": "0.5"}]')[0].strength == 0.5

    @pytest.mark.parametrize(
        "value, message",
        [
            ("not json", "not valid JSON"),
            ('{"lora": "a"}', "must be a list"),
            ('["a"]', "row 1 must be an object"),
            ('[{"lora": "a", "strength": "strong"}]', "row 1 has an invalid strength"),
        ],
    )
    def test_invalid(self, value, message):
        with pytest.raises(RowsFormatError, match=message):
            parse_rows(value)


class TestSplitAliases:
    def test_splits_and_trims(self):
        assert split_aliases(" mirabel , mirabel quill,,quill ") == [
            "mirabel",
            "mirabel quill",
            "quill",
        ]

    def test_empty(self):
        assert split_aliases("") == []
        assert split_aliases(" , ,") == []


class TestAliasPattern:
    @pytest.mark.parametrize(
        "prompt",
        [
            "a photo of Elara Voss",
            "a photo of elara voss",
            "a photo of ELARA-VOSS",
            "a photo of elara_voss",
            "a photo of elara   voss",
            "a photo of elara - voss",
            "(elara voss:1.2), smiling",
            "elara voss's portrait",
            "elara voss",
            "Elara Voss, outdoors",
        ],
    )
    def test_matches_variants(self, prompt):
        assert compiled("Elara Voss").search(prompt)

    @pytest.mark.parametrize(
        "prompt",
        [
            "a photo of elaravoss",
            "a photo of elara vossmere",
            "a photo of belara voss",
            "a photo of elara",
            "a photo of elara_voss2",
        ],
    )
    def test_rejects_partial_words(self, prompt):
        assert not compiled("Elara Voss").search(prompt)

    def test_alias_with_hyphen_matches_space(self):
        assert compiled("mary-jane").search("portrait of Mary Jane")

    def test_word_boundaries_for_short_alias(self):
        pattern = compiled("ann")
        assert pattern.search("ann walking")
        assert not pattern.search("anna walking")
        assert not pattern.search("planning a trip")

    def test_escapes_special_characters(self):
        pattern = compiled("c.j.")
        assert pattern.search("photo of c.j. smiling")
        assert not pattern.search("photo of cxjx smiling")

    def test_unicode_letters_are_word_characters(self):
        pattern = compiled("zoe")
        assert not pattern.search("zoeé")
        assert compiled("zoë").search("photo of Zoë")

    def test_blank_alias(self):
        assert alias_pattern("") is None
        assert alias_pattern(" - _ ") is None


class TestFindMatches:
    def test_no_rows(self):
        assert find_matches([], "anything") == ([], [])

    def test_no_trigger_in_prompt(self):
        rows = [LoraRow(lora="a.safetensors", triggers="alice")]
        assert matched_loras(rows, "a photo of bob") == []

    def test_empty_prompt(self):
        rows = [LoraRow(lora="a.safetensors", triggers="alice")]
        assert matched_loras(rows, "") == []
        assert matched_loras(rows, None) == []

    def test_matches_one_of_several(self):
        rows = [
            LoraRow(lora="a.safetensors", triggers="alice"),
            LoraRow(lora="b.safetensors", triggers="bob"),
        ]
        assert matched_loras(rows, "a photo of Bob") == ["b.safetensors"]

    def test_matches_multiple_in_row_order(self):
        rows = [
            LoraRow(lora="a.safetensors", triggers="alice"),
            LoraRow(lora="b.safetensors", triggers="bob"),
        ]
        assert matched_loras(rows, "bob and alice") == ["a.safetensors", "b.safetensors"]

    def test_any_alias_matches(self):
        rows = [LoraRow(lora="h.safetensors", triggers="mirabel, quill")]
        matches, _ = find_matches(rows, "portrait of Quill")
        assert [m.matched_text for m in matches] == ["Quill"]

    def test_same_lora_only_applied_once(self):
        rows = [
            LoraRow(lora="h.safetensors", triggers="mirabel", strength=0.7),
            LoraRow(lora="h.safetensors", triggers="quill", strength=0.9),
        ]
        matches, _ = find_matches(rows, "mirabel quill")
        assert [(m.row.lora, m.row.strength) for m in matches] == [("h.safetensors", 0.7)]

    def test_duplicate_row_can_match_if_first_did_not(self):
        rows = [
            LoraRow(lora="h.safetensors", triggers="mirabel", strength=0.7),
            LoraRow(lora="h.safetensors", triggers="quill", strength=0.9),
        ]
        matches, _ = find_matches(rows, "miss quill")
        assert [m.row.strength for m in matches] == [0.9]

    def test_disabled_row_never_matches(self):
        rows = [LoraRow(lora="a.safetensors", triggers="alice", enabled=False)]
        assert matched_loras(rows, "alice") == []

    def test_row_without_lora_or_triggers_never_matches(self):
        rows = [
            LoraRow(lora="", triggers="alice"),
            LoraRow(lora="b.safetensors", triggers=""),
            LoraRow(lora="c.safetensors", triggers=" , "),
        ]
        assert matched_loras(rows, "alice") == []

    def test_regex_row(self):
        rows = [LoraRow(lora="r.safetensors", triggers=r"\bmary[ -]?jane\b", regex=True)]
        assert matched_loras(rows, "portrait of MaryJane") == ["r.safetensors"]
        assert matched_loras(rows, "portrait of mary") == []

    def test_regex_commas_are_not_alias_separators(self):
        rows = [LoraRow(lora="r.safetensors", triggers=r"\bx{2,3}\b", regex=True)]
        assert matched_loras(rows, "a xxx b") == ["r.safetensors"]

    def test_regex_ignores_zero_length_matches(self):
        rows = [LoraRow(lora="r.safetensors", triggers="z*", regex=True)]
        assert matched_loras(rows, "abc") == []
        assert matched_loras(rows, "a zz b") == ["r.safetensors"]

    def test_invalid_regex_warns_and_skips(self):
        rows = [
            LoraRow(lora="bad.safetensors", triggers="(unclosed", regex=True),
            LoraRow(lora="ok.safetensors", triggers="ok"),
        ]
        matches, warnings = find_matches(rows, "ok (unclosed")
        assert [m.row.lora for m in matches] == ["ok.safetensors"]
        assert len(warnings) == 1
        assert "bad.safetensors" in warnings[0]

    def test_regex_is_case_insensitive(self):
        rows = [LoraRow(lora="r.safetensors", triggers="alice", regex=True)]
        assert matched_loras(rows, "ALICE") == ["r.safetensors"]
