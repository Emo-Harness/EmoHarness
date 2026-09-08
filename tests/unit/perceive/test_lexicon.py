from __future__ import annotations

from pathlib import Path

import pytest

from emoharness.common.errors import ValidationError
from emoharness.perceive.lexicon import (
    banned_words,
    contains_banned,
    filter_description,
    load_lexicon,
)


def test_load_lexicon_strips_comments_blanks_and_lowercases(tmp_path: Path) -> None:
    path = tmp_path / 'lex.txt'
    path.write_text(
        '# comment\n\nAngry\nSAD\n  \n# another comment\nhappy\n', encoding='utf-8'
    )
    assert load_lexicon(path) == frozenset({'angry', 'sad', 'happy'})


def test_load_lexicon_missing_file_raises() -> None:
    with pytest.raises(ValidationError):
        load_lexicon(Path('/no/such/lexicon.txt'))


def test_banned_words_splits_labels_on_space_slash_and_hyphen() -> None:
    lexicon = frozenset({'anger'})
    label_set = ('Doubt/Confusion', 'very-strong negative', 'Joy')
    result = banned_words(label_set, lexicon)
    assert result == frozenset(
        {'anger', 'doubt', 'confusion', 'very', 'strong', 'negative', 'joy'}
    )


def test_banned_words_does_not_mutate_input_lexicon() -> None:
    lexicon = frozenset({'anger'})
    banned_words(('Joy',), lexicon)
    assert lexicon == frozenset({'anger'})


def test_contains_banned_matches_whole_words_case_insensitively() -> None:
    banned = frozenset({'sad', 'angry'})
    assert contains_banned('She sounds ANGRY today.', banned) is True
    assert contains_banned('A neutral tone.', banned) is False


def test_contains_banned_does_not_match_substrings() -> None:
    banned = frozenset({'sad'})
    assert contains_banned('He spoke sadly but softly.', banned) is False


def test_contains_banned_empty_set_is_always_false() -> None:
    assert contains_banned('angry sad happy', frozenset()) is False


def test_filter_description_drops_sentences_with_banned_words() -> None:
    banned = frozenset({'angry'})
    text = 'The room is quiet. He sounds angry here. Background hum is present.'
    assert filter_description(text, banned) == (
        'The room is quiet. Background hum is present'
    )


def test_filter_description_keeps_substring_matches_intact() -> None:
    banned = frozenset({'sad'})
    text = 'He spoke sadly. The tone was flat.'
    assert filter_description(text, banned) == 'He spoke sadly. The tone was flat'


def test_filter_description_splits_on_question_exclamation_and_newline() -> None:
    banned = frozenset({'angry'})
    text = 'Is she angry?\nNo, she sounds calm! Great.'
    assert filter_description(text, banned) == 'No, she sounds calm. Great'


def test_filter_description_empty_text_is_empty() -> None:
    assert filter_description('', frozenset({'angry'})) == ''


def test_filter_description_all_sentences_dropped_is_empty() -> None:
    assert filter_description('So angry.', frozenset({'angry'})) == ''
