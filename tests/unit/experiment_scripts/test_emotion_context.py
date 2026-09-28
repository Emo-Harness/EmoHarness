"""The emotion-context experiment's scoring seams: answer parsing, context text, subsets.

These three decide what a reply scores and what the answering model sees, so a
silent change in any of them would move every number in the report.
"""

from __future__ import annotations

import pytest
from experiments.emotion_context import row, subset

OPTIONS = [
    'want to talk to him',
    'make music',
    'to energise him',
    'want to hit him',
    'to annoy him',
]


@pytest.mark.parametrize(
    ('reply', 'choice', 'rule'),
    [
        ('B', 1, 'leading_letter'),
        ('(C)', 2, 'leading_letter'),
        ('Answer: E', 4, 'leading_letter'),
        ('A\n', 0, 'leading_letter'),
        ('B make music', 1, 'leading_letter_text'),
        ('<answer>A</answer>', 0, 'answer_tag'),
        ('The answer is D.', 3, 'stated_answer'),
        (
            'Option A is unlikely.\n\nTherefore, the correct answer is **B. make music**.',
            1,
            'stated_answer',
        ),
        ('Analysis...\n\n**Final Answer: C**', 2, 'stated_answer'),
        ('Reasoning about A and C.\n\nB', 1, 'final_line_letter'),
        ('Reasoning.\n\n**E. to annoy him.**', 4, 'final_line_letter'),
        ('make music.', 1, 'exact_option_text'),
    ],
)
def test_parse_reads_the_committed_option(reply: str, choice: int, rule: str) -> None:
    assert row.parse_with_rule(reply, OPTIONS) == (choice, rule)


@pytest.mark.parametrize(
    'reply',
    [
        'I think',
        'A person walks in.',  # a sentence opening with the article "A"
        'the answer is a man playing',  # lower-case article, not an option letter
        'Reasoning ...\n- **A. want to talk to him**: unlikely',  # truncated option review
        'None of the options are supported by the video.',
        '',
    ],
)
def test_parse_never_guesses(reply: str) -> None:
    assert row.parse_with_rule(reply, OPTIONS) == (None, None)


def test_parse_rejects_letters_beyond_the_options() -> None:
    assert row.parse_letter('E', OPTIONS[:4]) is None


def test_context_keeps_label_alternatives_and_steps_one_to_four() -> None:
    output = {
        'labels': ['happy'],
        'hypotheses': {
            'hypotheses': [
                {'label': 'neutral', 'confidence': 0.3},
                {'label': 'happy', 'confidence': 0.6},
                {'label': 'sad', 'confidence': 0.0},
            ]
        },
        'response_text': (
            'Step 1: Setting and context: a kitchen [v1, v2].\n'
            'Step 2: Facial and bodily cues: smiling [v3].\n'
            'Step 3: Vocal and verbal cues: laughter [a1].\n'
            'Step 4: Cross-modal synthesis and strongest alternative: neutral is weaker.\n'
            'Step 5: Based on the analysis, the predicted emotion is happy.\n'
            '<answer>happy</answer>'
        ),
    }
    text = row.render_context(output)
    assert text.splitlines()[0] == 'Predicted emotion: happy (confidence 0.60)'
    assert 'Alternatives considered: neutral (0.30)' in text
    assert 'sad' not in text  # zero-confidence alternatives are dropped
    assert 'Step 4:' in text and 'Step 5' not in text and '<answer>' not in text
    assert '[' not in text  # evidence-id citations point at lists the reader never sees


def test_qa_prompt_differs_only_by_the_block() -> None:
    item = {'question': 'why does he go near the man', 'options': OPTIONS}
    bare = row.qa_prompt(item, 32, '')
    with_block = row.qa_prompt(
        item, 32, 'Audio evidence (from external tools):\nSpeech present: no'
    )
    assert bare.startswith('The 32 images are frames')
    assert 'E. to annoy him' in bare
    assert bare.endswith(
        "Answer with the option's letter from the given choices directly."
    )
    assert (
        with_block.replace(
            'Audio evidence (from external tools):\nSpeech present: no\n\n', ''
        )
        == bare
    )


@pytest.mark.parametrize(
    ('question', 'tier'),
    [
        ('How does the woman feel about the men?', 'core'),
        ('What is the tone of the doctor?', 'core'),
        ('Why is the woman so upset?', 'core'),
        ('Why do the two men laugh?', 'broad_only'),
        ('How did the boy react after the girl jumped', 'broad_only'),
        ('What happened after the man sat down?', None),  # "happen" is not "happy"
        ('Who is the dominant person in this video?', None),
    ],
)
def test_subset_rule(question: str, tier: str | None) -> None:
    assert subset.subset_of(question) == tier
