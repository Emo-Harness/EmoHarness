from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest

from emoharness.builder.clusters import FailureRecord, cluster_key
from emoharness.builder.config import PacketLimits
from emoharness.builder.packet import (
    TRUNCATION_MARKER,
    DiagnosisPacket,
    assert_no_forbidden_ids,
    build_packet,
    estimate_tokens,
    render_packet_text,
)
from emoharness.common.errors import LeakageError

Record = Callable[..., FailureRecord]
# Held-out capability names the Builder must never be shown (ADR-0005, ADR-0012).
FORBIDDEN_WORDS = ('multi-label', 'multilabel', 'fine-grained', 'intent', 'noise')
LIMITS = PacketLimits(
    max_failures_per_cluster=5,
    max_keyframe_samples=3,
    max_frames_per_sample=2,
    target_input_tokens=100_000,
)


def _records(make_record: Record) -> tuple[FailureRecord, ...]:
    frames = ('f1.jpg', 'f2.jpg', 'f3.jpg')
    return (
        *(
            make_record(f'a{i}', ('anger',), ('sadness',), keyframes=frames)
            for i in range(3)
        ),
        *(
            make_record(f'b{i}', ('happiness',), ('neutral',), keyframes=frames)
            for i in range(3)
        ),
    )


def _packet(
    make_record: Record, files: Mapping[str, str], stage: int = 2
) -> DiagnosisPacket:
    return build_packet(
        stage=stage,
        round_index=0,
        bundle_files=files,
        build_recognition=0.42,
        records=_records(make_record),
        limits=LIMITS,
    )


def test_packet_lists_only_the_paths_the_stage_unlocks(
    make_record: Record, bundle_files: Mapping[str, str]
) -> None:
    stage_two = _packet(make_record, bundle_files, stage=2)
    stage_three = _packet(make_record, bundle_files, stage=3)
    assert 'verification/cross_modal.py' not in stage_two.editable_paths
    assert 'verification/cross_modal.py' in stage_three.editable_paths
    assert 'manifest.json' not in stage_three.editable_paths


def test_keyframes_are_sampled_round_robin_and_capped(
    make_record: Record, bundle_files: Mapping[str, str]
) -> None:
    packet = _packet(make_record, bundle_files)
    assert [example_id for example_id, _ in packet.keyframe_samples] == [
        'a0',
        'b0',
        'a1',
    ]
    assert all(len(paths) == 2 for _, paths in packet.keyframe_samples)
    assert len(packet.image_paths()) == 6


def test_records_without_keyframes_are_skipped(
    make_record: Record, bundle_files: Mapping[str, str]
) -> None:
    packet = build_packet(
        stage=2,
        round_index=1,
        bundle_files=bundle_files,
        build_recognition=0.1,
        records=(make_record('x'), make_record('y', keyframes=('f.jpg',))),
        limits=LIMITS,
    )
    assert [example_id for example_id, _ in packet.keyframe_samples] == ['y']


def test_packet_text_states_the_contract(
    make_record: Record, bundle_files: Mapping[str, str]
) -> None:
    packet = _packet(make_record, bundle_files)
    text = render_packet_text(packet, LIMITS.target_input_tokens)
    for path in packet.editable_paths:
        assert f'- {path}' in text
    assert (
        'verification/cross_modal.py' not in text.split('## Current harness files')[0]
    )
    assert '"candidates"' in text and '"content"' in text
    assert 'targeted_cluster' in text
    assert 'at most 4 candidates' in text
    assert cluster_key(('anger',), ('sadness',), False) in text
    assert 'order: [visual, audio]' in text
    assert '| gold vs predicted |' in text


def test_packet_text_never_names_a_held_out_capability(
    make_record: Record, bundle_files: Mapping[str, str]
) -> None:
    text = render_packet_text(_packet(make_record, bundle_files), 100_000).lower()
    for word in FORBIDDEN_WORDS:
        assert word not in text


def test_an_empty_packet_still_renders(bundle_files: Mapping[str, str]) -> None:
    packet = build_packet(
        stage=2,
        round_index=0,
        bundle_files=bundle_files,
        build_recognition=1.0,
        records=(),
        limits=LIMITS,
    )
    text = render_packet_text(packet, 100_000)
    assert '(no failures recorded)' in text


def test_estimate_tokens_rounds_up() -> None:
    assert estimate_tokens('') == 0
    assert estimate_tokens('abc') == 1
    assert estimate_tokens('a' * 8) == 2


def _big_packet(bundle_files: Mapping[str, str]) -> DiagnosisPacket:
    record = FailureRecord(
        example_id='big',
        gold=('anger',),
        predicted=('sadness',),
        evidence_json='e' * 5000,
        stage_outputs={'hypotheses': 'h' * 3000, 'verification': 'v' * 10},
    )
    return build_packet(
        stage=2,
        round_index=0,
        bundle_files=bundle_files,
        build_recognition=0.3,
        records=(record,),
        limits=LIMITS,
    )


def test_samples_shrink_until_the_packet_fits(bundle_files: Mapping[str, str]) -> None:
    packet = _big_packet(bundle_files)
    full = render_packet_text(packet, 100_000)
    assert 'e' * 5000 in full
    budget = estimate_tokens(full) - 500
    tight = render_packet_text(packet, budget)
    assert estimate_tokens(tight) <= budget
    assert TRUNCATION_MARKER in tight
    assert 'e' * 5000 not in tight


def test_an_impossible_budget_returns_the_smallest_rendering(
    bundle_files: Mapping[str, str],
) -> None:
    packet = _big_packet(bundle_files)
    smallest = render_packet_text(packet, 1)
    assert '- evidence:' not in smallest
    assert '- hypotheses:' not in smallest
    assert '#### Item big' in smallest
    assert 'order: [visual, audio]' in smallest


def test_forbidden_ids_are_caught_in_the_builder_input(
    make_record: Record, bundle_files: Mapping[str, str]
) -> None:
    text = render_packet_text(_packet(make_record, bundle_files), 100_000)
    assert_no_forbidden_ids(text, ('mme_er_wild_000001', ''))
    assert_no_forbidden_ids(text, ())
    with pytest.raises(LeakageError, match='a0'):
        assert_no_forbidden_ids(text, ('a0',))


def test_round_robin_continues_when_a_cluster_runs_out(
    make_record: Record, bundle_files: Mapping[str, str]
) -> None:
    frames = ('f1.jpg',)
    records = (
        *(
            make_record(f'a{i}', ('anger',), ('sadness',), keyframes=frames)
            for i in range(3)
        ),
        make_record('b0', ('happiness',), ('neutral',), keyframes=frames),
    )
    packet = build_packet(
        stage=2,
        round_index=0,
        bundle_files=bundle_files,
        build_recognition=0.4,
        records=records,
        limits=PacketLimits(max_keyframe_samples=4, max_frames_per_sample=1),
    )
    assert [example_id for example_id, _ in packet.keyframe_samples] == [
        'a0',
        'b0',
        'a1',
        'a2',
    ]
