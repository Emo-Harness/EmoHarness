"""What the harness may cite: the evidence record and the images it sends."""

from __future__ import annotations

from collections.abc import Callable

from emoharness.data.manifest import Example
from emoharness.harness.loader import HarnessBundle
from emoharness.harness.schema import AudioEvidence, PerceivedInputs
from emoharness.target.fake import FakeTarget

from .conftest import run_once, scripted_responder

_NO_CUE = 'No reliable cue observed.'
_SAMPLING = 'face_crops:\n  enabled: {crops}\n  max_crops: 2\nextra_images:\n  enabled: {frames}\n'


def test_a_step_loses_its_unknown_ids_and_its_uncited_sentences(
    bundle: HarnessBundle, benchmark_example: Example, inputs: PerceivedInputs
) -> None:
    script = {
        'synthesize': {
            'steps': [
                'Grounded claim [v1]. Invented claim [zz]. Bare claim. Mixed [v2, zz].',
                'Nothing here points anywhere.',
                'The voice trembles [a2].',
                'The cues line up [v1, a2].',
            ],
            'justification': 'The tearful face wins [v1].',
        }
    }
    target = FakeTarget(responder=scripted_responder(script))

    output = run_once(bundle, target, benchmark_example, inputs)

    assert output.steps[0] == ('Setting and context: Grounded claim [v1]. Mixed [v2].')
    assert 'zz' not in ' '.join(output.steps)
    assert 'Invented claim' not in output.steps[0]
    assert 'Bare claim' not in output.steps[0]
    assert output.steps[1] == f'Facial and bodily cues: {_NO_CUE}'
    assert output.steps[3].endswith('The tearful face wins [v1].')


def test_missing_step_texts_become_the_no_cue_line(
    bundle: HarnessBundle, benchmark_example: Example, inputs: PerceivedInputs
) -> None:
    target = FakeTarget(responder=scripted_responder({'synthesize': {'steps': []}}))

    output = run_once(bundle, target, benchmark_example, inputs)

    assert all(step.endswith(_NO_CUE) for step in output.steps[:-1])
    assert output.steps[-1].startswith('Based on the analysis')


def test_face_crops_and_frames_ride_along_only_when_sampling_enables_them(
    edited_bundle: Callable[..., HarnessBundle],
    bundle: HarnessBundle,
    target: FakeTarget,
    benchmark_example: Example,
    inputs: PerceivedInputs,
) -> None:
    run_once(bundle, target, benchmark_example, inputs)
    assert target.requests[0].image_paths == ()

    cropped = FakeTarget(responder=scripted_responder())
    run_once(
        edited_bundle(
            'cropped', {'sampling.yaml': _SAMPLING.format(crops='true', frames='false')}
        ),
        cropped,
        benchmark_example,
        inputs,
    )
    assert cropped.requests[0].image_paths == inputs.face_crop_paths[:2]

    framed = FakeTarget(responder=scripted_responder())
    run_once(
        edited_bundle(
            'framed', {'sampling.yaml': _SAMPLING.format(crops='true', frames='true')}
        ),
        framed,
        benchmark_example,
        inputs,
    )
    assert framed.requests[0].image_paths == (
        inputs.face_crop_paths[:2] + inputs.frame_paths
    )


def test_the_record_holds_only_what_tools_observed(
    bundle: HarnessBundle,
    target: FakeTarget,
    benchmark_example: Example,
    inputs: PerceivedInputs,
) -> None:
    """No opinion channel: every heard item is a transcript sentence or a prosody phrase."""
    output = run_once(bundle, target, benchmark_example, inputs)

    modalities = {item.modality for item in output.evidence.items}
    assert modalities == {'visual', 'audio', 'text'}
    heard = [item.cue for item in output.evidence.items if item.modality == 'audio']
    assert heard == [
        'speech: I am fine.',
        'prosody: trembling voice',
        'prosody: low energy',
    ]
    assert not hasattr(output.evidence, 'audio_opinion')


def test_a_silent_clip_never_grows_a_transcript(
    bundle: HarnessBundle,
    target: FakeTarget,
    benchmark_example: Example,
    make_inputs: Callable[..., PerceivedInputs],
) -> None:
    silent = make_inputs(benchmark_example, audio=AudioEvidence(speech_present=False))

    output = run_once(bundle, target, benchmark_example, silent)

    heard = [item for item in output.evidence.items if item.modality == 'audio']
    assert [item.cue for item in heard] == ['no intelligible speech in the clip']
    assert output.evidence.audio.transcript == ''
    rendered = ' '.join(request.prompt for request in target.requests)
    assert 'Speech present: no' in rendered
    assert 'Transcript:' not in rendered


def test_a_long_transcript_is_truncated_before_it_is_cited(
    edited_bundle: Callable[..., HarnessBundle],
    target: FakeTarget,
    benchmark_example: Example,
    make_inputs: Callable[..., PerceivedInputs],
) -> None:
    bundle = edited_bundle(
        'terse',
        {'audio_pipeline.yaml': 'channels: [transcript]\ntranscript_max_chars: 40\n'},
    )
    chatty = make_inputs(
        benchmark_example,
        audio=AudioEvidence(speech_present=True, transcript='word ' * 200),
    )

    output = run_once(bundle, target, benchmark_example, chatty)

    assert len(output.evidence.audio.transcript) == 40


def test_a_loose_visual_answer_still_numbers_what_it_can(
    bundle: HarnessBundle, benchmark_example: Example, inputs: PerceivedInputs
) -> None:
    script = {
        'extract_visual': {
            'face': 'tearful eyes',
            'body': ['   '],
            'scene': None,
            'temporal_change': '',
        }
    }
    target = FakeTarget(responder=scripted_responder(script))

    output = run_once(bundle, target, benchmark_example, inputs)

    seen = [item for item in output.evidence.items if item.modality == 'visual']
    assert [(item.id, item.cue) for item in seen] == [('v1', 'tearful eyes')]
    assert output.evidence.visual.temporal_change is None


def test_the_question_itself_is_the_last_citable_item(
    bundle: HarnessBundle, target: FakeTarget, benchmark_example: Example
) -> None:
    output = run_once(
        bundle,
        target,
        benchmark_example,
        PerceivedInputs(
            example_id=benchmark_example.example_id,
            video_path=benchmark_example.media_path,
            duration_seconds=1.0,
            audio=AudioEvidence(speech_present=True, transcript='I am fine.'),
        ),
    )

    last = output.evidence.items[-1]
    assert (last.id, last.modality, last.cue) == ('t1', 'text', 'emotion')
    assert output.evidence.items[-2].modality == 'audio'
