"""Mechanism requests, branch activation, and unchanged-default invariants."""
import json
from emoharness.harness.bundle_models import MechanismScreenConfig
from emoharness.harness.pipeline import HarnessPipeline, PipelineConfig
from emoharness.target.fake import FakeTarget
from .conftest import scripted_responder, HYPOTHESES_JSON


def enabled(bundle, **fields):
    return bundle.model_copy(update={'routing': bundle.routing.model_copy(update={
        'mechanism_screen': MechanismScreenConfig(enabled=True, **fields)})})


def test_default_requests_and_labels_identical(bundle, make_example, make_inputs):
    ex=make_example(); inp=make_inputs(ex)
    a=FakeTarget(responder=scripted_responder()); b=FakeTarget(responder=scripted_responder())
    oa=HarnessPipeline(bundle,a).run(ex,inp)
    ob=HarnessPipeline(enabled(bundle),b).run(ex,inp)
    assert a.requests==b.requests
    assert oa.model_dump(exclude={'traces'})==ob.model_dump(exclude={'traces'})
    assert [t.model_dump(exclude={'latency_s'}) for t in oa.traces]==[t.model_dump(exclude={'latency_s'}) for t in ob.traces]


def test_source_table_preserves_real_sources_without_fake_timing(bundle, make_example, make_inputs):
    ex=make_example();inp=make_inputs(ex); t=FakeTarget(responder=scripted_responder())
    HarnessPipeline(enabled(bundle,evidence_layout='source_table'),t).run(ex,inp)
    p=t.requests[1].prompt
    assert inp.audio.transcript in p
    assert 'cached_asr' in p and 'cached_prosody' in p
    assert 'question_context_not_evidence' in p and 'unavailable_in_existing_cache' in p
    assert '"start_seconds": null' in p


def test_raw_frames_reach_decision_request(bundle, make_example, make_inputs):
    ex=make_example();inp=make_inputs(ex);t=FakeTarget(responder=scripted_responder())
    HarnessPipeline(enabled(bundle,decision_media='cached_frames',decision_frame_limit=3),t).run(ex,inp)
    assert t.requests[1].image_paths==(inp.frame_paths[0],inp.frame_paths[2],inp.frame_paths[5])
    assert not t.requests[2].image_paths


def test_margin_trigger_skips_high_confidence_but_reviews_low_margin_video(bundle, make_example, make_inputs):
    ex=make_example();inp=make_inputs(ex)
    b=enabled(bundle,verify_trigger='conflict_or_low_margin',margin_threshold=.15,review_media='video_on_trigger')
    high=FakeTarget(responder=scripted_responder());p=HarnessPipeline(b,high);out=p.run(ex,inp)
    assert not p.mechanism_trace['verify_executed'] and out.target_calls==3
    rows=[dict(x) for x in HYPOTHESES_JSON['hypotheses']]
    rows[0]['confidence']=.55;rows[1]['confidence']=.5
    low=FakeTarget(responder=scripted_responder({'generate_hypotheses':{'coarse':'uncertain','hypotheses':rows}}))
    p=HarnessPipeline(b,low);out=p.run(ex,inp)
    assert p.mechanism_trace['verify_executed'] and out.target_calls==4
    assert low.requests[2].video_path==inp.video_path


def test_top_two_filters_request_and_can_change_final_label(bundle, make_example, make_inputs):
    ex=make_example();inp=make_inputs(ex)
    rows=[dict(x) for x in HYPOTHESES_JSON['hypotheses']]
    rows.append({'label':'neutral','confidence':.1,'support':['v1'],'contradiction':[]})
    script={'generate_hypotheses':{'coarse':'test','hypotheses':rows},'verify':{'hypotheses':[{'label':'anger','confidence':.9,'support':['v1'],'contradiction':[]}]}}
    t=FakeTarget(responder=scripted_responder(script));p=HarnessPipeline(enabled(bundle,verify_style='top2_contrast'),t);out=p.run(ex,inp)
    assert p.mechanism_trace['verify_input_labels']==['sadness','anger']
    assert 'Compare the two leading hypotheses explicitly' in t.requests[2].prompt
    assert out.labels==('anger',)


def test_new_branches_still_obey_six_call_cap(bundle, make_example, make_inputs):
    ex=make_example();t=FakeTarget(default_text='not json')
    p=HarnessPipeline(enabled(bundle,verify_trigger='conflict_or_low_margin',review_media='video_on_trigger'),t,PipelineConfig(call_hard_cap=6))
    out=p.run(ex,make_inputs(ex))
    assert out.target_calls<=6 and out.parse_failures==6


def test_margin_uses_deciding_rank_not_unrelated_third_confidence():
    from emoharness.harness._mechanism_screen import confidence_margin
    from emoharness.harness.schema import HypothesisSet, Hypothesis
    hs=HypothesisSet(coarse='x',hypotheses=tuple(Hypothesis(label=str(i),confidence=c) for i,c in enumerate([.51,.50,.99])))
    assert abs(confidence_margin(hs)-.01)<1e-9


def test_empty_hypothesis_json_forces_review_even_at_zero_threshold(bundle, make_example, make_inputs):
    ex=make_example();t=FakeTarget(responder=scripted_responder({'generate_hypotheses':{}}))
    p=HarnessPipeline(enabled(bundle,verify_trigger='conflict_or_low_margin',margin_threshold=0),t)
    p.run(ex,make_inputs(ex))
    assert p.mechanism_trace['verify_executed']
