"""The inference-time harness: one deep module behind five public seams.

Import from here, or from the five modules re-exported below, and from nothing
else in this package:

- :mod:`emoharness.harness.schema` -- the records every subpackage exchanges;
- :mod:`emoharness.harness.loader` -- the bundle the Builder edits;
- :mod:`emoharness.harness.pipeline` -- ``HarnessPipeline.run``, one item in, one
  ``HarnessOutput`` out;
- :mod:`emoharness.harness.allowlist` -- the static check on Builder-written code;
- :mod:`emoharness.harness.fake_responder` -- the scripted Target that exercises
  the four stages, shipped beside them so no caller has to script them itself.

Modules whose name begins with an underscore are implementation: the evidence
builder, the stage callers, the verification fallbacks, and the output controller
are reachable only through ``run`` and may change without notice.
"""

from __future__ import annotations

from emoharness.harness.allowlist import (
    check_verification_source,
    load_verification_function,
)
from emoharness.harness.fake_responder import (
    ChooseLabel,
    label_chooser,
    scripted_responder,
)
from emoharness.harness.loader import (
    AudioPipelineConfig,
    BundleManifest,
    HarnessBundle,
    RoutingConfig,
    SamplingConfig,
    export_schemas,
    load_bundle,
    write_bundle_copy,
)
from emoharness.harness.pipeline import HarnessPipeline, PipelineConfig
from emoharness.harness.schema import (
    AudioEvidence,
    AudioOpinion,
    EvidenceItem,
    EvidenceRecord,
    HarnessOutput,
    Hypothesis,
    HypothesisSet,
    Modality,
    PerceivedInputs,
    StageTrace,
    VerificationResult,
    VisualEvidence,
)

__all__ = [
    'AudioEvidence',
    'AudioOpinion',
    'AudioPipelineConfig',
    'BundleManifest',
    'ChooseLabel',
    'EvidenceItem',
    'EvidenceRecord',
    'HarnessBundle',
    'HarnessOutput',
    'HarnessPipeline',
    'Hypothesis',
    'HypothesisSet',
    'Modality',
    'PerceivedInputs',
    'PipelineConfig',
    'RoutingConfig',
    'SamplingConfig',
    'StageTrace',
    'VerificationResult',
    'VisualEvidence',
    'check_verification_source',
    'export_schemas',
    'label_chooser',
    'load_bundle',
    'load_verification_function',
    'scripted_responder',
    'write_bundle_copy',
]
