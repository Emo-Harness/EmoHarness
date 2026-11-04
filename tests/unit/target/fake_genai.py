"""Fake google-genai client. Mirrors the real SDK's shape without importing it.

``GeminiDirectTarget`` builds request contents as plain dicts and only ever
touches ``client.models.generate_content`` and ``client.files.upload``, so this
fake records calls and returns canned, dict/attribute-compatible responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FakeUsageMetadata:
    prompt_token_count: int
    candidates_token_count: int


@dataclass(frozen=True)
class FakeGenerateContentResponse:
    text: str
    usage_metadata: FakeUsageMetadata
    response_id: str = 'fake-response-id'


@dataclass(frozen=True)
class FakeUploadedFile:
    uri: str
    mime_type: str


@dataclass
class FakeModelsAPI:
    response: FakeGenerateContentResponse
    calls: list[dict[str, Any]] = field(default_factory=list)

    def generate_content(self, **kwargs: Any) -> FakeGenerateContentResponse:
        self.calls.append(kwargs)
        return self.response


@dataclass
class FakeFilesAPI:
    uploaded: FakeUploadedFile
    uploads: list[str] = field(default_factory=list)

    def upload(self, *, file: str) -> FakeUploadedFile:
        self.uploads.append(file)
        return self.uploaded


class FakeGenAIClient:
    """Stands in for ``google.genai.Client``; exposes ``.models`` and ``.files``."""

    def __init__(
        self,
        response: FakeGenerateContentResponse | None = None,
        uploaded_file: FakeUploadedFile | None = None,
    ) -> None:
        self.models = FakeModelsAPI(
            response=response
            or FakeGenerateContentResponse(
                text='<answer>joy</answer>',
                usage_metadata=FakeUsageMetadata(
                    prompt_token_count=10, candidates_token_count=5
                ),
            )
        )
        self.files = FakeFilesAPI(
            uploaded=uploaded_file
            or FakeUploadedFile(uri='files/fake123', mime_type='video/mp4')
        )
