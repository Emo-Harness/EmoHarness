"""Target request/response contract and the protocol every client implements."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class TargetRequest(BaseModel):
    """One call to the Target. Media are paths; clients encode them as needed."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    prompt: str = Field(min_length=1)
    tag: str | None = Field(
        default=None,
        description='the harness stage or ladder row that issued this call',
    )
    system: str | None = None
    video_path: str | None = None
    image_paths: tuple[str, ...] = ()
    json_schema: Mapping[str, Any] | None = Field(
        default=None,
        description='when set, the client asks for JSON matching this schema',
    )
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int = Field(default=0, ge=0)


class TargetResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid', protected_namespaces=())

    text: str
    model_id: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_s: float = Field(ge=0.0)
    cached: bool = False


@runtime_checkable
class TargetModel(Protocol):
    """Anything that answers a TargetRequest. Implementations must be deterministic at temperature 0."""

    @property
    def model_id(self) -> str: ...

    def generate(self, request: TargetRequest) -> TargetResponse: ...
