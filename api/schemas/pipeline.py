from __future__ import annotations

from pydantic import BaseModel, Field


class BuildGraphRequest(BaseModel):
    organization_id: str = Field(..., description="The organization ID to build the graph for.")


class BuildCommunitiesRequest(BaseModel):
    organization_id: str = Field(..., description="The organization ID to build communities for.")
    min_community_size: int = Field(2, ge=1, description="Minimum number of nodes for a community to be included.")
    chunk_id: str | None = Field(
        None,
        description="If set, only build communities for this specific chunk (test mode). "
                    "If None, performs a full-graph community build.",
    )


class PipelineJobResponse(BaseModel):
    status: str = Field(..., description="'started' | 'completed' | 'error'")
    message: str
    organization_id: str
