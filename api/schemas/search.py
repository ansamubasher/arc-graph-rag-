from __future__ import annotations

from pydantic import BaseModel, Field


class PartialAnswer(BaseModel):
    community_id: int
    partial_answer: str
    score: int


class GlobalSearchRequest(BaseModel):
    graph_id: str = Field(..., description="The organization/graph ID to search within.")
    question: str = Field(..., description="The natural-language question to answer.")
    top_k: int = Field(5, ge=1, le=50, description="Number of top partial answers to include in the reduce step.")


class GlobalSearchResponse(BaseModel):
    answer: str = Field(..., description="The final synthesized answer in Markdown.")
    partial_answers: list[PartialAnswer] = Field(
        default_factory=list,
        description="Individual partial answers from each relevant community.",
    )
