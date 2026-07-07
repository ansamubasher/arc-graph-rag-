"""
Router: /api/v1/search
Exposes GlobalSearch (map-reduce over community reports) as a REST endpoint.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_global_search
from api.schemas.search import GlobalSearchRequest, GlobalSearchResponse, PartialAnswer
from global_search import GlobalSearch

router = APIRouter(
    prefix="/search",
    tags=["Search"],
)


@router.post(
    "/global",
    response_model=GlobalSearchResponse,
    summary="Global search over community reports",
    description=(
        "Runs a map-reduce query across all community reports for the given graph_id. "
        "Ideal for broad, document-spanning questions (e.g. 'summarise all lease terms', "
        "'what are the main risks?'). Community report embeddings are cached on the "
        "GlobalSearch singleton after the first request for each graph_id."
    ),
)
def global_search(
    body: GlobalSearchRequest,
    search: GlobalSearch = Depends(get_global_search),
) -> GlobalSearchResponse:
    # Override top_k on the singleton if the caller specified a custom value.
    search.top_k = body.top_k

    try:
        result = search.answer(question=body.question, graph_id=body.graph_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Search failed: {exc}",
        ) from exc

    partial_answers = [
        PartialAnswer(
            community_id=p.get("community_id", -1),
            partial_answer=p.get("partial_answer", ""),
            score=int(p.get("score", 0)),
        )
        for p in result.get("partial_answers", [])
    ]

    return GlobalSearchResponse(
        answer=result.get("answer", ""),
        partial_answers=partial_answers,
    )
