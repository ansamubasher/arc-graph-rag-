"""
Router: /api/v1/organizations
Lists all unique organization IDs stored in Cosmos DB.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_cosmos_reader
from api.schemas.documents import OrganizationListResponse
from cosmos_reader import CosmosReader

router = APIRouter(
    prefix="/organizations",
    tags=["Organizations"],
)


@router.get(
    "",
    response_model=OrganizationListResponse,
    summary="List all organization IDs",
    description="Returns every unique organization_id present in the Cosmos DB container.",
)
def list_organizations(
    reader: CosmosReader = Depends(get_cosmos_reader),
) -> OrganizationListResponse:
    try:
        org_ids = reader.get_organization_ids()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Cosmos DB error: {exc}") from exc

    return OrganizationListResponse(organizations=org_ids)
