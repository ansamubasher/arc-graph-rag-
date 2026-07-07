"""
Router: /api/v1/organizations/{org_id}/documents
Retrieves documents from Cosmos DB for a given organization.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path

from api.dependencies import get_cosmos_reader
from api.schemas.documents import DocumentSummary
from cosmos_reader import CosmosReader

router = APIRouter(
    prefix="/organizations",
    tags=["Documents"],
)


@router.get(
    "/{org_id}/documents",
    response_model=list[DocumentSummary],
    summary="List documents for an organization",
    description="Returns all documents belonging to the given organization_id.",
)
def list_documents(
    org_id: str = Path(..., description="The organization ID to fetch documents for."),
    reader: CosmosReader = Depends(get_cosmos_reader),
) -> list[DocumentSummary]:
    try:
        raw_docs = list(reader.load_documents_by_organization(org_id))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Cosmos DB error: {exc}") from exc

    if not raw_docs:
        raise HTTPException(
            status_code=404,
            detail=f"No documents found for organization '{org_id}'.",
        )

    return [
        DocumentSummary(
            id=doc.get("id", ""),
            organization_id=doc.get("organization_id", org_id),
            tenant_name=doc.get("tenant_name"),
            landlord_name=doc.get("landlord_name"),
            property_address=doc.get("property_address"),
            lease_term=doc.get("lease_term"),
            commencement_date=doc.get("commencement_date"),
            expiration_date=doc.get("expiration_date"),
            type=doc.get("type"),
        )
        for doc in raw_docs
    ]


@router.get(
    "/{org_id}/documents/{doc_id}",
    summary="Fetch a single document",
    description="Returns the full Cosmos DB document for the given org + document ID.",
)
def get_document(
    org_id: str = Path(..., description="The organization ID."),
    doc_id: str = Path(..., description="The document ID."),
    reader: CosmosReader = Depends(get_cosmos_reader),
) -> dict:
    try:
        doc = reader.load_document_by_id(org_id, doc_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Cosmos DB error: {exc}") from exc

    if doc is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_id}' not found for organization '{org_id}'.",
        )

    return doc
