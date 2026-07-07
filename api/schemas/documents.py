from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    id: str
    organization_id: str
    tenant_name: str | None = None
    landlord_name: str | None = None
    property_address: str | None = None
    lease_term: str | None = None
    commencement_date: str | None = None
    expiration_date: str | None = None
    type: str | None = None


class OrganizationListResponse(BaseModel):
    organizations: list[str] = Field(default_factory=list)
