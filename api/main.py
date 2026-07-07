"""
FastAPI application factory for the Cosmos Graph RAG backend.

Run with:
    uvicorn api.main:app --reload --app-dir <project_root>

Or from inside the project root:
    uvicorn api.main:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import documents, organizations, pipeline, search


# ---------------------------------------------------------------------------
# Lifespan  (startup / shutdown hooks)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup: eagerly validate that environment variables for Cosmos DB and
    Azure OpenAI are present by importing the dependency module (which reads
    them at import time). Any missing-var ValueError surfaces immediately
    rather than on the first request.
    """
    import api.dependencies  # noqa: F401  – triggers env-var validation
    yield
    # Shutdown: nothing to clean up for these stateless SDK clients.


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Cosmos Graph RAG API",
    description=(
        "REST API exposing the full GraphRAG pipeline: document ingestion, "
        "knowledge-graph construction, community detection, and global search "
        "over community reports backed by Azure Cosmos DB and Azure OpenAI."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS  — permissive in dev; tighten allow_origins in production
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

API_PREFIX = "/api/v1"

app.include_router(organizations.router, prefix=API_PREFIX)
app.include_router(documents.router, prefix=API_PREFIX)
app.include_router(pipeline.router, prefix=API_PREFIX)
app.include_router(search.router, prefix=API_PREFIX)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"], summary="Health check")
def health() -> dict:
    return {"status": "ok"}
