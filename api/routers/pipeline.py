"""
Router: /api/v1/pipeline
Triggers long-running pipeline stages (graph build, community build) as
FastAPI BackgroundTasks so the endpoint returns 202 Accepted immediately
while the work continues in the background.

Progress is logged to stdout (visible in the uvicorn console). A lightweight
in-memory job store tracks whether each job is running / completed / errored
so clients can poll /pipeline/jobs/{job_id} for status.
"""
from __future__ import annotations

import sys
import os
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

# Ensure project root is importable (dependencies.py already handles this,
# but we guard here too in case routers are imported in isolation).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from api.dependencies import get_community_builder, GRAPHS_FOLDER
from api.schemas.pipeline import (
    BuildGraphRequest,
    BuildCommunitiesRequest,
    PipelineJobResponse,
)
from community_builder_final import CommunityBuilder

router = APIRouter(
    prefix="/pipeline",
    tags=["Pipeline"],
)

# ---------------------------------------------------------------------------
# In-memory job store  (suitable for single-process dev / small deployments)
# ---------------------------------------------------------------------------
JobStatus = Literal["started", "completed", "error"]

_jobs: dict[str, dict] = {}


def _make_job(organization_id: str, job_type: str) -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "job_type": job_type,
        "organization_id": organization_id,
        "status": "started",
        "message": "Job started.",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
    }
    return job_id


def _finish_job(job_id: str, *, error: str | None = None) -> None:
    if job_id not in _jobs:
        return
    _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
    if error:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["message"] = error
    else:
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["message"] = "Job completed successfully."


# ---------------------------------------------------------------------------
# Background task functions
# ---------------------------------------------------------------------------

def _run_build_graph(organization_id: str, job_id: str) -> None:
    """Runs GraphBuilder.build() in the background."""
    try:
        # Import here to avoid circular deps and keep startup fast.
        from graph_builder_final import GraphBuilder  # noqa: PLC0415
        builder = GraphBuilder(organization_id=organization_id)
        builder.build()
        _finish_job(job_id)
    except Exception as exc:  # noqa: BLE001
        _finish_job(job_id, error=str(exc))
        raise


def _run_build_communities(
    organization_id: str,
    min_community_size: int,
    chunk_id: str | None,
    job_id: str,
    graphs_folder: str,
) -> None:
    """Runs CommunityBuilder.build() or .build_for_chunk() in the background."""
    try:
        builder = CommunityBuilder(
            graph_folder=graphs_folder,
            min_community_size=min_community_size,
        )
        if chunk_id:
            builder.build_for_chunk(organization_id, chunk_id)
        else:
            builder.build(organization_id)
        _finish_job(job_id)
    except Exception as exc:  # noqa: BLE001
        _finish_job(job_id, error=str(exc))
        raise


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/build-graph",
    status_code=202,
    response_model=PipelineJobResponse,
    summary="Build the knowledge graph for an organization",
    description=(
        "Triggers GraphBuilder.build() in the background. "
        "Returns 202 Accepted immediately. "
        "Poll GET /pipeline/jobs/{job_id} to track progress."
    ),
)
def build_graph(
    body: BuildGraphRequest,
    background_tasks: BackgroundTasks,
) -> PipelineJobResponse:
    job_id = _make_job(body.organization_id, "build-graph")
    background_tasks.add_task(
        _run_build_graph,
        body.organization_id,
        job_id,
    )
    return PipelineJobResponse(
        status="started",
        message=f"Graph build started (job_id={job_id}). Poll /pipeline/jobs/{job_id} for status.",
        organization_id=body.organization_id,
    )


@router.post(
    "/build-communities",
    status_code=202,
    response_model=PipelineJobResponse,
    summary="Build community reports for an organization",
    description=(
        "Triggers CommunityBuilder.build() (full graph) or "
        "CommunityBuilder.build_for_chunk() (chunk-only test mode) in the background. "
        "Returns 202 Accepted immediately. "
        "Poll GET /pipeline/jobs/{job_id} to track progress."
    ),
)
def build_communities(
    body: BuildCommunitiesRequest,
    background_tasks: BackgroundTasks,
    builder: CommunityBuilder = Depends(get_community_builder),
) -> PipelineJobResponse:
    job_id = _make_job(body.organization_id, "build-communities")
    background_tasks.add_task(
        _run_build_communities,
        body.organization_id,
        body.min_community_size,
        body.chunk_id,
        job_id,
        GRAPHS_FOLDER,
    )
    mode = f"chunk '{body.chunk_id}'" if body.chunk_id else "full graph"
    return PipelineJobResponse(
        status="started",
        message=(
            f"Community build ({mode}) started (job_id={job_id}). "
            f"Poll /pipeline/jobs/{job_id} for status."
        ),
        organization_id=body.organization_id,
    )


@router.get(
    "/jobs/{job_id}",
    summary="Poll pipeline job status",
    description="Returns the current status of a background pipeline job.",
)
def get_job_status(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@router.get(
    "/jobs",
    summary="List all pipeline jobs",
    description="Returns all pipeline jobs tracked since the server started.",
)
def list_jobs() -> list[dict]:
    return list(_jobs.values())
