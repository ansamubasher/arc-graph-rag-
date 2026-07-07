"""
Shared dependency singletons for the FastAPI application.

All heavy objects (Cosmos client, OpenAI client, etc.) are instantiated once
at module level and injected into route handlers via FastAPI's Depends() system.
This avoids re-creating SDK clients on every request.
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that the existing modules
# (cosmos_reader, global_search, etc.) can be imported regardless of the
# working directory from which uvicorn is launched.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from cosmos_reader import CosmosReader  # noqa: E402
from global_search import GlobalSearch  # noqa: E402
from community_builder_final import CommunityBuilder  # noqa: E402

# ---------------------------------------------------------------------------
# Graphs folder — configurable via env var, defaults to ./graphs
# ---------------------------------------------------------------------------
GRAPHS_FOLDER: str = os.getenv("GRAPHS_FOLDER", os.path.join(_PROJECT_ROOT, "graphs"))

# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------
_cosmos_reader: CosmosReader | None = None
_global_search: GlobalSearch | None = None
_community_builder: CommunityBuilder | None = None


def get_cosmos_reader() -> CosmosReader:
    """Return the shared CosmosReader singleton, creating it on first call."""
    global _cosmos_reader
    if _cosmos_reader is None:
        _cosmos_reader = CosmosReader()
    return _cosmos_reader


def get_global_search() -> GlobalSearch:
    """Return the shared GlobalSearch singleton, creating it on first call.

    The singleton caches community report embeddings per graph_id across
    requests, so repeated queries to the same graph are fast after the
    first warm-up.
    """
    global _global_search
    if _global_search is None:
        _global_search = GlobalSearch(graph_folder=GRAPHS_FOLDER)
    return _global_search


def get_community_builder() -> CommunityBuilder:
    """Return the shared CommunityBuilder singleton, creating it on first call."""
    global _community_builder
    if _community_builder is None:
        _community_builder = CommunityBuilder(graph_folder=GRAPHS_FOLDER)
    return _community_builder
