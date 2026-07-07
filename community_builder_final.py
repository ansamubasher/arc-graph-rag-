import argparse
import json
import os

import networkx as nx
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()


def _env(*keys, default=None):
    for key in keys:
        value = os.getenv(key)
        if value is None:
            continue
        value = value.strip()
        if value:
            return value
    return default


AZURE_OPENAI_ENDPOINT = _env("AZURE_OPENAI_ENDPOINT", "ENDPOINT_URL")
AZURE_OPENAI_KEY = _env("AZURE_OPENAI_KEY", "AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = _env("AZURE_OPENAI_DEPLOYMENT", "DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = _env("AZURE_OPENAI_API_VERSION", default="2024-02-01")


class CommunityBuilder:
    """
    Detects communities (clusters of related entities) in an organization's
    graph and generates an LLM summary ("community report") for each one.
    These reports are what global search queries against, instead of
    walking the raw graph node-by-node.

    Run this once after your graph is built (and re-run whenever the graph
    changes materially). Output is cached to
    graphs/{organization_id}_communities.json.
    """

    def __init__(self, graph_folder="graphs", min_community_size=2):
        self.graph_folder = graph_folder
        self.min_community_size = min_community_size

        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_ENDPOINT/ENDPOINT_URL": AZURE_OPENAI_ENDPOINT,
                "AZURE_OPENAI_KEY/AZURE_OPENAI_API_KEY": AZURE_OPENAI_KEY,
                "AZURE_OPENAI_DEPLOYMENT/DEPLOYMENT_NAME": AZURE_OPENAI_DEPLOYMENT,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                "Missing required environment variables: " + ", ".join(missing)
            )

        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )

    def _load_graph(self, organization_id):
        path = os.path.join(self.graph_folder, f"{organization_id}.graphml")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Graph not found: {path}")

        graph = nx.read_graphml(path)

        # FIX: read_graphml does not reliably preserve MultiDiGraph type on
        # round-trip (it depends on whether the file happens to contain
        # parallel edges with distinct keys). graph_builder.py stores
        # multiple edges of different relationship types between the same
        # node pair (e.g. HAS_TENANT and REFERENCES both from Landlord ->
        # Tenant), which requires MultiDiGraph semantics. If read_graphml
        # returns a plain DiGraph here, parallel edges silently collapse to
        # one, and community reports lose relationship facts without any
        # warning. Force it back explicitly, same fix as entity_resolver.py.
        if not graph.is_multigraph():
            graph = nx.MultiDiGraph(graph)

        return graph

    @staticmethod
    def _has_chunk(node_data, chunk_id):
        value = node_data.get("source_chunks")
        if value is None:
            return False

        if isinstance(value, list):
            return str(chunk_id) in {str(v).strip() for v in value}

        if isinstance(value, str):
            parts = {p.strip() for p in value.split(",") if p.strip()}
            return str(chunk_id) in parts

        return False

    def _extract_chunk_subgraph(self, graph, chunk_id):
        node_ids = [
            node_id
            for node_id, data in graph.nodes(data=True)
            if self._has_chunk(data, chunk_id)
        ]

        subgraph = graph.subgraph(node_ids).copy()

        # Keep only edges that also came from this chunk when provenance exists.
        if subgraph.is_multigraph():
            edges_to_remove = []
            for source, target, key, edge_data in subgraph.edges(keys=True, data=True):
                edge_chunk = edge_data.get("source_chunk")
                if edge_chunk and str(edge_chunk).strip() != str(chunk_id):
                    edges_to_remove.append((source, target, key))

            for source, target, key in edges_to_remove:
                subgraph.remove_edge(source, target, key=key)
        else:
            edges_to_remove = []
            for source, target, edge_data in subgraph.edges(data=True):
                edge_chunk = edge_data.get("source_chunk")
                if edge_chunk and str(edge_chunk).strip() != str(chunk_id):
                    edges_to_remove.append((source, target))

            for source, target in edges_to_remove:
                subgraph.remove_edge(source, target)

        return subgraph

    @staticmethod
    def _first_chunk_id_from_graph(graph):
        for _, data in graph.nodes(data=True):
            value = data.get("source_chunks")
            if isinstance(value, list):
                for chunk in value:
                    chunk = str(chunk).strip()
                    if chunk:
                        return chunk
            elif isinstance(value, str):
                for chunk in value.split(","):
                    chunk = chunk.strip()
                    if chunk:
                        return chunk
        return None

    def detect_communities(self, graph):
        """
        Runs Louvain community detection on the undirected projection of
        the graph (community detection doesn't care about edge direction
        here — we just want groups of tightly-connected entities).
        Returns a list of sets of node_ids, filtered to a minimum size so
        singleton/noise clusters don't get their own (expensive, low-value)
        LLM summary call.
        """

        undirected = graph.to_undirected()

        communities = list(
            nx.algorithms.community.louvain_communities(undirected, seed=42)
        )

        return [c for c in communities if len(c) >= self.min_community_size]

    @staticmethod
    def _community_to_text(graph, community):
        """Render a community's entities + internal relationships as text for the prompt."""

        lines = ["Entities:"]
        for node_id in community:
            data = graph.nodes[node_id]
            name = data.get("display_name", node_id)
            etype = data.get("type", "")
            lines.append(f"- {name} ({etype})")

        lines.append("\nRelationships:")
        has_edges = False
        for source, target, edge_data in graph.edges(data=True):
            if source in community and target in community:
                has_edges = True
                source_name = graph.nodes[source].get("display_name", source)
                target_name = graph.nodes[target].get("display_name", target)
                rel = edge_data.get("relationship", "RELATED_TO")
                lines.append(f"- {source_name} --[{rel}]--> {target_name}")

        if not has_edges:
            lines.append("(none)")

        return "\n".join(lines)

    def _summarize_community(self, community_text):

        prompt = f"""You are given a set of entities and relationships that belong to the
        same cluster within a larger knowledge graph extracted from lease /
        real-estate documents.

        Write a community report as JSON with these fields:
        - "title": a short (<=8 word) descriptive title for this cluster
        - "summary": a 2-4 sentence summary of what this cluster is about and the
        key facts it contains
        - "rating": an integer 1-10 for how important/information-dense this
        cluster is (10 = critical deal terms like rent/dates/parties, 1 = trivial)

                IMPORTANT NAME-PRESERVATION RULES:
                - For every fact or relationship you mention in "summary", include the
                    specific entity names involved (e.g., tenant name, lease ID,
                    property/building/suite name, exact date term label).
                - Do NOT generalize named entities into vague phrases like "a tenant",
                    "several parties", "a property", or "key terms" when a specific name
                    exists in the cluster data.
                - If the cluster data does not provide a specific name for a fact,
                    state only what is explicitly present and do not invent one.

        Return ONLY valid JSON, no markdown fences, no commentary.

        Cluster data:

        {community_text}
        """

        try:
            response = self.client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You summarize clusters of a knowledge graph "
                            "extracted from legal/real-estate text. Return "
                            "only valid JSON. Preserve specific entity names "
                            "for each fact; do not abstract them away."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
        except Exception as e:
            print(f"    [warn] community summary failed: {e}")
            return None

        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"    [warn] failed to parse community summary JSON: {e}")
            return None

    def build(self, organization_id):
        """
        Detects communities for an organization and writes
        graphs/{organization_id}_communities.json with one report per
        community, including its member entity display names.
        """

        graph = self._load_graph(organization_id)
        communities = self.detect_communities(graph)

        print(f"Found {len(communities)} communities (size >= {self.min_community_size}).")

        reports = []

        for i, community in enumerate(communities):

            community_text = self._community_to_text(graph, community)
            summary = self._summarize_community(community_text)

            if summary is None:
                continue

            member_names = [
                graph.nodes[n].get("display_name", n) for n in community
            ]

            # Extract document IDs from the members of the community
            doc_ids = set()
            for node_id in community:
                node_data = graph.nodes[node_id]
                chunks_val = node_data.get("source_chunks")
                if chunks_val:
                    chunk_list = []
                    if isinstance(chunks_val, list):
                        chunk_list = chunks_val
                    elif isinstance(chunks_val, str):
                        chunk_list = [c.strip() for c in chunks_val.split(",") if c.strip()]
                    
                    for chunk_id in chunk_list:
                        if "_" in chunk_id:
                            doc_ids.add(chunk_id.rsplit("_", 1)[0])
                        else:
                            doc_ids.add(chunk_id)

            reports.append({
                "community_id": i,
                "title": summary.get("title", f"Community {i}"),
                "summary": summary.get("summary", ""),
                "rating": summary.get("rating", 5),
                "size": len(community),
                "members": member_names,
                "document_ids": sorted(list(doc_ids)),
            })

            print(f"  [{i}] {summary.get('title')} ({len(community)} entities)")

        out_path = os.path.join(self.graph_folder, f"{organization_id}_communities.json")
        with open(out_path, "w") as f:
            json.dump(reports, f, indent=2)

        print(f"\nSaved {len(reports)} community reports to {out_path}")

        return reports

    def build_for_chunk(self, organization_id, chunk_id):
        """
        Builds community summaries for only one chunk (testing mode).
        It loads the full organization graph and narrows to the nodes/edges
        tagged with the target chunk id.
        """

        graph = self._load_graph(organization_id)
        chunk_graph = self._extract_chunk_subgraph(graph, chunk_id)

        if chunk_graph.number_of_nodes() == 0:
            print(f"No nodes found for chunk '{chunk_id}' in {organization_id}.")
            return []

        communities = list(
            nx.algorithms.community.louvain_communities(
                chunk_graph.to_undirected(),
                seed=42,
            )
        )

        print(f"Found {len(communities)} chunk communities for '{chunk_id}'.")

        reports = []
        for i, community in enumerate(communities):
            community_text = self._community_to_text(chunk_graph, community)
            summary = self._summarize_community(community_text)

            if summary is None:
                continue

            member_names = [
                chunk_graph.nodes[n].get("display_name", n) for n in community
            ]

            # Extract document IDs from the members of the community
            doc_ids = set()
            for node_id in community:
                node_data = chunk_graph.nodes[node_id]
                chunks_val = node_data.get("source_chunks")
                if chunks_val:
                    chunk_list = []
                    if isinstance(chunks_val, list):
                        chunk_list = chunks_val
                    elif isinstance(chunks_val, str):
                        chunk_list = [c.strip() for c in chunks_val.split(",") if c.strip()]
                    
                    for chunk_id_item in chunk_list:
                        if "_" in chunk_id_item:
                            doc_ids.add(chunk_id_item.rsplit("_", 1)[0])
                        else:
                            doc_ids.add(chunk_id_item)

            reports.append({
                "chunk_id": str(chunk_id),
                "community_id": i,
                "title": summary.get("title", f"Chunk Community {i}"),
                "summary": summary.get("summary", ""),
                "rating": summary.get("rating", 5),
                "size": len(community),
                "members": member_names,
                "document_ids": sorted(list(doc_ids)),
            })

            print(f"  [{i}] {summary.get('title')} ({len(community)} entities)")

        safe_chunk_id = "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in str(chunk_id)
        )
        out_path = os.path.join(
            self.graph_folder,
            f"{organization_id}_chunk_{safe_chunk_id}_communities.json",
        )
        with open(out_path, "w") as f:
            json.dump(reports, f, indent=2)

        print(f"\nSaved {len(reports)} chunk community reports to {out_path}")

        return reports


def main():
    parser = argparse.ArgumentParser(description="Build community reports for global search.")
    parser.add_argument("organization_id")
    parser.add_argument("--graph-folder", default="graphs")
    parser.add_argument("--min-size", type=int, default=2)
    parser.add_argument("--chunk-id", default=None, help="Specific chunk id to process")
    parser.add_argument("--full", action="store_true", help="Run full-graph community build instead of chunk-only test mode")
    args = parser.parse_args()

    builder = CommunityBuilder(
        graph_folder=args.graph_folder,
        min_community_size=args.min_size,
    )

    if args.full:
        builder.build(args.organization_id)
        return

    chunk_id = args.chunk_id
    if not chunk_id:
        graph = builder._load_graph(args.organization_id)
        chunk_id = builder._first_chunk_id_from_graph(graph)
        if not chunk_id:
            print("No chunk ids found in graph metadata; unable to run chunk-only test mode.")
            return
        print(f"No --chunk-id provided; using first available chunk: {chunk_id}")

    builder.build_for_chunk(args.organization_id, chunk_id)


if __name__ == "__main__":
    main()