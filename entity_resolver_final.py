import json
import os
import pickle
import sys

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


class EntityResolver:
    """
    Post-build cleanup pass: uses LLM to detect near-duplicates and
    invalid entities that slipped through alias-based dedup.
    
    For any two Property-type nodes with similar names or related edges,
    asks: "are these the same property?"

    Also catches generic role nouns (e.g. "Tenant", "Landlord") that were
    never merged with the specific named Party they refer to (e.g. "Agios
    Pharmaceuticals, Inc."), even when the generic node has few or no
    edges to compare against.
    """

    # FIX (Bug 2): generic contractual role nouns. These frequently appear
    # as their own Party node (because the entity extractor sees "Tenant"
    # used as a standalone term in some chunk) even though they refer to a
    # specific named Party defined elsewhere in the document. The
    # shared-edges heuristic below can't catch this case when the generic
    # node has few/no edges of its own (e.g. it was extracted in isolation
    # or the alias wasn't picked up), so it needs its own candidate rule.
    GENERIC_ROLE_NAMES = {
        "tenant", "landlord", "party", "buyer", "seller",
        "lessor", "lessee", "grantor", "grantee",
    }

    def __init__(self):
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

    def _similarity_score(self, name1, name2):
        """Quick heuristic for string similarity (0-1)."""
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()
        
        if n1 == n2:
            return 1.0
        
        common = sum(1 for c in n1 if c in n2)
        return common / max(len(n1), len(n2))

    def _candidates_for_merge(self, graph, threshold=0.7):
        """
        Return (source_id, target_id, reason, score) tuples for node pairs
        that might be duplicates. source_id is always the node that should
        be absorbed; target_id is the node that should remain canonical.

        Three heuristics:
        - Two Property-typed nodes with name similarity > threshold
        - A generic role noun (e.g. "Tenant") paired with any other named
          Party node — always proposed regardless of shared edges, since
          the generic node may be isolated or nearly isolated
        - Two nodes with many shared edges (indicating they describe the
          same thing)
        """
        candidates = []
        nodes = list(graph.nodes(data=True))
        
        for i, (n1_id, n1_data) in enumerate(nodes):
            for n2_id, n2_data in nodes[i+1:]:
                
                # Same-type similar-name heuristic
                if (n1_data.get("type") == "Property" and 
                    n2_data.get("type") == "Property"):
                    score = self._similarity_score(
                        n1_data.get("display_name", ""),
                        n2_data.get("display_name", "")
                    )
                    if score >= threshold:
                        candidates.append((n1_id, n2_id, "similar_name", score))

                # FIX (Bug 2): generic role noun vs. any other named Party.
                # Proposed unconditionally (no edge-overlap requirement)
                # because the generic node commonly has zero or few edges.
                # Ordered so the generic node is always the merge source
                # and the named node stays canonical.
                n1_is_party = n1_data.get("type") == "Party"
                n2_is_party = n2_data.get("type") == "Party"
                if n1_is_party and n2_is_party:
                    n1_generic = n1_id in self.GENERIC_ROLE_NAMES
                    n2_generic = n2_id in self.GENERIC_ROLE_NAMES
                    if n1_generic and not n2_generic:
                        candidates.append((n1_id, n2_id, "generic_role_vs_named", 1.0))
                        continue
                    elif n2_generic and not n1_generic:
                        candidates.append((n2_id, n1_id, "generic_role_vs_named", 1.0))
                        continue

                # Shared edges heuristic
                n1_neighbors = set(graph.successors(n1_id)).union(set(graph.predecessors(n1_id)))
                n2_neighbors = set(graph.successors(n2_id)).union(set(graph.predecessors(n2_id)))
                if n1_neighbors and n2_neighbors:
                    overlap = len(n1_neighbors & n2_neighbors) / min(len(n1_neighbors), len(n2_neighbors))
                    if overlap > 0.5:
                        candidates.append((n1_id, n2_id, "shared_edges", overlap))
        
        return candidates

    def _ask_merge(self, graph, node1_id, node2_id, reason, score):
        """Use LLM to decide if two nodes should be merged."""
        n1_data = graph.nodes[node1_id]
        n2_data = graph.nodes[node2_id]
        
        n1_name = n1_data.get("display_name", node1_id)
        n2_name = n2_data.get("display_name", node2_id)
        n1_type = n1_data.get("type", "Unknown")
        n2_type = n2_data.get("type", "Unknown")
        
        n1_neighbors = self._describe_neighbors(graph, node1_id)
        n2_neighbors = self._describe_neighbors(graph, node2_id)
        
        prompt = f"""Given two entities from a real-estate document knowledge graph, decide if they refer to the same thing.

If they ARE the same entity, respond with JSON: {{"merge": true, "reason": "brief explanation"}}
If they are NOT the same, respond with JSON: {{"merge": false, "reason": "brief explanation"}}

Entity 1:
- Name: {n1_name}
- Type: {n1_type}
- Related to: {n1_neighbors}

Entity 2:
- Name: {n2_name}
- Type: {n2_type}
- Related to: {n2_neighbors}

Merge reason hint: {reason} (confidence: {score:.2f})

Decide:"""
        
        try:
            response = self.client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a data deduplication expert. Compare entity pairs "
                            "and decide if they refer to the same real-world thing. "
                            "Return only valid JSON, no markdown, no explanation."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            
            raw = response.choices[0].message.content.strip()
            try:
                result = json.loads(raw)
                return result.get("merge", False)
            except json.JSONDecodeError:
                return False
        except Exception as e:
            print(f"    [warn] LLM merge check failed: {e}")
            return False

    def _describe_neighbors(self, graph, node_id):
        """Get a readable list of related entities."""
        neighbors = []
        for target, edge_data in graph[node_id].items():
            rel = edge_data.get(0, {}).get("relationship", "RELATED") if isinstance(edge_data, dict) else "RELATED"
            target_name = graph.nodes[target].get("display_name", target)
            neighbors.append(f"{target_name} ({rel})")
        return ", ".join(neighbors[:5]) if neighbors else "(none)"

    def resolve(self, graph):
        """
        Detect and merge duplicate entities in the graph.
        Returns a list of merge operations performed.
        """
        merges = []
        candidates = self._candidates_for_merge(graph, threshold=0.7)
        
        print(f"EntityResolver: Found {len(candidates)} potential duplicates to review.")
        
        for node1_id, node2_id, reason, score in candidates:
            # Skip if either node no longer exists (merged already).
            if not graph.has_node(node1_id) or not graph.has_node(node2_id):
                continue
            
            should_merge = self._ask_merge(graph, node1_id, node2_id, reason, score)
            
            if should_merge:
                merged = self._merge_nodes(graph, node1_id, node2_id)
                merges.append(merged)
                print(f"  Merged: {merged['source']} -> {merged['target']}")
        
        return merges

  
    def _merge_nodes(self, graph, source_id, target_id):
        """
        Merge source_id into target_id: copy all attributes, redirect edges.
        Keep target_id as canonical; remove source_id.
        """
        source_data = graph.nodes[source_id]
        target_data = graph.nodes[target_id]

        # source_chunks may be a list (fresh graph) or a comma-joined string
        # (round-tripped through GraphML) — normalize both to a list before merging.
        def _as_list(value):
            if isinstance(value, list):
                return value
            if isinstance(value, str) and value:
                return [v.strip() for v in value.split(",")]
            return []

        target_chunks = _as_list(target_data.get("source_chunks", []))
        source_chunks = _as_list(source_data.get("source_chunks", []))
        for chunk in source_chunks:
            if chunk not in target_chunks:
                target_chunks.append(chunk)
        target_data["source_chunks"] = ", ".join(target_chunks)

        # Redirect all edges from source to target
        for predecessor in list(graph.predecessors(source_id)):
            edges = graph.get_edge_data(predecessor, source_id) or {}
            for key, edge_data in edges.items():
                graph.add_edge(predecessor, target_id, **edge_data)

        for successor in list(graph.successors(source_id)):
            edges = graph.get_edge_data(source_id, successor) or {}
            for key, edge_data in edges.items():
                graph.add_edge(target_id, successor, **edge_data)

        source_name = source_data.get("display_name", source_id)
        target_name = target_data.get("display_name", target_id)

        # Remove source node
        graph.remove_node(source_id)

        return {
            "source": source_name,
            "target": target_name,
        }


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python entity_resolver.py <organization_id> [graph_folder]")
        sys.exit(1)
    
    graph_id = sys.argv[1]
    graph_folder = sys.argv[2] if len(sys.argv) > 2 else "graphs"

    graph_path = os.path.join(graph_folder, f"{graph_id}.graphml")

    if not os.path.exists(graph_path):
        print(f"Graph not found: {graph_path}")
        sys.exit(1)
    print(f"Loading graph: {graph_path}")
    graph = nx.read_graphml(graph_path)

    # read_graphml doesn't reliably preserve MultiDiGraph type on
    # round-trip — force it back so edge redirection logic (which
    # expects {key: {attrs}} from get_edge_data) works correctly.
    if not graph.is_multigraph():
        graph = nx.MultiDiGraph(graph)

    print(f"Loaded {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")

    # FIX (Bug 1): main() previously loaded the graph and stopped — resolve()
    # was never called and nothing was ever saved back, so this module was
    # dead code. Now it actually runs resolution and persists the result.
    resolver = EntityResolver()
    merges = resolver.resolve(graph)
    print(f"\nPerformed {len(merges)} merges.")
    print(f"After resolution: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")

    # Save back — pickle first (safe backup), then sanitized graphml,
    # mirroring the save pattern used in graph_builder.py.
    pickle_path = os.path.join(graph_folder, f"{graph_id}.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"Saved backup: {pickle_path}")

    export_graph = graph.copy()

    def _sanitize(data):
        for key, value in list(data.items()):
            if isinstance(value, (list, tuple, set)):
                data[key] = ", ".join(str(v) for v in value)
            elif isinstance(value, dict):
                data[key] = str(value)
            elif value is None:
                data[key] = ""

    for _, data in export_graph.nodes(data=True):
        _sanitize(data)
    for _, _, data in export_graph.edges(data=True):
        _sanitize(data)

    nx.write_graphml(export_graph, graph_path)
    print(f"Saved resolved graph: {graph_path}")


if __name__ == "__main__":
    main()