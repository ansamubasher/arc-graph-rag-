import networkx as nx
import os
import pickle
import argparse

from cosmos_reader import CosmosReader
from chunker import Chunker
from azure_extraction import EntityExtractor
from azure_rel2 import RelationshipExtractor


class GraphBuilder:

    # Entity types that SHOULD merge across different documents when the
    # name matches exactly (e.g. the same tenant company leasing multiple
    # properties, or the same city appearing in multiple leases). All other
    # types get namespaced per document — see _target_node_key below —
    # because generic recurring lease terms ("Base Rent", "Term",
    # "Months 1-12", "Security Deposit") are common across MANY unrelated
    # leases and must NOT collapse into one node just because the text
    # matches. This only matters once multiple documents share one graph
    # (--organization-id runs); it's invisible when testing on one document.
    PORTFOLIO_SHARED_TYPES = {"Party", "Location"}

    def __init__(self, organization_id):

        self.organization_id = organization_id
        self.graph = nx.MultiDiGraph()
        self.alias_map = {}
        self.current_document_id = None

    @staticmethod
    def _normalize_key(name):
        if not isinstance(name, str):
            return None
        return " ".join(name.strip().lower().split())

    def _register_aliases(self, node_key, aliases):
        if not isinstance(aliases, list):
            return
        for alias in aliases:
            if alias:
                alias_key = self._normalize_key(alias)
                if alias_key:
                    self.alias_map[alias_key] = node_key

    def _resolve_alias(self, node_key):
        return self.alias_map.get(node_key, node_key)

    def _find_existing_node_key(self, plain_key):
        """
        Look up a node that may have been created under its plain key
        (portfolio-shared types, or an earlier chunk in this same
        document) or under a document-scoped key (non-shared types
        created earlier in this same document). Returns the key that
        actually exists in the graph, or the plain key if neither does
        (caller handles the "doesn't exist" case as before).
        """
        if self.graph.has_node(plain_key):
            return plain_key
        if self.current_document_id:
            scoped_key = f"doc_{self.current_document_id}::{plain_key}"
            if self.graph.has_node(scoped_key):
                return scoped_key
        return plain_key

    def _target_node_key_for_new_entity(self, plain_key, entity_type):
        """
        Decide the key to use when CREATING a brand-new node (plain_key
        was not found by _find_existing_node_key). Portfolio-shared types
        get the plain key, so a matching entity in a later document merges
        into it on purpose. Everything else gets namespaced to the current
        document, so an unrelated document's identically-worded generic
        term ("Base Rent", "Term", etc.) doesn't accidentally merge into
        this node later.
        """
        if entity_type in self.PORTFOLIO_SHARED_TYPES or not self.current_document_id:
            return plain_key
        return f"doc_{self.current_document_id}::{plain_key}"

    def seed_document_metadata(self, document):
        """
        Pre-seed canonical nodes from document-level metadata that's
        already available on the Cosmos document (and was already being
        attached to every chunk by chunker.py, but never used downstream).

        This runs BEFORE any chunk in the document is processed, so that
        'tenant'/'landlord' are registered as aliases from the start —
        any later chunk that extracts a bare "Tenant"/"Landlord" entity
        will resolve straight to the correct named Party node via the
        alias map, instead of creating an isolated generic node that has
        to be fixed later by entity_resolver's LLM merge pass.

        This also sidesteps the cross-chunk relationship gap for these
        specific fields: tenant_name/landlord_name/property_address/etc.
        are document-level ground truth, not something extracted from a
        single chunk's text, so they're available and correct regardless
        of which chunk mentions "the Tenant" or "the Property".
        """

        self.current_document_id = document.get("id")

        tenant_name = document.get("tenant_name")
        landlord_name = document.get("landlord_name")
        property_address = document.get("property_address")
        lease_term = document.get("lease_term")
        commencement_date = document.get("commencement_date")
        expiration_date = document.get("expiration_date")

        doc_chunk_id = f"{document.get('id')}_metadata"

        tenant_key = None
        landlord_key = None
        property_key = None
        term_key = None
        commencement_key = None
        expiration_key = None

        def _seed_node(raw_name, entity_type):
            key = self._normalize_key(raw_name)
            if key is None:
                return None
            if not self.graph.has_node(key):
                self.graph.add_node(
                    key,
                    display_name=raw_name,
                    type=entity_type,
                    value=None,
                    unit=None,
                    related_to=None,
                    source_chunks=[doc_chunk_id],
                    type_conflicts=[],
                    aliases=[],
                )
            return key

        if tenant_name:
            tenant_key = _seed_node(tenant_name, "Party")
            self.alias_map["tenant"] = tenant_key

        if landlord_name:
            landlord_key = _seed_node(landlord_name, "Party")
            self.alias_map["landlord"] = landlord_key

        if property_address:
            property_key = _seed_node(property_address, "Property")
            # common generic terms used in lease text for "the leased space"
            self.alias_map["property"] = property_key
            self.alias_map["the property"] = property_key
            self.alias_map["premises"] = property_key
            self.alias_map["the premises"] = property_key

        if lease_term:
            term_key = _seed_node(lease_term, "Duration")

        if commencement_date:
            commencement_key = _seed_node(commencement_date, "Date")

        if expiration_date:
            expiration_key = _seed_node(expiration_date, "Date")

        def _seed_edge(source_key, target_key, relationship):
            if not source_key or not target_key or source_key == target_key:
                return
            existing = self.graph.get_edge_data(source_key, target_key) or {}
            if any(data.get("relationship") == relationship for data in existing.values()):
                return
            self.graph.add_edge(
                source_key,
                target_key,
                relationship=relationship,
                source_chunk=doc_chunk_id,
            )

        # Deterministic backbone edges — using the same relationship
        # vocabulary as RelationshipExtractor.ALLOWED_RELATIONSHIPS so
        # downstream code (community_builder, inspect_graph) treats them
        # identically to LLM-extracted edges.
        _seed_edge(property_key, tenant_key, "HAS_TENANT")
        _seed_edge(property_key, landlord_key, "HAS_LANDLORD")
        _seed_edge(tenant_key, property_key, "LEASES")
        _seed_edge(property_key, term_key, "APPLIES_TO")
        _seed_edge(term_key, commencement_key, "EFFECTIVE_ON")
        _seed_edge(term_key, expiration_key, "HAS_DEADLINE")

    def add_entities(self, entities, chunk_id=None):

        for entity in entities:

            if not isinstance(entity, dict):
                print(f"    [warn] Skipping malformed entity (not a dict): {entity}")
                continue

            raw_name = entity.get("entity")
            entity_type = entity.get("type")

            if not raw_name or not entity_type:
                print(f"    [warn] Skipping entity missing 'entity' or 'type': {entity}")
                continue

            node_key = self._normalize_key(raw_name)
            if node_key is None:
                print(f"    [warn] Skipping entity with invalid name: {entity}")
                continue

            aliases = entity.get("aliases", [])
            self._register_aliases(node_key, aliases)

            # FIX (Bug 3): resolve node_key through the alias map before
            # checking has_node/creating a node. Previously only
            # add_relationships() did this resolution, so a bare "Tenant"
            # entity extracted from a later chunk would create its own
            # separate "tenant" node instead of merging into the named
            # entity ("Agios Pharmaceuticals, Inc.") that an earlier chunk
            # had already registered "Tenant" as an alias for.
            node_key = self._resolve_alias(node_key)

            # Look for an existing node under either its plain key or its
            # document-scoped key before deciding whether to create a new
            # one — see _find_existing_node_key / _target_node_key_for_new_entity.
            node_key = self._find_existing_node_key(node_key)

            if not self.graph.has_node(node_key):
                node_key = self._target_node_key_for_new_entity(node_key, entity_type)

            if self.graph.has_node(node_key):
                node = self.graph.nodes[node_key]

                if chunk_id and chunk_id not in node["source_chunks"]:
                    node["source_chunks"].append(chunk_id)

                if node.get("value") is None and entity.get("value") is not None:
                    node["value"] = entity.get("value")
                if node.get("unit") is None and entity.get("unit") is not None:
                    node["unit"] = entity.get("unit")

                if entity_type != node.get("type"):
                    node.setdefault("type_conflicts", [])
                    if entity_type not in node["type_conflicts"]:
                        node["type_conflicts"].append(entity_type)

            else:
                self.graph.add_node(
                    node_key,
                    display_name=raw_name,
                    type=entity_type,
                    value=entity.get("value"),
                    unit=entity.get("unit"),
                    related_to=entity.get("related_to"),
                    source_chunks=[chunk_id] if chunk_id else [],
                    type_conflicts=[],
                    aliases=aliases,
                )

    def add_related_to_edges(self, entities, chunk_id=None):
        """
        Build edges directly from each entity's own 'related_to' field,
        in addition to whatever RelationshipExtractor separately produces.

        The entity extraction prompt already requires the model to set
        related_to whenever a value clearly belongs to another entity in
        the same chunk (e.g. a Rate related_to a Duration, a Measurement
        related_to a Party). Previously this was stored as a node
        attribute but never converted into a graph edge, so it was
        silently discarded unless the separate relationship-extraction
        call also happened to notice the same connection. Since community
        detection (and therefore global search) drops any node with zero
        edges entirely, this was quietly making some facts invisible to
        global search even though they'd already been correctly
        extracted. Must be called after add_entities() for the same
        batch, since related_to targets are expected to be other entities
        from the same chunk.
        """

        for entity in entities:

            if not isinstance(entity, dict):
                continue

            raw_name = entity.get("entity")
            related_to = entity.get("related_to")

            if not raw_name or not related_to:
                continue

            source_key = self._normalize_key(raw_name)
            target_key = self._normalize_key(related_to)

            if source_key is None or target_key is None:
                continue

            source_key = self._resolve_alias(source_key)
            target_key = self._resolve_alias(target_key)

            source_key = self._find_existing_node_key(source_key)
            target_key = self._find_existing_node_key(target_key)

            if source_key == target_key:
                continue

            if not self.graph.has_node(source_key):
                continue

            if not self.graph.has_node(target_key):
                # related_to pointed at something that wasn't itself
                # extracted as an entity in this chunk (e.g. a typo, or a
                # generic phrase the extractor rejected) — skip rather
                # than fabricate a node for it.
                continue

            existing = self.graph.get_edge_data(source_key, target_key) or {}
            if any(data.get("relationship") == "RELATED_TO" for data in existing.values()):
                continue

            self.graph.add_edge(
                source_key,
                target_key,
                relationship="RELATED_TO",
                source_chunk=chunk_id,
            )

    def add_relationships(self, relationships, chunk_id=None):

        for relationship in relationships:

            if not isinstance(relationship, dict):
                print(f"    [warn] Skipping malformed relationship (not a dict): {relationship}")
                continue

            source = relationship.get("source")
            target = relationship.get("target")
            relation = relationship.get("relationship")

            if not source or not target or not relation:
                print(f"    [warn] Skipping relationship missing a required field: {relationship}")
                continue

            source_key = self._normalize_key(source)
            target_key = self._normalize_key(target)

            source_key = self._resolve_alias(source_key)
            target_key = self._resolve_alias(target_key)

            source_key = self._find_existing_node_key(source_key)
            target_key = self._find_existing_node_key(target_key)

            if not self.graph.has_node(source_key):
                print(f"    [warn] Skipping edge — unknown source entity: '{source}'")
                continue

            if not self.graph.has_node(target_key):
                print(f"    [warn] Skipping edge — unknown target entity: '{target}'")
                continue

            existing = self.graph.get_edge_data(source_key, target_key) or {}
            if any(data.get("relationship") == relation for data in existing.values()):
                continue

            self.graph.add_edge(
                source_key,
                target_key,
                relationship=relation,
                source_chunk=chunk_id,
            )

    def get_graph(self):
        return self.graph

    def number_of_nodes(self):
        return self.graph.number_of_nodes()

    def number_of_edges(self):
        return self.graph.number_of_edges()

    def save_graph(self, output_folder="graphs"):

        os.makedirs(output_folder, exist_ok=True)

        graphml_path = f"{output_folder}/{self.organization_id}.graphml"
        pickle_path = f"{output_folder}/{self.organization_id}.pkl"

        # Save a pickle FIRST, before any export logic that could fail.
        # Pickle handles arbitrary Python objects (lists, dicts, None)
        # natively, so this always succeeds and protects the extraction
        # work regardless of what happens in the GraphML export below.
        with open(pickle_path, "wb") as f:
            pickle.dump(self.graph, f)
        print(f"Saved backup: {pickle_path}")

        # graphml doesn't support list/dict-valued attributes — sanitize
        # ANY such attribute generically, not just specific keys, so new
        # list-valued fields (like 'aliases') don't silently break this
        # again in the future.
        export_graph = self.graph.copy()

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

        try:
            nx.write_graphml(export_graph, graphml_path)
            print(f"Saved graph: {graphml_path}")
        except Exception as e:
            # Keep the run non-fatal at the end; pickle backup above is
            # already saved and can still be used for recovery.
            print(f"[warn] Failed to write GraphML at end of run: {e}")
            print(f"[warn] You can recover from pickle backup: {pickle_path}")


def main():

    parser = argparse.ArgumentParser(description="Build graph for all docs or a single document.")
    parser.add_argument("--organization-id", default=None, help="Restrict run to one organization id")
    parser.add_argument("--document-id", default=None, help="Process only this document id")
    parser.add_argument(
        "--document-ids",
        nargs="+",
        default=None,
        help="Process only these document ids (space-separated), e.g. --document-ids doc1 doc2",
    )
    parser.add_argument("--first-doc", action="store_true", help="Process only the first document in each selected organization")
    args = parser.parse_args()

    selected_modes = sum(
        bool(x)
        for x in [args.document_id, args.document_ids, args.first_doc]
    )
    if selected_modes > 1:
        raise ValueError("Use only one of --document-id, --document-ids, or --first-doc.")

    reader = CosmosReader()
    chunker = Chunker()
    entity_extractor = EntityExtractor()
    relationship_extractor = RelationshipExtractor()

    for organization_id in reader.get_organization_ids():

        if args.organization_id and organization_id != args.organization_id:
            continue

        print(f"\nBuilding graph for organization: {organization_id}")

        builder_id = organization_id

        if args.document_id:
            safe_doc_id = "".join(
                c if c.isalnum() or c in ("-", "_") else "_" for c in args.document_id
            )
            builder_id = f"{organization_id}_{safe_doc_id}"
        elif args.document_ids:
            safe_joined_ids = "_".join(
                "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in doc_id)
                for doc_id in args.document_ids
            )
            builder_id = f"{organization_id}_docs_{safe_joined_ids}"
        elif args.first_doc:
            builder_id = f"{organization_id}_firstdoc"

        builder = GraphBuilder(builder_id)

        if args.document_id:
            document = reader.load_document_by_id(organization_id, args.document_id)
            documents = [document] if document else []
            if not documents:
                print(f"  [warn] Document not found: {args.document_id}")
        elif args.document_ids:
            all_documents = reader.load_documents_by_organization(organization_id)
            doc_lookup = {doc.get("id"): doc for doc in all_documents}
            documents = [doc_lookup[doc_id] for doc_id in args.document_ids if doc_id in doc_lookup]
            missing = [doc_id for doc_id in args.document_ids if doc_id not in doc_lookup]
            for missing_id in missing:
                print(f"  [warn] Document not found: {missing_id}")
        else:
            documents = reader.load_documents_by_organization(organization_id)
            if args.first_doc:
                documents = documents[:1]

        for document in documents:

            try:
                builder.seed_document_metadata(document)

                chunks = chunker.chunk_document(document)

                print(f"\n  Document: {document['id']} — {len(chunks)} chunks")

                for chunk in chunks:
                    print(f"    Processing chunk: {chunk['chunk_id']}")

                    try:
                        entities = entity_extractor.extract(chunk)
                        print(f"      ✓ Extracted {len(entities)} entities")
                        relationships = relationship_extractor.extract(chunk, entities)
                        print(f"      ✓ Extracted {len(relationships)} relationships")

                        builder.add_entities(entities, chunk_id=chunk["chunk_id"])
                        builder.add_related_to_edges(entities, chunk_id=chunk["chunk_id"])
                        builder.add_relationships(relationships, chunk_id=chunk["chunk_id"])
                    except Exception as e:
                        print(
                            f"      [warn] Chunk failed ({chunk.get('chunk_id')}): {e}. "
                            "Skipping this chunk and continuing."
                        )
                        continue
            except Exception as e:
                print(
                    f"  [warn] Document failed ({document.get('id')}): {e}. "
                    "Skipping this document and continuing."
                )
                continue

        print(
            f"\n  Graph complete — "
            f"{builder.number_of_nodes()} nodes, "
            f"{builder.number_of_edges()} edges"
        )

        builder.save_graph()


if __name__ == "__main__":
    main()