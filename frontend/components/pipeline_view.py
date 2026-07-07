import streamlit as st
import os
import sys

from cosmos_reader import CosmosReader
from chunker import Chunker
from azure_extraction import EntityExtractor
from azure_rel2 import RelationshipExtractor
from graph_builder_final import GraphBuilder
from community_builder_final import CommunityBuilder

def build_graph_for_org(organization_id, status_placeholder):
    status_placeholder.info(f"Starting Graph Build for organization: {organization_id}...")
    
    try:
        reader = CosmosReader()
        chunker = Chunker()
        entity_extractor = EntityExtractor()
        relationship_extractor = RelationshipExtractor()
        
        builder = GraphBuilder(organization_id)
        documents = list(reader.load_documents_by_organization(organization_id))
        
        if not documents:
            status_placeholder.warning("No documents found for this organization in Cosmos DB.")
            return False
            
        status_placeholder.info(f"Loaded {len(documents)} documents. Starting extraction...")
        
        progress_bar = st.progress(0)
        for idx, document in enumerate(documents):
            doc_id = document.get("id", "Unknown")
            status_placeholder.info(f"Processing document {idx+1}/{len(documents)}: {doc_id}...")
            
            builder.seed_document_metadata(document)
            chunks = chunker.chunk_document(document)
            
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    entities = entity_extractor.extract(chunk)
                    relationships = relationship_extractor.extract(chunk, entities)
                    
                    builder.add_entities(entities, chunk_id=chunk["chunk_id"])
                    builder.add_related_to_edges(entities, chunk_id=chunk["chunk_id"])
                    builder.add_relationships(relationships, chunk_id=chunk["chunk_id"])
                except Exception as chunk_err:
                    st.warning(f"Error processing chunk {chunk.get('chunk_id')}: {chunk_err}")
                    continue
            
            progress_bar.progress((idx + 1) / len(documents))
            
        status_placeholder.info("Running entity resolution pass to merge duplicates and generic roles...")
        try:
            from entity_resolver_final import EntityResolver
            resolver = EntityResolver()
            resolver.resolve(builder.graph)
        except Exception as resolve_err:
            st.warning(f"Entity resolution pass had warnings: {resolve_err}")
            
        status_placeholder.info("Finalizing and saving resolved knowledge graph...")
        builder.save_graph()
        status_placeholder.success(f"Graph successfully built with {builder.number_of_nodes()} nodes and {builder.number_of_edges()} edges!")
        return True
        
    except Exception as e:
        status_placeholder.error(f"Graph build failed: {e}")
        return False


def build_communities_for_org(organization_id, status_placeholder):
    status_placeholder.info(f"Starting Community Detection and Summarization for: {organization_id}...")
    
    try:
        builder = CommunityBuilder(graph_folder="graphs", min_community_size=2)
        
        # We will wrap the execution to show stdout/logs or just let it run.
        # Since build() prints to stdout, we can call it directly and show success.
        reports = builder.build(organization_id)
        
        status_placeholder.success(f"Successfully generated {len(reports)} community reports for {organization_id}!")
        return True
    except Exception as e:
        status_placeholder.error(f"Community build failed: {e}")
        return False


def render_pipeline_view():
    st.subheader("Data Pipeline Control Panel")
    st.markdown("Trigger graph building and community report summarization for the selected organization.")
    
    org_id = st.session_state.get("selected_org_id")
    
    if not org_id:
        st.info("Please select an organization in the sidebar first.")
        return
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 1. Knowledge Graph Builder")
        st.write("Extracts entities/relationships from Cosmos DB documents and saves the GraphML representation.")
        if st.button("Build Knowledge Graph", key="btn_build_graph", use_container_width=True):
            status = st.empty()
            build_graph_for_org(org_id, status)
            st.rerun()
            
    with col2:
        st.write("### 2. Community Detection & Summary")
        st.write("Clusters the graph using Louvain community detection and generates LLM summaries for each cluster.")
        if st.button("Detect & Summarize Communities", key="btn_build_communities", use_container_width=True):
            status = st.empty()
            build_communities_for_org(org_id, status)
            st.rerun()
