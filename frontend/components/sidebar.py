import streamlit as st
import os
from cosmos_reader import CosmosReader

def render_sidebar():
    st.sidebar.title("Configuration")
    
    # Initialize CosmosReader
    try:
        reader = CosmosReader()
        org_ids = reader.get_organization_ids()
    except Exception as e:
        st.sidebar.error(f"Failed to connect to Cosmos DB: {e}")
        org_ids = []
        
    if not org_ids:
        st.sidebar.warning("No organizations found in Cosmos DB.")
        st.session_state["selected_org_id"] = None
        return
        
    # Filter out organization IDs starting with '1e', '1d', or equal to 'id'
    org_ids = [
        org for org in org_ids
        if not str(org).lower().startswith("1e")
        and not str(org).lower().startswith("1d")
        and str(org).lower() != "id"
    ]
    
    if not org_ids:
        st.sidebar.warning("No valid organizations found in Cosmos DB.")
        st.session_state["selected_org_id"] = None
        return
        
    selected_org_id = st.sidebar.selectbox(
        "Select Organization",
        options=org_ids,
        index=0 if "selected_org_id" not in st.session_state or st.session_state["selected_org_id"] not in org_ids else org_ids.index(st.session_state["selected_org_id"])
    )
    
    st.session_state["selected_org_id"] = selected_org_id
    
    # Check graph status
    if selected_org_id:
        graph_path = os.path.join("graphs", f"{selected_org_id}.graphml")
        communities_path = os.path.join("graphs", f"{selected_org_id}_communities.json")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Organization Status")
        
        if os.path.exists(graph_path):
            st.sidebar.success("✅ Knowledge Graph Built")
        else:
            st.sidebar.warning("⚠️ Knowledge Graph Missing")
            
        if os.path.exists(communities_path):
            st.sidebar.success("✅ Communities Detected")
        else:
            st.sidebar.warning("⚠️ Communities Missing")
