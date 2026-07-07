import streamlit as st
from global_search import GlobalSearch

def render_search_view():
    st.subheader("Global Search Engine")
    st.markdown(
        "Ask broad, document-spanning questions (e.g. *'Summarize the lease terms'*, *'What are the main risks?'*) "
        "across all communities in the organization's knowledge graph."
    )
    
    org_id = st.session_state.get("selected_org_id")
    
    if not org_id:
        st.info("Please select an organization in the sidebar first.")
        return
        
    # Check if communities exist first
    import os
    communities_path = os.path.join("graphs", f"{org_id}_communities.json")
    if not os.path.exists(communities_path):
        st.warning(
            f"No community summaries found for organization '{org_id}'. "
            "Please build the graph and detect/summarize communities in the 'Pipeline' tab first."
        )
        return
        
    top_k = 5
    batch_size = 20
    filter_limit = 30
            
    # Simple query input
    query = st.text_input("Enter your question:", placeholder="e.g. What are the key lease terms and risks mentioned across all documents?")
    
    if st.button("Run Search", key="btn_run_search", use_container_width=True) and query:
        with st.spinner("Searching and synthesizing answers..."):
            try:
                # Instantiate search class
                search = GlobalSearch(
                    graph_folder="graphs",
                    batch_size=batch_size,
                    top_k=top_k,
                    filter_limit=filter_limit
                )
                
                # Run search
                result = search.answer(query, graph_id=org_id)
                
                # Show results
                st.markdown("### Answer")
                st.markdown(result["answer"])
                            
            except Exception as e:
                st.error(f"Failed to perform search: {e}")
