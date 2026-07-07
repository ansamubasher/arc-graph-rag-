import streamlit as st
import sys
import os

# Ensure the root folder is in the Python path so we can import from core files
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frontend.components.sidebar import render_sidebar
from frontend.components.search_view import render_search_view

# Setup Page Configuration
st.set_page_config(
    page_title="Cosmos Graph RAG Studio",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* Premium aesthetics styling */
    /* Hide top-right menu and deploy buttons */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1E3A8A;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
    }
    h2 {
        color: #2563EB;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("🕸️ Cosmos Graph RAG Studio")
st.markdown(
    "Construct knowledge graphs from lease and real-estate documents stored in "
    "**Azure Cosmos DB** and query them globally using **Azure OpenAI**."
)

# Render Sidebar
render_sidebar()

# Render Global Search View directly
render_search_view()
