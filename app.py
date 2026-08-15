import streamlit as st

from ui.sidebar import render_sidebar
from pipeline.ingestion_pipeline import run_ingestion


st.set_page_config(
    page_title="Universal RAG",
    page_icon="📚",
    layout="wide",
)

st.title("Universal RAG")
st.write("Upload documents to build your knowledge base.")

config = render_sidebar()

if config["ingest"]:
    run_ingestion(config)


