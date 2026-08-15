import streamlit as st

from ui.document_config import render_document_config
from ui.chunking_config import render_chunking_config
from ui.embedding_config import render_embedding_config
from ui.faiss_config import render_faiss_config
from ui.retrieval_config import render_retrieval_config


def render_sidebar():

    with st.sidebar:

        st.header("Knowledge Base")

        document_config = render_document_config()

        chunking_config = render_chunking_config()

        embedding_config = render_embedding_config()

        faiss_config = render_faiss_config()

        retrieval_config = render_retrieval_config()

        ingest = st.button(
            "Start Ingestion",
            use_container_width=True,
        )

    return {
        **document_config,
        **chunking_config,
        **embedding_config,
        **faiss_config,
        **retrieval_config,
        "ingest": ingest,
    }