import streamlit as st
from db.db_registry import SUPPORTED_EMBEDDING_MODELS


def render_embedding_config():

    st.subheader("Embedding Configuration")

    embedding_model = st.selectbox(
        "Embedding Model",
        options=list(SUPPORTED_EMBEDDING_MODELS.keys()),
    )

    return {
        "embedding_model": SUPPORTED_EMBEDDING_MODELS.get(embedding_model),
    }