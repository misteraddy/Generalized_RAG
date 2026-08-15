import streamlit as st
from utils.constants import DOCUMENT_TYPES


def render_document_config():

    st.subheader("Document Configuration")

    document_type = st.selectbox(
        "Document Type",
        options=list(DOCUMENT_TYPES.keys()),
    )

    url = None
    uploaded_files = None

    if document_type == "Web":

        url = st.text_input(
            "Website URL",
            placeholder="https://example.com",
        )

        uploaded_files = None

    else:

        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=DOCUMENT_TYPES[document_type],
            accept_multiple_files=True,
        )

        url = None

    return {
        "document_type": document_type,
        "url": url,
        "uploaded_files": uploaded_files,
    }