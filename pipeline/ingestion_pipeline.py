import streamlit as st

from ingestion.loader import load_document
from ingestion.validator import validate_files
from chunking.chunker import chunk_documents
from db.db_utils import save_chunks_to_vector_store
from utils.constants import DOCUMENT_TYPES

document_types = [
    extension
    for extensions in DOCUMENT_TYPES.values()
    for extension in extensions
]

def run_ingestion(config):

    if not config["ingest"]:
        return

    files = config["uploaded_files"]
    chunking_strategy = config["chunking_strategy"]

    # Validate the uploaded files first
    with st.status("🔍 Validating uploaded files...", expanded=True) as status:

        validation_errors = validate_files(
            uploaded_files=files,
            allowed_extensions=document_types
        )

        if validation_errors:
            status.update(
                label="❌ File validation failed",
                state="error"
            )
            st.error(f"Validation errors: {validation_errors}")
            return

        st.write(f"📁 Files received: **{len(files)}**")

        status.update(
            label="✅ File validation completed",
            state="complete"
        )

    # Load the files into documents
    with st.status("📄 Loading documents...", expanded=True) as status:

        try:
            documents = load_document(files)

            st.write(f"📄 Documents loaded: **{len(documents)}**")

            status.update(
                label="✅ Document loading completed",
                state="complete"
            )

        except Exception as e:
            status.update(
                label="❌ Document loading failed",
                state="error"
            )
            st.exception(e)
            return

    # Split the documents into chunks
    with st.status(
        f"✂️ Chunking documents using **{chunking_strategy}**...",
        expanded=True
    ) as status:

        try:
            chunks = chunk_documents(
                documents,
                strategy=chunking_strategy,
                config=config
            )

            st.write(f"🧩 Chunks created: **{len(chunks)}**")

            status.update(
                label="✅ Document chunking completed",
                state="complete"
            )

        except Exception as e:
            status.update(
                label="❌ Chunking failed",
                state="error"
            )
            st.exception(e)
            return

    # Save the chunks to the vector store
    with st.status("🗄️ Creating vector store...", expanded=True) as status:

        try:
            vector_store = save_chunks_to_vector_store(
                chunks,
                config
            )

            st.write(f"📌 Vectors indexed: **{len(chunks)}**")

            status.update(
                label="✅ Vector store creation completed",
                state="complete"
            )

        except Exception as e:
            status.update(
                label="❌ Vector store creation failed",
                state="error"
            )
            st.exception(e)
            return

    st.success("🎉 Ingestion completed successfully!")

