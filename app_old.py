from pathlib import Path

import streamlit as st

from chunking.chunker import chunk_documents
from db.db_registry import (
DISTANCE_METRICS,
EMBEDDING_MODELS,
INDEXING_TECHNIQUES,
)
from ingestion.loader import load_document
from utils.constants import DOCUMENT_TYPES
from chunking.chunk_registry import CHUNK_OVERLAP_OPTIONS, CHUNK_SIZE_OPTIONS, CHUNKING_STRATEGIES

# ================================================================

# PAGE CONFIGURATION

# ================================================================

st.set_page_config(
page_title="Universal RAG",
page_icon="📚",
layout="wide",
)

st.title("Universal RAG")

st.write(
"Upload documents to build your knowledge base."
)

# ================================================================

# SIDEBAR

# ================================================================

with st.sidebar:

    st.header("Knowledge Base")

    # ============================================================
    # DOCUMENT CONFIGURATION
    # ============================================================

    document_type = st.selectbox(
        "Document Type",
        options=list(DOCUMENT_TYPES.keys()),
    )

    chunking_strategy = st.selectbox(
        "Chunking Strategy",
        options=list(CHUNKING_STRATEGIES.keys()),
    )

    # ============================================================
    # INITIALIZE CHUNKING PARAMETERS
    # ============================================================

    chunk_size = None
    overlap = None
    breakpoint_percentile = None
    buffer_size = None

    # ============================================================
    # CHARACTER-BASED CHUNKING PARAMETERS
    # ============================================================

    if chunking_strategy in [
        "Fixed",
        "Markdown",
        "Recursive Character",
        "Token",
        "Content Aware",
    ]:

        chunk_size = st.selectbox(
            "Chunk Size",
            options=CHUNK_SIZE_OPTIONS,
            index=2,
            help="Maximum size of each chunk.",
        )

        overlap_options = [
            value
            for value in
                CHUNK_OVERLAP_OPTIONS
            if value < chunk_size
        ]

        if not overlap_options:
            overlap_options = [0]

        default_overlap = (
            100
            if 100 in overlap_options
            else overlap_options[0]
        )

        overlap = st.selectbox(
            "Chunk Overlap",
            options=overlap_options,
            index=overlap_options.index(default_overlap),
            help="Amount of overlap between consecutive chunks.",
        )

    # ============================================================
    # SEMANTIC CHUNKING PARAMETERS
    # ============================================================

    elif chunking_strategy == "Semantic":

        breakpoint_percentile = st.slider(
            "Breakpoint Percentile",
            min_value=50,
            max_value=100,
            value=95,
            help="Higher values create fewer, larger chunks.",
        )

        buffer_size = st.number_input(
            "Buffer Size",
            min_value=0,
            max_value=5,
            value=1,
            step=1,
            help=(
                "Number of neighboring sentences considered "
                "while computing semantic similarity."
            ),
        )

    # ============================================================
    # WEB / FILE INPUT
    # ============================================================

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

    # ============================================================
    # EMBEDDING CONFIGURATION
    # ============================================================

    st.subheader("Embedding Configuration")

    embedding_model = st.selectbox(
        "Embedding Model",
        options=list(EMBEDDING_MODELS.keys()),
        index=0,
        help="Sentence Transformer model used to generate embeddings.",
    )

    # ============================================================
    # FAISS CONFIGURATION
    # ============================================================

    st.subheader("FAISS Configuration")

    indexing_technique = st.selectbox(
        "Indexing Technique",
        options=list(INDEXING_TECHNIQUES.keys()),
        index=0,
        help="FAISS index used for vector similarity search.",
    )

    # Initialize FAISS parameters

    distance_metric = None

    nlist = None
    nprobe = None

    hnsw_m = None
    ef_construction = None
    ef_search = None

    pq_m = None
    nbits = None

    # ============================================================
    # FLAT INDEX
    # ============================================================

    if indexing_technique == "Flat":

        distance_metric = st.selectbox(
            "Distance Metric",
            options=list(DISTANCE_METRICS.keys()),
            index=0,
            help="Similarity metric used by the FAISS Flat index.",
        )

    # ============================================================
    # IVF INDEX
    # ============================================================

    elif indexing_technique == "IVF":

        distance_metric = st.selectbox(
            "Distance Metric",
            options=list(DISTANCE_METRICS.keys()),
            index=0,
            help="Similarity metric used by the IVF index.",
        )

        nlist = st.number_input(
            "Number of Clusters (nlist)",
            min_value=1,
            max_value=4096,
            value=100,
            step=1,
            help=(
                "Number of Voronoi clusters used "
                "during IVF index construction."
            ),
        )

        nprobe = st.number_input(
            "Search Clusters (nprobe)",
            min_value=1,
            max_value=4096,
            value=10,
            step=1,
            help=(
                "Number of IVF clusters searched "
                "during retrieval."
            ),
        )

    # ============================================================
    # HNSW INDEX
    # ============================================================

    elif indexing_technique == "HNSW":

        distance_metric = st.selectbox(
            "Distance Metric",
            options=list(DISTANCE_METRICS.keys()),
            index=0,
            help="Similarity metric used by the HNSW index.",
        )

        hnsw_m = st.number_input(
            "HNSW M",
            min_value=4,
            max_value=128,
            value=32,
            step=1,
            help=(
                "Number of connections per node "
                "in the HNSW graph."
            ),
        )

        ef_construction = st.number_input(
            "efConstruction",
            min_value=4,
            max_value=512,
            value=40,
            step=1,
            help=(
                "Controls the quality of the HNSW "
                "graph during construction."
            ),
        )

        ef_search = st.number_input(
            "efSearch",
            min_value=4,
            max_value=512,
            value=40,
            step=1,
            help=(
                "Controls the number of candidates "
                "explored during search."
            ),
        )

    # ============================================================
    # IVF + PQ INDEX
    # ============================================================

    elif indexing_technique == "IVF + PQ":

        distance_metric = st.selectbox(
            "Distance Metric",
            options=list(DISTANCE_METRICS.keys()),
            index=0,
            help="Similarity metric used by IVF-PQ.",
        )

        nlist = st.number_input(
            "Number of Clusters (nlist)",
            min_value=1,
            max_value=4096,
            value=100,
            step=1,
            help=(
                "Number of Voronoi clusters used "
                "during IVF construction."
            ),
        )

        nprobe = st.number_input(
            "Search Clusters (nprobe)",
            min_value=1,
            max_value=4096,
            value=10,
            step=1,
            help=(
                "Number of IVF clusters searched "
                "during retrieval."
            ),
        )

        pq_m = st.number_input(
            "PQ M",
            min_value=1,
            max_value=32,
            value=8,
            step=1,
            help=(
                "Number of subquantizers used "
                "by Product Quantization."
            ),
        )

        nbits = st.number_input(
            "PQ nbits",
            min_value=1,
            max_value=8,
            value=8,
            step=1,
            help=(
                "Number of bits per subquantizer."
            ),
        )

    # ============================================================
    # RETRIEVAL CONFIGURATION
    # ============================================================

    st.subheader("Retrieval Configuration")

    top_k = st.number_input(
        "Top K",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
        help="Number of chunks to retrieve for each query.",
    )

    # ============================================================
    # INGESTION BUTTON
    # ============================================================

    ingest = st.button(
        "Start Ingestion",
        use_container_width=True,
    )

# ================================================================

# INGESTION

# ================================================================

if ingest:
    # ============================================================
    # WEB INGESTION
    # ============================================================

    if document_type == "Web":

        if not url:
            st.warning("Please enter a website URL.")
            st.stop()

        with st.spinner("Loading website..."):

            documents = load_document(url)

        st.success(
            f"Website loaded successfully. "
            f"Loaded {len(documents)} documents."
        )

    # ============================================================
    # FILE INGESTION
    # ============================================================

    else:

        if not uploaded_files:

            st.warning(
                "Please upload at least one document."
            )

            st.stop()

        # ========================================================
        # FILE VALIDATION
        # ========================================================

        allowed_extensions = DOCUMENT_TYPES[
            document_type
        ]

        invalid_files = []

        for file in uploaded_files:

            extension = (
                Path(file.name)
                .suffix
                .lower()
                .replace(".", "")
            )

            if extension not in allowed_extensions:

                invalid_files.append(
                    file.name
                )

        if invalid_files:

            st.error(
                "Uploaded file(s) do not match "
                "the selected document type."
            )

            for file in invalid_files:

                st.write(
                    f"• {file}"
                )

            st.stop()

        # ========================================================
        # DOCUMENT LOADING
        # ========================================================

        with st.spinner("Ingesting documents..."):

            documents = load_document(
                uploaded_files
            )

        # ========================================================
        # CHUNKING PARAMETERS
        # ========================================================

        kwargs = {}

        if chunking_strategy in [
            "Fixed",
            "Recursive Character",
            "Markdown",
            "Token",
            "Content Aware",
        ]:

            kwargs["chunk_size"] = chunk_size
            kwargs["overlap"] = overlap

        elif chunking_strategy == "Semantic":

            kwargs["breakpoint_percentile"] = (
                breakpoint_percentile
            )

            kwargs["buffer_size"] = (
                buffer_size
            )

        # ========================================================
        # CHUNK DOCUMENTS
        # ========================================================

        with st.spinner("Chunking documents..."):

            chunks = chunk_documents(
                documents=documents,
                strategy=chunking_strategy,
                **kwargs,
            )

        st.success(
            f"Documents processed successfully! "
            f"Generated {len(chunks)} chunks."
        )

        # ========================================================
        # CREATE EMBEDDINGS
        # ========================================================

        with st.spinner(
            f"Generating embeddings using "
            f"{embedding_model}..."
        ):

            # ----------------------------------------------------
            # TODO:
            # Call your Sentence Transformer embedding function
            # here.
            #
            # Example future structure:
            #
            # embedding_model_instance = create_embedding_model(
            #     model_name=embedding_model
            # )
            #
            # embeddings = generate_embeddings(
            #     texts=[
            #         chunk.page_content
            #         for chunk in chunks
            #     ],
            #     model=embedding_model_instance,
            # )
            # ----------------------------------------------------

            embeddings = None

        # ========================================================
        # CREATE FAISS INDEX
        # ========================================================

        with st.spinner(
            f"Creating FAISS {indexing_technique} index..."
        ):

            # ----------------------------------------------------
            # TODO:
            #
            # Call your FAISS index creation function here.
            #
            # The parameters passed should depend on the
            # selected indexing technique.
            # ----------------------------------------------------

            faiss_index = None

        # ========================================================
        # DISPLAY CONFIGURATION
        # ========================================================

        st.subheader("Vector Store Configuration")

        config_data = {
            "Embedding Model": embedding_model,
            "Indexing Technique": indexing_technique,
            "Distance Metric": distance_metric,
            "Top K": top_k,
        }

        if indexing_technique in [
            "IVF",
            "IVF + PQ",
        ]:

            config_data["nlist"] = nlist
            config_data["nprobe"] = nprobe

        if indexing_technique == "HNSW":

            config_data["HNSW M"] = hnsw_m
            config_data["efConstruction"] = (
                ef_construction
            )
            config_data["efSearch"] = ef_search

        if indexing_technique == "IVF + PQ":

            config_data["PQ M"] = pq_m
            config_data["PQ nbits"] = nbits

        st.json(config_data)

        # ========================================================
        # VIEW CHUNKS
        # ========================================================

        with st.expander("View Chunks"):

            for i, chunk in enumerate(
                chunks,
                start=1,
            ):

                st.markdown(
                    f"### Chunk {i}"
                )

                if hasattr(
                    chunk,
                    "page_content",
                ):

                    st.write(
                        chunk.page_content
                    )

                else:

                    st.write(chunk)

                st.divider()
