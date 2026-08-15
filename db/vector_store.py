from db.db_utils import create_vector_store, save_vector_store_to_disk, generate_embeddings, create_embedding_model, save_vector_store_to_disk
from langchain_core.documents import Document
from typing import List, Dict, Any
from utils import constants as const

def save_chunks_to_vector_store(
    chunks: List[Document],
    config: Dict[str, Any],
):
    """
    Generate embeddings for chunks and create a FAISS vector store.

    Expected config:

        {
            "embedding_model": "all-MiniLM-L6-v2",
            "indexing_technique": "Flat",
            "distance_metric": "L2",

            # IVF
            "nlist": 100,
            "nprobe": 10,

            # HNSW
            "hnsw_m": 32,
            "ef_construction": 200,
            "ef_search": 50,

            # IVF + PQ
            "pq_m": 8,
            "nbits": 8,
        }
    """

    if not chunks:
        raise ValueError("No chunks provided.")

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    embedding_model_name = config.get("embedding_model")
    indexing_technique = config.get("indexing_technique")
    distance_metric = config.get("distance_metric")

    # --------------------------------------------------------
    # Create embedding model
    # --------------------------------------------------------

    normalize_embeddings = distance_metric == "Inner Product"

    embedding_model = create_embedding_model(
        model_name=embedding_model_name,
        normalize_embeddings=normalize_embeddings,
    )

    # --------------------------------------------------------
    # Extract chunk text
    # --------------------------------------------------------

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embeddings = generate_embeddings(
        texts=texts,
        embedding_model=embedding_model,
    )

    # --------------------------------------------------------
    # Create vector store
    # --------------------------------------------------------

    vector_store = create_vector_store(
        chunks=chunks,
        embeddings=embeddings,
        embedding_model=embedding_model,
        indexing_technique=indexing_technique,
        distance_metric=distance_metric,
        config=config,
    )

    save_vector_store_to_disk(
        vector_store=vector_store,
        file_path=const.VECTOR_STORE_PATH
    )

    return vector_store