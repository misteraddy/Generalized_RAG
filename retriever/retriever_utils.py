from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


def create_dense_retriever(vector_store, config):
    """
    Create a dense retriever from the vector store.

    Args:
        vector_store: FAISS vector store.
        config (dict): Retrieval configuration.

    Returns:
        Retriever: Configured dense retriever.
    """

    search_type = config.get(
        "search_type",
        "similarity",
    )

    search_kwargs = {
        "k": config.get("k", 4),
    }

    if search_type == "mmr":
        search_kwargs["fetch_k"] = config.get(
            "fetch_k",
            20,
        )

    elif search_type == "similarity_score_threshold":
        search_kwargs["score_threshold"] = config.get(
            "score_threshold",
            0.5,
        )

    if config.get("metadata_filtering", False):

        metadata_filter = config.get(
            "metadata_filter"
        )

        if metadata_filter:
            search_kwargs["filter"] = metadata_filter

    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def create_sparse_retriever(documents, config):
    """
    Create a BM25 sparse retriever.

    Args:
        documents (list): Original chunked Documents.
        config (dict): Retrieval configuration.

    Returns:
        BM25Retriever: Configured sparse retriever.
    """

    sparse_retriever = BM25Retriever.from_documents(
        documents
    )

    sparse_retriever.k = config.get(
        "k",
        4,
    )

    return sparse_retriever


def create_hybrid_retriever(
    vector_store,
    documents,
    config,
):
    """
    Create a hybrid retriever using dense and sparse
    retrieval.

    Supports weighted fusion and reciprocal rank
    fusion.

    Args:
        vector_store: FAISS vector store.
        documents (list): Original chunked Documents.
        config (dict): Retrieval configuration.

    Returns:
        EnsembleRetriever: Configured hybrid retriever.
    """

    sparse_retriever = create_sparse_retriever(
        documents=documents,
        config=config,
    )

    dense_retriever = create_dense_retriever(
        vector_store=vector_store,
        config=config,
    )

    fusion_type = config.get(
        "fusion_type",
        "weighted",
    )

    if fusion_type == "weighted":

        sparse_weight = config.get(
            "sparse_weight",
            0.4,
        )

        dense_weight = config.get(
            "dense_weight",
            0.6,
        )

    elif fusion_type == "rrf":

        sparse_weight = 1.0
        dense_weight = 1.0

    else:

        raise ValueError(
            f"Unsupported fusion type: {fusion_type}"
        )

    return EnsembleRetriever(
        retrievers=[
            sparse_retriever,
            dense_retriever,
        ],
        weights=[
            sparse_weight,
            dense_weight,
        ],
    )


def remove_duplicate_documents(documents):
    """
    Remove duplicate documents from retrieval results.

    Args:
        documents (list): Retrieved documents.

    Returns:
        list: Unique documents.
    """

    unique_documents = []
    seen = set()

    for document in documents:

        document_id = get_document_id(document)

        if document_id not in seen:
            seen.add(document_id)
            unique_documents.append(document)

    return unique_documents


def get_document_id(document):
    """
    Generate a unique identifier for a document.

    Uses metadata when available and falls back
    to document content.
    """

    metadata = document.metadata or {}

    source = metadata.get("source")
    page = metadata.get("page")
    chunk_id = metadata.get("chunk_id")

    if source is not None:
        return (
            source,
            page,
            chunk_id,
        )

    return document.page_content


