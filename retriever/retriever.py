from langchain_core.documents import Document
from retriever_utils import (
    create_dense_retriever,
    create_sparse_retriever,
    create_hybrid_retriever,
    remove_duplicate_documents,
    create_sentence_window_retriever,
)
    

def create_retriever(vector_store, config):
    """
    Create a retriever from the vector store using the
    retrieval configuration.

    Args:
        vector_store: Vector store containing document embeddings.
        config (dict): Retrieval configuration.

    Returns:
        Retriever: Configured LangChain retriever.
    """

    retriever_type = config.get("retriever_type")

    if retriever_type == "dense":
        return create_dense_retriever(
            vector_store=vector_store,
            config=config,
        )

    if retriever_type == "sparse":
        return create_sparse_retriever(
            vector_store=vector_store,
            config=config,
        )

    if retriever_type == "hybrid":
        return create_hybrid_retriever(
            vector_store=vector_store,
            config=config,
        )

    if retriever_type == "sentence_window":
        return create_sentence_window_retriever(
            vector_store=vector_store,
            config=config,
        )

    raise ValueError(
        f"Unsupported retriever type: {retriever_type}"
    )


def retrieve_documents(retriever, queries, config):
    """
    Retrieve documents for one or more queries.

    Args:
        retriever: Configured retriever.
        queries: A single query or list of queries.
        config (dict): Retrieval configuration.

    Returns:
        list: Retrieved documents.
    """

    if isinstance(queries, str):
        queries = [queries]

    documents = []

    for query in queries:

        retrieved_documents = retriever.invoke(query)

        documents.extend(retrieved_documents)

    return remove_duplicate_documents(documents)