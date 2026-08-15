from langchain_classic.chains.hyde.base import (
    HypotheticalDocumentEmbedder,
)
from langchain_classic.retrievers.multi_query import (
    MultiQueryRetriever,
)


def rewrite_query(query, llm):
    """
    Rewrite the user query into a clearer retrieval query.

    Args:
        query (str): Original user query.
        llm: Configured LLM.

    Returns:
        str: Rewritten query.
    """

    prompt = (
        "Rewrite the following user query so that it is "
        "clear, precise, and suitable for semantic retrieval.\n\n"
        f"Query: {query}"
    )

    response = llm.invoke(prompt)

    return response.content.strip()


def expand_query(query, llm):
    """
    Expand the query with additional relevant terms.

    Args:
        query (str): Original query.
        llm: Configured LLM.

    Returns:
        str: Expanded query.
    """

    prompt = (
        "Expand the following query with relevant terms and "
        "concepts that may improve document retrieval.\n\n"
        f"Query: {query}"
    )

    response = llm.invoke(prompt)

    return response.content.strip()


def decompose_query(query, llm):
    """
    Decompose a complex query into smaller queries.

    Args:
        query (str): Original query.
        llm: Configured LLM.

    Returns:
        list: List of sub-queries.
    """

    prompt = (
        "Break the following complex question into smaller, "
        "independent questions that can be answered from "
        "a document collection.\n\n"
        f"Question: {query}\n\n"
        "Return each question on a separate line."
    )

    response = llm.invoke(prompt)

    queries = [
        line.strip()
        for line in response.content.splitlines()
        if line.strip()
    ]

    return queries


def hyde_transformation(
    query,
    vector_store,
    llm,
    config,
):
    """
    Perform Hypothetical Document Embeddings (HyDE).

    The LLM generates a hypothetical answer/document.
    That hypothetical document is then embedded and used
    for vector retrieval.

    Args:
        query (str): Original user query.
        vector_store: Vector store.
        llm: Configured LLM.
        config (dict): Retrieval configuration.

    Returns:
        list: Retrieved documents.
    """

    prompt = (
        "Write a hypothetical document that would contain "
        "the information needed to answer the following "
        "question. Do not explain your reasoning.\n\n"
        f"Question: {query}"
    )

    response = llm.invoke(prompt)

    hypothetical_document = response.content.strip()

    hyde_vector = vector_store.embedding_function.embed_query(
        hypothetical_document
    )

    k = config.get("k", 4)

    documents = vector_store.similarity_search_by_vector(
        embedding=hyde_vector,
        k=k,
    )

    return documents


def multi_query_transformation(
    query,
    retriever,
    llm,
):
    """
    Generate multiple query variations and retrieve
    documents for each query.

    Args:
        query (str): Original user query.
        retriever: Configured LangChain retriever.
        llm: Configured LLM.

    Returns:
        list: Retrieved documents.
    """

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
        include_original=True,
    )

    return multi_query_retriever.invoke(query)


def transform_query(
    query,
    config,
    llm,
    retriever=None,
    vector_store=None,
):
    """
    Apply the configured query transformation.

    Args:
        query (str): Original user query.
        config (dict): Retrieval configuration.
        llm: Configured LLM.
        retriever: Configured retriever.
        vector_store: Vector store.

    Returns:
        str | list: Transformed query or retrieved documents.
    """

    transformation = config.get(
        "query_transformation"
    )

    if not transformation:
        return query

    if transformation == "query_rewriting":
        return rewrite_query(
            query=query,
            llm=llm,
        )

    if transformation == "query_expansion":
        return expand_query(
            query=query,
            llm=llm,
        )

    if transformation == "query_decomposition":
        return decompose_query(
            query=query,
            llm=llm,
        )

    if transformation == "hyde":
        if vector_store is None:
            raise ValueError(
                "Vector store is required for HYDE."
            )

        return hyde_transformation(
            query=query,
            vector_store=vector_store,
            llm=llm,
            config=config,
        )

    if transformation == "multi_query":
        if retriever is None:
            raise ValueError(
                "Retriever is required for Multi Query."
            )

        return multi_query_transformation(
            query=query,
            retriever=retriever,
            llm=llm,
        )

    raise ValueError(
        f"Unsupported query transformation: {transformation}"
    )