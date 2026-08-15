from db.vector_store import load_vector_store

from retriever.retriever import (
    create_retriever,
    retrieve_documents,
)

from retriever.query_transformer import transform_query
from retriever.reranker import rerank_documents
from retriever.compressor import compress_documents

from llm import llm


def run_retrieval(query, config):

    vector_store = load_vector_store(config)

    retriever = create_retriever(
        vector_store=vector_store,
        config=config,
    )

    transformed_queries = query

    if config.get("query_transformation"):

        llm_for_qt = llm.prepare_llm()

        transformed_queries = transform_query(
            query=query,
            config=config,
            llm=llm_for_qt,
            retriever=retriever,
            vector_store=vector_store,
        )

    documents = retrieve_documents(
        retriever=retriever,
        queries=transformed_queries,
    )

    if config.get("reranking", False):

        documents = rerank_documents(
            query=query,
            documents=documents,
            config=config,
        )

    if config.get("contextual_compression", False):

        documents = compress_documents(
            query=query,
            documents=documents,
            config=config,
        )

    return documents