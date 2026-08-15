from langchain_classic.retrievers import (
    ContextualCompressionRetriever
)
from langchain_classic.retrievers.document_compressors import (
    LLMChainExtractor
)
from llm import llm

def create_compression_retriever(
    retriever,
    config,
):
    """
    Wrap a retriever with contextual compression.
    """

    if not config.get(
        "contextual_compression",
        False,
    ):
        return retriever

    llm_for_compression = llm.prepare_llm()

    compressor = LLMChainExtractor.from_llm(
        llm_for_compression
    )

    return ContextualCompressionRetriever(
        base_retriever=retriever,
        base_compressor=compressor,
    )
