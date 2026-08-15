from chunking.chunk_registry import CHUNKING_STRATEGIES


def chunk_documents(
    documents,
    strategy=None,
    config=None
):
    """
    Chunk the documents based on the selected strategy.

    Args:
        documents (list): List of documents to be chunked.
        strategy (str): Selected chunking strategy.
        config (dict): Configuration parameters for the chunking strategy.

    Returns:
        list: List of chunked documents.
    """

    chunks = []

    chunking_strategy = CHUNKING_STRATEGIES.get(strategy)

    if chunking_strategy is None:
        raise ValueError(
            f"No chunking strategy registered for '{strategy}'."
        )

    for document in documents:
        chunks.extend(
            chunking_strategy(
                document,
                config=config
            )
        )

    return chunks