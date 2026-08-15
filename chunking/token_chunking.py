from langchain_text_splitters import TokenTextSplitter

def token_chunking(text: str, config) -> list[str]:
    """
    Split the input text into chunks based on token count.

    Args:
        text (str): The input text to be chunked.
        config (dict): Configuration parameters for the chunking strategy.

    Returns:
        list[str]: A list of text chunks.
    """
    chunk_size = config.get("chunk_size")
    overlap = config.get("overlap")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must satisfy 0 <= overlap < chunk_size."
        )

    token_splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    return token_splitter.split_text(text)