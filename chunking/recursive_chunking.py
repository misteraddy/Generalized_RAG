from langchain_text_splitters import RecursiveCharacterTextSplitter

def recursive_chunking(document, config) -> list[str]:
    """
    Split the input text into chunks using a recursive character-based approach.

    Args:
        document: The input document to be chunked.
        config (dict): Configuration parameters for the chunking strategy.

    Returns:
        list[str]: A list of text chunks.
    """
    text = document.page_content

    chunk_size = config.get("chunk_size")
    overlap = config.get("overlap")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must satisfy 0 <= overlap < chunk_size."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )

    return text_splitter.split_text(text)