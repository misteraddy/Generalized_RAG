from langchain_core.documents import Document


def fixed_chunking(
    document,
    config,
):
    """
    Split a LangChain Document into fixed-size character chunks.

    Args:
        document:
            LangChain Document object.

        config:
            Configuration parameters for the chunking strategy.

    Returns:
        list[Document]:
            List of chunked LangChain Document objects.
    """
    
    chunk_size = config.get("chunk_size")
    overlap = config.get("overlap")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    # Get actual text from LangChain Document
    text = document.page_content

    if not text:
        return []

    # Calculate step between chunks
    step = chunk_size - overlap

    chunks = []

    for start in range(0, len(text), step):

        chunk_text = text[
            start:start + chunk_size
        ]

        if not chunk_text.strip():
            continue

        # Preserve original metadata
        metadata = document.metadata.copy()

        # Add chunk information
        metadata["chunk_start"] = start
        metadata["chunk_end"] = start + len(chunk_text)

        chunks.append(
            Document(
                page_content=chunk_text,
                metadata=metadata,
            )
        )

    return chunks