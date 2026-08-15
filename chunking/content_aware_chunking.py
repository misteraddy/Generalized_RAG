import re

def split_sentences(text: str) -> list[str]:
    """Split the input text into sentences using a simple regex-based approach.

    Args:
        text (str): The input text to be split.

    Returns:
        list[str]: A list of sentences.
    """
    return [
        sentence.strip()
        for sentence in re.split(r'(?<=[.!?])\s+', text.strip())
        if sentence.strip()
    ]

def content_aware_chunking(document, config) -> list[str]:
    """
    Split the input text into chunks based on content-aware strategies.

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

    max_characters = chunk_size

    text = document.page_content

    paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\\n\\n")
            if paragraph.strip()
        ]

    logical_units: list[str] = []

    for paragraph in paragraphs:
        if len(paragraph) <= max_characters:
            logical_units.append(paragraph)
        else:
            logical_units.extend(split_sentences(paragraph))

    chunks: list[str] = []
    current_units: list[str] = []
    current_length = 0

    for unit in logical_units:
        separator_length = 2 if current_units else 0
        proposed_length = current_length + len(unit) + separator_length

        if current_units and proposed_length > max_characters:
            chunks.append("\\n\\n".join(current_units))
            current_units = [unit]
            current_length = len(unit)
        else:
            current_units.append(unit)
            current_length = proposed_length

    if current_units:
        chunks.append("\\n\\n".join(current_units))

    return chunks