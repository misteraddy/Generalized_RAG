import numpy as np
import re
from chunking.content_aware_chunking import split_sentences
from langchain_core.documents import Document

def create_sentence_windows(sentences: list[str], buffer_size: int = 1) -> list[str]:
    """
    Create overlapping windows of sentences.

    Args:
        sentences (list[str]): A list of sentences.
        buffer_size (int): The number of sentences to overlap between windows.

    Returns:
        list[str]: A list of sentence windows.
    """
    windows = []

    for index in range(len(sentences)):
        start_index = max(0, index - buffer_size)
        end_index = min(len(sentences), index + buffer_size + 1)
        window = " ".join(sentences[start_index:end_index])
        windows.append(window)

    return windows

def create_local_embeddings(texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2",) -> tuple[np.ndarray, str]:
    """
    Prefer a local Sentence Transformers embedding model.

    If the model cannot be loaded, fall back to TF-IDF so that the notebook
    remains runnable. TF-IDF is lexical, not a true semantic replacement.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True)
        return embeddings, model_name
    except Exception as e:
        print("Sentence Transformers could not be loaded.")
        print("Using TF-IDF fallback for demonstration only.")
        print("Reason:", repr(e))
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(texts).toarray()

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        return embeddings, "TF-IDF fallback"


def semantic_chunking(
    document,
    config
) -> list[Document]:
    """
    Split a LangChain Document into semantic chunks.

    Args:
        document: LangChain Document.
        config (dict): Configuration parameters for the chunking strategy.

    Returns:
        List[Document]
    """

    text = document.page_content

    buffer_size = config.get("buffer_size", 1)
    breakpoint_percentile = config.get("breakpoint_percentile", 90)

    sentences = split_sentences(
        re.sub(r"\n+", " ", text)
    )

    if len(sentences) <= 1:
        return [document]

    # Build overlapping sentence windows
    windows = create_sentence_windows(
        sentences,
        buffer_size=buffer_size,
    )

    # Create embeddings
    embeddings, embedding_method = create_local_embeddings(
        windows
    )

    # Cosine similarity (embeddings are normalized)
    similarities = np.sum(
        embeddings[:-1] * embeddings[1:],
        axis=1,
    )

    # Convert similarity -> distance
    distances = 1 - similarities

    # Break threshold
    threshold = np.percentile(
        distances,
        breakpoint_percentile,
    )

    break_after_indices = {
        i
        for i, distance in enumerate(distances)
        if distance > threshold
    }

    chunks = []
    current_chunk = []

    for i, sentence in enumerate(sentences):
        current_chunk.append(sentence)

        if i in break_after_indices:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Convert chunks to LangChain Documents
    chunk_documents = []

    for chunk_id, chunk in enumerate(chunks):
        metadata = document.metadata.copy()

        metadata.update(
            {
                "chunk_id": chunk_id,
                "embedding_method": embedding_method,
            }
        )

        chunk_documents.append(
            Document(
                page_content=chunk,
                metadata=metadata,
            )
        )

    return chunk_documents