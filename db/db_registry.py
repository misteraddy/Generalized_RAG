from sentence_transformers import SentenceTransformer


INDEXING_TECHNIQUES = {
    "Flat": "Flat",
    "IVF": "IVF",
    "HNSW": "HNSW",
    "IVF + PQ": "IVF + PQ",
}

DISTANCE_METRICS = {
    "Cosine Similarity": "Cosine",
    "Euclidean Distance": "Euclidean",
    "Inner Product": "IP",
}


SUPPORTED_EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "all-mpnet-base-v2",
    "paraphrase-MiniLM-L3-v2": "paraphrase-MiniLM-L3-v2",
}