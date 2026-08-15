import uuid
from typing import List, Dict, Any

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from db.db_registry import (
    SUPPORTED_EMBEDDING_MODELS,
)
from utils import constants as const

# ============================================================
# Sentence Transformer -> LangChain Embeddings Adapter
# ============================================================

class SentenceTransformerEmbeddings(Embeddings):
    """
    Adapter that allows SentenceTransformer to work with
    LangChain VectorStores.
    """

    def __init__(
        self,
        model_name: str,
        normalize_embeddings: bool = False,
    ):
        self.model = SentenceTransformer(model_name)
        self.normalize_embeddings = normalize_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        return embeddings.astype(np.float32).tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        return embedding.astype(np.float32).tolist()


# ============================================================
# Embedding Model
# ============================================================


def create_embedding_model(
    model_name: str,
    normalize_embeddings: bool = False,
) -> SentenceTransformerEmbeddings:
    """
    Create a SentenceTransformer embedding model.

    Args:
        model_name: Supported SentenceTransformer model name.
        normalize_embeddings:
            Normalize vectors to unit length. Recommended when
            using cosine similarity through Inner Product.

    Returns:
        SentenceTransformerEmbeddings instance.
    """

    if model_name not in SUPPORTED_EMBEDDING_MODELS:
        raise ValueError(
            f"Unsupported embedding model: {model_name}. "
            f"Supported models: {list(SUPPORTED_EMBEDDING_MODELS.keys())}"
        )

    return SentenceTransformerEmbeddings(
        model_name=SUPPORTED_EMBEDDING_MODELS[model_name],
        normalize_embeddings=normalize_embeddings,
    )


# ============================================================
# Generate Embeddings
# ============================================================

def generate_embeddings(
    texts: List[str],
    embedding_model: SentenceTransformerEmbeddings,
) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Returns:
        NumPy array with shape:

        (number_of_texts, embedding_dimension)
    """

    if not texts:
        raise ValueError("No texts provided for embedding generation.")

    embeddings = embedding_model.model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=embedding_model.normalize_embeddings,
    )

    return embeddings.astype(np.float32)


# ============================================================
# Utility
# ============================================================

def get_embedding_dimension(embeddings: np.ndarray) -> int:
    """
    Get embedding dimension from an embedding matrix.
    """

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embedding matrix, got shape {embeddings.shape}"
        )

    return embeddings.shape[1]


def get_faiss_metric(distance_metric: str) -> int:
    """
    Convert user-friendly distance metric to FAISS metric.
    """

    if distance_metric == "L2":
        return faiss.METRIC_L2

    if distance_metric == "IP":
        return faiss.METRIC_INNER_PRODUCT

    raise ValueError(
        f"Unsupported distance metric: {distance_metric}. "
        "Use 'L2' or 'IP'."
    )


# ============================================================
# Flat Index
# ============================================================

def create_flat_index(
    embedding_dimension: int,
    distance_metric: str,
):
    """
    Create a FAISS Flat index.

    Flat indexes perform exact nearest-neighbor search.
    """

    metric = get_faiss_metric(distance_metric)

    if metric == faiss.METRIC_L2:
        return faiss.IndexFlatL2(embedding_dimension)

    return faiss.IndexFlatIP(embedding_dimension)


# ============================================================
# IVF Index
# ============================================================

def create_ivf_index(
    embedding_dimension: int,
    distance_metric: str,
    nlist: int,
    nprobe: int,
):
    """
    Create an IVF-Flat index.

    Note:
        IVF index must be trained before vectors are added.
    """

    if nlist <= 0:
        raise ValueError("nlist must be greater than 0.")

    if nprobe <= 0:
        raise ValueError("nprobe must be greater than 0.")

    metric = get_faiss_metric(distance_metric)

    if metric == faiss.METRIC_L2:
        quantizer = faiss.IndexFlatL2(embedding_dimension)
    else:
        quantizer = faiss.IndexFlatIP(embedding_dimension)

    index = faiss.IndexIVFFlat(
        quantizer,
        embedding_dimension,
        nlist,
        metric,
    )

    index.nprobe = nprobe

    return index


# ============================================================
# HNSW Index
# ============================================================

def create_hnsw_index(
    embedding_dimension: int,
    distance_metric: str,
    hnsw_m: int,
    ef_construction: int,
    ef_search: int,
):
    """
    Create an HNSW index.

    Args:
        hnsw_m:
            Number of connections per node.

        ef_construction:
            Search depth during graph construction.

        ef_search:
            Search depth during retrieval.
    """

    if hnsw_m <= 0:
        raise ValueError("hnsw_m must be greater than 0.")

    metric = get_faiss_metric(distance_metric)

    index = faiss.IndexHNSWFlat(
        embedding_dimension,
        hnsw_m,
        metric,
    )

    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search

    return index


# ============================================================
# IVF + PQ Index
# ============================================================

def create_ivf_pq_index(
    embedding_dimension: int,
    distance_metric: str,
    nlist: int,
    nprobe: int,
    pq_m: int,
    nbits: int,
):
    """
    Create an IVF + Product Quantization index.

    Note:
        embedding_dimension must be divisible by pq_m.
    """

    if nlist <= 0:
        raise ValueError("nlist must be greater than 0.")

    if nprobe <= 0:
        raise ValueError("nprobe must be greater than 0.")

    if pq_m <= 0:
        raise ValueError("pq_m must be greater than 0.")

    if embedding_dimension % pq_m != 0:
        raise ValueError(
            f"Embedding dimension ({embedding_dimension}) must be "
            f"divisible by pq_m ({pq_m})."
        )

    metric = get_faiss_metric(distance_metric)

    if metric == faiss.METRIC_L2:
        quantizer = faiss.IndexFlatL2(embedding_dimension)
    else:
        quantizer = faiss.IndexFlatIP(embedding_dimension)

    index = faiss.IndexIVFPQ(
        quantizer,
        embedding_dimension,
        nlist,
        pq_m,
        nbits,
        metric,
    )

    index.nprobe = nprobe

    return index


# ============================================================
# FAISS Index Factory
# ============================================================

def create_faiss_index(
    embedding_dimension: int,
    indexing_technique: str,
    distance_metric: str,
    config: Dict[str, Any],
):
    """
    Create the requested FAISS index.
    """

    if indexing_technique == "Flat":

        return create_flat_index(
            embedding_dimension=embedding_dimension,
            distance_metric=distance_metric,
        )

    elif indexing_technique == "IVF":

        return create_ivf_index(
            embedding_dimension=embedding_dimension,
            distance_metric=distance_metric,
            nlist=config["nlist"],
            nprobe=config["nprobe"],
        )

    elif indexing_technique == "HNSW":

        return create_hnsw_index(
            embedding_dimension=embedding_dimension,
            distance_metric=distance_metric,
            hnsw_m=config["hnsw_m"],
            ef_construction=config["ef_construction"],
            ef_search=config["ef_search"],
        )

    elif indexing_technique == "IVF + PQ":

        return create_ivf_pq_index(
            embedding_dimension=embedding_dimension,
            distance_metric=distance_metric,
            nlist=config["nlist"],
            nprobe=config["nprobe"],
            pq_m=config["pq_m"],
            nbits=config["nbits"],
        )

    else:
        raise ValueError(
            f"Unsupported indexing technique: {indexing_technique}"
        )


# ============================================================
# Create LangChain FAISS Vector Store
# ============================================================

def create_vector_store(
    chunks: List[Document],
    embeddings: np.ndarray,
    embedding_model: SentenceTransformerEmbeddings,
    indexing_technique: str,
    distance_metric: str,
    config: Dict[str, Any],
):
    """
    Create a LangChain FAISS vector store from chunks and embeddings.
    """

    embedding_dimension = get_embedding_dimension(embeddings)

    index = create_faiss_index(
        embedding_dimension=embedding_dimension,
        indexing_technique=indexing_technique,
        distance_metric=distance_metric,
        config=config,
    )

    # --------------------------------------------------------
    # IVF / IVF-PQ require training before adding vectors
    # --------------------------------------------------------

    if isinstance(
        index,
        (faiss.IndexIVFFlat, faiss.IndexIVFPQ),
    ):

        if not index.is_trained:
            index.train(embeddings)

    # --------------------------------------------------------
    # Add vectors
    # --------------------------------------------------------

    index.add(embeddings)

    # --------------------------------------------------------
    # Create LangChain document store
    # --------------------------------------------------------

    docstore = InMemoryDocstore()

    index_to_docstore_id = {}

    for position, chunk in enumerate(chunks):

        doc_id = str(uuid.uuid4())

        docstore.add({
            doc_id: chunk
        })

        index_to_docstore_id[position] = doc_id

    # --------------------------------------------------------
    # Create LangChain FAISS VectorStore
    # --------------------------------------------------------

    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    return vector_store


def save_vector_store_to_disk(vector_store: FAISS, file_path: str):
    """
    Save the FAISS vector store to disk.
    """

    vector_store.save_local(file_path)