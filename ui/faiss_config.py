import streamlit as st

from db.db_registry import (
    DISTANCE_METRICS,
    INDEXING_TECHNIQUES,
)


def render_faiss_config():

    st.subheader("FAISS Configuration")

    indexing_technique = st.selectbox(
        "Indexing Technique",
        options=list(INDEXING_TECHNIQUES.keys()),
    )

    db_config = {
        "indexing_technique": INDEXING_TECHNIQUES.get(indexing_technique),
        "distance_metric": None,
        "nlist": None,
        "nprobe": None,
        "hnsw_m": None,
        "ef_construction": None,
        "ef_search": None,
        "pq_m": None,
        "nbits": None,
    }

    distance_metric_value = st.selectbox(
        "Distance Metric",
        options=list(DISTANCE_METRICS.keys()),
    )

    db_config["distance_metric"] = DISTANCE_METRICS.get(distance_metric_value)

    if indexing_technique in ["IVF", "IVF + PQ"]:

        db_config["nlist"] = st.number_input(
            "Number of Clusters (nlist)",
            min_value=1,
            max_value=4096,
            value=100,
        )

        db_config["nprobe"] = st.number_input(
            "Search Clusters (nprobe)",
            min_value=1,
            max_value=4096,
            value=10,
        )

    if indexing_technique == "HNSW":

        db_config["hnsw_m"] = st.number_input(
            "HNSW M",
            min_value=4,
            max_value=128,
            value=32,
        )

        db_config["ef_construction"] = st.number_input(
            "efConstruction",
            min_value=4,
            max_value=512,
            value=40,
        )

        db_config["ef_search"] = st.number_input(
            "efSearch",
            min_value=4,
            max_value=512,
            value=40,
        )

    if indexing_technique == "IVF + PQ":

        db_config["pq_m"] = st.number_input(
            "PQ M",
            min_value=1,
            max_value=32,
            value=8,
        )

        db_config["nbits"] = st.number_input(
            "PQ nbits",
            min_value=1,
            max_value=8,
            value=8,
        )

    return db_config