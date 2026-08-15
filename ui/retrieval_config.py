import streamlit as st

from retriever.retriever_registry import (
    RETRIEVER_TYPES,
    QUERY_TRANSFORMATIONS,
    SEARCH_TYPES,
)

def render_retrieval_config():

    st.subheader("Retrieval Configuration")

    retriever_name = st.selectbox(
        "Retriever Type",
        options=list(RETRIEVER_TYPES.keys()),
    )

    retriever_type = RETRIEVER_TYPES.get(retriever_name)

    retrieval_config = {
        "retriever_type": retriever_type,
        "search_type": "similarity",
        "k": 4,
        "fetch_k": None,
        "score_threshold": None,
        "query_transformation": None,
        "reranking": False,
        "metadata_filtering": False,
        "metadata_filter": None,
        "contextual_compression": False,
    }

    if retriever_type == "dense":

        search_type_name = st.selectbox(
            "Search Type",
            options=list(SEARCH_TYPES.keys()),
        )

        search_type = SEARCH_TYPES.get(search_type_name)

        retrieval_config["search_type"] = search_type

        retrieval_config["k"] = st.number_input(
            "Number of Documents (k)",
            min_value=1,
            max_value=100,
            value=4,
        )

        if search_type == "mmr":

            retrieval_config["fetch_k"] = st.number_input(
                "Fetch K",
                min_value=1,
                max_value=100,
                value=20,
            )

        elif search_type == "similarity_score_threshold":

            retrieval_config["score_threshold"] = st.slider(
                "Score Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
            )

    query_transformation = st.selectbox(
        "Query Transformation",
        options=["No", "Yes"],
    )

    if query_transformation == "Yes":

        transformation_name = st.selectbox(
            "Transformation Strategy",
            options=list(QUERY_TRANSFORMATIONS.keys()),
        )

        retrieval_config["query_transformation"] = (
            QUERY_TRANSFORMATIONS.get(transformation_name)
        )

    reranking = st.selectbox(
        "Reranking",
        options=["No", "Yes"],
    )

    retrieval_config["reranking"] = reranking == "Yes"

    metadata_filtering = st.selectbox(
        "Metadata Filtering",
        options=["No", "Yes"],
    )

    retrieval_config["metadata_filtering"] = metadata_filtering == "Yes"

    if metadata_filtering == "Yes":

        metadata_key = st.text_input(
            "Metadata Field"
        )

        metadata_value = st.text_input(
            "Metadata Value"
        )

        if metadata_key and metadata_value:

            retrieval_config["metadata_filter"] = {
                metadata_key: metadata_value
            }

    contextual_compression = st.selectbox(
        "Contextual Compression",
        options=["No", "Yes"],
    )

    retrieval_config["contextual_compression"] = (
        contextual_compression == "Yes"
    )

    return retrieval_config