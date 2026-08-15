import streamlit as st

from chunking.chunk_registry import (
    CHUNK_OVERLAP_OPTIONS,
    CHUNK_SIZE_OPTIONS,
    CHUNKING_STRATEGIES,
)


CHARACTER_BASED_STRATEGIES = [
    "Fixed",
    "Markdown",
    "Recursive Character",
    "Token",
    "Content Aware",
]


def render_chunking_config():

    chunking_strategy = st.selectbox(
        "Chunking Strategy",
        options=list(CHUNKING_STRATEGIES.keys()),
    )

    chunk_size = None
    overlap = None
    breakpoint_percentile = None
    buffer_size = None

    if chunking_strategy in CHARACTER_BASED_STRATEGIES:

        chunk_size = st.selectbox(
            "Chunk Size",
            options=CHUNK_SIZE_OPTIONS,
            index=2,
        )

        overlap_options = [
            value
            for value in CHUNK_OVERLAP_OPTIONS
            if value < chunk_size
        ]

        if not overlap_options:
            overlap_options = [0]

        default_overlap = (
            100
            if 100 in overlap_options
            else overlap_options[0]
        )

        overlap = st.selectbox(
            "Chunk Overlap",
            options=overlap_options,
            index=overlap_options.index(default_overlap),
        )

    elif chunking_strategy == "Semantic":

        breakpoint_percentile = st.slider(
            "Breakpoint Percentile",
            min_value=50,
            max_value=100,
            value=95,
        )

        buffer_size = st.number_input(
            "Buffer Size",
            min_value=0,
            max_value=5,
            value=1,
            step=1,
        )

    return {
        "chunking_strategy": chunking_strategy,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "breakpoint_percentile": breakpoint_percentile,
        "buffer_size": buffer_size,
    }