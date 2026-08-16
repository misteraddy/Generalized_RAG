from chunking.content_aware_chunking import content_aware_chunking
from chunking.fixed_size_chunking import fixed_chunking
from chunking.recursive_chunking import recursive_chunking
from chunking.token_chunking import token_chunking
from chunking.markdown_chunking import markdown_chunking
from chunking.semantic_chunking import semantic_chunking
from chunking.sentence_window_chunking import sentence_window_chunking

CHUNKING_STRATEGIES = {
    "Fixed": fixed_chunking,
    "Recursive Character": recursive_chunking,
    "Token": token_chunking,
    "Content Aware": content_aware_chunking,
    "Markdown Header": markdown_chunking,
    "Semantic": semantic_chunking,
    "Sentence Window": sentence_window_chunking,
}

CHUNK_SIZE_OPTIONS = [
    200,
                300,
                500,
                700,
                1000,
                1500,
                2000,
                3000,
                5000,
]

CHUNK_OVERLAP_OPTIONS = [
                0,
                25,
                50,
                75,
                100,
                150,
                200,
                250,
                300,
                400,
                500,
            ]
