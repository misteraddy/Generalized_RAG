import re
from langchain_core.documents import Document


def split_into_sentences(text:str) -> list[str]:

    text = text.strip()

    if not text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+',text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

def sentence_window_chunking(document,config):

    chunks = []
    
    window_size = config.get('window_size')

    text = document.page_content

    sentences = split_into_sentences(
        text
    )

    for sentence_index, sentence in enumerate(sentences):

        start_index = max(
            0,
            sentence_index - window_size
        )

        end_index = min(
            len(sentences),
            sentence_index + window_size + 1
        )

        window_sentences = sentences[
            start_index:end_index
        ]

        sentence_window = " ".join(
            window_sentences
        )

        metadata = dict(
            document.metadata,
        )

        metadata.update(
            {
                "sentence_index": sentence_index,
                "window_start": start_index,
                "window_end": end_index - 1,
                "sentence_window": sentence_window,
                "original_sentence": sentence,
                "window_size": window_size,
            }
        )

        sentence_document = Document(
            page_content=sentence,
            metadata=metadata
        )

        chunks.append(
            sentence_document
        )

    




    

