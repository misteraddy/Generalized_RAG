from pathlib import Path

from langchain_core.documents import Document


def load_txt(uploaded_file):

    text = uploaded_file.read().decode("utf-8")

    return [
        Document(
            page_content=text,
            metadata={
                "file_name": uploaded_file.name,
                "file_type": "txt",
                "source": uploaded_file.name,
            },
        )
    ]