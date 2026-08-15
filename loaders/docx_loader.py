import os
from tempfile import NamedTemporaryFile

from langchain_community.document_loaders import Docx2txtLoader


def load_word(uploaded_file):

    with NamedTemporaryFile(
        delete=False,
        suffix=".docx"
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:

        loader = Docx2txtLoader(temp_path)

        documents = loader.load()

        for document in documents:

            document.metadata.update(
                {
                    "file_name": uploaded_file.name,
                    "file_type": "docx",
                    "source": uploaded_file.name,
                }
            )

        return documents

    finally:

        os.remove(temp_path)