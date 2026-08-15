import os
from tempfile import NamedTemporaryFile

from langchain_community.document_loaders import JSONLoader


def load_json(uploaded_file):

    with NamedTemporaryFile(
        delete=False,
        suffix=".json"
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:

        loader = JSONLoader(
            file_path=temp_path,
            jq_schema=".",
            text_content=False,
        )

        documents = loader.load()

        for document in documents:

            document.metadata.update(
                {
                    "file_name": uploaded_file.name,
                    "file_type": "json",
                    "source": uploaded_file.name,
                }
            )

        return documents

    finally:

        os.remove(temp_path)