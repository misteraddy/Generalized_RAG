from tempfile import NamedTemporaryFile
import os

from langchain_community.document_loaders import CSVLoader


def load_csv(uploaded_file):

    with NamedTemporaryFile(
        delete=False,
        suffix=".csv"
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:

        loader = CSVLoader(
            file_path=temp_path,
            encoding="utf-8",
        )

        documents = loader.load()

        for index, document in enumerate(documents, start=1):

            document.metadata.update(
                {
                    "file_name": uploaded_file.name,
                    "file_type": "csv",
                    "row": index,
                    "source": uploaded_file.name,
                }
            )

        return documents

    finally:

        os.remove(temp_path)