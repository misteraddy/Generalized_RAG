from pathlib import Path
from loaders.loader_registry import PARSER_REGISTRY


def load_document(uploaded_files):
    """
    Routes uploaded files to their respective parser based on file extension.
    """

    documents = []

    for uploaded_file in uploaded_files:

        extension = Path(uploaded_file.name).suffix.lower()

        parser = PARSER_REGISTRY.get(extension)

        if parser is None:
            raise ValueError(
                f"No parser registered for '{extension}' files."
            )

        documents.extend(parser(uploaded_file))

    return documents