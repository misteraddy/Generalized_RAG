from langchain_community.document_loaders import WebBaseLoader


def load_web(url: str):

    loader = WebBaseLoader(
        web_path=url,
        requests_per_second=2,
        raise_for_status=True,
    )

    documents = loader.load()

    for document in documents:

        document.metadata.update(
            {
                "file_type": "web",
                "source": url,
            }
        )

    return documents