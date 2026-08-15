from langchain_text_splitters import MarkdownHeaderTextSplitter

def markdown_chunking(document, config) -> list[str]:
    """
    Split the input text into chunks based on Markdown headers.

    Args:
        document: The input document to be chunked.
        config (dict): Configuration parameters for the chunking strategy.

    Returns:
        list[str]: A list of text chunks.
    """

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#","Header 1"), ("##","Header 2"), ("###","Header 3")],
        strip_headers=True,
    )

    text = document.page_content

    return markdown_splitter.split_text(text)