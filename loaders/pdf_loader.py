import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling_core.types.doc import ImageRefMode

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from utils import constants as const


def extract_text_with_ocr(pdf_path):
    """
    Perform OCR using Docling and return LangChain Documents.
    """

    pipeline_options = PdfPipelineOptions()

    pipeline_options.do_ocr = True

    pipeline_options.ocr_options = TesseractCliOcrOptions(
        tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        lang=["eng"],
        force_full_page_ocr=False,
    )

    pipeline_options.do_table_structure = True

    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True,
    )

    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    conversion_result = converter.convert(pdf_path)

    docling_document = conversion_result.document

    print("PDF parsed successfully.")

    # Save markdown
    markdown_output_path = const.OUTPUT_DIR / "rag_ready_document.md"

    docling_document.save_as_markdown(
        markdown_output_path,
        image_mode=ImageRefMode.REFERENCED,
    )

    # Save tables
    for table_index, table in enumerate(docling_document.tables, start=1):

        table_df = table.export_to_dataframe(doc=docling_document)

        markdown_path = (
            const.TABLE_DIR /
            f"table_{table_index:03d}.md"
        )

        table_df.to_markdown(
            markdown_path,
            index=False,
        )

    print("Tables saved:", const.TABLE_DIR)

    # Read markdown back into LangChain Document
    markdown_text = markdown_output_path.read_text(
        encoding="utf-8"
    )

    documents = [
        Document(
            page_content=markdown_text,
            metadata={
                "source": str(pdf_path),
                "parser": "docling",
            },
        )
    ]

    return documents


def load_pdf(uploaded_file):

    with NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    loader = None

    try:

        # ---------------------------------------------------------
        # Normal text extraction
        # ---------------------------------------------------------

        loader = PyMuPDFLoader(temp_path)

        documents = loader.load()

        extracted_text = "".join(
            document.page_content.strip()
            for document in documents
        )

        # ---------------------------------------------------------
        # OCR fallback for scanned PDFs
        # ---------------------------------------------------------

        if len(extracted_text) < 50:

            print("Scanned PDF detected. Running OCR...")

            # Close the normal loader before OCR
            # so that the temporary file is not locked.
            if hasattr(loader, "close"):
                loader.close()

            loader = None

            documents = extract_text_with_ocr(temp_path)

        # ---------------------------------------------------------
        # Update metadata
        # ---------------------------------------------------------

        for document in documents:

            document.metadata.update(
                {
                    "file_name": uploaded_file.name,
                    "file_type": "pdf",
                    "source": uploaded_file.name,
                }
            )

        return documents

    finally:

        # Close PyMuPDF loader
        if loader is not None:

            try:
                if hasattr(loader, "close"):
                    loader.close()
            except Exception:
                pass


        if os.path.exists(temp_path):

            try:
                os.remove(temp_path)

            except PermissionError:

                print(
                    f"Warning: Could not delete temporary file: "
                    f"{temp_path}"
                )