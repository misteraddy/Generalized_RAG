from pathlib import Path

DOCUMENT_TYPES = {
    "PDF": ["pdf"],
    "Word": ["doc", "docx"],
    "Image": [".png", ".jpg", ".jpeg"],
    "PowerPoint": ["ppt", "pptx"],
    "Excel": ["xls", "xlsx"],
    "CSV": ["csv"],
    "Text": ["txt"],
    "Markdown": ["md"],
    "HTML": ["html"],
    "XML": ["xml"],
    "JSON": ["json"],
    "Web": []
}


OUTPUT_DIR = Path(
    r"D:\projects\RAG\data"
)

PAGE_IMAGE_DIR = OUTPUT_DIR / "page_images"
PICTURE_DIR = OUTPUT_DIR / "extracted_pictures"
TABLE_DIR = OUTPUT_DIR / "extracted_tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PAGE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
PICTURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


VECTOR_STORE_PATH = Path(r"D:\projects\RAG\db\vector_db")
