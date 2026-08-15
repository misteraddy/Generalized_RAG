from loaders.txt_loader import load_txt
from loaders.pdf_loader import load_pdf
from loaders.docx_loader import load_word
from loaders.csv_loader import load_csv
from loaders.html_loader import load_html
from loaders.webBase_loader import load_web


PARSER_REGISTRY = {
    ".txt": load_txt,
    ".pdf": load_pdf,
    ".docx": load_word,
    ".csv": load_csv,
    ".html": load_html,
     "web": load_web,
     ".jpg": load_pdf,
    ".jpeg": load_pdf,
    ".png": load_pdf
}