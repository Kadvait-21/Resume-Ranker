import io
from PyPDF2 import PdfReader

def extract_text_from_filelike(file_like):
    file_like.seek(0)
    reader = PdfReader(file_like)
    text_chunks = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            text_chunks.append(txt)
    return "\n".join(text_chunks).strip()
