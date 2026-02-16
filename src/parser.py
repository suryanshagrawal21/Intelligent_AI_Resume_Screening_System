import logging
import os

from pdfminer.high_level import extract_text as extract_text_pdf
import docx

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path):
    """Reads all text from a PDF using pdfminer."""
    try:
        text = extract_text_pdf(pdf_path)
        if not text.strip():
            logger.warning("No text found in %s — might be an image-based PDF.", pdf_path)
            return ""
        return text
    except Exception as e:
        logger.error("Failed to extract PDF (%s): %s", pdf_path, e)
        return ""


def extract_text_from_docx(docx_path):
    """Reads all text from a DOCX using python-docx."""
    try:
        doc = docx.Document(docx_path)
        full_text = "\n".join(para.text for para in doc.paragraphs)
        return full_text
    except Exception as e:
        logger.error("Failed to extract DOCX (%s): %s", docx_path, e)
        return ""


def extract_text(file_path):
    """Routes to the right extractor based on file extension."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        logger.error("Unsupported file format: %s", ext)
        return ""
