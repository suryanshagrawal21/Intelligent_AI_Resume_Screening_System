import logging
from pdfminer.high_level import extract_text as extract_text_pdf
import docx
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using pdfminer.six.
    """
    try:
        text = extract_text_pdf(pdf_path)
        if not text.strip():
            logging.warning(f"No text extracted from {pdf_path}. It might be an image-based PDF.")
            return ""
        return text
    except Exception as e:
        logging.error(f"Error extracting PDF {pdf_path}: {e}")
        return ""

def extract_text_from_docx(docx_path):
    """
    Extracts text from a DOCX file using python-docx.
    """
    try:
        doc = docx.Document(docx_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        logging.error(f"Error extracting DOCX {docx_path}: {e}")
        return ""

def extract_text(file_path):
    """
    Generic function to extract text based on file extension.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    else:
        logging.error(f"Unsupported file format: {ext}")
        return ""
