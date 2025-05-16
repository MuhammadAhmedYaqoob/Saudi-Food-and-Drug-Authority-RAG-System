import hashlib
import re
from pathlib import Path
from typing import List, Dict
import pdfplumber
import os
from .config import RAGS_DIR, SUPPORTED_LANGUAGES

def is_arabic(text):
    """Check if text contains Arabic characters."""
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    return bool(arabic_pattern.search(text))

def detect_language(text):
    """Detect if text is primarily Arabic or English."""
    if is_arabic(text):
        return "ar"
    return "en"

def clean_text(text, language):
    """Clean and normalize text based on language."""
    if not text:
        return ""
        
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    if language == "ar":
        # Normalize Arabic characters
        text = re.sub(r'[ًٌٍَُِّْ]', '', text)  # Remove diacritics
        
    return text

def load_multiple_pdfs() -> List[Dict]:
    """
    Extract content from all PDF files in the RAGS_DIR directory.
    Returns a list of message dictionaries with source file tracking.
    """
    print(f"[DEBUG] Loading PDFs from directory: {RAGS_DIR}")
    
    if not RAGS_DIR.exists():
        print(f"[ERROR] Directory not found: {RAGS_DIR}")
        return []
    
    messages = []
    pdf_files = list(RAGS_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"[ERROR] No PDF files found in {RAGS_DIR}")
        return []
        
    print(f"[INFO] Found {len(pdf_files)} PDF files")
    
    for pdf_path in pdf_files:
        file_name = pdf_path.name
        print(f"[INFO] Processing {file_name}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Process each page
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    # Detect primary language
                    lang = detect_language(text)
                    
                    # Split into paragraphs or sections
                    paragraphs = re.split(r'\n\s*\n', text)
                    
                    for para_num, paragraph in enumerate(paragraphs):
                        paragraph = paragraph.strip()
                        if not paragraph:
                            continue
                        
                        # Clean text based on language
                        cleaned_paragraph = clean_text(paragraph, lang)
                        
                        # Create a unique ID for this content chunk
                        content_id = hashlib.md5(
                            f"{file_name}_page_{page_num}_para_{para_num}:{cleaned_paragraph}".encode()
                        ).hexdigest()
                        
                        # Determine if it's a table by checking for multiple aligned rows
                        is_table = bool(re.search(r'\n\s+\w+.*\n\s+\w+.*', paragraph))
                        
                        # Extract section header if available
                        header_match = re.match(r'^([\w\s\u0600-\u06FF]+)[:|-]', paragraph)
                        section = header_match.group(1).strip() if header_match else "General"
                        
                        messages.append({
                            "id": content_id,
                            "source_file": file_name,
                            "page": page_num + 1,
                            "sender": "SFDA_Doc",  # Source attribution
                            "section": section,
                            "is_table": is_table,
                            "language": lang,
                            "text": cleaned_paragraph
                        })
                    
                    # Extract tables as separate entities
                    tables = page.extract_tables()
                    for table_num, table in enumerate(tables):
                        if table:
                            # Convert table to text representation
                            table_text = "\n".join([" | ".join([cell or "" for cell in row]) for row in table])
                            
                            # Skip empty tables
                            if not table_text.strip():
                                continue
                            
                            # Detect language for table
                            table_lang = detect_language(table_text)
                            
                            # Clean text
                            cleaned_table = clean_text(table_text, table_lang)
                            
                            table_id = hashlib.md5(
                                f"{file_name}_page_{page_num}_table_{table_num}:{cleaned_table}".encode()
                            ).hexdigest()
                            
                            messages.append({
                                "id": table_id,
                                "source_file": file_name,
                                "page": page_num + 1,
                                "sender": "SFDA_Table",
                                "section": "Table Data",
                                "is_table": True,
                                "language": table_lang,
                                "text": cleaned_table
                            })
        
        except Exception as e:
            print(f"[ERROR] Failed to process PDF {file_name}: {e}")
    
    print(f"[INFO] Successfully extracted {len(messages)} content chunks from all PDFs")
    
    # Log language distribution
    ar_chunks = sum(1 for m in messages if m.get("language") == "ar")
    en_chunks = sum(1 for m in messages if m.get("language") == "en")
    print(f"[INFO] Language distribution: Arabic: {ar_chunks}, English: {en_chunks}")
    
    return messages