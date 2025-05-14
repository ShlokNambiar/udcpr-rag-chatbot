"""
PDF Text Extraction Module

This module handles the extraction of text from PDF documents using PyMuPDF (fitz).
It provides functions to extract text with page tracking and metadata.
"""

import os
import json
import fitz  # PyMuPDF
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


def extract_text_from_pdf(pdf_path: str, output_path: Optional[str] = None) -> List[Dict]:
    """
    Extract text from a PDF file with page numbers and basic metadata.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path to save the extracted text as JSON
        
    Returns:
        List of dictionaries containing text and metadata for each page
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"Extracting text from {pdf_path}...")
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # Extract filename without extension for metadata
    filename = os.path.basename(pdf_path)
    base_filename = os.path.splitext(filename)[0]
    
    pages_data = []
    
    for page_num in tqdm(range(total_pages), desc="Extracting pages"):
        page = doc[page_num]
        text = page.get_text()
        
        # Skip empty pages
        if not text.strip():
            continue
        
        # Extract potential section/chapter titles (simple heuristic)
        lines = text.split('\n')
        potential_title = lines[0] if lines and len(lines[0]) < 100 else ""
        
        # Create page data with metadata
        page_data = {
            "page_num": page_num + 1,  # 1-based page numbering
            "text": text,
            "source": base_filename,
            "potential_title": potential_title,
            "total_pages": total_pages
        }
        
        pages_data.append(page_data)
    
    print(f"Extracted {len(pages_data)} pages with content from {total_pages} total pages")
    
    # Save to JSON if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pages_data, f, ensure_ascii=False, indent=2)
        print(f"Saved extracted text to {output_path}")
    
    return pages_data


def extract_text_with_sections(pdf_path: str, output_path: Optional[str] = None) -> List[Dict]:
    """
    Extract text from PDF with attempt to identify sections and structure.
    This is a more advanced version that tries to identify document structure.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path to save the extracted text as JSON
        
    Returns:
        List of dictionaries containing text and metadata with section information
    """
    # Basic extraction first
    pages_data = extract_text_from_pdf(pdf_path)
    
    # TODO: Implement more sophisticated section detection
    # This would involve analyzing text formatting, headers, etc.
    
    # Save to JSON if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pages_data, f, ensure_ascii=False, indent=2)
        print(f"Saved extracted text with sections to {output_path}")
    
    return pages_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract text from PDF files")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--sections", "-s", action="store_true", 
                        help="Attempt to extract section information")
    
    args = parser.parse_args()
    
    if args.sections:
        extract_text_with_sections(args.pdf_path, args.output)
    else:
        extract_text_from_pdf(args.pdf_path, args.output)
