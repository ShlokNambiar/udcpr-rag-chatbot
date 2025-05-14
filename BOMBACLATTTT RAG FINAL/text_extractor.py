"""
Text Extraction Module

This module handles the extraction of text from various file formats including:
- Text files (.txt)
- PDF files (via pdf_extractor)

It provides functions to extract text with appropriate metadata.
"""

import os
import json
from typing import Dict, List, Optional
from tqdm import tqdm
from pdf_extractor import extract_text_from_pdf


def extract_text_from_txt(txt_path: str, output_path: Optional[str] = None) -> List[Dict]:
    """
    Extract text from a text file with basic metadata.
    
    Args:
        txt_path: Path to the text file
        output_path: Optional path to save the extracted text as JSON
        
    Returns:
        List of dictionaries containing text and metadata
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Text file not found: {txt_path}")
    
    print(f"Extracting text from {txt_path}...")
    
    # Extract filename without extension for metadata
    filename = os.path.basename(txt_path)
    base_filename = os.path.splitext(filename)[0]
    
    # Read the text file
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split text into pages (for text files, we'll create artificial pages)
    # Using a reasonable page size of ~3000 characters
    page_size = 3000
    text_chunks = []
    
    # Check if the text contains table-like structures
    has_tables = '|' in text or '\t' in text
    
    # If it has tables, we'll be more careful with splitting
    if has_tables:
        # Split by lines first
        lines = text.split('\n')
        current_chunk = ""
        table_section = False
        
        for line in lines:
            # Check if this line is part of a table
            is_table_line = '|' in line or '\t' in line
            
            # If we're transitioning between table and non-table sections,
            # or the chunk is getting too large, start a new chunk
            if (is_table_line != table_section) or len(current_chunk) > page_size:
                if current_chunk:
                    text_chunks.append(current_chunk)
                current_chunk = line + '\n'
                table_section = is_table_line
            else:
                current_chunk += line + '\n'
        
        # Add the last chunk
        if current_chunk:
            text_chunks.append(current_chunk)
    else:
        # Simple splitting for non-table text
        for i in range(0, len(text), page_size):
            chunk = text[i:i + page_size]
            # Try to end at a paragraph or sentence boundary
            if i + page_size < len(text):
                # Look for paragraph breaks
                para_break = chunk.rfind('\n\n')
                if para_break != -1 and para_break > page_size * 0.5:
                    chunk = chunk[:para_break]
                else:
                    # Look for sentence breaks
                    sentence_break = max(chunk.rfind('. '), chunk.rfind('.\n'))
                    if sentence_break != -1 and sentence_break > page_size * 0.5:
                        chunk = chunk[:sentence_break + 1]
            
            text_chunks.append(chunk)
    
    # Create page data with metadata
    pages_data = []
    for i, chunk in enumerate(text_chunks):
        # Extract potential section/chapter titles (simple heuristic)
        lines = chunk.split('\n')
        potential_title = lines[0] if lines and len(lines[0]) < 100 else ""
        
        page_data = {
            "page_num": i + 1,  # 1-based page numbering
            "text": chunk,
            "source": base_filename,
            "potential_title": potential_title,
            "total_pages": len(text_chunks),
            "has_tables": has_tables
        }
        
        pages_data.append(page_data)
    
    print(f"Created {len(pages_data)} text chunks from the file")
    
    # Save to JSON if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pages_data, f, ensure_ascii=False, indent=2)
        print(f"Saved extracted text to {output_path}")
    
    return pages_data


def extract_text_from_file(file_path: str, output_path: Optional[str] = None) -> List[Dict]:
    """
    Extract text from a file based on its extension.
    
    Args:
        file_path: Path to the file
        output_path: Optional path to save the extracted text as JSON
        
    Returns:
        List of dictionaries containing text and metadata
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine file type by extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == '.pdf':
        return extract_text_from_pdf(file_path, output_path)
    elif ext == '.txt':
        return extract_text_from_txt(file_path, output_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract text from various file formats")
    parser.add_argument("file_path", help="Path to the file")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    
    args = parser.parse_args()
    
    extract_text_from_file(args.file_path, args.output)
