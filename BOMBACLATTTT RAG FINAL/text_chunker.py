"""
Text Chunking Module

This module handles the chunking of extracted text using LangChain's text splitters.
It provides functions to split text into chunks with appropriate overlap.
"""

import os
import json
from typing import Dict, List, Optional
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken


def num_tokens_from_string(string: str, model_name: str = "text-embedding-3-small") -> int:
    """
    Calculate the number of tokens in a string for a specific model.

    Args:
        string: The text to calculate tokens for
        model_name: The name of the model to use for tokenization

    Returns:
        Number of tokens in the string
    """
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(string))


def chunk_text(
    pages_data: List[Dict],
    chunk_size: int = 512,
    chunk_overlap: int = 77,  # ~15% of 512
    output_path: Optional[str] = None
) -> List[Dict]:
    """
    Split text into chunks with metadata preserved.

    Args:
        pages_data: List of dictionaries containing text and metadata
        chunk_size: Target size of chunks in tokens
        chunk_overlap: Number of tokens to overlap between chunks
        output_path: Optional path to save the chunked text as JSON

    Returns:
        List of dictionaries containing chunked text with metadata
    """
    print(f"Chunking text with chunk size {chunk_size} tokens and {chunk_overlap} tokens overlap...")

    all_chunks = []

    for page_data in tqdm(pages_data, desc="Chunking pages"):
        text = page_data["text"]

        # Check if the page contains tables
        has_tables = page_data.get("has_tables", False) or '|' in text or '\t' in text

        # Use different chunking strategies based on content
        if has_tables:
            # For tables, use a more careful splitting approach
            # First, identify table sections
            lines = text.split('\n')
            table_sections = []
            current_section = []
            in_table = False

            for line in lines:
                is_table_line = '|' in line or '\t' in line

                # If we're transitioning between table and non-table
                if is_table_line != in_table:
                    if current_section:
                        section_text = '\n'.join(current_section)
                        table_sections.append((in_table, section_text))
                        current_section = []
                    in_table = is_table_line

                current_section.append(line)

            # Add the last section
            if current_section:
                section_text = '\n'.join(current_section)
                table_sections.append((in_table, section_text))

            # Process each section
            for is_table, section_text in table_sections:
                if is_table:
                    # For table sections, keep them intact if possible
                    # If too large, split at row boundaries
                    token_count = num_tokens_from_string(section_text)

                    if token_count <= chunk_size:
                        # Table fits in one chunk
                        chunks = [section_text]
                    else:
                        # Need to split the table
                        table_lines = section_text.split('\n')
                        chunks = []
                        current_chunk = []
                        current_tokens = 0

                        # Try to keep header row with data rows
                        header = table_lines[0] if table_lines else ""
                        header_tokens = num_tokens_from_string(header)

                        for line in table_lines:
                            line_tokens = num_tokens_from_string(line)

                            # If adding this line would exceed chunk size, start a new chunk
                            if current_tokens + line_tokens > chunk_size and current_chunk:
                                chunk_text = '\n'.join(current_chunk)
                                chunks.append(chunk_text)
                                # Start new chunk with header for context
                                current_chunk = [header, line] if header else [line]
                                current_tokens = header_tokens + line_tokens
                            else:
                                current_chunk.append(line)
                                current_tokens += line_tokens

                        # Add the last chunk
                        if current_chunk:
                            chunk_text = '\n'.join(current_chunk)
                            chunks.append(chunk_text)
                else:
                    # For non-table sections, use the standard text splitter
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size * 4,  # Approximate character count (1 token ≈ 4 chars)
                        chunk_overlap=chunk_overlap * 4,
                        length_function=lambda text: num_tokens_from_string(text),
                        separators=["\n\n", "\n", ". ", " ", ""]
                    )
                    chunks = text_splitter.split_text(section_text)

                # Create chunk data with metadata for this section
                for i, chunk in enumerate(chunks):
                    # Skip empty chunks
                    if not chunk.strip():
                        continue

                    chunk_data = {
                        "chunk_id": f"{page_data['page_num']}_{len(all_chunks)}",
                        "text": chunk,
                        "page_num": page_data["page_num"],
                        "source": page_data["source"],
                        "potential_title": page_data["potential_title"],
                        "is_table": is_table,
                        "total_pages": page_data["total_pages"],
                        "token_count": num_tokens_from_string(chunk)
                    }

                    all_chunks.append(chunk_data)
        else:
            # For regular text, use the standard text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size * 4,  # Approximate character count (1 token ≈ 4 chars)
                chunk_overlap=chunk_overlap * 4,
                length_function=lambda text: num_tokens_from_string(text),
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_text(text)

            # Create chunk data with metadata
            for i, chunk in enumerate(chunks):
                # Skip empty chunks
                if not chunk.strip():
                    continue

                chunk_data = {
                    "chunk_id": f"{page_data['page_num']}_{i}",
                    "text": chunk,
                    "page_num": page_data["page_num"],
                    "source": page_data["source"],
                    "potential_title": page_data["potential_title"],
                    "chunk_index": i,
                    "total_chunks_in_page": len(chunks),
                    "total_pages": page_data["total_pages"],
                    "token_count": num_tokens_from_string(chunk)
                }

                all_chunks.append(chunk_data)

    print(f"Created {len(all_chunks)} chunks from {len(pages_data)} pages")

    # Validate token counts
    token_counts = [chunk["token_count"] for chunk in all_chunks]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0

    print(f"Average tokens per chunk: {avg_tokens:.1f}")
    print(f"Maximum tokens in a chunk: {max_tokens}")

    # Save to JSON if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved chunked text to {output_path}")

    return all_chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk extracted text from PDF files")
    parser.add_argument("input_json", help="Path to the JSON file with extracted text")
    parser.add_argument("--output", "-o", help="Output JSON file path for chunks")
    parser.add_argument("--chunk-size", "-c", type=int, default=512,
                        help="Target chunk size in tokens (default: 512)")
    parser.add_argument("--overlap", type=int, default=77,
                        help="Chunk overlap in tokens (default: 77, ~15% of 512)")

    args = parser.parse_args()

    # Load the extracted text
    with open(args.input_json, 'r', encoding='utf-8') as f:
        pages_data = json.load(f)

    chunk_text(pages_data, args.chunk_size, args.overlap, args.output)
