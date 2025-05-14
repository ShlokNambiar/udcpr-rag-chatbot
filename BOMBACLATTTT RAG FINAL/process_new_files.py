"""
Process New Files Script

This script processes additional files for the RAG system:
1. Extract text from various file formats
2. Chunk the text with special handling for tables
3. Generate embeddings
4. Upload to Pinecone
"""

import os
import json
import argparse
from typing import List, Dict
from dotenv import load_dotenv

# Import pipeline components
from text_extractor import extract_text_from_file
from text_chunker import chunk_text
from embeddings_generator import generate_embeddings
from pinecone_uploader import upload_to_pinecone

# Load environment variables
load_dotenv()

# Create output directories
os.makedirs("output", exist_ok=True)


def process_file(
    file_path: str,
    skip_extraction: bool = False,
    skip_chunking: bool = False,
    skip_embeddings: bool = False,
    skip_upload: bool = False,
    resume: bool = False
):
    """
    Process a single file through the RAG pipeline.
    
    Args:
        file_path: Path to the file
        skip_extraction: Skip the text extraction step
        skip_chunking: Skip the text chunking step
        skip_embeddings: Skip the embeddings generation step
        skip_upload: Skip the Pinecone upload step
        resume: Resume from checkpoints where possible
    """
    # Define output paths
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    extracted_path = f"output/{base_filename}_extracted.json"
    chunked_path = f"output/{base_filename}_chunked.json"
    embeddings_path = f"output/{base_filename}_embeddings.json"
    embeddings_checkpoint = f"output/{base_filename}_embeddings_checkpoint.json"
    upload_checkpoint = f"output/{base_filename}_upload_checkpoint.json"
    
    # Step 1: Extract text from file
    if not skip_extraction:
        print(f"\n=== Step 1: Extracting text from {file_path} ===")
        pages_data = extract_text_from_file(file_path, extracted_path)
    elif os.path.exists(extracted_path):
        print(f"\n=== Step 1: Loading extracted text from file {extracted_path} ===")
        with open(extracted_path, 'r', encoding='utf-8') as f:
            pages_data = json.load(f)
    else:
        raise FileNotFoundError(f"Extracted text file not found: {extracted_path}")
    
    # Step 2: Chunk the text
    if not skip_chunking:
        print("\n=== Step 2: Chunking text ===")
        chunks_data = chunk_text(pages_data, output_path=chunked_path)
    elif os.path.exists(chunked_path):
        print(f"\n=== Step 2: Loading chunked text from file {chunked_path} ===")
        with open(chunked_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
    else:
        raise FileNotFoundError(f"Chunked text file not found: {chunked_path}")
    
    # Step 3: Generate embeddings
    if not skip_embeddings:
        print("\n=== Step 3: Generating embeddings ===")
        chunks_with_embeddings = generate_embeddings(
            chunks_data,
            output_path=embeddings_path,
            checkpoint_path=embeddings_checkpoint,
            resume=resume
        )
    elif os.path.exists(embeddings_path):
        print(f"\n=== Step 3: Loading embeddings from file {embeddings_path} ===")
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            chunks_with_embeddings = json.load(f)
    else:
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    # Step 4: Upload to Pinecone
    if not skip_upload:
        print("\n=== Step 4: Uploading to Pinecone ===")
        upload_to_pinecone(
            chunks_with_embeddings,
            checkpoint_path=upload_checkpoint,
            resume=resume
        )
    
    print(f"\n=== Pipeline completed successfully for {file_path} ===")
    print(f"Processed {len(pages_data)} pages")
    print(f"Created {len(chunks_data)} chunks")
    print(f"Generated {len(chunks_with_embeddings)} embeddings")
    print("Data uploaded to Pinecone")
    
    return True


def process_multiple_files(
    file_paths: List[str],
    skip_extraction: bool = False,
    skip_chunking: bool = False,
    skip_embeddings: bool = False,
    skip_upload: bool = False,
    resume: bool = False
):
    """
    Process multiple files through the RAG pipeline.
    
    Args:
        file_paths: List of paths to the files
        skip_extraction: Skip the text extraction step
        skip_chunking: Skip the text chunking step
        skip_embeddings: Skip the embeddings generation step
        skip_upload: Skip the Pinecone upload step
        resume: Resume from checkpoints where possible
    """
    results = {}
    
    for file_path in file_paths:
        print(f"\n\n=== Processing file: {file_path} ===")
        try:
            success = process_file(
                file_path,
                skip_extraction,
                skip_chunking,
                skip_embeddings,
                skip_upload,
                resume
            )
            results[file_path] = "Success" if success else "Failed"
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            results[file_path] = f"Error: {str(e)}"
    
    print("\n\n=== Processing Summary ===")
    for file_path, status in results.items():
        print(f"{file_path}: {status}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process files for the RAG pipeline")
    parser.add_argument("--files", nargs='+', help="Paths to the files to process")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip text extraction")
    parser.add_argument("--skip-chunking", action="store_true", help="Skip text chunking")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embeddings generation")
    parser.add_argument("--skip-upload", action="store_true", help="Skip Pinecone upload")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    
    args = parser.parse_args()
    
    if args.files:
        process_multiple_files(
            args.files,
            args.skip_extraction,
            args.skip_chunking,
            args.skip_embeddings,
            args.skip_upload,
            args.resume
        )
    else:
        # Default files to process if none specified
        default_files = [
            "rag context files/list_documents_ca_services.txt",
            "rag context files/comprehensive_regulatory_services.txt",
            "rag context files/MRTP-act_1966-Modified_2015.pdf"
        ]
        
        print("No files specified, using default files:")
        for file in default_files:
            print(f"- {file}")
        
        process_multiple_files(
            default_files,
            args.skip_extraction,
            args.skip_chunking,
            args.skip_embeddings,
            args.skip_upload,
            args.resume
        )
