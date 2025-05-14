"""
Main RAG Pipeline Script

This script orchestrates the entire RAG pipeline:
1. Extract text from PDF
2. Chunk the text
3. Generate embeddings
4. Upload to Pinecone
5. Provide a query interface
"""

import os
import argparse
import json
from dotenv import load_dotenv

# Import pipeline components
from pdf_extractor import extract_text_from_pdf
from text_chunker import chunk_text
from embeddings_generator import generate_embeddings
from pinecone_uploader import upload_to_pinecone
from query_interface import query_rag_system

# Load environment variables
load_dotenv()

# Create output directories
os.makedirs("output", exist_ok=True)


def run_pipeline(
    pdf_path: str,
    skip_extraction: bool = False,
    skip_chunking: bool = False,
    skip_embeddings: bool = False,
    skip_upload: bool = False,
    resume: bool = False
):
    """
    Run the complete RAG pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        skip_extraction: Skip the text extraction step
        skip_chunking: Skip the text chunking step
        skip_embeddings: Skip the embeddings generation step
        skip_upload: Skip the Pinecone upload step
        resume: Resume from checkpoints where possible
    """
    # Define output paths
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    extracted_path = f"output/{base_filename}_extracted.json"
    chunked_path = f"output/{base_filename}_chunked.json"
    embeddings_path = f"output/{base_filename}_embeddings.json"
    embeddings_checkpoint = f"output/{base_filename}_embeddings_checkpoint.json"
    upload_checkpoint = f"output/{base_filename}_upload_checkpoint.json"
    
    # Step 1: Extract text from PDF
    if not skip_extraction:
        print("\n=== Step 1: Extracting text from PDF ===")
        pages_data = extract_text_from_pdf(pdf_path, extracted_path)
    elif os.path.exists(extracted_path):
        print("\n=== Step 1: Loading extracted text from file ===")
        with open(extracted_path, 'r', encoding='utf-8') as f:
            pages_data = json.load(f)
    else:
        raise FileNotFoundError(f"Extracted text file not found: {extracted_path}")
    
    # Step 2: Chunk the text
    if not skip_chunking:
        print("\n=== Step 2: Chunking text ===")
        chunks_data = chunk_text(pages_data, output_path=chunked_path)
    elif os.path.exists(chunked_path):
        print("\n=== Step 2: Loading chunked text from file ===")
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
        print("\n=== Step 3: Loading embeddings from file ===")
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
    
    print("\n=== Pipeline completed successfully ===")
    print(f"Processed {len(pages_data)} pages")
    print(f"Created {len(chunks_data)} chunks")
    print(f"Generated {len(chunks_with_embeddings)} embeddings")
    print("Data uploaded to Pinecone")
    
    return True


def interactive_query():
    """Run an interactive query session."""
    print("\n=== RAG Query Interface ===")
    print("Enter your questions about the document (or 'exit' to quit):")
    
    while True:
        query = input("\nQuery: ")
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        results = query_rag_system(query, top_k=5)
        
        print("\nSearch Results:")
        for result in results:
            print(f"\nRank {result['rank']} (Score: {result['score']:.4f}, Page: {result['page']})")
            print(f"Source: {result['source']}")
            print(f"Text: {result['text'][:300]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG pipeline")
    parser.add_argument("--pdf", help="Path to the PDF file")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip text extraction")
    parser.add_argument("--skip-chunking", action="store_true", help="Skip text chunking")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embeddings generation")
    parser.add_argument("--skip-upload", action="store_true", help="Skip Pinecone upload")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    parser.add_argument("--query", action="store_true", help="Run interactive query interface")
    
    args = parser.parse_args()
    
    if args.query:
        interactive_query()
    elif args.pdf:
        run_pipeline(
            args.pdf,
            args.skip_extraction,
            args.skip_chunking,
            args.skip_embeddings,
            args.skip_upload,
            args.resume
        )
    else:
        parser.print_help()
