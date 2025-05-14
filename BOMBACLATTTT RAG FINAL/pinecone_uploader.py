"""
Pinecone Uploader Module

This module handles the uploading of embeddings to Pinecone vector database.
It includes batch processing, error handling, and metadata management.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pinecone constants
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "udcpr-rag-index"
VECTOR_DIMENSION = 1024
BATCH_SIZE = 100  # Number of vectors to upsert in one batch


def initialize_pinecone():
    """Initialize Pinecone client and return the index."""
    if not PINECONE_API_KEY:
        raise ValueError("Pinecone API key not set. Check your .env file.")

    # Initialize Pinecone
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists, create if it doesn't
    index_list = [index.name for index in pc.list_indexes()]
    if INDEX_NAME not in index_list:
        print(f"Creating new Pinecone index: {INDEX_NAME}")
        try:
            # Try with Starter (free) environment
            pc.create_index(
                name=INDEX_NAME,
                dimension=VECTOR_DIMENSION,
                metric="cosine",
                spec=pinecone.PodSpec(
                    environment="gcp-starter"
                )
            )
        except Exception as e:
            print(f"Error creating index with PodSpec: {str(e)}")
            print("Trying with ServerlessSpec...")
            # Try with Serverless
            pc.create_index(
                name=INDEX_NAME,
                dimension=VECTOR_DIMENSION,
                metric="cosine",
                spec=pinecone.ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # Try a different region
                )
            )
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        time.sleep(10)

    # Connect to the index
    index = pc.Index(INDEX_NAME)
    return index


def prepare_vectors(chunks_with_embeddings: List[Dict]) -> List[Dict]:
    """
    Prepare vectors for Pinecone upsert.

    Args:
        chunks_with_embeddings: List of dictionaries containing chunks with embeddings

    Returns:
        List of dictionaries formatted for Pinecone upsert
    """
    vectors = []

    for chunk in chunks_with_embeddings:
        # Skip if no embedding
        if "embedding" not in chunk:
            continue

        # Create metadata (excluding the embedding and limiting text size)
        metadata = {k: v for k, v in chunk.items() if k != "embedding"}

        # Limit text size in metadata (Pinecone has metadata size limits)
        if "text" in metadata and len(metadata["text"]) > 8000:
            metadata["text"] = metadata["text"][:8000] + "..."

        vector = {
            "id": chunk["chunk_id"],
            "values": chunk["embedding"],
            "metadata": metadata
        }

        vectors.append(vector)

    return vectors


def upload_to_pinecone(
    chunks_with_embeddings: List[Dict],
    batch_size: int = BATCH_SIZE,
    checkpoint_path: Optional[str] = None,
    resume: bool = False
) -> None:
    """
    Upload vectors to Pinecone with batch processing and error handling.

    Args:
        chunks_with_embeddings: List of dictionaries containing chunks with embeddings
        batch_size: Number of vectors to upsert in one batch
        checkpoint_path: Optional path to save checkpoints during processing
        resume: Whether to resume from a checkpoint
    """
    # Initialize Pinecone
    index = initialize_pinecone()

    # Get index stats
    stats = index.describe_index_stats()
    print(f"Index stats before upload: {stats}")

    # Resume from checkpoint if requested
    start_idx = 0
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            uploaded_ids = json.load(f)

        # Filter out already uploaded chunks
        chunks_to_upload = [chunk for chunk in chunks_with_embeddings
                           if chunk["chunk_id"] not in uploaded_ids]

        print(f"Resuming upload: {len(uploaded_ids)} vectors already uploaded, "
              f"{len(chunks_to_upload)} vectors remaining")

        # If all chunks are uploaded, just return
        if not chunks_to_upload:
            print("All vectors already uploaded")
            return
    else:
        chunks_to_upload = chunks_with_embeddings
        uploaded_ids = []

    # Prepare vectors
    vectors = prepare_vectors(chunks_to_upload)

    # Upload vectors in batches
    print(f"Uploading {len(vectors)} vectors to Pinecone in batches of {batch_size}...")

    for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading batches"):
        # Get the current batch
        batch = vectors[i:i + batch_size]

        try:
            # Upsert the batch
            upsert_response = index.upsert(vectors=batch)

            # Add uploaded IDs to the list
            for vector in batch:
                uploaded_ids.append(vector["id"])

            # Save checkpoint
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(uploaded_ids, f, ensure_ascii=False)

            # Print batch stats
            print(f"Batch {i//batch_size + 1}: {len(batch)} vectors, "
                  f"upserted: {upsert_response.get('upserted_count', 0)}")

            # Small delay between batches
            time.sleep(0.5)

        except Exception as e:
            print(f"Error uploading batch starting at index {i}: {str(e)}")
            # Save progress before raising the exception
            if checkpoint_path and uploaded_ids:
                os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(uploaded_ids, f, ensure_ascii=False)
                print(f"Progress saved to {checkpoint_path}")
            raise

    # Get updated index stats
    stats = index.describe_index_stats()
    print(f"Index stats after upload: {stats}")
    print(f"Successfully uploaded {len(uploaded_ids)} vectors to Pinecone")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload embeddings to Pinecone")
    parser.add_argument("input_json", help="Path to the JSON file with chunks and embeddings")
    parser.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE,
                        help=f"Batch size for Pinecone upserts (default: {BATCH_SIZE})")
    parser.add_argument("--checkpoint", "-c", help="Checkpoint file path")
    parser.add_argument("--resume", "-r", action="store_true",
                        help="Resume from checkpoint")

    args = parser.parse_args()

    # Load the chunks with embeddings
    with open(args.input_json, 'r', encoding='utf-8') as f:
        chunks_with_embeddings = json.load(f)

    upload_to_pinecone(
        chunks_with_embeddings,
        args.batch_size,
        args.checkpoint,
        args.resume
    )
