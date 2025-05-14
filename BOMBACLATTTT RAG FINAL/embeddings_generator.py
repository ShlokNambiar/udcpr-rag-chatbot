"""
Embeddings Generator Module

This module handles the generation of embeddings using OpenAI's API.
It includes rate limit handling, batch processing, and retry logic.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import openai
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import tiktoken
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1024
BATCH_SIZE = 50  # Number of chunks to process in one batch
RATE_LIMIT_DELAY = 1.5  # Seconds to wait between batches


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIError, openai.APIConnectionError))
)
def get_embeddings_with_retry(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """
    Get embeddings for a list of texts with retry logic.
    
    Args:
        texts: List of text strings to embed
        model: OpenAI embedding model to use
        
    Returns:
        List of embedding vectors
    """
    response = openai.embeddings.create(
        input=texts,
        model=model,
        dimensions=EMBEDDING_DIMENSIONS
    )
    
    # Extract embeddings from response
    embeddings = [item.embedding for item in response.data]
    return embeddings


def calculate_batch_token_count(texts: List[str], model: str = EMBEDDING_MODEL) -> int:
    """
    Calculate the total number of tokens in a batch of texts.
    
    Args:
        texts: List of text strings
        model: Model name for tokenization
        
    Returns:
        Total token count
    """
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = sum(len(encoding.encode(text)) for text in texts)
    return total_tokens


def generate_embeddings(
    chunks_data: List[Dict],
    batch_size: int = BATCH_SIZE,
    output_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    resume: bool = False
) -> List[Dict]:
    """
    Generate embeddings for text chunks with rate limit handling.
    
    Args:
        chunks_data: List of dictionaries containing chunked text with metadata
        batch_size: Number of chunks to process in one batch
        output_path: Optional path to save the embeddings as JSON
        checkpoint_path: Optional path to save checkpoints during processing
        resume: Whether to resume from a checkpoint
        
    Returns:
        List of dictionaries containing text chunks with embeddings
    """
    # Check if OpenAI API key is set
    if not openai.api_key:
        raise ValueError("OpenAI API key is not set. Set the OPENAI_API_KEY environment variable.")
    
    # Resume from checkpoint if requested
    start_idx = 0
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            processed_chunks = json.load(f)
        
        start_idx = len(processed_chunks)
        print(f"Resuming from checkpoint with {start_idx} already processed chunks")
        
        # If we've processed all chunks, just return them
        if start_idx >= len(chunks_data):
            print("All chunks already processed")
            return processed_chunks
    else:
        processed_chunks = []
    
    # Process chunks in batches
    print(f"Generating embeddings for {len(chunks_data) - start_idx} chunks in batches of {batch_size}...")
    
    for i in tqdm(range(start_idx, len(chunks_data), batch_size), desc="Processing batches"):
        # Get the current batch
        batch = chunks_data[i:i + batch_size]
        batch_texts = [chunk["text"] for chunk in batch]
        
        # Calculate token count for the batch
        batch_token_count = calculate_batch_token_count(batch_texts)
        
        try:
            # Get embeddings for the batch
            embeddings = get_embeddings_with_retry(batch_texts)
            
            # Add embeddings to chunks
            for j, embedding in enumerate(embeddings):
                chunk_with_embedding = batch[j].copy()
                chunk_with_embedding["embedding"] = embedding
                processed_chunks.append(chunk_with_embedding)
            
            # Save checkpoint
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_chunks, f, ensure_ascii=False)
            
            # Print batch stats
            print(f"Batch {i//batch_size + 1}: {len(batch)} chunks, {batch_token_count} tokens")
            
            # Rate limit delay
            time.sleep(RATE_LIMIT_DELAY)
            
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {str(e)}")
            # Save progress before raising the exception
            if checkpoint_path and processed_chunks:
                os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_chunks, f, ensure_ascii=False)
                print(f"Progress saved to {checkpoint_path}")
            raise
    
    print(f"Generated embeddings for {len(processed_chunks)} chunks")
    
    # Save to JSON if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_chunks, f, ensure_ascii=False)
        print(f"Saved chunks with embeddings to {output_path}")
    
    return processed_chunks


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for text chunks")
    parser.add_argument("input_json", help="Path to the JSON file with chunked text")
    parser.add_argument("--output", "-o", help="Output JSON file path for embeddings")
    parser.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE,
                        help=f"Batch size for API calls (default: {BATCH_SIZE})")
    parser.add_argument("--checkpoint", "-c", help="Checkpoint file path")
    parser.add_argument("--resume", "-r", action="store_true",
                        help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Load the chunked text
    with open(args.input_json, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    generate_embeddings(
        chunks_data,
        args.batch_size,
        args.output,
        args.checkpoint,
        args.resume
    )
