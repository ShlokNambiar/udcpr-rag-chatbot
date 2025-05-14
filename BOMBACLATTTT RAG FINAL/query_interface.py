"""
Query Interface Module

This module provides a simple interface to query the RAG system.
It handles query embedding, vector search, and result formatting.
"""

import os
import json
from typing import Dict, List, Optional, Any
import openai
import pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Pinecone constants
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "udcpr-rag-index"

# OpenAI constants
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1024


def initialize_pinecone():
    """Initialize Pinecone client and return the index."""
    if not PINECONE_API_KEY:
        raise ValueError("Pinecone API key not set. Check your .env file.")

    # Initialize Pinecone
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

    # Connect to the index
    index_list = [index.name for index in pc.list_indexes()]
    if INDEX_NAME not in index_list:
        raise ValueError(f"Index {INDEX_NAME} does not exist. Run the uploader first.")

    # Connect to the index
    index = pc.Index(INDEX_NAME)
    return index


def get_query_embedding(query: str) -> List[float]:
    """
    Get embedding for a query string.

    Args:
        query: Query string

    Returns:
        Embedding vector
    """
    response = openai.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS
    )

    return response.data[0].embedding


def search_pinecone(query: str, top_k: int = 5, include_metadata: bool = True) -> List[Dict]:
    """
    Search Pinecone index with a query string.

    Args:
        query: Query string
        top_k: Number of results to return
        include_metadata: Whether to include metadata in results

    Returns:
        List of search results
    """
    # Initialize Pinecone
    index = initialize_pinecone()

    # Get query embedding
    query_embedding = get_query_embedding(query)

    # Search Pinecone
    search_response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=include_metadata
    )

    return search_response["matches"]


def format_search_results(results: List[Dict]) -> List[Dict]:
    """
    Format search results for display.

    Args:
        results: List of search results from Pinecone

    Returns:
        Formatted results
    """
    formatted_results = []

    for i, result in enumerate(results):
        formatted_result = {
            "rank": i + 1,
            "score": result["score"],
            "page": result["metadata"].get("page_num", "Unknown"),
            "text": result["metadata"].get("text", ""),
            "source": result["metadata"].get("source", "Unknown"),
            "chunk_id": result["id"]
        }

        formatted_results.append(formatted_result)

    return formatted_results


def query_rag_system(query: str, top_k: int = 5) -> List[Dict]:
    """
    Query the RAG system with a natural language query.

    Args:
        query: Natural language query
        top_k: Number of results to return

    Returns:
        Formatted search results
    """
    print(f"Searching for: {query}")

    # Search Pinecone
    results = search_pinecone(query, top_k)

    # Format results
    formatted_results = format_search_results(results)

    return formatted_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("--top-k", "-k", type=int, default=5,
                        help="Number of results to return (default: 5)")
    parser.add_argument("--output", "-o", help="Output JSON file path for results")

    args = parser.parse_args()

    # Query the RAG system
    results = query_rag_system(args.query, args.top_k)

    # Print results
    print("\nSearch Results:")
    for result in results:
        print(f"\nRank {result['rank']} (Score: {result['score']:.4f}, Page: {result['page']})")
        print(f"Source: {result['source']}")
        print(f"Text: {result['text'][:300]}...")

    # Save to JSON if output path is provided
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved results to {args.output}")
