"""
Web Search Module

This module provides web search functionality for the RAG chatbot.
It is used when the RAG system doesn't have relevant information in its knowledge base.
This implementation uses a simple web scraping approach without requiring API keys.
"""

import os
import re
import requests
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
MAX_SEARCH_RESULTS = 5  # Maximum number of search results to return
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"


def perform_web_search(query: str, num_results: int = MAX_SEARCH_RESULTS) -> List[Dict]:
    """
    Perform a web search using a simple scraping approach.

    Args:
        query: Search query
        num_results: Number of results to return

    Returns:
        List of search results with title, link, and snippet
    """
    # For UDCPR-specific queries, add "UDCPR" to the query if not already present
    if "udcpr" not in query.lower() and any(keyword in query.lower() for keyword in [
        "regulation", "building", "development", "control", "promotion", "maharashtra"
    ]):
        query = f"UDCPR {query}"

    # For queries about recent updates, make the query more specific
    if any(phrase in query.lower() for phrase in ["recent", "latest", "new", "update", "amendment"]):
        if "udcpr" not in query.lower():
            query = f"latest UDCPR updates Maharashtra {query}"
        else:
            query = f"latest {query} Maharashtra"

    print(f"Searching web for: {query}")

    # Prepare the search query
    encoded_query = quote_plus(query)

    # Try multiple search engines in sequence
    search_engines = [
        # Bing
        {
            "url": f"https://www.bing.com/search?q={encoded_query}",
            "result_selector": "li.b_algo",
            "title_selector": "h2 a",
            "snippet_selector": "p",
            "name": "Bing"
        },
        # DuckDuckGo
        {
            "url": f"https://duckduckgo.com/html/?q={encoded_query}",
            "result_selector": ".result",
            "title_selector": ".result__title a",
            "snippet_selector": ".result__snippet",
            "name": "DuckDuckGo"
        },
        # Google (fallback with simple selector)
        {
            "url": f"https://www.google.com/search?q={encoded_query}",
            "result_selector": ".g",
            "title_selector": "h3",
            "snippet_selector": ".VwiC3b",
            "name": "Google"
        }
    ]

    # Set up headers to mimic a browser
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

    results = []

    # Try each search engine until we get results
    for engine in search_engines:
        if results:
            break

        try:
            print(f"Trying {engine['name']} search...")
            headers["Referer"] = engine["url"].split("?")[0]

            # Make the request
            response = requests.get(engine["url"], headers=headers, timeout=15)
            response.raise_for_status()

            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all search result containers
            search_results = soup.select(engine["result_selector"])
            print(f"Found {len(search_results)} raw results from {engine['name']}")

            for result in search_results[:num_results]:
                # Extract title and link
                title_elem = result.select_one(engine["title_selector"])
                if not title_elem:
                    continue

                title = title_elem.get_text().strip()

                # Handle link extraction based on search engine
                if engine["name"] == "Google":
                    # For Google, links are in a parent element with an href attribute
                    link_elem = title_elem.find_parent("a")
                    link = link_elem.get("href", "") if link_elem else ""
                    # Clean up Google's redirect links
                    if link.startswith("/url?q="):
                        link = link.split("/url?q=")[1].split("&")[0]
                else:
                    link = title_elem.get("href", "")

                # Extract snippet
                snippet_elem = result.select_one(engine["snippet_selector"])
                snippet = snippet_elem.get_text().strip() if snippet_elem else ""

                # Add to results if we have both title and link
                if title and link:
                    # Avoid duplicate results
                    if not any(r["title"] == title for r in results):
                        results.append({
                            "title": title,
                            "link": link,
                            "snippet": snippet,
                            "source": engine["name"]
                        })

            if results:
                print(f"Successfully retrieved {len(results)} results from {engine['name']}")

        except Exception as e:
            print(f"Error with {engine['name']} search: {str(e)}")

    # If we still have no results, create a fallback result
    if not results:
        print("All search engines failed. Creating fallback result.")
        results = [{
            "title": "Maharashtra Urban Development Department",
            "link": "https://urban.maharashtra.gov.in/",
            "snippet": "Official website of Maharashtra Urban Development Department where you can find the latest UDCPR updates and notifications.",
            "source": "Fallback"
        }]

    return results


def format_search_results_for_context(results: List[Dict]) -> str:
    """
    Format web search results into a context string for the chatbot.

    Args:
        results: List of search results

    Returns:
        Formatted context string
    """
    if not results:
        return "No relevant information found from web search."

    context = "Information from web search about UDCPR updates and regulations:\n\n"

    for i, result in enumerate(results):
        source = result.get('source', 'Web')
        context += f"Source {i+1} [{source}]: {result['title']}\n"
        context += f"URL: {result['link']}\n"

        # Clean up and enhance the snippet
        snippet = result['snippet']
        if snippet:
            # If snippet is very short, add a note
            if len(snippet) < 50:
                snippet += " (Note: Limited information available from this source. Please check the URL for more details.)"
            context += f"Summary: {snippet}\n\n"
        else:
            context += "Summary: No summary available. Please check the URL for information.\n\n"

    # Add a note about using this information
    context += "\nIMPORTANT: The above information is from web search results and may not be complete or up-to-date. "
    context += "For official and authoritative information about UDCPR, please refer to the official Maharashtra government "
    context += "publications and notifications. When answering questions about recent updates or changes to the UDCPR, "
    context += "make sure to mention the source and date of the information if available.\n"

    return context


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Perform a web search")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--num-results", "-n", type=int, default=MAX_SEARCH_RESULTS,
                        help=f"Number of results to return (default: {MAX_SEARCH_RESULTS})")

    args = parser.parse_args()

    # Check if API keys are set
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        print("Error: Google API key or CSE ID not set.")
        print("Set the GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables.")
        exit(1)

    # Perform the search
    results = perform_web_search(args.query, args.num_results)

    # Print the results
    if results:
        print(f"\nSearch results for '{args.query}':\n")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['title']}")
            print(f"   URL: {result['link']}")
            print(f"   {result['snippet']}\n")
    else:
        print(f"No results found for '{args.query}'")
