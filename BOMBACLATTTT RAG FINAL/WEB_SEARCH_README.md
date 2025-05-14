# Web Search Feature for RAG Chatbot

This document explains how to set up and use the web search feature in the RAG chatbot.

## Overview

The web search feature allows the chatbot to search the web for information when it doesn't have relevant information in its knowledge base. This is particularly useful for questions that are outside the scope of the UDCPR document.

## How It Works

1. When a user asks a question, the chatbot first searches its knowledge base (Pinecone vector database) for relevant information.
2. If the relevance score of the best match is below a certain threshold, the chatbot will automatically search the web for information.
3. The web search results are then used to generate a response to the user's question.
4. The chatbot clearly indicates when it's using information from the web rather than from the UDCPR document.

## Setup

### 1. Install Required Packages

Make sure you have the required packages installed:

```bash
pip install requests beautifulsoup4
```

These packages are already included in the requirements.txt file, so if you've installed all requirements, you should be good to go.

### 2. Configure Environment Variables

To enable web search by default, add the following to your `.env` file:

```
# Web Search Configuration
ENABLE_WEB_SEARCH=true
```

## Usage

### Command Line Interface

To enable web search in the command line interface:

```bash
python rag_chatbot.py --web-search
```

Or for a single query:

```bash
python rag_chatbot.py --query "What is the height of Mount Everest?" --web-search
```

### Web Interface

In the web interface, you can toggle web search on/off using the checkbox in the sidebar.

## Configuration

You can adjust the following parameters in `rag_chatbot.py`:

- `WEB_SEARCH_THRESHOLD`: The minimum relevance score threshold for RAG results (default: 0.75)
- `WEB_SEARCH_RESULTS`: Number of web search results to retrieve (default: 3)

## How the Web Search Works

The web search feature uses a simple web scraping approach to search for information on the internet:

1. It first tries to search using Bing's search engine
2. If that fails, it falls back to DuckDuckGo
3. The search results are parsed using BeautifulSoup to extract titles, links, and snippets
4. These results are then formatted and provided to the chatbot as additional context

This approach doesn't require any API keys or special setup, making it easy to use out of the box.

## Important Notes

- Web search is only used when there's no relevant information in the knowledge base
- The chatbot clearly indicates when it's using information from the web
- Web search requires an internet connection
- Be aware that web scraping may be against the terms of service of some websites
- This implementation is for educational purposes only
