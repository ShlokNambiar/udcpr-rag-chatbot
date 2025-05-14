# RAG Pipeline for UDCPR Document

This project implements a Retrieval-Augmented Generation (RAG) pipeline for processing the UDCPR document. The pipeline extracts text from the PDF, chunks it into manageable pieces, generates embeddings, and stores them in a vector database for semantic search.

## Features

- **PDF Text Extraction**: Uses PyMuPDF for efficient extraction from large PDFs
- **Smart Text Chunking**: Implements 512-token chunks with 15% overlap for context preservation
- **Rate-Limited Embedding Generation**: Handles OpenAI API rate limits with batch processing
- **Vector Database Storage**: Uses Pinecone for efficient vector search
- **Checkpointing**: Supports resuming from checkpoints for long-running processes
- **Query Interface**: Simple interface for semantic search
- **Web Interface**: Clean, user-friendly interface built with Streamlit
- **Cloud Deployment**: Ready for deployment on Streamlit Cloud

## Requirements

- Python 3.8+
- OpenAI API key
- Pinecone API key

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys (see `.env.example`)

## Usage

### Running the Full Pipeline

```bash
python main.py --pdf "UDCPR Updated 30.01.25 with earlier provisions & corrections.pdf"
```

### Running Individual Steps

```bash
# Extract text from PDF
python pdf_extractor.py "UDCPR Updated 30.01.25 with earlier provisions & corrections.pdf" -o output/udcpr_extracted.json

# Chunk the text
python text_chunker.py output/udcpr_extracted.json -o output/udcpr_chunked.json

# Generate embeddings
python embeddings_generator.py output/udcpr_chunked.json -o output/udcpr_embeddings.json -c output/embeddings_checkpoint.json

# Upload to Pinecone
python pinecone_uploader.py output/udcpr_embeddings.json -c output/upload_checkpoint.json
```

### Querying the RAG System

```bash
# Interactive query mode
python main.py --query

# Single query
python query_interface.py "What are the building height regulations?"
```

## Pipeline Components

1. **PDF Extraction** (`pdf_extractor.py`): Extracts text with page numbers and metadata
2. **Text Chunking** (`text_chunker.py`): Splits text into semantic chunks with overlap
3. **Embeddings Generation** (`embeddings_generator.py`): Creates vector embeddings with rate limit handling
4. **Pinecone Upload** (`pinecone_uploader.py`): Uploads vectors to Pinecone with metadata
5. **Query Interface** (`query_interface.py`): Provides semantic search functionality

## Optimization Features

- **Rate Limit Handling**: Implements delays and retries to avoid API rate limits
- **Batch Processing**: Processes data in batches to optimize API calls
- **Token Calculation**: Validates token counts before API calls
- **Checkpointing**: Saves progress to resume long-running processes
- **Error Recovery**: Handles errors gracefully with progress saving

## Web Interface and Chatbot

This project includes a web interface and chatbot for interacting with the RAG system:

- **Command Line Interface**: Run `python rag_chatbot.py` or use `chat.bat`
- **Web Interface**: Run `streamlit run chatbot_web.py` or use `run_web_chatbot.bat`
- **Streamlit Cloud Deployment**: See [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md) for instructions

## License

MIT
