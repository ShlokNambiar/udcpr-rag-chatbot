# UDCPR RAG Chatbot

A minimalist, intelligent chatbot for the Unified Development Control and Promotion Regulations (UDCPR) for Maharashtra State. This chatbot uses Retrieval Augmented Generation (RAG) to provide accurate information from the UDCPR document.

## Features

- **Intelligent Responses**: Powered by OpenAI's GPT-4o model
- **Document-Grounded Answers**: Uses RAG to retrieve relevant information from the UDCPR document
- **Minimalist Web Interface**: Clean, user-friendly interface built with Streamlit
- **Conversation History**: Maintains context across multiple interactions
- **Source Citations**: Cites page numbers when providing information

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key

### Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Make sure your `.env` file contains the necessary API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

### Running the Chatbot

#### Web Interface (Recommended)

Run the web interface using:
```
run_web_chatbot.bat
```
or
```
streamlit run chatbot_web.py
```

This will open a browser window with the chatbot interface.

#### Command Line Interface

For a command-line experience:
```
chat.bat
```
or
```
python rag_chatbot.py
```

For a single query:
```
python rag_chatbot.py --query "What are the building height regulations?"
```

### Deploying to Streamlit Cloud

This chatbot is ready for deployment on Streamlit Cloud. For detailed instructions, see [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md).

Quick steps:
1. Push your code to GitHub
2. Deploy on Streamlit Cloud using `streamlit_app.py` as the main file
3. Configure your API keys in Streamlit Cloud secrets

## How It Works

1. The chatbot takes your question and searches for relevant sections in the UDCPR document using vector similarity search.
2. It retrieves the most relevant sections and uses them as context for generating a response.
3. GPT-4o generates a comprehensive answer based on the retrieved context.
4. The answer is displayed in a user-friendly format with citations to the source material.

## Sample Questions

- What are the parking requirements for residential buildings?
- What are the fire safety regulations for high-rise buildings?
- What is the definition of Floor Space Index (FSI)?
- What are the requirements for green buildings?
- What are the setback requirements for different types of buildings?

## Customization

You can customize the chatbot by modifying these parameters in the `rag_chatbot.py` file:
- `MAX_CONTEXT_TOKENS`: Maximum tokens for context to send to OpenAI
- `MODEL`: OpenAI model to use (currently set to "gpt-4o")
- `TOP_K_RESULTS`: Number of results to retrieve from Pinecone

## Acknowledgments

- This chatbot is built on top of the RAG pipeline for the UDCPR document
- Uses OpenAI's GPT-4o for natural language understanding and generation
- Pinecone for vector search capabilities
- Streamlit for the web interface
