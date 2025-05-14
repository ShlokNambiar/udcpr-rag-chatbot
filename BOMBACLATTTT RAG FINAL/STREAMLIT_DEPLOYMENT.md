# Deploying UDCPR RAG Chatbot to Streamlit Cloud

This guide explains how to deploy the UDCPR RAG Chatbot to Streamlit Cloud.

## Prerequisites

Before deploying to Streamlit Cloud, you need:

1. A GitHub account
2. An OpenAI API key
3. A Pinecone API key
4. (Optional) A Supabase account for persistent chat memory

## Deployment Steps

### 1. Push your code to GitHub

First, create a GitHub repository and push your code to it:

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/udcpr-rag-chatbot.git
git push -u origin main
```

### 2. Create a Streamlit Cloud account

If you don't already have one, sign up for a Streamlit Cloud account at [https://streamlit.io/cloud](https://streamlit.io/cloud).

### 3. Deploy your app

1. Log in to Streamlit Cloud
2. Click "New app"
3. Connect your GitHub repository
4. Select the repository and branch
5. Set the main file path to `streamlit_app.py`
6. Click "Deploy"

### 4. Configure Secrets

After deploying, you need to configure your API keys and other secrets:

1. In the Streamlit Cloud dashboard, go to your app settings
2. Click on "Secrets"
3. Add the following secrets:

```toml
OPENAI_API_KEY = "your_openai_api_key_here"
PINECONE_API_KEY = "your_pinecone_api_key_here"
STREAMLIT_DEPLOYED_URL = "https://your-app-name.streamlit.app"

# Optional: Supabase configuration for persistent chat memory
SUPABASE_URL = "your_supabase_url_here"
SUPABASE_API_KEY = "your_supabase_api_key_here"

# Optional: Enable web search
ENABLE_WEB_SEARCH = "true"
```

4. Save your secrets

### 5. Reboot your app

After configuring secrets, reboot your app from the Streamlit Cloud dashboard.

## Local Development with Streamlit Cloud Configuration

For local development that matches your Streamlit Cloud deployment:

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
2. Edit `.streamlit/secrets.toml` with your API keys and configuration
3. Run the app locally with:

```bash
streamlit run streamlit_app.py
```

## Troubleshooting

If you encounter issues with your deployment:

1. Check the app logs in the Streamlit Cloud dashboard
2. Verify that all required secrets are correctly configured
3. Make sure your Pinecone index is accessible from Streamlit Cloud
4. Check that your OpenAI API key has sufficient quota

## Limitations

When deployed to Streamlit Cloud:

1. The app will restart after a period of inactivity, which will clear in-memory chat history
2. File uploads and downloads may be restricted
3. Long-running operations might time out

For these reasons, it's recommended to enable Supabase integration for persistent chat memory when deploying to Streamlit Cloud.
