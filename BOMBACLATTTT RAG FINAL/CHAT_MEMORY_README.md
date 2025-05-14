# Enhanced Chat Memory Feature for UDCPR RAG Chatbot

This document explains how to set up and use the enhanced chat memory feature for the UDCPR RAG Chatbot.

## Overview

The chat memory feature allows the chatbot to remember previous conversations, providing a more natural and contextual interaction experience. It uses Supabase as a backend database to store and retrieve conversation history, with robust error handling and fallback mechanisms.

## Features

- **Persistent Chat History**: Conversations are stored in Supabase and can be retrieved across sessions
- **Session Management**: Each conversation has a unique session ID and metadata
- **Shareable Conversations**: Users can share their conversations with others via a URL
- **Fallback to In-Memory**: If Supabase is not available, the system falls back to in-memory chat history
- **Automatic Retry Logic**: Built-in retry mechanism for handling temporary connection issues
- **System Messages**: Support for system messages to control chatbot behavior
- **Session Management**: List, retrieve, and delete chat sessions
- **Realtime Updates**: Supabase Realtime enabled for live updates across clients

## Setup Instructions

### 1. Create a Supabase Account

1. Go to [Supabase](https://supabase.com/) and sign up for an account
2. Create a new project
3. Note your project URL and API key (found in Project Settings > API)

### 2. Set Up Database Tables

1. In your Supabase project, go to the SQL Editor
2. Copy and paste the contents of `supabase_schema.sql` into the SQL editor
3. Run the SQL to create the necessary tables, indexes, and security policies

### 3. Configure Environment Variables

Add your Supabase credentials to the `.env` file:

```
# Supabase Configuration for Chat Memory
SUPABASE_URL="https://your-project-id.supabase.co"
SUPABASE_API_KEY="your_supabase_api_key"
```

### 4. Install Dependencies

Make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
# Start a new conversation with memory
python rag_chatbot.py

# Continue an existing conversation
python rag_chatbot.py --session "your-session-id"

# Disable memory (use in-memory only)
python rag_chatbot.py --no-memory
```

### Web Interface

1. Start the web interface:
   ```bash
   streamlit run chatbot_web.py
   ```

2. Use the interface to ask questions
3. Your conversation will be automatically saved
4. You can share your conversation by copying the session ID or URL from the sidebar
5. To continue a conversation, add `?session_id=your-session-id` to the URL

## Advanced Features

### System Messages

You can add system messages to control the behavior of the chatbot:

```python
from supabase_config import initialize_supabase, add_system_message

supabase = initialize_supabase()
session_id = "your-session-id"
add_system_message(supabase, session_id, "You are a helpful assistant specialized in UDCPR regulations.")
```

### Managing Sessions

List all chat sessions for a user:

```python
from supabase_config import initialize_supabase, list_chat_sessions

supabase = initialize_supabase()
sessions = list_chat_sessions(supabase, user_id="anonymous", limit=10)
for session in sessions:
    print(f"Session ID: {session['session_id']}, Updated: {session['updated_at']}")
```

Delete a chat session:

```python
from supabase_config import initialize_supabase, delete_chat_session

supabase = initialize_supabase()
success = delete_chat_session(supabase, session_id="your-session-id")
```

## How It Works

1. When a user starts a conversation, a new session is created in Supabase
2. Each message (user, assistant, or system) is stored in the database with metadata
3. When continuing a conversation, the system retrieves the chat history
4. The chat history is used to provide context for new responses
5. If Supabase is unavailable, the system falls back to in-memory chat history
6. Retry logic handles temporary connection issues automatically

## Database Schema

The implementation uses two main tables:

1. **chat_memories**: Stores session information
   - `session_id`: Unique identifier for the conversation
   - `user_id`: Identifier for the user (default: "anonymous")
   - `title`: Optional title for the conversation
   - `metadata`: JSON field for additional data
   - `created_at`: Timestamp when the session was created
   - `updated_at`: Timestamp when the session was last updated

2. **chat_messages**: Stores individual messages
   - `id`: Unique identifier for the message
   - `session_id`: Reference to the chat session
   - `role`: Message role (user/assistant/system)
   - `content`: Message content
   - `metadata`: JSON field for additional data
   - `timestamp`: When the message was created

## Troubleshooting

- **Connection Issues**: Make sure your Supabase URL and API key are correct
- **Missing Tables**: Verify that you've run the SQL schema to create the necessary tables
- **Permissions Errors**: Check that the Row Level Security policies are properly configured
- **Package Missing**: If you see "Supabase package not installed" errors, run `pip install supabase`

## Security Considerations

- The current implementation uses anonymous access for simplicity
- For production use, consider implementing proper authentication
- Update the Row Level Security policies in `supabase_schema.sql` if you implement authentication
- The SQL functions use `SECURITY DEFINER` to ensure they run with appropriate permissions

## Performance Optimization

- Indexes are created on frequently queried columns for better performance
- Messages are retrieved with a limit to prevent loading too much data
- Retry logic with exponential backoff helps handle temporary connection issues
- Realtime is enabled for live updates without polling
