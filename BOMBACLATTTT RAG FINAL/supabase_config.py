"""
Supabase Configuration Module

This module handles the configuration and connection to Supabase
for storing and retrieving chat memory.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import uuid
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Try to import Supabase
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    # Create a dummy Client class for type hints when Supabase is not available
    class Client:
        pass

# Load environment variables
load_dotenv()

# Supabase constants
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_API_KEY")
CHAT_MEMORY_TABLE = "chat_memories"
CHAT_MESSAGES_TABLE = "chat_messages"
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

def initialize_supabase() -> Optional[Client]:
    """
    Initialize Supabase client and return the client object.

    Returns:
        Supabase client or None if not available

    Raises:
        ValueError: If Supabase URL or API key is not set
        ImportError: If Supabase package is not installed
    """
    if not SUPABASE_AVAILABLE:
        raise ImportError(
            "Supabase package not installed. Install it with 'pip install supabase'"
        )

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase URL or API key not set. Check your .env file.")

    # Initialize Supabase client
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        return supabase
    except Exception as e:
        print(f"Error initializing Supabase client: {str(e)}")
        return None


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type(Exception)
)
def create_chat_session(supabase: Client, user_id: str = "anonymous") -> str:
    """
    Create a new chat session and return the session ID.

    Args:
        supabase: Supabase client
        user_id: User identifier (default: "anonymous")

    Returns:
        Session ID

    Raises:
        Exception: If there's an error creating the session
    """
    session_id = str(uuid.uuid4())
    current_time = datetime.now().isoformat()

    try:
        # Insert new chat session
        result = supabase.table(CHAT_MEMORY_TABLE).insert({
            "session_id": session_id,
            "user_id": user_id,
            "created_at": current_time,
            "updated_at": current_time
        }).execute()

        # Verify the insert was successful
        if not result.data:
            raise Exception("Failed to create chat session")

        return session_id
    except Exception as e:
        print(f"Error creating chat session: {str(e)}")
        raise


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type(Exception)
)
def save_message(
    supabase: Client,
    session_id: str,
    role: str,
    content: str
) -> Dict:
    """
    Save a chat message to the database.

    Args:
        supabase: Supabase client
        session_id: Chat session ID
        role: Message role (user/assistant/system)
        content: Message content

    Returns:
        Saved message data

    Raises:
        ValueError: If role is invalid
        Exception: If there's an error saving the message
    """
    # Validate role
    if role not in ["user", "assistant", "system"]:
        raise ValueError("Role must be one of: user, assistant, system")

    message_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    try:
        # Insert message
        result = supabase.table(CHAT_MESSAGES_TABLE).insert({
            "id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": timestamp
        }).execute()

        # Update session last activity
        supabase.table(CHAT_MEMORY_TABLE).update({
            "updated_at": timestamp
        }).eq("session_id", session_id).execute()

        return result.data[0] if result.data else {}
    except Exception as e:
        print(f"Error saving message: {str(e)}")
        raise


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type(Exception)
)
def get_chat_history(
    supabase: Client,
    session_id: str,
    limit: int = 10
) -> List[Dict]:
    """
    Retrieve chat history for a session.

    Args:
        supabase: Supabase client
        session_id: Chat session ID
        limit: Maximum number of messages to retrieve

    Returns:
        List of chat messages

    Raises:
        Exception: If there's an error retrieving the chat history
    """
    try:
        result = supabase.table(CHAT_MESSAGES_TABLE)\
            .select("*")\
            .eq("session_id", session_id)\
            .order("timestamp", desc=False)\
            .limit(limit)\
            .execute()

        messages = []
        for msg in result.data:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        return messages
    except Exception as e:
        print(f"Error retrieving chat history: {str(e)}")
        raise


def format_chat_history_for_openai(messages: List[Dict]) -> List[Dict]:
    """
    Format chat history for OpenAI API.

    Args:
        messages: List of chat messages

    Returns:
        Formatted messages for OpenAI
    """
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
    ]


def get_session_info(
    supabase: Client,
    session_id: str
) -> Optional[Dict]:
    """
    Get information about a chat session.

    Args:
        supabase: Supabase client
        session_id: Chat session ID

    Returns:
        Session information or None if not found
    """
    try:
        result = supabase.table(CHAT_MEMORY_TABLE)\
            .select("*")\
            .eq("session_id", session_id)\
            .execute()

        return result.data[0] if result.data else None
    except Exception as e:
        print(f"Error retrieving session info: {str(e)}")
        return None


def delete_chat_session(
    supabase: Client,
    session_id: str
) -> bool:
    """
    Delete a chat session and all its messages.

    Args:
        supabase: Supabase client
        session_id: Chat session ID

    Returns:
        True if successful, False otherwise
    """
    try:
        # Delete messages first (due to foreign key constraint)
        supabase.table(CHAT_MESSAGES_TABLE)\
            .delete()\
            .eq("session_id", session_id)\
            .execute()

        # Delete session
        supabase.table(CHAT_MEMORY_TABLE)\
            .delete()\
            .eq("session_id", session_id)\
            .execute()

        return True
    except Exception as e:
        print(f"Error deleting chat session: {str(e)}")
        return False


def list_chat_sessions(
    supabase: Client,
    user_id: str = "anonymous",
    limit: int = 10
) -> List[Dict]:
    """
    List chat sessions for a user.

    Args:
        supabase: Supabase client
        user_id: User identifier
        limit: Maximum number of sessions to retrieve

    Returns:
        List of chat sessions
    """
    try:
        result = supabase.table(CHAT_MEMORY_TABLE)\
            .select("*")\
            .eq("user_id", user_id)\
            .order("updated_at", desc=True)\
            .limit(limit)\
            .execute()

        return result.data if result.data else []
    except Exception as e:
        print(f"Error listing chat sessions: {str(e)}")
        return []


def add_system_message(
    supabase: Client,
    session_id: str,
    content: str
) -> Dict:
    """
    Add a system message to the chat history.

    Args:
        supabase: Supabase client
        session_id: Chat session ID
        content: System message content

    Returns:
        Saved message data
    """
    return save_message(supabase, session_id, "system", content)


if __name__ == "__main__":
    # Test the Supabase connection
    if not SUPABASE_AVAILABLE:
        print("Supabase package not installed. Install it with 'pip install supabase'")
        exit(1)

    try:
        supabase = initialize_supabase()
        if not supabase:
            print("Failed to initialize Supabase client")
            exit(1)

        print("Successfully connected to Supabase!")

        # Test creating a session
        session_id = create_chat_session(supabase)
        print(f"Created test session: {session_id}")

        # Test saving messages
        save_message(supabase, session_id, "system", "You are a helpful assistant.")
        save_message(supabase, session_id, "user", "Hello, how are you?")
        save_message(supabase, session_id, "assistant", "I'm doing well, thank you for asking!")

        # Test retrieving messages
        messages = get_chat_history(supabase, session_id)
        print(f"Retrieved {len(messages)} messages:")
        for msg in messages:
            print(f"{msg['role']}: {msg['content']}")

        # Test getting session info
        session_info = get_session_info(supabase, session_id)
        if session_info:
            print(f"Session info: Created at {session_info['created_at']}, Updated at {session_info['updated_at']}")

        # Test listing sessions
        sessions = list_chat_sessions(supabase)
        print(f"Found {len(sessions)} sessions")

        # Test deleting the session
        if delete_chat_session(supabase, session_id):
            print(f"Successfully deleted session {session_id}")
        else:
            print(f"Failed to delete session {session_id}")

    except Exception as e:
        print(f"Error: {str(e)}")
