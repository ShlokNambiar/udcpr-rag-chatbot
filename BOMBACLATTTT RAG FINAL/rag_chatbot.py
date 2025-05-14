"""
RAG Chatbot for UDCPR Document

This module implements a conversational interface to the UDCPR document using
the RAG (Retrieval Augmented Generation) approach with OpenAI's API.
It includes chat memory functionality using Supabase for persistence and
web search capability for questions outside the document's scope.
"""

import os
import json
import uuid
from typing import Dict, List, Optional, Any
import openai
import pinecone
from dotenv import load_dotenv
from query_interface import initialize_pinecone, get_query_embedding, search_pinecone

# Import web search functionality
try:
    from web_search import perform_web_search, format_search_results_for_context
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False

    # Define dummy functions for when web search is not available
    def perform_web_search(query, num_results=5):
        print("Web search package not installed or configured.")
        return []

    def format_search_results_for_context(results):
        return "Web search is not available."

# Try to import Supabase functions, but provide fallbacks if not available
try:
    from supabase_config import (
        initialize_supabase, create_chat_session, save_message,
        get_chat_history, format_chat_history_for_openai
    )
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

    # Define dummy functions for when Supabase is not available
    def initialize_supabase():
        print("Supabase package not installed. Using in-memory chat history only.")
        return None

    def create_chat_session(supabase, user_id="anonymous"):
        return str(uuid.uuid4())

    def save_message(supabase, session_id, role, content):
        return {"id": str(uuid.uuid4()), "role": role, "content": content}

    def get_chat_history(supabase, session_id, limit=10):
        return []

    def format_chat_history_for_openai(messages):
        return messages

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Constants
MAX_CONTEXT_TOKENS = 2000  # Maximum tokens for context to send to OpenAI (reduced for speed)
MODEL = "gpt-4o"  # Using GPT-4o for better quality responses
TOP_K_RESULTS = 3  # Number of results to retrieve from Pinecone (reduced for speed)
MAX_HISTORY_MESSAGES = 6  # Maximum number of messages to keep in history (reduced for speed)
RESPONSE_STREAMING = True  # Enable streaming responses for better user experience
WEB_SEARCH_ENABLED = os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true"  # Enable web search
WEB_SEARCH_THRESHOLD = 0.75  # Minimum relevance score threshold for RAG results
WEB_SEARCH_RESULTS = 3  # Number of web search results to retrieve


def format_context_from_results(results: List[Dict]) -> str:
    """
    Format search results into a context string for the chatbot.

    Args:
        results: List of search results from Pinecone

    Returns:
        Formatted context string
    """
    context = "Relevant UDCPR sections:\n\n"

    # Sort results by score to prioritize most relevant content
    sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)

    # Track total token count (rough estimate: 4 chars ≈ 1 token)
    total_chars = len(context)
    max_chars = MAX_CONTEXT_TOKENS * 4

    for i, result in enumerate(sorted_results):
        metadata = result["metadata"]
        section_text = metadata.get('text', '')

        # Skip very short sections (likely not useful)
        if len(section_text) < 50:
            continue

        # Create section header
        section_header = f"Section {i+1} (Page {metadata.get('page_num', 'Unknown')}):\n"

        # Check if adding this section would exceed our token limit
        if total_chars + len(section_header) + len(section_text) > max_chars:
            break

        context += section_header + section_text + "\n\n"
        total_chars += len(section_header) + len(section_text) + 2

    return context


def create_chat_prompt(
    query: str,
    context: str,
    web_search_context: str = None,
    chat_history: List[Dict] = None
) -> List[Dict]:
    """
    Create a chat prompt with system message, context, history, and user query.

    Args:
        query: User's question
        context: Retrieved context from the document
        web_search_context: Context information from web search (if available)
        chat_history: Previous conversation history

    Returns:
        List of message dictionaries for the OpenAI chat API
    """
    system_message = {
        "role": "system",
        "content": (
            "You are an expert assistant for the Unified Development Control and Promotion "
            "Regulations (UDCPR) for Maharashtra State. Your task is to provide accurate, "
            "helpful information based on the UDCPR document in a conversational and engaging manner. "
            "When answering questions, use only the context provided, but present the information in a "
            "natural, conversational way rather than directly quoting the document. Use a professional "
            "and legally appropriate tone, but make your responses feel like they're coming from a "
            "knowledgeable expert having a conversation, not just reading from a book. "

            "IMPORTANT INSTRUCTIONS FOR HANDLING WEB SEARCH RESULTS: "
            "For questions about recent updates, amendments, or changes to the UDCPR, you MUST use the "
            "web search information when provided. When using web search information, clearly indicate that "
            "the information comes from external web sources rather than from your internal UDCPR document. "
            "Cite the specific sources from the web search results (e.g., 'According to [Source Name]...'). "
            "For questions specifically asking about 'most recent' or 'latest' information, prioritize the "
            "web search results even if you think you have some information in your knowledge base, as your "
            "knowledge base may not contain the most up-to-date information."

            "If the web search results don't provide clear or specific information about the question, "
            "acknowledge this limitation and suggest where the user might find more detailed or official "
            "information, such as the Maharashtra Urban Development Department website or official "
            "government notifications."

            "If no relevant information is found in either the UDCPR context or web search, "
            "say that you don't have enough information to answer the question completely. "
            "When providing information from the UDCPR document, always cite the page numbers, "
            "but integrate these citations naturally into your response. "
            "Use transitional phrases, clear explanations, and a helpful tone throughout."
        )
    }

    context_message = {
        "role": "system",
        "content": f"Context information from the UDCPR document:\n\n{context}"
    }

    messages = [system_message, context_message]

    # Add web search context if provided
    if web_search_context:
        web_context_message = {
            "role": "system",
            "content": f"Additional information from web search:\n\n{web_search_context}"
        }
        messages.append(web_context_message)

    # Add chat history if provided
    if chat_history:
        messages.extend(chat_history)

    # Add the current user query
    messages.append({"role": "user", "content": query})

    return messages


def generate_response(
    query: str,
    session_id: Optional[str] = None,
    chat_history: List[Dict] = None,
    use_supabase: bool = True,
    use_web_search: bool = None
) -> Dict:
    """
    Generate a response to the user's query using RAG with chat memory.

    Args:
        query: User's question
        session_id: Supabase chat session ID (if None, memory won't be persisted)
        chat_history: Previous conversation history (used if not using Supabase)
        use_supabase: Whether to use Supabase for chat memory
        use_web_search: Whether to use web search (overrides WEB_SEARCH_ENABLED)

    Returns:
        Dictionary with response, updated chat history, and session ID
    """
    # Check if Supabase is available
    use_supabase = use_supabase and SUPABASE_AVAILABLE

    # Determine whether to use web search
    if use_web_search is None:
        use_web_search = WEB_SEARCH_ENABLED and WEB_SEARCH_AVAILABLE

    # Initialize Supabase if using it
    supabase = None
    if use_supabase:
        try:
            supabase = initialize_supabase()

            # Create a new session if none provided
            if not session_id:
                session_id = create_chat_session(supabase)

            # Get chat history from Supabase if we have a session
            if session_id:
                db_chat_history = get_chat_history(supabase, session_id, MAX_HISTORY_MESSAGES)
                chat_history = format_chat_history_for_openai(db_chat_history)

        except Exception as e:
            print(f"Supabase error: {str(e)}. Falling back to in-memory chat history.")
            use_supabase = False

    # Search for relevant context
    results = search_pinecone(query, top_k=TOP_K_RESULTS)

    # Format context from results
    context = format_context_from_results(results)

    # Check if we need to use web search
    web_search_context = None
    if use_web_search:
        # First, check if this is a simple greeting or basic interaction
        greeting_patterns = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon",
                            "good evening", "how are you", "what's up", "howdy"]
        is_greeting = any(pattern in query.lower() for pattern in greeting_patterns)
        print(f"Is greeting: {is_greeting}")

        # Check if the query is likely about the UDCPR document but also check for special cases
        udcpr_keywords = ["udcpr", "regulation", "building", "development", "control", "promotion",
                         "maharashtra", "construction", "zoning", "fsi", "floor space", "height",
                         "setback", "plot", "land", "urban", "planning", "architect"]

        # Keywords that suggest we might need external information even for UDCPR-related queries
        external_info_keywords = ["recent", "latest", "new", "update", "amendment", "change",
                                 "modified", "revision", "current", "2023", "2024", "added", "removed",
                                 "most", "latest", "changes", "amendments", "notifications"]

        # Explicit phrases that should always trigger web search
        force_web_search_phrases = [
            "most recent", "latest update", "new rules", "recent changes",
            "latest amendment", "current version", "updated regulation",
            "what are the most recent", "what are the latest", "recent notification"
        ]

        is_udcpr_related = any(keyword in query.lower() for keyword in udcpr_keywords)
        needs_external_info = any(keyword in query.lower() for keyword in external_info_keywords)
        force_web_search = any(phrase in query.lower() for phrase in force_web_search_phrases)

        print(f"Is UDCPR related: {is_udcpr_related}")
        print(f"Needs external info: {needs_external_info}")
        print(f"Force web search: {force_web_search}")

        # Check if we have any relevant results at all, regardless of score
        has_any_results = len(results) > 0
        print(f"Has any results: {has_any_results}")

        # Print the top result score for debugging
        if results:
            top_score = max(result.get("score", 0) for result in results)
            print(f"Top result score: {top_score}")
        else:
            print("No results from knowledge base")

        # ALWAYS use web search for queries that explicitly ask for recent/latest information
        if force_web_search:
            print(f"Forcing web search due to explicit request for recent information: {query}")
            has_relevant_results = False
        # Don't use web search for greetings, very short queries, or standard UDCPR queries
        elif is_greeting or len(query.strip()) < 10 or (is_udcpr_related and not needs_external_info and has_any_results):
            print(f"Basic interaction or standard UDCPR query detected. Not using web search for: {query}")
            has_relevant_results = True  # Pretend we have relevant results to skip web search
        else:
            # Check if RAG results are relevant enough
            has_relevant_results = False
            if results:
                # Check if any result has a score above the threshold
                for result in results:
                    if result.get("score", 0) > WEB_SEARCH_THRESHOLD:
                        has_relevant_results = True
                        print(f"Found relevant result with score {result.get('score', 0)}")
                        break

        # If no relevant results or we're forcing web search, use web search
        if not has_relevant_results:
            print(f"No relevant results found in RAG. Using web search for: {query}")
            web_results = perform_web_search(query, num_results=WEB_SEARCH_RESULTS)
            if web_results:
                web_search_context = format_search_results_for_context(web_results)
                print(f"Found {len(web_results)} web search results")
                # Print the first result for debugging
                if web_results:
                    print(f"First web result: {web_results[0]['title']}")
            else:
                print("No web search results found")

    # Initialize chat history if None
    if chat_history is None:
        chat_history = []

    # Create chat prompt with web search context if available
    messages = create_chat_prompt(query, context, web_search_context, chat_history)

    # Generate response using OpenAI
    if RESPONSE_STREAMING:
        # For web interface, we'll handle streaming in the web app
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.5,  # Balanced temperature for natural but accurate responses
            max_tokens=800,  # Reduced max tokens for faster responses
            stream=True  # Enable streaming for faster perceived response time
        )

        # Collect the streaming response
        response_text = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
    else:
        # Non-streaming mode
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.5,  # Balanced temperature for natural but accurate responses
            max_tokens=800  # Reduced max tokens for faster responses
        )

        # Extract response text
        response_text = response.choices[0].message.content

    # Save to Supabase if using it
    if use_supabase and supabase and session_id:
        try:
            # Save user message
            save_message(supabase, session_id, "user", query)

            # Save assistant response
            save_message(supabase, session_id, "assistant", response_text)
        except Exception as e:
            print(f"Error saving to Supabase: {str(e)}")

    # Update in-memory chat history
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": response_text})

    # Limit chat history to prevent context overflow
    if len(chat_history) > MAX_HISTORY_MESSAGES:
        chat_history = chat_history[-MAX_HISTORY_MESSAGES:]

    return {
        "response": response_text,
        "chat_history": chat_history,
        "session_id": session_id
    }


def interactive_chat(use_supabase: bool = True, use_web_search: bool = None):
    """
    Run an interactive chat session with the RAG chatbot.

    Args:
        use_supabase: Whether to use Supabase for chat memory
        use_web_search: Whether to use web search (overrides WEB_SEARCH_ENABLED)
    """
    # Check if Supabase is available
    use_supabase = use_supabase and SUPABASE_AVAILABLE

    # Determine whether to use web search
    if use_web_search is None:
        use_web_search = WEB_SEARCH_ENABLED and WEB_SEARCH_AVAILABLE

    print("\n=== UDCPR RAG Chatbot with Memory ===")
    print("Ask questions about the UDCPR document. Type 'exit' to quit.")
    print("This chatbot uses RAG to provide accurate information from the document.")

    if use_web_search:
        print("Web search is ENABLED for questions outside the document's scope.")

    if not SUPABASE_AVAILABLE and use_supabase:
        print("Supabase package not installed. Using in-memory chat history only.")
        use_supabase = False

    print(f"Chat memory: {'Enabled (Supabase)' if use_supabase else 'In-memory only'}\n")

    chat_history = []
    session_id = None

    # Try to initialize Supabase if using it
    if use_supabase:
        try:
            supabase = initialize_supabase()
            session_id = create_chat_session(supabase)
            print(f"Chat session created: {session_id}")
        except Exception as e:
            print(f"Supabase initialization error: {str(e)}")
            print("Falling back to in-memory chat history.")
            use_supabase = False

    while True:
        query = input("\nYou: ")
        if query.lower() in ['exit', 'quit', 'q']:
            print("\nThank you for using the UDCPR RAG Chatbot!")
            if session_id:
                print(f"Your chat session ID: {session_id}")
                print("Your conversation has been saved.")
            break

        try:
            result = generate_response(
                query=query,
                session_id=session_id,
                chat_history=chat_history,
                use_supabase=use_supabase,
                use_web_search=use_web_search
            )

            # Update local variables
            chat_history = result["chat_history"]
            session_id = result.get("session_id", session_id)

            print(f"\nAssistant: {result['response']}")
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a different question.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UDCPR RAG Chatbot with Memory and Web Search")
    parser.add_argument("--query", help="Single query mode (non-interactive)")
    parser.add_argument("--session", help="Chat session ID to continue a conversation")
    parser.add_argument("--no-memory", action="store_true", help="Disable Supabase chat memory")
    parser.add_argument("--web-search", action="store_true", help="Enable web search for questions outside document scope")
    parser.add_argument("--no-web-search", action="store_true", help="Disable web search even if enabled in environment")

    args = parser.parse_args()

    # Determine whether to use Supabase
    use_supabase = not args.no_memory

    # Determine whether to use web search
    use_web_search = None
    if args.web_search:
        use_web_search = True
    elif args.no_web_search:
        use_web_search = False
    # Otherwise, use the default from environment variables

    # Check if Supabase is available
    if not SUPABASE_AVAILABLE and use_supabase:
        print("Warning: Supabase package not installed. Using in-memory chat history only.")
        print("To enable persistent chat memory, install the Supabase package:")
        print("pip install supabase")
        use_supabase = False

    # Check if web search is available
    if use_web_search and not WEB_SEARCH_AVAILABLE:
        print("Warning: Web search module not installed or configured.")
        print("To enable web search, install the requests package and set up Google API keys:")
        print("pip install requests")
        print("Set GOOGLE_API_KEY and GOOGLE_CSE_ID in your .env file")
        use_web_search = False

    if args.query:
        # Single query mode
        result = generate_response(
            query=args.query,
            session_id=args.session if use_supabase else None,
            use_supabase=use_supabase,
            use_web_search=use_web_search
        )
        print(f"\nResponse to '{args.query}':\n")
        print(result["response"])

        # Print session ID if using Supabase
        if use_supabase and result.get("session_id"):
            print(f"\nChat session ID: {result['session_id']}")
            print("Use --session [ID] to continue this conversation.")
    else:
        # Interactive mode
        interactive_chat(use_supabase=use_supabase, use_web_search=use_web_search)
