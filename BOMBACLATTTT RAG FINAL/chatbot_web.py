"""
Minimalist Web Interface for UDCPR RAG Chatbot

This module provides a clean, minimalist web interface for the UDCPR RAG Chatbot
using Streamlit with persistent chat memory via Supabase.
Optimized for deployment on Streamlit Cloud.
"""

import os
import uuid
import streamlit as st
import time
import openai
from datetime import datetime

# Configure environment variables from Streamlit secrets if available
if hasattr(st, 'secrets'):
    # Set environment variables from Streamlit secrets
    for key in st.secrets:
        if key != '_streamlit_config':  # Skip Streamlit's internal config
            os.environ[key] = st.secrets[key]

# Try to import Supabase functions, but provide fallbacks if not available
try:
    from rag_chatbot import (
        generate_response, create_chat_prompt, format_context_from_results,
        MODEL, MAX_HISTORY_MESSAGES, TOP_K_RESULTS, WEB_SEARCH_ENABLED, WEB_SEARCH_AVAILABLE
    )
    from query_interface import search_pinecone
    from supabase_config import initialize_supabase, get_chat_history, format_chat_history_for_openai, save_message
    SUPABASE_AVAILABLE = True
except ImportError:
    # Fallback to basic functionality without Supabase
    from rag_chatbot import generate_response, WEB_SEARCH_ENABLED, WEB_SEARCH_AVAILABLE
    SUPABASE_AVAILABLE = False

    # Define dummy functions
    def initialize_supabase():
        st.warning("Supabase package not installed. Using in-memory chat history only.")
        return None

    def get_chat_history(supabase, session_id, limit=10):
        return []

    def format_chat_history_for_openai(messages):
        return messages

    def save_message(supabase, session_id, role, content):
        return {}

# Set page configuration
st.set_page_config(
    page_title="UDCPR RAG Chatbot",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a minimalist design
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    .stButton>button {
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e6f7ff;
        border-left: 5px solid #1890ff;
    }
    .chat-message.assistant {
        background-color: #f6ffed;
        border-left: 5px solid #52c41a;
    }
    .chat-message .message-content {
        display: flex;
        margin-top: 0;
    }
    .avatar {
        min-width: 20px;
        margin-right: 10px;
        font-size: 20px;
    }
    .message {
        flex-grow: 1;
    }
    h1, h2, h3 {
        color: #333;
    }
    .stMarkdown a {
        color: #1890ff;
        text-decoration: none;
    }
    .stMarkdown a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "use_supabase" not in st.session_state:
    # Default to using Supabase if available and credentials exist
    st.session_state.use_supabase = SUPABASE_AVAILABLE and bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_API_KEY"))

if "use_web_search" not in st.session_state:
    # Default to using web search if enabled in environment
    st.session_state.use_web_search = WEB_SEARCH_ENABLED and WEB_SEARCH_AVAILABLE

# Try to initialize Supabase and load existing chat if we have a session ID
if st.session_state.use_supabase and not st.session_state.messages:
    try:
        supabase = initialize_supabase()

        # Create a new session if we don't have one
        if not st.session_state.session_id:
            # Check URL parameters for session_id
            if "session_id" in st.query_params:
                st.session_state.session_id = st.query_params["session_id"]

                # Load chat history from Supabase
                db_messages = get_chat_history(supabase, st.session_state.session_id)
                if db_messages:
                    st.session_state.messages = db_messages
                    st.session_state.chat_history = format_chat_history_for_openai(db_messages)
            else:
                # Generate a new session ID and create the session in Supabase
                from supabase_config import create_chat_session
                try:
                    session_id = create_chat_session(supabase)
                    st.session_state.session_id = session_id
                    st.info(f"Created new chat session: {session_id}")
                except Exception as e:
                    st.error(f"Failed to create chat session: {str(e)}")
                    st.session_state.use_supabase = False

    except Exception as e:
        st.warning(f"Failed to connect to Supabase: {str(e)}")
        st.session_state.use_supabase = False

# App header
st.title("📚 UDCPR Document Assistant")
st.markdown("""
This chatbot uses Retrieval Augmented Generation (RAG) to provide accurate information from the
Unified Development Control and Promotion Regulations (UDCPR) for Maharashtra State.
""")

# Sidebar with information
with st.sidebar:
    st.header("About")

    # Adjust the description based on available features
    features = []
    features.append("**OpenAI GPT-4o** for generating responses")
    features.append("**Pinecone** vector database for document retrieval")
    features.append("**RAG (Retrieval Augmented Generation)** to provide accurate information")

    if SUPABASE_AVAILABLE and st.session_state.use_supabase:
        features.append("**Supabase** for persistent chat memory")
    else:
        features.append("**In-memory chat history**")

    if WEB_SEARCH_AVAILABLE and st.session_state.use_web_search:
        features.append("**Web search** for questions outside the document's scope")

    features_text = "\n".join([f"- {feature}" for feature in features])

    st.markdown(f"""
    This chatbot uses:
    {features_text}

    The chatbot has access to the complete UDCPR document and can answer questions about:
    - Building regulations
    - Zoning requirements
    - Development control rules
    - And more...
    """)

    # Display chat memory status
    st.header("Chat Memory")

    if not SUPABASE_AVAILABLE:
        st.warning("Supabase package not installed. Using in-memory chat history only.")
        memory_status = "In-memory only (Supabase not available)"
    else:
        memory_status = "Enabled (Supabase)" if st.session_state.use_supabase else "In-memory only"

    st.markdown(f"**Status:** {memory_status}")

    # Display web search status
    st.header("Web Search")

    if not WEB_SEARCH_AVAILABLE:
        st.warning("Web search not available. Make sure Google API keys are configured.")
        web_search_status = "Not available"
    else:
        web_search_status = "Enabled" if st.session_state.use_web_search else "Disabled"

        # Toggle for web search
        if st.checkbox("Enable web search for questions outside document scope",
                       value=st.session_state.use_web_search):
            st.session_state.use_web_search = True
        else:
            st.session_state.use_web_search = False

    st.markdown(f"**Status:** {web_search_status}")

    if st.session_state.use_web_search:
        st.info("Web search will be used ONLY when the document doesn't contain relevant information.")

    # Display session management if Supabase is available
    if SUPABASE_AVAILABLE and st.session_state.use_supabase:
        # Display current session info
        if st.session_state.session_id:
            st.markdown(f"**Current Session ID:** `{st.session_state.session_id}`")

            # Create a shareable link
            # For Streamlit Cloud, we can use the deployed URL
            if hasattr(st, 'secrets') and 'STREAMLIT_DEPLOYED_URL' in st.secrets:
                base_url = st.secrets['STREAMLIT_DEPLOYED_URL']
            else:
                base_url = st.query_params.get("host", "")
                if not base_url:
                    base_url = "http://localhost:8501"  # Default Streamlit URL
            share_url = f"{base_url}?session_id={st.session_state.session_id}"

            st.markdown("**Share this conversation:**")
            st.code(share_url, language="text")

        # Session management expander
        with st.expander("Session Management"):
            # Try to list available sessions
            try:
                from supabase_config import initialize_supabase, list_chat_sessions

                supabase = initialize_supabase()
                if supabase:
                    sessions = list_chat_sessions(supabase, limit=5)

                    if sessions:
                        st.markdown("**Recent Sessions:**")
                        for session in sessions:
                            # Skip current session in the list
                            if session["session_id"] == st.session_state.session_id:
                                continue

                            # Format the timestamp
                            updated_at = session.get("updated_at", "")
                            if updated_at:
                                try:
                                    # Convert to datetime and format
                                    dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                                    updated_at = dt.strftime("%Y-%m-%d %H:%M")
                                except:
                                    pass

                            # Create a button to load this session
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"Session from {updated_at}")
                            with col2:
                                if st.button("Load", key=f"load_{session['session_id']}"):
                                    # Update URL parameter and reload
                                    st.experimental_set_query_params(session_id=session["session_id"])
                                    st.session_state.session_id = session["session_id"]
                                    st.session_state.chat_history = []
                                    st.session_state.messages = []
                                    st.experimental_rerun()
                    else:
                        st.info("No previous sessions found.")
            except Exception as e:
                st.error(f"Error loading sessions: {str(e)}")

        # Toggle for Supabase usage
        if st.checkbox("Disable persistent memory", value=False):
            st.session_state.use_supabase = False
            st.experimental_rerun()

    st.header("Sample Questions")
    st.markdown("""
    - What are the parking requirements for residential buildings?
    - What are the fire safety regulations for high-rise buildings?
    - What is the definition of Floor Space Index (FSI)?
    - What are the requirements for green buildings?
    """)

# Function to display chat messages
def display_chat_message(_, content, avatar=None):
    # Note: role parameter is kept for backward compatibility but not used
    with st.container():
        col1, col2 = st.columns([1, 12])
        with col1:
            if avatar:
                st.markdown(f"<div class='avatar'>{avatar}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='message'>{content}</div>", unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about UDCPR..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)

    # Display assistant response with streaming
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        try:
            # Check if we should use Supabase
            use_supabase = SUPABASE_AVAILABLE and st.session_state.use_supabase

            # Initialize variables
            full_response = ""
            session_id = st.session_state.session_id if use_supabase else None

            # Use streaming for faster perceived response time
            try:
                # Check if we should use web search
                use_web_search = st.session_state.use_web_search and WEB_SEARCH_AVAILABLE

                # Get relevant context from Pinecone
                results = search_pinecone(prompt, top_k=TOP_K_RESULTS)
                context = format_context_from_results(results)

                # Initialize web search context
                web_search_context = None

                # Check if we need to use web search
                if use_web_search:
                    # First, check if this is a simple greeting or basic interaction
                    greeting_patterns = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon",
                                        "good evening", "how are you", "what's up", "howdy"]
                    is_greeting = any(pattern in prompt.lower() for pattern in greeting_patterns)

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

                    is_udcpr_related = any(keyword in prompt.lower() for keyword in udcpr_keywords)
                    needs_external_info = any(keyword in prompt.lower() for keyword in external_info_keywords)
                    force_web_search = any(phrase in prompt.lower() for phrase in force_web_search_phrases)

                    # Check if we have any relevant results at all, regardless of score
                    has_any_results = len(results) > 0

                    # ALWAYS use web search for queries that explicitly ask for recent/latest information
                    if force_web_search:
                        has_relevant_results = False
                    # Don't use web search for greetings, very short queries, or standard UDCPR queries
                    elif is_greeting or len(prompt.strip()) < 10 or (is_udcpr_related and not needs_external_info and has_any_results):
                        has_relevant_results = True  # Pretend we have relevant results to skip web search
                    else:
                        # Check if RAG results are relevant enough
                        has_relevant_results = False
                        if results:
                            for result in results:
                                if result.get("score", 0) > 0.75:  # Using threshold from rag_chatbot.py
                                    has_relevant_results = True
                                    break

                    # If no relevant results and not a basic interaction, use web search
                    if not has_relevant_results:
                        from web_search import perform_web_search, format_search_results_for_context
                        message_placeholder.markdown("Searching the web for information...")
                        web_results = perform_web_search(prompt, num_results=3)
                        if web_results:
                            web_search_context = format_search_results_for_context(web_results)

                # Create chat prompt with web search context if available
                messages = create_chat_prompt(prompt, context, web_search_context, st.session_state.chat_history)

                # Stream the response
                message_placeholder.empty()
                stream = openai.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.5,
                    max_tokens=800,
                    stream=True
                )

                # Display the streaming response
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.01)  # Small delay for smoother streaming

                # Display final response without cursor
                message_placeholder.markdown(full_response)

                # Save to Supabase if using it
                if use_supabase and session_id:
                    try:
                        # Save user message
                        save_message(initialize_supabase(), session_id, "user", prompt)

                        # Save assistant response
                        save_message(initialize_supabase(), session_id, "assistant", full_response)
                    except Exception as e:
                        st.warning(f"Failed to save to Supabase: {str(e)}")

                # Update in-memory chat history
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})

                # Limit chat history
                if len(st.session_state.chat_history) > MAX_HISTORY_MESSAGES:
                    st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY_MESSAGES:]

                # Add assistant message to display history
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                # Fallback to non-streaming if streaming fails
                st.warning("Streaming failed, falling back to standard response")

                # Check if we should use web search
                use_web_search = st.session_state.use_web_search and WEB_SEARCH_AVAILABLE

                # Generate response with or without Supabase integration
                if use_supabase:
                    result = generate_response(
                        query=prompt,
                        session_id=session_id,
                        chat_history=st.session_state.chat_history,
                        use_supabase=True,
                        use_web_search=use_web_search
                    )
                else:
                    # Fallback to in-memory chat history
                    result = generate_response(
                        query=prompt,
                        chat_history=st.session_state.chat_history,
                        use_supabase=False,
                        use_web_search=use_web_search
                    )

                # Update session state
                st.session_state.chat_history = result["chat_history"]
                if use_supabase and "session_id" in result:
                    st.session_state.session_id = result["session_id"]

                # Display the response
                message_placeholder.markdown(result["response"])

                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": result["response"]})

        except Exception as e:
            message_placeholder.markdown(f"Error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

# Add buttons for conversation management
col1, col2 = st.columns(2)

# Clear conversation button
if col1.button("Clear Conversation"):
    # Reset all conversation state
    st.session_state.chat_history = []
    st.session_state.messages = []

    # Create a new session if using Supabase
    if SUPABASE_AVAILABLE and st.session_state.use_supabase:
        try:
            from supabase_config import create_chat_session
            supabase = initialize_supabase()
            session_id = create_chat_session(supabase)
            st.session_state.session_id = session_id
        except Exception as e:
            st.warning(f"Failed to create new session: {str(e)}")
            st.session_state.use_supabase = False

    st.experimental_rerun()

# New conversation button (keeps Supabase enabled but starts fresh)
if SUPABASE_AVAILABLE and st.session_state.use_supabase and col2.button("New Conversation"):
    # Reset conversation but keep Supabase enabled
    st.session_state.chat_history = []
    st.session_state.messages = []

    # Create a new session
    try:
        from supabase_config import create_chat_session
        supabase = initialize_supabase()
        session_id = create_chat_session(supabase)
        st.session_state.session_id = session_id
    except Exception as e:
        st.warning(f"Failed to create new session: {str(e)}")
        st.session_state.use_supabase = False

    st.experimental_rerun()

# Footer
footer_components = ["OpenAI GPT-4o", "Pinecone"]

if SUPABASE_AVAILABLE and st.session_state.use_supabase:
    footer_components.append("Supabase")

if WEB_SEARCH_AVAILABLE and st.session_state.use_web_search:
    footer_components.append("Web Search")

footer_text = ", ".join(footer_components)
st.markdown(f"""
---
*Powered by {footer_text}*
""")
