"""
Streamlit Cloud Entry Point for UDCPR RAG Chatbot

This file serves as the main entry point for the Streamlit Cloud deployment.
It imports and runs the chatbot_web.py module from the BOMBACLATTTT RAG FINAL directory.
"""

import os
import sys

# Add the BOMBACLATTTT RAG FINAL directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "BOMBACLATTTT RAG FINAL"))

# Import the chatbot web interface
import chatbot_web  # This will run the Streamlit app automatically
