@echo off
echo UDCPR RAG Chatbot Web Interface
echo ==============================

REM Install Streamlit if not already installed
pip install streamlit

REM Run the Streamlit app
streamlit run chatbot_web.py

pause
