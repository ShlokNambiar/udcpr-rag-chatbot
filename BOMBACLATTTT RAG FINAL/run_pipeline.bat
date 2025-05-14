@echo off
echo RAG Pipeline for UDCPR Document
echo ==============================

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file. Please edit it with your API keys.
    copy .env.example .env
    echo.
    echo Please edit the .env file with your API keys and run this script again.
    pause
    exit
)

REM Create output directory
mkdir output 2>nul

REM Run the pipeline
python main.py --pdf "UDCPR Updated 30.01.25 with earlier provisions & corrections.pdf"

echo.
echo Pipeline completed. You can now query the system.
echo.

REM Run interactive query
python main.py --query

pause
