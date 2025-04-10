@echo off
echo This will uninstall all packages listed in requirements.txt
echo.
set /p confirm=Are you sure you want to proceed? (y/n): 
if /i "%confirm%" neq "y" goto :end

echo.
echo Uninstalling packages...
pip uninstall -y streamlit pandas numpy groq langchain langchain-groq langchain-community langchain-huggingface langchain-chroma chromadb python-dotenv python-docx PyPDF2 sentence-transformers
echo.
echo Uninstallation complete.

:end
pause
