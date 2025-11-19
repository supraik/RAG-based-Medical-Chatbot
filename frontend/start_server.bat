@echo off
REM HaleAI Frontend Server Launcher
REM Starts a simple HTTP server on port 5500

echo ============================================
echo   HaleAI Medical Chatbot - Frontend Server
echo ============================================
echo.
echo Starting server on http://localhost:5500
echo Press Ctrl+C to stop the server
echo.
echo Opening browser in 3 seconds...
echo.

timeout /t 3 /nobreak >nul
start http://localhost:5500/index.html

python -m http.server 5500
