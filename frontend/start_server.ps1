# HaleAI Frontend Server Launcher (PowerShell)
# Starts a simple HTTP server on port 5500

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  HaleAI Medical Chatbot - Frontend Server" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting server on http://localhost:5500" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
Write-Host "Opening browser in 3 seconds..." -ForegroundColor White
Write-Host ""

Start-Sleep -Seconds 3
Start-Process "http://localhost:5500/index.html"

python -m http.server 5500
