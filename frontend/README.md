# HaleAI Frontend

A simple, standalone HTML/React chatbot interface for the HaleAI Medical Chatbot.

## Features

- ✅ Pure HTML + React CDN (no build step required)
- ✅ Real-time streaming chat responses
- ✅ Markdown rendering (bold, italic, code, lists)
- ✅ Conversation history
- ✅ Analytics dashboard
- ✅ Dark mode
- ✅ Responsive design with Tailwind CSS

## Quick Start

### Option 1: Python HTTP Server (Recommended)

```bash
# From the frontend folder
python -m http.server 5500
```

Then open: http://localhost:5500/index.html

### Option 2: VS Code Live Server

1. Install "Live Server" extension in VS Code
2. Right-click `index.html` → "Open with Live Server"

### Option 3: Any static file server

Just serve the `index.html` file - no compilation needed!

## Requirements

- Backend API running on http://localhost:8000
- Modern web browser (Chrome, Firefox, Edge, Safari)

## Backend Setup

Make sure your FastAPI backend is running:

```bash
cd ../backend
python backend_api.py
```

## File Structure

```
frontend/
├── index.html          # Main application (self-contained)
├── index.html.backup   # Backup copy
└── README.md          # This file
```

## API Endpoints Used

- `GET  /api/health` - Health check
- `POST /api/chat` - Non-streaming chat
- `POST /api/chat/stream` - Streaming chat (SSE)
- `GET  /api/analytics` - Analytics data
- `GET  /api/conversations/{id}/export` - Export conversation

## Configuration

Update `API_BASE_URL` in `index.html` (line ~48) if your backend is on a different port:

```javascript
const API_BASE_URL = "http://localhost:8000";
```

## Features in Detail

### Streaming Chat

Real-time token-by-token responses using Server-Sent Events (SSE).

### Markdown Rendering

Automatically renders:

- **Bold text** (`**text**`)
- _Italic text_ (`*text*`)
- `Inline code` (`` `code` ``)
- Lists (bullet points)
- Headings (### Heading)

### Session Management

Conversation history is maintained server-side using session IDs.

### Export

Export conversations as JSON or TXT format.

## Troubleshooting

### "Failed to fetch" errors

- Ensure backend is running: `cd backend && python backend_api.py`
- Check CORS is enabled for your frontend port in `backend/backend_api.py`

### Streaming not working

- Check browser console for errors
- Verify `/api/chat/stream` endpoint is accessible

### Markdown not rendering

- Clear browser cache and refresh
- Check `renderMarkdown()` function is defined in index.html

## Development Notes

This frontend uses CDN-loaded libraries for simplicity:

- React 18 (production build)
- Tailwind CSS 3
- Babel Standalone (for JSX transformation)

No npm, webpack, or build process required! Perfect for quick deployment and testing.
