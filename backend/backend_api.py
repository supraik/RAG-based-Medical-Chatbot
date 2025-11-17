"""
FastAPI Backend for HaleAI - Replaces Gradio
Integrates with existing chatbot.py backend
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import asyncio
import logging
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing chatbot
from chatbot import HaleAI
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="HaleAI Medical Chatbot API",
    description="REST API for HaleAI Medical Assistant",
    version="1.0.0"
)

# CORS Configuration - Allow your Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://localhost:3001",
        "http://localhost:5500",  # Live Server
        "http://127.0.0.1:5500",  # Live Server alternate
        "https://yourdomain.com"  # Production domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chatbot instance
chatbot: Optional[HaleAI] = None
# Store conversation histories per session
conversations: Dict[str, List[Dict]] = {}
# Store analytics data
analytics_data = {
    "total_queries": 0,
    "average_latency": 0,
    "model_accuracy": 94.2,
    "active_sessions": 0
}

# Pydantic Models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    temperature: Optional[float] = 0.3
    max_tokens: Optional[int] = 1024

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict]
    num_sources: int
    status: str
    session_id: str
    metadata: Dict

class AnalyticsResponse(BaseModel):
    accuracy: float
    avg_latency: float
    total_queries: int
    active_users: int
    recent_accuracy: List[Dict]
    recent_latency: List[Dict]
    weekly_queries: List[Dict]

# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup"""
    global chatbot
    try:
        logger.info("Initializing HaleAI backend...")
        chatbot = HaleAI()
        chatbot.connect()
        logger.info("✅ HaleAI backend ready")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down HaleAI backend...")

# Health Check
@app.get("/api/health")
async def health_check():
    """Check if the backend is running"""
    return {
        "status": "healthy" if chatbot else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model": Config.GEMINI_MODEL,
        "vector_store": Config.PINECONE_INDEX_NAME
    }

# Chat Endpoint (Non-streaming)
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Handle chat requests
    Returns complete response at once
    """
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        # Get or create session
        session_id = request.session_id or f"session_{int(datetime.now().timestamp())}"
        history = conversations.get(session_id, [])
        
        # Query chatbot
        start_time = datetime.now()
        response = chatbot.query(
            user_question=request.message,
            history=history,
            use_reranking=True
        )
        latency = (datetime.now() - start_time).total_seconds()
        
        # Update conversation history
        conversations[session_id] = response.get("history", history)
        
        # Update analytics
        analytics_data["total_queries"] += 1
        analytics_data["average_latency"] = (
            (analytics_data["average_latency"] * (analytics_data["total_queries"] - 1) + latency * 1000) 
            / analytics_data["total_queries"]
        )
        
        # Format sources for frontend
        sources = [
            {
                "content": doc.page_content[:200] + "...",
                "page": doc.metadata.get("page", "N/A"),
                "source": doc.metadata.get("source", "Unknown"),
                "chunk": doc.metadata.get("chunk", 0)
            }
            for doc in response.get("sources", [])
        ]
        
        return ChatResponse(
            answer=response["answer"],
            sources=sources,
            num_sources=response["num_sources"],
            status=response["status"],
            session_id=session_id,
            metadata={
                **response.get("metadata", {}),
                "latency_ms": round(latency * 1000, 2)
            }
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Streaming Chat Endpoint
@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Handle chat requests with streaming response
    Sends chunks of the response as they're generated
    """
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    async def generate_stream():
        try:
            session_id = request.session_id or f"session_{int(datetime.now().timestamp())}"
            history = conversations.get(session_id, [])
            
            # Get response with conversation history for context
            response = chatbot.query(
                user_question=request.message,
                history=history,  # Pass existing history for context
                use_reranking=True
            )
            
            # Update conversation history with new exchange
            updated_history = response.get("history", history)
            conversations[session_id] = updated_history
            
            # Stream the answer preserving formatting (paragraphs, newlines)
            answer = response["answer"]
            
            # Split by sentences/paragraphs while preserving structure
            import re
            # Split on sentence endings but keep the structure
            parts = re.split(r'(\n\n|\n|\. )', answer)
            
            for part in parts:
                if part:  # Skip empty parts
                    chunk = {
                        "type": "token",
                        "content": part if part in ['\n\n', '\n', '. '] else part + ('. ' if not part.endswith(('\n', '.', '!', '?', ':', ',')) else ''),
                        "done": False
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.03)  # Faster streaming
            
            # Send final metadata
            final_chunk = {
                "type": "complete",
                "done": True,
                "session_id": session_id,
                "num_sources": response["num_sources"],
                "sources": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "page": doc.metadata.get("page", "N/A"),
                        "source": doc.metadata.get("source", "Unknown")
                    }
                    for doc in response.get("sources", [])
                ]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            
            # Update conversation
            conversations[session_id] = response.get("history", history)
            
        except Exception as e:
            error_chunk = {
                "type": "error",
                "content": str(e),
                "done": True
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# WebSocket Chat Endpoint (Real-time)
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat
    """
    await websocket.accept()
    session_id = f"ws_session_{int(datetime.now().timestamp())}"
    analytics_data["active_sessions"] += 1
    
    try:
        logger.info(f"WebSocket connected: {session_id}")
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            if not message:
                continue
            
            # Get history
            history = conversations.get(session_id, [])
            
            # Send typing indicator
            await websocket.send_json({"type": "typing", "status": True})
            
            # Query chatbot
            response = chatbot.query(
                user_question=message,
                history=history,
                use_reranking=True
            )
            
            # Stop typing
            await websocket.send_json({"type": "typing", "status": False})
            
            # Send response
            await websocket.send_json({
                "type": "message",
                "answer": response["answer"],
                "sources": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "page": doc.metadata.get("page", "N/A"),
                        "source": doc.metadata.get("source", "Unknown")
                    }
                    for doc in response.get("sources", [])
                ],
                "num_sources": response["num_sources"],
                "status": response["status"],
                "session_id": session_id
            })
            
            # Update history
            conversations[session_id] = response.get("history", history)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        analytics_data["active_sessions"] -= 1
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await websocket.send_json({"type": "error", "message": str(e)})

# Analytics Endpoint
@app.get("/api/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """
    Get real-time analytics data
    """
    import random
    
    # Generate mock time-series data (replace with real metrics)
    recent_accuracy = [
        {"time": f"{i*2}m", "value": 85 + random.random() * 10}
        for i in range(10)
    ]
    
    recent_latency = [
        {"time": f"{i*2}m", "value": 200 + random.random() * 100}
        for i in range(10)
    ]
    
    weekly_queries = [
        {"day": day, "count": random.randint(50, 150)}
        for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ]
    
    return AnalyticsResponse(
        accuracy=analytics_data["model_accuracy"],
        avg_latency=analytics_data["average_latency"],
        total_queries=analytics_data["total_queries"],
        active_users=analytics_data["active_sessions"],
        recent_accuracy=recent_accuracy,
        recent_latency=recent_latency,
        weekly_queries=weekly_queries
    )

# Get Conversation History
@app.get("/api/conversations/{session_id}")
async def get_conversation(session_id: str):
    """
    Retrieve conversation history for a session
    """
    history = conversations.get(session_id, [])
    return {
        "session_id": session_id,
        "messages": history,
        "count": len(history)
    }

# Clear Conversation
@app.delete("/api/conversations/{session_id}")
async def clear_conversation(session_id: str):
    """
    Clear conversation history for a session
    """
    if session_id in conversations:
        del conversations[session_id]
        return {"status": "success", "message": "Conversation cleared"}
    return {"status": "not_found", "message": "Session not found"}

# Export Conversation
@app.get("/api/conversations/{session_id}/export")
async def export_conversation(session_id: str, format: str = "json"):
    """
    Export conversation in JSON or TXT format
    """
    history = conversations.get(session_id, [])
    
    if not history:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if format == "json":
        return {
            "session_id": session_id,
            "messages": history,
            "exported_at": datetime.now().isoformat()
        }
    elif format == "txt":
        text = f"HaleAI Conversation - Session: {session_id}\n"
        text += f"Exported: {datetime.now().isoformat()}\n\n"
        text += "=" * 70 + "\n\n"
        
        for msg in history:
            text += f"[{msg.get('timestamp', 'N/A')}] You: {msg.get('user', '')}\n"
            text += f"[{msg.get('timestamp', 'N/A')}] Dr. Hale: {msg.get('bot', '')}\n\n"
        
        return {"content": text, "format": "txt"}
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Use 'json' or 'txt'")

# System Information
@app.get("/api/system/info")
async def get_system_info():
    """
    Get system configuration and status
    """
    return {
        "model": Config.GEMINI_MODEL,
        "temperature": Config.GEMINI_TEMPERATURE,
        "max_tokens": Config.GEMINI_MAX_OUTPUT_TOKENS,
        "vector_store": {
            "provider": "Pinecone",
            "index": Config.PINECONE_INDEX_NAME,
            "dimension": Config.PINECONE_DIMENSION,
            "region": Config.PINECONE_REGION
        },
        "embeddings": {
            "model": Config.EMBEDDING_MODEL
        },
        "rag": {
            "initial_retrieval": Config.RETRIEVER_K,
            "after_reranking": Config.RERANK_TOP_K,
            "chunk_size": Config.CHUNK_SIZE,
            "chunk_overlap": Config.CHUNK_OVERLAP
        },
        "status": "online" if chatbot else "offline"
    }

# Run with: uvicorn backend_api:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )