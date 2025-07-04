import os
import time
import uuid
import logging
from collections import defaultdict
from typing import Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, validator
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "1000"))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "20"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "20"))
SESSION_TIMEOUT_HOURS = int(os.getenv("SESSION_TIMEOUT_HOURS", "24"))

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Setup FastAPI
app = FastAPI(
    title="Moodly Chat API",
    version="1.0.0",
    description="AI-powered mental health chatbot API",
    docs_url="/docs" if ENVIRONMENT == "development" else None,  # Hide docs in production
    redoc_url="/redoc" if ENVIRONMENT == "development" else None
)

# Security middleware
if ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["yourdomain.com", "*.yourdomain.com"]  # Replace with your domain
    )

# CORS middleware
if ENVIRONMENT == "development":
    # Development: Allow all origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
else:
    # Production: Restrict origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )

# Rate limiting storage
rate_limit_storage = defaultdict(list)

# Session storage with timestamps
chat_sessions = {}
session_timestamps = {}

# Models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError(f'Message too long (max {MAX_MESSAGE_LENGTH} characters)')
        # Basic content filtering
        banned_words = ['spam', 'test' * 10]  # Add your banned words
        if any(word in v.lower() for word in banned_words):
            raise ValueError('Message contains inappropriate content')
        return v.strip()

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str

# Helper functions
def clean_old_sessions():
    """Remove sessions older than SESSION_TIMEOUT_HOURS"""
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, timestamp in session_timestamps.items():
        if current_time - timestamp > timedelta(hours=SESSION_TIMEOUT_HOURS):
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        if session_id in chat_sessions:
            del chat_sessions[session_id]
        if session_id in session_timestamps:
            del session_timestamps[session_id]
    
    if expired_sessions:
        logger.info(f"Cleaned {len(expired_sessions)} expired sessions")

def get_client_identifier(request: Request) -> str:
    """Get client identifier for rate limiting"""
    # Try to get real IP if behind proxy
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    return request.client.host

# Dependencies
async def rate_limit_check(request: Request):
    """Rate limiting middleware"""
    client_id = get_client_identifier(request)
    current_time = time.time()
    
    # Clean old requests
    rate_limit_storage[client_id] = [
        req_time for req_time in rate_limit_storage[client_id] 
        if current_time - req_time < RATE_LIMIT_WINDOW
    ]
    
    # Check rate limit
    if len(rate_limit_storage[client_id]) >= RATE_LIMIT_REQUESTS:
        logger.warning(f"Rate limit exceeded for client: {client_id}")
        raise HTTPException(
            status_code=429, 
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds"
        )
    
    # Add current request
    rate_limit_storage[client_id].append(current_time)

async def verify_firebase_token(authorization: str = Header(None)):
    """Verify Firebase authentication token (implement based on your needs)"""
    if ENVIRONMENT == "development":
        return True  # Skip auth in development
    
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = authorization.split(' ')[1]
    
    # TODO: Implement Firebase token verification
    # try:
    #     decoded_token = auth.verify_id_token(token)
    #     return decoded_token['uid']
    # except Exception as e:
    #     raise HTTPException(status_code=401, detail="Invalid token")
    
    return True  # Placeholder - implement Firebase verification

# Routes
@app.get("/")
async def root():
    return {
        "service": "Moodly Chat API",
        "status": "healthy",
        "version": "1.0.0",
        "environment": ENVIRONMENT
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        # Test OpenAI connection
        test_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        openai_status = "healthy"
    except Exception as e:
        logger.error(f"OpenAI health check failed: {e}")
        openai_status = "unhealthy"
    
    # Clean old sessions
    clean_old_sessions()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "openai": openai_status,
            "sessions": len(chat_sessions),
            "rate_limits": len(rate_limit_storage)
        },
        "config": {
            "environment": ENVIRONMENT,
            "max_message_length": MAX_MESSAGE_LENGTH,
            "rate_limit": f"{RATE_LIMIT_REQUESTS}/{RATE_LIMIT_WINDOW}s"
        }
    }

@app.post("/chat")
async def chat(
    req: ChatRequest, 
    request: Request,
    _rate_limit: None = Depends(rate_limit_check),
    _auth: bool = Depends(verify_firebase_token)
):
    """Main chat endpoint"""
    session_id = req.session_id or str(uuid.uuid4())
    client_id = get_client_identifier(request)
    
    try:
        # Clean old sessions periodically
        if len(chat_sessions) > 100:  # Clean when we have too many sessions
            clean_old_sessions()
        
        # Initialize or get session
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
            logger.info(f"New chat session created: {session_id}")
        
        session_timestamps[session_id] = datetime.now()
        
        # Add user message
        user_message = {"role": "user", "content": req.message}
        chat_sessions[session_id].append(user_message)
        
        # Limit conversation history
        if len(chat_sessions[session_id]) > MAX_CONVERSATION_HISTORY:
            chat_sessions[session_id] = chat_sessions[session_id][-MAX_CONVERSATION_HISTORY:]
            logger.info(f"Trimmed conversation history for session: {session_id}")

        # Call OpenAI API
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:fyp-moodly:moodly-bot:BYjVg5Sx",
            messages=chat_sessions[session_id],
            max_tokens=500,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        
        reply = response.choices[0].message.content
        assistant_message = {"role": "assistant", "content": reply}
        chat_sessions[session_id].append(assistant_message)
        
        # Log successful interaction
        logger.info(f"Chat response generated - Session: {session_id}, Client: {client_id}")
        
        return {
            "session_id": session_id,
            "response": reply,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chat error - Session: {session_id}, Client: {client_id}, Error: {str(e)}")
        
        # Return appropriate error based on exception type
        if "rate_limit" in str(e).lower():
            raise HTTPException(status_code=429, detail="OpenAI rate limit exceeded. Please try again later.")
        elif "insufficient_quota" in str(e).lower():
            raise HTTPException(status_code=503, detail="Service temporarily unavailable. Please try again later.")
        else:
            raise HTTPException(status_code=500, detail="Failed to generate response. Please try again.")

@app.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str,
    _auth: bool = Depends(verify_firebase_token)
):
    """Clear a specific chat session"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        if session_id in session_timestamps:
            del session_timestamps[session_id]
        logger.info(f"Session cleared: {session_id}")
        return {"message": f"Session {session_id} cleared successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/sessions/stats")
async def get_session_stats(_auth: bool = Depends(verify_firebase_token)):
    """Get session statistics (admin only)"""
    clean_old_sessions()
    
    return {
        "total_active_sessions": len(chat_sessions),
        "total_rate_limited_clients": len(rate_limit_storage),
        "oldest_session": min(session_timestamps.values()).isoformat() if session_timestamps else None,
        "newest_session": max(session_timestamps.values()).isoformat() if session_timestamps else None
    }

# Error handlers
@app.exception_handler(ValueError)
async def validation_exception_handler(request: Request, exc: ValueError):
    return ErrorResponse(
        error="validation_error",
        message=str(exc),
        timestamp=datetime.now().isoformat()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return ErrorResponse(
        error="http_error",
        message=exc.detail,
        timestamp=datetime.now().isoformat()
    )

if __name__ == "__main__":
    import uvicorn
    
    # Configuration based on environment
    if ENVIRONMENT == "production":
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            workers=4,
            access_log=True,
            log_level="info"
        )
    else:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="debug"
        )