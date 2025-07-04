import os
import time
import uuid
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
import redis
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

# Redis configuration (Upstash)
REDIS_URL = os.getenv("REDIS_URL")  # Your Upstash Redis URL
if not REDIS_URL:
    raise ValueError("REDIS_URL environment variable is required")

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "1000"))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "20"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "20"))
SESSION_TIMEOUT_HOURS = int(os.getenv("SESSION_TIMEOUT_HOURS", "24"))

# Initialize Redis client
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    # Test connection
    redis_client.ping()
    logger.info("Redis connection established successfully")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    raise

try:
        # Check if we're in production (Railway) or local development
        firebase_credentials_json = os.getenv("FIREBASE_CREDENTIALS")
             
        if firebase_credentials_json:
            # Production: Load from environment variable (JSON string)
            logger.info("Loading Firebase credentials from environment variable")
            firebase_credentials = json.loads(firebase_credentials_json)
            cred = credentials.Certificate(firebase_credentials)
        else:
            # Local development: Load from file
            logger.info("Loading Firebase credentials from file")
            firebase_service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "app/firebase-service-account.json")
            cred = credentials.Certificate(firebase_service_account_path)
        
        # Initialize Firebase Admin
        firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin initialized successfully")
        
except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin: {e}")
        raise

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Setup FastAPI
app = FastAPI(
    title="Moodly Chat API",
    version="2.0.0",
    description="AI-powered mental health chatbot API with Redis and Authentication",
    docs_url="/docs" if ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if ENVIRONMENT == "development" else None
)

# Security middleware
if ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
    )

# CORS middleware
if ENVIRONMENT == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )

# Redis key patterns
REDIS_KEYS = {
    "chat_session": "chat_session:{user_id}:{session_id}",
    "user_sessions": "user_sessions:{user_id}",
    "session_timestamp": "session_timestamp:{user_id}:{session_id}",
    "rate_limit": "rate_limit:{client_id}",
    "user_metadata": "user_metadata:{user_id}"
}

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
        banned_words = ['spam', 'test' * 10]
        if any(word in v.lower() for word in banned_words):
            raise ValueError('Message contains inappropriate content')
        return v.strip()

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str

class UserInfo(BaseModel):
    uid: str
    email: Optional[str] = None
    name: Optional[str] = None

# Redis Helper Functions
class RedisHelper:
    @staticmethod
    def get_chat_session(user_id: str, session_id: str) -> list:
        """Get chat session from Redis"""
        try:
            key = REDIS_KEYS["chat_session"].format(user_id=user_id, session_id=session_id)
            data = redis_client.get(key)
            if data:
                return json.loads(data)
            return []
        except Exception as e:
            logger.error(f"Error getting chat session: {e}")
            return []
    
    @staticmethod
    def save_chat_session(user_id: str, session_id: str, messages: list):
        """Save chat session to Redis"""
        try:
            key = REDIS_KEYS["chat_session"].format(user_id=user_id, session_id=session_id)
            redis_client.setex(key, SESSION_TIMEOUT_HOURS * 3600, json.dumps(messages))
            
            # Update session timestamp
            timestamp_key = REDIS_KEYS["session_timestamp"].format(user_id=user_id, session_id=session_id)
            redis_client.setex(timestamp_key, SESSION_TIMEOUT_HOURS * 3600, datetime.now().isoformat())
            
            # Add to user's session list
            user_sessions_key = REDIS_KEYS["user_sessions"].format(user_id=user_id)
            redis_client.sadd(user_sessions_key, session_id)
            redis_client.expire(user_sessions_key, SESSION_TIMEOUT_HOURS * 3600)
            
        except Exception as e:
            logger.error(f"Error saving chat session: {e}")
    
    @staticmethod
    def get_user_sessions(user_id: str) -> list:
        """Get all sessions for a user"""
        try:
            key = REDIS_KEYS["user_sessions"].format(user_id=user_id)
            sessions = redis_client.smembers(key)
            return list(sessions) if sessions else []
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []
    
    @staticmethod
    def delete_chat_session(user_id: str, session_id: str):
        """Delete a specific chat session"""
        try:
            # Delete session data
            session_key = REDIS_KEYS["chat_session"].format(user_id=user_id, session_id=session_id)
            timestamp_key = REDIS_KEYS["session_timestamp"].format(user_id=user_id, session_id=session_id)
            
            redis_client.delete(session_key)
            redis_client.delete(timestamp_key)
            
            # Remove from user's session list
            user_sessions_key = REDIS_KEYS["user_sessions"].format(user_id=user_id)
            redis_client.srem(user_sessions_key, session_id)
            
        except Exception as e:
            logger.error(f"Error deleting chat session: {e}")
    
    @staticmethod
    def check_rate_limit(client_id: str) -> bool:
        """Check and update rate limit"""
        try:
            key = REDIS_KEYS["rate_limit"].format(client_id=client_id)
            current_time = int(time.time())
            
            # Use Redis sorted set for efficient rate limiting
            pipe = redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, current_time - RATE_LIMIT_WINDOW)
            pipe.zcard(key)
            pipe.zadd(key, {str(current_time): current_time})
            pipe.expire(key, RATE_LIMIT_WINDOW)
            
            results = pipe.execute()
            request_count = results[1]
            
            return request_count < RATE_LIMIT_REQUESTS
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True  # Allow request if Redis fails

# Helper functions
def get_client_identifier(request: Request) -> str:
    """Get client identifier for rate limiting"""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    return request.client.host

# REPLACE the existing verify_firebase_token function with this:
def verify_firebase_token(authorization: str = Header(None)) -> UserInfo:
    """Verify Firebase authentication token"""
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = authorization.split(' ')[1]
    
    try:
        # Verify the Firebase token
        decoded_token = firebase_auth.verify_id_token(token)
        return UserInfo(
            uid=decoded_token['uid'],
            email=decoded_token.get('email'),
            name=decoded_token.get('name', decoded_token.get('email', 'User'))
        )
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    # TODO: Implement actual Firebase token verification
    # For now, we'll simulate token verification
    # In production, use Firebase Admin SDK:
    """
    try:
        import firebase_admin
        from firebase_admin import auth
        
        decoded_token = auth.verify_id_token(token)
        return UserInfo(
            uid=decoded_token['uid'],
            email=decoded_token.get('email'),
            name=decoded_token.get('name')
        )
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")
    """
    
    # Placeholder - replace with actual Firebase verification
    if token == "valid_token":
        return UserInfo(uid="test_user_123", email="test@example.com", name="Test User")
    else:
        raise HTTPException(status_code=401, detail="Invalid token")

# Dependencies
async def rate_limit_check(request: Request):
    """Rate limiting middleware"""
    client_id = get_client_identifier(request)
    
    if not RedisHelper.check_rate_limit(client_id):
        logger.warning(f"Rate limit exceeded for client: {client_id}")
        raise HTTPException(
            status_code=429, 
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds"
        )

# Routes
@app.get("/")
async def root():
    return {
        "service": "Moodly Chat API",
        "status": "healthy",
        "version": "2.0.0",
        "environment": ENVIRONMENT,
        "features": ["Redis Storage", "Firebase Auth", "Rate Limiting"]
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
    
    # Test Redis connection
    try:
        redis_client.ping()
        redis_status = "healthy"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        redis_status = "unhealthy"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "openai": openai_status,
            "redis": redis_status
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
    user: UserInfo = Depends(verify_firebase_token),
    _rate_limit: None = Depends(rate_limit_check)
):
    """Main chat endpoint with user authentication"""
    session_id = req.session_id or str(uuid.uuid4())
    client_id = get_client_identifier(request)
    
    try:
        # Get existing conversation from Redis
        messages = RedisHelper.get_chat_session(user.uid, session_id)
        
        if not messages:
            logger.info(f"New chat session created: {session_id} for user: {user.uid}")
        
        # Add user message
        user_message = {"role": "user", "content": req.message}
        messages.append(user_message)
        
        # Limit conversation history
        if len(messages) > MAX_CONVERSATION_HISTORY:
            messages = messages[-MAX_CONVERSATION_HISTORY:]
            logger.info(f"Trimmed conversation history for session: {session_id}")

        # Call OpenAI API
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:fyp-moodly:moodly-bot:BYjVg5Sx",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        
        reply = response.choices[0].message.content
        assistant_message = {"role": "assistant", "content": reply}
        messages.append(assistant_message)
        
        # Save to Redis
        RedisHelper.save_chat_session(user.uid, session_id, messages)
        
        logger.info(f"Chat response generated - User: {user.uid}, Session: {session_id}, Client: {client_id}")
        
        return {
            "session_id": session_id,
            "response": reply,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chat error - User: {user.uid}, Session: {session_id}, Client: {client_id}, Error: {str(e)}")
        
        if "rate_limit" in str(e).lower():
            raise HTTPException(status_code=429, detail="OpenAI rate limit exceeded. Please try again later.")
        elif "insufficient_quota" in str(e).lower():
            raise HTTPException(status_code=503, detail="Service temporarily unavailable. Please try again later.")
        else:
            raise HTTPException(status_code=500, detail="Failed to generate response. Please try again.")

@app.get("/sessions")
async def get_user_sessions(user: UserInfo = Depends(verify_firebase_token)):
    """Get all sessions for the authenticated user"""
    try:
        sessions = RedisHelper.get_user_sessions(user.uid)
        session_data = []
        
        for session_id in sessions:
            # Get session metadata
            timestamp_key = REDIS_KEYS["session_timestamp"].format(user_id=user.uid, session_id=session_id)
            timestamp = redis_client.get(timestamp_key)
            
            # Get first message as preview
            messages = RedisHelper.get_chat_session(user.uid, session_id)
            preview = ""
            if messages and len(messages) > 0:
                preview = messages[0].get("content", "")[:50] + "..." if len(messages[0].get("content", "")) > 50 else messages[0].get("content", "")
            
            session_data.append({
                "session_id": session_id,
                "last_activity": timestamp,
                "message_count": len(messages),
                "preview": preview
            })
        
        return {
            "sessions": session_data,
            "total": len(sessions)
        }
    except Exception as e:
        logger.error(f"Error getting user sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")

@app.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str,
    user: UserInfo = Depends(verify_firebase_token)
):
    """Clear a specific chat session for the authenticated user"""
    try:
        # Verify session belongs to user
        user_sessions = RedisHelper.get_user_sessions(user.uid)
        if session_id not in user_sessions:
            raise HTTPException(status_code=404, detail="Session not found or access denied")
        
        RedisHelper.delete_chat_session(user.uid, session_id)
        logger.info(f"Session cleared: {session_id} for user: {user.uid}")
        
        return {"message": f"Session {session_id} cleared successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear session")

@app.delete("/sessions/all")
async def clear_all_sessions(user: UserInfo = Depends(verify_firebase_token)):
    """Clear all sessions for the authenticated user"""
    try:
        sessions = RedisHelper.get_user_sessions(user.uid)
        for session_id in sessions:
            RedisHelper.delete_chat_session(user.uid, session_id)
        
        logger.info(f"All sessions cleared for user: {user.uid}")
        return {"message": f"All {len(sessions)} sessions cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing all sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear sessions")

# ADD this new endpoint after the existing /sessions endpoint
@app.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    user: UserInfo = Depends(verify_firebase_token)
):
    """Get messages for a specific session"""
    try:
        # Verify session belongs to user
        user_sessions = RedisHelper.get_user_sessions(user.uid)
        if session_id not in user_sessions:
            raise HTTPException(status_code=404, detail="Session not found or access denied")
        
        messages = RedisHelper.get_chat_session(user.uid, session_id)
        return {
            "session_id": session_id,
            "messages": messages,
            "total": len(messages)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session messages: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session messages")

@app.get("/stats")
async def get_user_stats(user: UserInfo = Depends(verify_firebase_token)):
    """Get statistics for the authenticated user"""
    try:
        sessions = RedisHelper.get_user_sessions(user.uid)
        total_messages = 0
        
        for session_id in sessions:
            messages = RedisHelper.get_chat_session(user.uid, session_id)
            total_messages += len(messages)
        
        return {
            "user_id": user.uid,
            "total_sessions": len(sessions),
            "total_messages": total_messages,
            "avg_messages_per_session": total_messages / len(sessions) if sessions else 0
        }
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

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