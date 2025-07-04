import os
import time
import uuid
import json
import logging
from collections import defaultdict
from typing import Optional, List, Dict
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, validator
from openai import OpenAI
from dotenv import load_dotenv
import redis
import firebase_admin
from firebase_admin import credentials, auth

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

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Firebase configuration
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")

# Initialize Firebase Admin SDK
if FIREBASE_CREDENTIALS_PATH and os.path.exists(FIREBASE_CREDENTIALS_PATH):
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred)
    logger.info("Firebase Admin SDK initialized")
else:
    logger.warning("Firebase credentials not found - auth will be disabled in development")

# Initialize Redis
try:
    redis_client = redis.Redis.from_url(REDIS_URL, password=REDIS_PASSWORD, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connected successfully")
except Exception as e:
    logger.error(f"Redis connection failed: {e}")
    redis_client = None

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Setup FastAPI
app = FastAPI(
    title="Moodly Chat API",
    version="1.0.0",
    description="AI-powered mental health chatbot API",
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

# Rate limiting storage (keep in-memory for simplicity)
rate_limit_storage = defaultdict(list)

# Models
class ChatRequest(BaseModel):
    message: str
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError(f'Message too long (max {MAX_MESSAGE_LENGTH} characters)')
        
        # Crisis detection keywords
        crisis_keywords = [
            'suicide', 'kill myself', 'end it all', 'not worth living',
            'want to die', 'better off dead', 'harm myself', 'kill me',
            'no point in living', 'rather be dead', 'end my life'
        ]
        
        # Check for crisis keywords
        detected_keywords = [keyword for keyword in crisis_keywords if keyword in v.lower()]
        if detected_keywords:
            logger.critical(f"CRISIS ALERT - Keywords detected: {detected_keywords} - Message: {v[:100]}...")
            # This will be handled in the chat endpoint to return crisis response
        
        # Basic content filtering
        banned_words = ['spam', 'test' * 10]
        if any(word in v.lower() for word in banned_words):
            raise ValueError('Message contains inappropriate content')
        
        return v.strip()

class ChatResponse(BaseModel):
    response: str
    status: str
    timestamp: str
    conversation_id: str
    crisis_detected: bool = False
    crisis_resources: Optional[Dict] = None

class ConversationHistory(BaseModel):
    conversations: List[Dict]
    total_conversations: int

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str

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

def detect_crisis_keywords(message: str) -> tuple[bool, list]:
    """Detect crisis-related keywords in message"""
    crisis_keywords = [
        'suicide', 'kill myself', 'end it all', 'not worth living',
        'want to die', 'better off dead', 'harm myself', 'kill me',
        'no point in living', 'rather be dead', 'end my life',
        'overdose', 'cutting myself', 'self harm', 'hurt myself'
    ]
    
    detected = [keyword for keyword in crisis_keywords if keyword in message.lower()]
    return len(detected) > 0, detected

def get_crisis_resources() -> Dict:
    """Get crisis intervention resources"""
    return {
        "message": "I'm concerned about you. Please reach out for immediate help.",
        "hotlines": {
            "National Suicide Prevention Lifeline": {
                "phone": "988",
                "text": "Text HOME to 741741",
                "available": "24/7"
            },
            "Crisis Text Line": {
                "text": "Text HOME to 741741",
                "available": "24/7"
            },
            "International Association for Suicide Prevention": {
                "website": "https://www.iasp.info/resources/Crisis_Centres/",
                "note": "Find help in your country"
            }
        },
        "immediate_actions": [
            "Call emergency services (911) if in immediate danger",
            "Go to your nearest emergency room",
            "Call a trusted friend or family member",
            "Contact your therapist or counselor if you have one"
        ],
        "online_resources": {
            "National Suicide Prevention Lifeline": "https://suicidepreventionlifeline.org/",
            "Crisis Text Line": "https://www.crisistextline.org/",
            "Mental Health America": "https://www.mhanational.org/find-help"
        }
    }
    """Generate Redis key for user conversation"""
    if conversation_id:
        return f"user:{user_id}:conversation:{conversation_id}"
    else:
        return f"user:{user_id}:conversations"

def get_user_conversations_list_key(user_id: str) -> str:
    """Generate Redis key for user's conversation list"""
    return f"user:{user_id}:conversation_list"

def store_conversation_message(user_id: str, conversation_id: str, message: dict):
    """Store a single message in Redis"""
    if not redis_client:
        logger.error("Redis not available")
        return
    
    try:
        # Get existing conversation
        conversation_key = get_user_conversation_key(user_id, conversation_id)
        conversation_data = redis_client.get(conversation_key)
        
        if conversation_data:
            conversation = json.loads(conversation_data)
        else:
            conversation = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "messages": []
            }
            
            # Add to user's conversation list
            list_key = get_user_conversations_list_key(user_id)
            redis_client.sadd(list_key, conversation_id)
        
        # Add message
        conversation["messages"].append({
            **message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit conversation history
        if len(conversation["messages"]) > MAX_CONVERSATION_HISTORY:
            conversation["messages"] = conversation["messages"][-MAX_CONVERSATION_HISTORY:]
        
        conversation["updated_at"] = datetime.now().isoformat()
        
        # Store back to Redis with TTL
        redis_client.setex(
            conversation_key, 
            timedelta(hours=SESSION_TIMEOUT_HOURS).total_seconds(),
            json.dumps(conversation)
        )
        
        logger.info(f"Message stored for user {user_id}, conversation {conversation_id}")
        
    except Exception as e:
        logger.error(f"Failed to store conversation message: {e}")

def get_conversation_history(user_id: str, conversation_id: str) -> List[dict]:
    """Get conversation history from Redis"""
    if not redis_client:
        return []
    
    try:
        conversation_key = get_user_conversation_key(user_id, conversation_id)
        conversation_data = redis_client.get(conversation_key)
        
        if conversation_data:
            conversation = json.loads(conversation_data)
            return conversation.get("messages", [])
        
        return []
        
    except Exception as e:
        logger.error(f"Failed to get conversation history: {e}")
        return []

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

async def verify_firebase_token(authorization: str = Header(None)) -> str:
    """Verify Firebase authentication token and return user ID"""
    if ENVIRONMENT == "development" and not FIREBASE_CREDENTIALS_PATH:
        # Return a test user ID in development
        return "test_user_123"
    
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(
            status_code=401, 
            detail="Missing or invalid authorization header"
        )
    
    token = authorization.split(' ')[1]
    
    try:
        # Verify the Firebase ID token
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token['uid']
        logger.info(f"Authenticated user: {user_id}")
        return user_id
        
    except auth.InvalidIdTokenError:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")
    except auth.ExpiredIdTokenError:
        raise HTTPException(status_code=401, detail="Firebase token expired")
    except Exception as e:
        logger.error(f"Firebase token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

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
    
    # Test Redis connection
    redis_status = "healthy" if redis_client and redis_client.ping() else "unhealthy"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "openai": openai_status,
            "redis": redis_status,
            "firebase": "configured" if FIREBASE_CREDENTIALS_PATH else "development_mode"
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
    conversation_id: str = None,  # Optional conversation ID
    _rate_limit: None = Depends(rate_limit_check),
    user_id: str = Depends(verify_firebase_token)
) -> ChatResponse:
    """Main chat endpoint with user authentication"""
    
    # Generate conversation ID if not provided
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    client_id = get_client_identifier(request)
    
    try:
        # Check for crisis keywords in user message
        crisis_detected, detected_keywords = detect_crisis_keywords(req.message)
        
        if crisis_detected:
            logger.critical(f"CRISIS ALERT - User: {user_id}, Keywords: {detected_keywords}")
            
            # Store the user message (important for continuity)
            user_message = {"role": "user", "content": req.message}
            store_conversation_message(user_id, conversation_id, user_message)
            
            # Generate empathetic crisis response
            crisis_response = (
                "I'm really concerned about what you're going through right now. "
                "Your feelings are valid, but I want you to know that you don't have to face this alone. "
                "Please consider reaching out to a mental health professional or crisis helpline immediately. "
                "They have trained counselors who can provide the support you need right now."
            )
            
            # Store assistant crisis response
            assistant_message = {"role": "assistant", "content": crisis_response}
            store_conversation_message(user_id, conversation_id, assistant_message)
            
            return ChatResponse(
                response=crisis_response,
                status="crisis_detected",
                timestamp=datetime.now().isoformat(),
                conversation_id=conversation_id,
                crisis_detected=True,
                crisis_resources=get_crisis_resources()
            )
        
        # Normal conversation flow
        # Get existing conversation history
        conversation_history = get_conversation_history(user_id, conversation_id)
        
        # Add user message to history
        user_message = {"role": "user", "content": req.message}
        conversation_history.append(user_message)
        
        # Store user message
        store_conversation_message(user_id, conversation_id, user_message)
        
        # Call OpenAI API with conversation history
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:fyp-moodly:moodly-bot:BYjVg5Sx",
            messages=conversation_history,
            max_tokens=500,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        
        reply = response.choices[0].message.content
        assistant_message = {"role": "assistant", "content": reply}
        
        # Store assistant message
        store_conversation_message(user_id, conversation_id, assistant_message)
        
        # Log successful interaction
        logger.info(f"Chat response generated - User: {user_id}, Conversation: {conversation_id}")
        
        return ChatResponse(
            response=reply,
            status="success",
            timestamp=datetime.now().isoformat(),
            conversation_id=conversation_id,
            crisis_detected=False,
            crisis_resources=None
        )
        
    except Exception as e:
        logger.error(f"Chat error - User: {user_id}, Conversation: {conversation_id}, Error: {str(e)}")
        
        if "rate_limit" in str(e).lower():
            raise HTTPException(status_code=429, detail="OpenAI rate limit exceeded. Please try again later.")
        elif "insufficient_quota" in str(e).lower():
            raise HTTPException(status_code=503, detail="Service temporarily unavailable. Please try again later.")
        else:
            raise HTTPException(status_code=500, detail="Failed to generate response. Please try again.")

@app.get("/conversations")
async def get_user_conversations(
    user_id: str = Depends(verify_firebase_token)
) -> ConversationHistory:
    """Get all conversations for authenticated user"""
    
    if not redis_client:
        raise HTTPException(status_code=503, detail="Database service unavailable")
    
    try:
        # Get user's conversation list
        list_key = get_user_conversations_list_key(user_id)
        conversation_ids = redis_client.smembers(list_key)
        
        conversations = []
        for conversation_id in conversation_ids:
            conversation_key = get_user_conversation_key(user_id, conversation_id)
            conversation_data = redis_client.get(conversation_key)
            
            if conversation_data:
                conversation = json.loads(conversation_data)
                # Return summary instead of full messages for list view
                conversations.append({
                    "conversation_id": conversation["conversation_id"],
                    "created_at": conversation.get("created_at"),
                    "updated_at": conversation.get("updated_at"),
                    "message_count": len(conversation.get("messages", [])),
                    "last_message": conversation.get("messages", [])[-1] if conversation.get("messages") else None
                })
        
        # Sort by updated_at (most recent first)
        conversations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        return ConversationHistory(
            conversations=conversations,
            total_conversations=len(conversations)
        )
        
    except Exception as e:
        logger.error(f"Failed to get user conversations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversations")

@app.get("/conversations/{conversation_id}")
async def get_conversation_detail(
    conversation_id: str,
    user_id: str = Depends(verify_firebase_token)
):
    """Get detailed conversation history"""
    
    try:
        conversation_history = get_conversation_history(user_id, conversation_id)
        
        if not conversation_history:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {
            "conversation_id": conversation_id,
            "messages": conversation_history,
            "message_count": len(conversation_history)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation detail: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation")

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user_id: str = Depends(verify_firebase_token)
):
    """Delete a specific conversation"""
    
    if not redis_client:
        raise HTTPException(status_code=503, detail="Database service unavailable")
    
    try:
        # Remove from conversation list
        list_key = get_user_conversations_list_key(user_id)
        redis_client.srem(list_key, conversation_id)
        
        # Delete conversation data
        conversation_key = get_user_conversation_key(user_id, conversation_id)
        redis_client.delete(conversation_key)
        
        logger.info(f"Conversation deleted - User: {user_id}, Conversation: {conversation_id}")
        
        return {"message": f"Conversation {conversation_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Failed to delete conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete conversation")

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