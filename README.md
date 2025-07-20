# 🧠 Moodly Backend – FastAPI

This is the backend for **Moodly**, an AI-powered psychotherapist chatbot. Built with **FastAPI**, this backend manages user authentication via Firebase, handles chat sessions with **Redis**, and enforces questionnaire-based access control. It connects to **OpenAI** for intelligent chat responses.

---

## 🚀 Features

- 🔐 **Firebase Authentication** – Secure login & access control using Firebase ID tokens.
- 🧪 **Questionnaire System** – Users complete a questionnaire before accessing the chatbot.
- ⏳ **Cooldown Logic** – 24-hour cooldown enforced before users can retake the quiz.
- 💬 **Chat Session Management** – Stores session data in Redis for efficient retrieval.
- 🧠 **OpenAI Integration** – Sends user messages to GPT-based AI and returns smart responses.
- 🧾 **Session History API** – Allows retrieval of past chat sessions.
- 🌐 **CORS Support** – Allows safe cross-origin requests from frontend (React, etc.).

---

## 🔧 Tech Stack

- **FastAPI** – Python web framework
- **Redis** – In-memory store for session management
- **Firebase** – Auth system (Google Identity Platform)
- **OpenAI Fine-Tuned GPT-3.5 Turbo** – AI chat responses
- **Uvicorn** – ASGI server
- **Python-dotenv** – Load `.env` securely

---

## 🔌 Environment Variables

```env
OPENAI_API_KEY=your-openai-api-key
REDIS_URL=your-redis-url
FIREBASE_SERVICE_ACCOUNT_PATH=./firebase-service-account.json
ALLOWED_ORIGINS=http://localhost:3000,http://yourfrontend.com
