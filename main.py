import logging
import json
import os
from typing import Optional, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from requests.help import main
from config import (
    LOG_LEVEL, LOG_FORMAT, APP_TITLE, APP_VERSION,
    TEMPLATES_DIR, validate_config
)
from user_manager import UserManager, ChatHistoryManager
from state_graph import run_conversation
from conversation_state import ConversationState, ConversationMessage, SessionManager
from weather_api import WeatherAPI, WeatherAPIError, get_weather, get_forecast
from recommender import is_good_weather, WeatherFilters

# Cấu hình logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
# Validate configuration on startup
config_errors = validate_config()
if config_errors:
    for error in config_errors:
        logger.warning(error)

# Initialize database on startup
UserManager.init_db()

# Initialize weather cache on startup
app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình thư mục templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Load dữ liệu địa điểm từ file JSON
def load_locations_task1() -> List[dict]:
    """Load locations.json file for Task 1 (Weather queries)"""
    try:
        with open("D:\Data\Learn 4n2\LLM\chatbot_weather\data\locations.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f" Loaded {len(data)} locations from locations.json for Task 1")
            return data
    except FileNotFoundError:
        logger.warning("⚠ File locations.json not found! Using empty data.")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"✗ JSON parse error in locations.json: {e}")
        return []

def load_suggested_locations_task3() -> List[dict]:
    """Load Suggested_locations.json file for Task 3 (Smart Recommendations)"""
    try:
        with open("D:\Data\Learn 4n2\LLM\chatbot_weather\data\Suggested_locations.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f" Loaded {len(data)} suggestions from Suggested_locations.json for Task 3")
            return data
    except FileNotFoundError:
        logger.warning("⚠ File Suggested_locations.json not found! Using empty data.")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"✗ JSON parse error in Suggested_locations.json: {e}")
        return []

data_task1 = load_locations_task1()
data_task3 = load_suggested_locations_task3()

# Pydantic Models
class ChatRequest(BaseModel):
    user_id: int = Field(..., description="User ID")
    message: str = Field(..., min_length=1, max_length=500)

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3)
    email: str = Field(...)
    password: str = Field(..., min_length=6)

class LoginRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    success: bool
    user_id: Optional[int] = None
    message: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Home page with chat interface"""
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={}
    )

@app.post("/register", response_model=AuthResponse)
def register(req: RegisterRequest):
    """Register new user"""
    result = UserManager.register(req.username, req.email, req.password)
    return AuthResponse(
        success=result["success"],
        user_id=result.get("user_id"),
        message=result["message"]
    )

@app.post("/login", response_model=AuthResponse)
def login(req: LoginRequest):
    """Login user"""
    result = UserManager.login(req.username, req.password)
    return AuthResponse(
        success=result["success"],
        user_id=result.get("user_id"),
        message=result["message"]
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint using stateful conversation with LangGraph
    Integrates: Weather, POI, and Smart Recommendation tasks
    """
    try:
        user_id = req.user_id
        message_text = req.message.strip()
        logger.info(f"📨 User {user_id}: {message_text}")
        
        # 1. Get or create user session
        session_id = f"user_{user_id}"
        
        # Check if session exists
        state = SessionManager.get_session(session_id)
        if state is None:
            # Create new session
            state = SessionManager.create_session(user_id=user_id, session_id=session_id)
            logger.info(f" Created new session: {session_id}")
        else:
            logger.info(f" Retrieved existing session: {session_id}")
        
        # 2. Add user message to conversation state
        state.add_message(role="user", content=message_text)
        
        # 3. Run stateful conversation workflow
        logger.info(f" Running LangGraph stateful workflow...")
        final_state = await run_conversation(state) 
        
        # 4. Extract response from final state
        if final_state.messages:
            last_message = final_state.messages[-1]
            response = last_message.content
            intent = final_state.current_intent.value
        else:
            response = "Xin lỗi, không thể xử lý yêu cầu của bạn."
            intent = "unknown"
        
        logger.info(f" Bot response generated (intent: {intent})")
        
        # 5. Save to chat history
        ChatHistoryManager.save_message(user_id, message_text, response, intent)
        
        # 6. Update session in memory
        SessionManager.update_session(session_id, final_state)
        
        logger.info(f" Conversation completed and saved")
        return ChatResponse(response=response)
    
    except WeatherAPIError as e:
        logger.error(f"✗ Weather API error: {e}")
        response = "Xin lỗi, tôi không thể lấy dữ liệu thời tiết. Vui lòng thử lại sau."
        return ChatResponse(response=response, status="error")
    
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}", exc_info=True)
        response = "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại."
        return ChatResponse(response=response, status="error")
        
@app.post("/session/restore/{user_id}/{history_id}")
async def restore_session_from_history(user_id: int, history_id: int):
    """Lấy lịch sử từ DB và nạp lại vào Session hiện tại để chat tiếp"""
    try:
        # 1. Lấy thread hội thoại từ DB
        result = ChatHistoryManager.get_conversation_thread(user_id, history_id)
        if not result.get("success"):
            return {"success": False, "message": "Không tìm thấy lịch sử"}
            
        # 2. Tạo ID phiên
        session_id = f"user_{user_id}"
        
        # 3. Tạo một State mới và nạp các tin nhắn cũ vào
        state = SessionManager.create_session(user_id=user_id, session_id=session_id)
        
        for msg in result["conversation"]:
            # Nạp cặp câu hỏi - trả lời cũ vào bộ nhớ
            state.add_message(role="user", content=msg["user_message"])
            state.add_message(role="assistant", content=msg["bot_response"])
            
        # 4. Cập nhật lại Session
        SessionManager.update_session(session_id, state)
        
        return {"success": True, "message": "Đã khôi phục phiên chat thành công"}
    except Exception as e:
        logger.error(f"Error restoring session: {e}")
        return {"success": False, "message": str(e)}
# ============================================================================
# LEGACY HANDLER FUNCTIONS REMOVED
# All handling is now done via stateful conversation in run_conversation()
# from state_graph.py which properly handles async/await
# ============================================================================

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "locations_loaded": len(data_task1),
        "suggestions_loaded": len(data_task3),
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }

@app.get("/history/{user_id}")
def get_history(user_id: int, limit: int = 50):
    """Get chat history for user"""
    try:
        user_info = UserManager.get_user_info(user_id)
        if not user_info:
            raise HTTPException(status_code=404, detail="User not found")
        
        history = ChatHistoryManager.get_history(user_id, limit)
        return {
            "success": True,
            "user_id": user_id,
            "message_count": len(history),
            "history": history
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Error retrieving history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/{user_id}")
def get_stats(user_id: int):
    """Get chat statistics for user"""
    try:
        user_info = UserManager.get_user_info(user_id)
        if not user_info:
            raise HTTPException(status_code=404, detail="User not found")
        
        stats = ChatHistoryManager.get_statistics(user_id)
        return {
            "success": True,
            "user_id": user_id,
            "username": user_info["username"],
            "created_at": user_info["created_at"],
            "last_login": user_info["last_login"],
            "statistics": stats
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Error retrieving stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/history/{user_id}")
def clear_history(user_id: int):
    """Clear chat history for user"""
    try:
        user_info = UserManager.get_user_info(user_id)
        if not user_info:
            raise HTTPException(status_code=404, detail="User not found")
        
        success = ChatHistoryManager.clear_history(user_id)
        return {
            "success": success,
            "message": "Chat history cleared" if success else "Failed to clear history"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Stateful Conversation Session Management Endpoints
# ============================================================================

@app.get("/session/{user_id}")
def get_session(user_id: int):
    """Get current conversation session for user"""
    try:
        session_id = f"user_{user_id}"
        state = SessionManager.get_session(session_id)
        
        if not state:
            return {
                "success": False,
                "message": "No active session for this user"
            }
        
        return {
            "success": True,
            "session_id": session_id,
            "user_id": user_id,
            "status": state.current_status.value,
            "current_intent": state.current_intent.value,
            "target_location": state.target_location,
            "target_activity": state.target_activity,
            "message_count": len(state.messages),
            "created_at": state.created_at,
            "updated_at": state.updated_at
        }
    except Exception as e:
        logger.error(f"✗ Error retrieving session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{user_id}/history")
def get_session_history(user_id: int):
    """Get conversation history from current session"""
    try:
        session_id = f"user_{user_id}"
        state = SessionManager.get_session(session_id)
        
        if not state:
            return {
                "success": False,
                "message": "No active session for this user"
            }
        
        messages = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "intent": msg.intent
            }
            for msg in state.messages
        ]
        
        return {
            "success": True,
            "session_id": session_id,
            "user_id": user_id,
            "message_count": len(messages),
            "messages": messages
        }
    except Exception as e:
        logger.error(f"✗ Error retrieving session history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{user_id}")
def clear_session(user_id: int):
    """Clear conversation session for user"""
    try:
        session_id = f"user_{user_id}"
        SessionManager.delete_session(session_id)
        # --- THÊM DÒNG NÀY ĐỂ TẠO VÁCH NGĂN GIỮA CÁC PHIÊN TRÒ CHUYỆN, TRÁNH NHẦM LẪN LỊCH SỬ ---
        ChatHistoryManager.save_message(user_id, "---NEW_CHAT---", "---NEW_CHAT---", "system")
        return {
            "success": True,
            "message": f"Session {session_id} cleared"
        }
    except Exception as e:
        logger.error(f"✗ Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Chat History Viewing Endpoints (Xem lịch sử chi tiết)
# ============================================================================

@app.get("/conversation/{user_id}/{history_id}")
def view_conversation_thread(user_id: int, history_id: int):
    """
    View a full conversation thread around a specific message
    Lấy toàn bộ cuộc trò chuyện liên quan đến một tin nhắn cụ thể
    """
    try:
        user_info = UserManager.get_user_info(user_id)
        if not user_info:
            raise HTTPException(status_code=404, detail="User not found")
        
        result = ChatHistoryManager.get_conversation_thread(user_id, history_id)
        
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("message", "Conversation not found"))
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Error retrieving conversation thread: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{user_id}")
def get_user_sessions(user_id: int, limit: int = 20):
    """
    Get all chat sessions for user grouped by date/time
    Lấy tất cả các phiên trò chuyện được nhóm theo thời gian
    """
    try:
        user_info = UserManager.get_user_info(user_id)
        if not user_info:
            raise HTTPException(status_code=404, detail="User not found")
        
        sessions = ChatHistoryManager.get_all_sessions(user_id, limit)
        
        return {
            "success": True,
            "user_id": user_id,
            "session_count": len(sessions),
            "sessions": sessions
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Error retrieving sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/all")
def get_all_sessions():
    """Get all active sessions (for monitoring/debugging)"""
    try:
        sessions = SessionManager.get_all_sessions()
        
        session_info = []
        for session_id, state in sessions.items():
            session_info.append({
                "session_id": session_id,
                "user_id": state.user_id,
                "status": state.current_status.value,
                "intent": state.current_intent.value,
                "message_count": len(state.messages),
                "created_at": state.created_at,
                "updated_at": state.updated_at
            })
        
        return {
            "success": True,
            "total_sessions": len(sessions),
            "sessions": session_info
        }
    except Exception as e:
        logger.error(f"✗ Error retrieving all sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))