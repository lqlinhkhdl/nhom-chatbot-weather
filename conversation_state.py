"""
Conversation State Management for Stateful Chatbot
Dùng LangGraph State để lưu ngữ cảnh cuộc trò chuyện
"""

import logging
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class ConversationIntent(str, Enum):
    """Loại ý định người dùng"""
    WEATHER = "weather"
    ACTIVITY = "activity"
    FORECAST = "forecast"
    SMALL_TALK = "small_talk"
    UNKNOWN = "unknown"
    CLARIFICATION_NEEDED = "clarification_needed"

class ConversationStatus(str, Enum):
    """Trạng thái cuộc trò chuyện"""
    STARTING = "starting"
    IN_PROGRESS = "in_progress"
    WAITING_FOR_INPUT = "waiting_for_input"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class WeatherData:
    """Dữ liệu thời tiết"""
    location: str = ""
    temperature_c: float = 0.0
    condition: str = ""
    humidity: int = 0
    wind_kph: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WeatherData':
        return cls(**{k: v for k, v in data.items() if k in asdict(cls())})

@dataclass
class ConversationMessage:
    """Một message trong cuộc trò chuyện"""
    role: str  # "user" hoặc "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    intent: str = ""  # Ý định khi message này được gửi
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ConversationState:
    """
    State của một cuộc trò chuyện - LƯU TRỮ NGỮ CẢNH
    Cái này được truyền xuyên suốt các node trong LangGraph
    """
    
    # Session info
    user_id: int
    session_id: str  # Unique conversation session
    
    # Conversation history
    messages: List[ConversationMessage] = field(default_factory=list)
    
    # Current state
    current_status: ConversationStatus = ConversationStatus.STARTING
    current_intent: ConversationIntent = ConversationIntent.UNKNOWN
    
    # Context (nhớ được thông tin từ các lần trước)
    target_location: Optional[str] = None  # Địa điểm user quan tâm
    target_activity: Optional[str] = None  # Loại hoạt động (leo núi, biển, v.v)
    weather_data: Optional[WeatherData] = None  # Thời tiết đã fetch
    # Thông tin đã biết về thời tiết, hoạt động, địa điểm (nếu có)
    forecast_data: Optional[dict] = None
    # Pending question from bot
    pending_question: Optional[str] = None  # Cái mà bot đang chờ trả lời
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    recommendation_data: list = None  # Dữ liệu gợi ý đã fetch (nếu có)
    # History of intents (để theo dõi conversational flow)
    intent_history: List[str] = field(default_factory=list)
    
    # Error tracking
    error_message: Optional[str] = None
    error_count: int = 0
    
    def add_message(self, role: str, content: str, intent: str = "") -> None:
        """Thêm message vào history"""
        msg = ConversationMessage(role=role, content=content, intent=intent)
        self.messages.append(msg)
        self.updated_at = datetime.now().isoformat()
        logger.debug(f"[{self.session_id}] Added {role} message")
    
    def update_location(self, location: str) -> None:
        """Update location context"""
        self.target_location = location
        self.updated_at = datetime.now().isoformat()
        logger.info(f"[{self.session_id}] Updated target_location: {location}")
    
    def update_intent(self, intent: ConversationIntent) -> None:
        """Update current intent"""
        self.current_intent = intent
        self.intent_history.append(intent.value)
        self.updated_at = datetime.now().isoformat()
        logger.info(f"[{self.session_id}] Updated intent: {intent.value}")
    
    def update_status(self, status: ConversationStatus) -> None:
        """Update conversation status"""
        self.current_status = status
        self.updated_at = datetime.now().isoformat()
        logger.debug(f"[{self.session_id}] Status: {status.value}")
    
    def get_context_summary(self) -> str:
        """Tóm tắt context hiện tại để dùng cho LLM prompt"""
        context_parts = []
        
        if self.target_location:
            context_parts.append(f"Địa điểm: {self.target_location}")
        
        if self.target_activity:
            context_parts.append(f"Hoạt động: {self.target_activity}")
        
        if self.current_intent != ConversationIntent.UNKNOWN:
            context_parts.append(f"Ý định: {self.current_intent.value}")
        
        if self.weather_data and self.weather_data.location:
            context_parts.append(
                f"Thời tiết: {self.weather_data.location} - "
                f"{self.weather_data.temperature_c}°C, {self.weather_data.condition}"
            )
        
        if context_parts:
            return "Ngữ cảnh hiện tại: " + " | ".join(context_parts)
        return ""
    
    def get_conversation_history_text(self, max_turns: int = 5) -> str:
        """Lấy lịch sử trò chuyện để dùng cho LLM context"""
        recent_messages = self.messages[-max_turns*2:]  # Mỗi turn = 1 user + 1 bot = 2 messages
        
        history_text = ""
        for msg in recent_messages:
            role_prefix = "User:" if msg.role == "user" else "Assistant:"
            history_text += f"{role_prefix} {msg.content}\n"
        
        return history_text.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization"""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "current_status": self.current_status.value,
            "current_intent": self.current_intent.value,
            "target_location": self.target_location,
            "target_activity": self.target_activity,
            "weather_data": self.weather_data.to_dict() if self.weather_data else None,
            "pending_question": self.pending_question,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "intent_history": self.intent_history,
            "error_message": self.error_message,
            "error_count": self.error_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationState':
        """Restore from dict"""
        state = cls(
            user_id=data.get("user_id", 0),
            session_id=data.get("session_id", ""),
        )
        
        # Restore messages
        for msg_data in data.get("messages", []):
            msg = ConversationMessage(
                role=msg_data.get("role", ""),
                content=msg_data.get("content", ""),
                timestamp=msg_data.get("timestamp", ""),
                intent=msg_data.get("intent", ""),
            )
            state.messages.append(msg)
        
        # Restore enums
        state.current_status = ConversationStatus(data.get("current_status", "starting"))
        state.current_intent = ConversationIntent(data.get("current_intent", "unknown"))
        
        # Restore other fields
        state.target_location = data.get("target_location")
        state.target_activity = data.get("target_activity")
        state.pending_question = data.get("pending_question")
        state.created_at = data.get("created_at", datetime.now().isoformat())
        state.updated_at = data.get("updated_at", datetime.now().isoformat())
        state.intent_history = data.get("intent_history", [])
        state.error_message = data.get("error_message")
        state.error_count = data.get("error_count", 0)
        
        # Restore weather data
        weather_dict = data.get("weather_data")
        if weather_dict:
            state.weather_data = WeatherData.from_dict(weather_dict)
        
        return state


class SessionManager:
    """
    Quản lý conversation sessions cho tất cả users
    Lưu state vào memory (có thể extend thành Redis sau)
    """
    
    # In-memory store: {session_id: ConversationState}
    _sessions: Dict[str, ConversationState] = {}
    
    @classmethod
    def create_session(cls, user_id: int, session_id: str) -> ConversationState:
        """Tạo mới conversation session"""
        state = ConversationState(user_id=user_id, session_id=session_id)
        cls._sessions[session_id] = state
        logger.info(f"✓ Created session {session_id} for user {user_id}")
        return state
    
    @classmethod
    def get_session(cls, session_id: str) -> Optional[ConversationState]:
        """Lấy session"""
        return cls._sessions.get(session_id)
    
    @classmethod
    def update_session(cls, session_id: str, state: ConversationState) -> None:
        """Update session state"""
        cls._sessions[session_id] = state
        state.updated_at = datetime.now().isoformat()
        logger.debug(f"✓ Updated session {session_id}")
    
    @classmethod
    def delete_session(cls, session_id: str) -> None:
        """Delete session"""
        if session_id in cls._sessions:
            del cls._sessions[session_id]
            logger.info(f"✓ Deleted session {session_id}")
    
    @classmethod
    def get_all_sessions(cls) -> Dict[str, ConversationState]:
        """Get all sessions (for monitoring)"""
        return cls._sessions
    
    @classmethod
    def persist_session(cls, session_id: str, filepath: str) -> None:
        """Persist session to file (for recovery)"""
        state = cls.get_session(session_id)
        if state:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"✓ Persisted session {session_id} to {filepath}")
    
    @classmethod
    def load_session(cls, filepath: str, session_id: str) -> Optional[ConversationState]:
        """Load session from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            state = ConversationState.from_dict(data)
            cls._sessions[session_id] = state
            logger.info(f"✓ Loaded session {session_id} from {filepath}")
            return state
        except Exception as e:
            logger.error(f"✗ Failed to load session: {e}")
            return None
    
    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Get session manager statistics"""
        return {
            "active_sessions": len(cls._sessions),
            "sessions": {
                sess_id: {
                    "user_id": state.user_id,
                    "messages": len(state.messages),
                    "status": state.current_status.value,
                    "intent": state.current_intent.value,
                    "location": state.target_location,
                }
                for sess_id, state in cls._sessions.items()
            }
        }
