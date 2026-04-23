"""
User authentication and management system
"""

import sqlite3
import hashlib
import os
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

DB_PATH = "chatbot_users.db"

class UserManager:
    """Manage user accounts and authentication"""
    
    @staticmethod
    def init_db():
        """Initialize database schema"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            ''')
            
            # Create chat_history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    user_message TEXT NOT NULL,
                    bot_response TEXT NOT NULL,
                    intent TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(" Database initialized successfully")
        except Exception as e:
            logger.error(f" Database initialization failed: {e}")
            raise
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def register(username: str, email: str, password: str) -> Dict[str, Any]:
        """
        Register new user
        
        Args:
            username: Username (must be unique)
            email: Email address (must be unique)
            password: Password (will be hashed)
        
        Returns:
            {
                "success": bool,
                "user_id": int (if success),
                "message": str
            }
        """
        try:
            # Validate input
            if not username or len(username) < 3:
                return {"success": False, "message": "Tên đăng nhập phải có ít nhất 3 ký tự"}
            
            if not email or "@" not in email:
                return {"success": False, "message": "Địa chỉ email không hợp lệ"}
            
            if not password or len(password) < 6:
                return {"success": False, "message": "Mật khẩu phải có ít nhất 6 ký tự"}
            
            # Hash password
            password_hash = UserManager.hash_password(password)
            
            # Insert into database
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO users (username, email, password_hash)
                    VALUES (?, ?, ?)
                ''', (username, email, password_hash))
                
                conn.commit()
                user_id = cursor.lastrowid
                
                logger.info(f" User registered: {username} (ID: {user_id})")
                return {
                    "success": True,
                    "user_id": user_id,
                    "message": "Đăng ký thành công"
                }
            
            except sqlite3.IntegrityError as e:
                if "username" in str(e):
                    return {"success": False, "message": "Tên đăng nhập đã tồn tại"}
                elif "email" in str(e):
                    return {"success": False, "message": "Địa chỉ email đã tồn tại"}
                else:
                    return {"success": False, "message": str(e)}
            
            finally:
                conn.close()
        
        except Exception as e:
            logger.error(f" Registration failed: {e}")
            return {"success": False, "message": f"Đăng ký thất bại: {str(e)}"}
    
    @staticmethod
    def login(username: str, password: str) -> Dict[str, Any]:
        """
        Login user
        
        Args:
            username: Username
            password: Password
        
        Returns:
            {
                "success": bool,
                "user_id": int (if success),
                "message": str
            }
        """
        try:
            password_hash = UserManager.hash_password(password)
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id FROM users
                WHERE username = ? AND password_hash = ?
            ''', (username, password_hash))
            
            result = cursor.fetchone()
            
            if result:
                user_id = result[0]
                
                # Update last_login
                cursor.execute('''
                    UPDATE users SET last_login = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (user_id,))
                conn.commit()
                
                logger.info(f" User logged in: {username} (ID: {user_id})")
                return {
                    "success": True,
                    "user_id": user_id,
                    "message": "Đăng nhập thành công"
                }
            else:
                logger.warning(f"⚠ Login failed for: {username}")
                return {"success": False, "message": "Tên đăng nhập hoặc mật khẩu không đúng"}
            
            conn.close()
        
        except Exception as e:
            logger.error(f" Login failed: {e}")
            return {"success": False, "message": f"Đăng nhập thất bại: {str(e)}"}
    
    @staticmethod
    def get_user_info(user_id: int) -> Optional[Dict[str, Any]]:
        """Get user information"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id, username, email, created_at, last_login
                FROM users WHERE user_id = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    "user_id": result[0],
                    "username": result[1],
                    "email": result[2],
                    "created_at": result[3],
                    "last_login": result[4]
                }
            return None
        
        except Exception as e:
            logger.error(f" Error fetching user info: {e}")
            return None


class ChatHistoryManager:
    """Manage chat history storage and retrieval"""
    
    @staticmethod
    def save_message(user_id: int, user_message: str, bot_response: str, intent: str = None) -> bool:
        """
        Save chat message to history
        
        Args:
            user_id: User ID
            user_message: User's message
            bot_response: Bot's response
            intent: Detected intent (weather/activity/none)
        
        Returns:
            True if saved successfully
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO chat_history (user_id, user_message, bot_response, intent)
                VALUES (?, ?, ?, ?)
            ''', (user_id, user_message, bot_response, intent))
            
            conn.commit()
            conn.close()
            
            logger.info(f"📝 Chat saved for user {user_id}")
            return True
        
        except Exception as e:
            logger.error(f" Failed to save chat: {e}")
            return False
    
    @staticmethod
    def get_history(user_id: int, limit: int = 50) -> list:
        """
        Get chat history for user
        
        Args:
            user_id: User ID
            limit: Maximum number of messages to retrieve
        
        Returns:
            List of chat messages in chronological order
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT history_id, user_message, bot_response, intent, timestamp
                FROM chat_history
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (user_id, limit))
            
            results = cursor.fetchall()
            conn.close()
            
            history = []
            for row in reversed(results):  # Reverse to get chronological order
                history.append({
                    "id": row[0],
                    "user_message": row[1],
                    "bot_response": row[2],
                    "intent": row[3],
                    "timestamp": row[4]
                })
            
            logger.info(f"📖 Retrieved {len(history)} messages for user {user_id}")
            return history
        
        except Exception as e:
            logger.error(f" Failed to retrieve history: {e}")
            return []
    
    @staticmethod
    def clear_history(user_id: int) -> bool:
        """
        Clear all chat history for user
        
        Args:
            user_id: User ID
        
        Returns:
            True if cleared successfully
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM chat_history WHERE user_id = ?
            ''', (user_id,))
            
            count = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"🗑️  Cleared {count} messages for user {user_id}")
            return True
        
        except Exception as e:
            logger.error(f" Failed to clear history: {e}")
            return False
    
    @staticmethod
    def get_statistics(user_id: int) -> Dict[str, Any]:
        """
        Get chat statistics for user
        
        Args:
            user_id: User ID
        
        Returns:
            Statistics dictionary
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Total messages
            cursor.execute('''
                SELECT COUNT(*) FROM chat_history WHERE user_id = ?
            ''', (user_id,))
            total_messages = cursor.fetchone()[0]
            
            # Messages by intent
            cursor.execute('''
                SELECT intent, COUNT(*) as count
                FROM chat_history
                WHERE user_id = ?
                GROUP BY intent
            ''', (user_id,))
            intent_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            conn.close()
            
            return {
                "total_messages": total_messages,
                "by_intent": intent_counts
            }
        
        except Exception as e:
            logger.error(f" Failed to get statistics: {e}")
            return {"total_messages": 0, "by_intent": {}}
    
    @staticmethod
    def get_conversation_thread(user_id: int, history_id: int) -> Dict[str, Any]:
        """Get full conversation thread starting from a specific message"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''SELECT history_id, user_message, bot_response, intent, timestamp FROM chat_history WHERE user_id = ? AND history_id = ?''', (user_id, history_id))
            target = cursor.fetchone()
            if not target: return {"success": False, "message": "Message not found"}
            
            target_timestamp = target[4]
            from datetime import datetime, timedelta
            target_time = datetime.fromisoformat(target_timestamp.replace(" ", "T"))
            
            start_time = (target_time - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
            end_time = (target_time + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
            
            cursor.execute('''
                SELECT history_id, user_message, bot_response, intent, timestamp
                FROM chat_history
                WHERE user_id = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            ''', (user_id, start_time, end_time))
            
            results = cursor.fetchall()
            conn.close()
            
            # --- THUẬT TOÁN TÌM VÀ CẮT VÁCH NGĂN ---
            target_index = next((i for i, r in enumerate(results) if r[0] == history_id), -1)
            
            # 1. Quét ngược lên để tìm vách chặn trên
            start_index = 0
            for i in range(target_index, -1, -1):
                if results[i][1] == "---NEW_CHAT---":
                    start_index = i + 1
                    break
                    
            # 2. Quét xuống dưới để tìm vách chặn dưới
            end_index = len(results)
            for i in range(target_index, len(results)):
                if results[i][1] == "---NEW_CHAT---":
                    end_index = i
                    break
                    
            # Lấy đúng mảng tin nhắn ở giữa 2 vách ngăn
            valid_results = results[start_index:end_index]
            
            conversation = []
            for row in valid_results:
                conversation.append({
                    "id": row[0],
                    "user_message": row[1],
                    "bot_response": row[2],
                    "intent": row[3],
                    "timestamp": row[4]
                })
            
            return {
                "success": True,
                "target_id": history_id,
                "message_count": len(conversation),
                "conversation": conversation
            }
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    @staticmethod
    def get_all_sessions(user_id: int, limit: int = 20) -> list:
        """Get all chat sessions grouped by date and boundary markers"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT history_id, user_message, bot_response, intent, timestamp
                FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?
            ''', (user_id, limit * 5))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results: return []
            
            from datetime import datetime, timedelta
            sessions = []
            current_session = None
            session_gap = timedelta(minutes=30)
            
            for row in reversed(results):
                # NẾU GẶP VÁCH NGĂN -> ĐÓNG GÓI PHIÊN CŨ, BỎ QUA TIN NHẮN NÀY
                if row[1] == "---NEW_CHAT---":
                    if current_session is not None:
                        sessions.append(current_session)
                        current_session = None
                    continue
                
                msg_time = datetime.fromisoformat(row[4].replace(" ", "T"))
                
                if current_session is None:
                    current_session = {
                        "start_time": row[4],
                        "end_time": row[4],
                        "first_message": row[1][:40] + "..." if len(row[1]) > 40 else row[1],
                        "message_count": 1,
                        "latest_history_id": row[0]
                    }
                else:
                    prev_time = datetime.fromisoformat(current_session["end_time"].replace(" ", "T"))
                    if msg_time - prev_time > session_gap:
                        sessions.append(current_session)
                        if len(sessions) >= limit: break
                        current_session = {
                            "start_time": row[4], "end_time": row[4],
                            "first_message": row[1][:40] + "..." if len(row[1]) > 40 else row[1],
                            "message_count": 1, "latest_history_id": row[0]
                        }
                    else:
                        current_session["end_time"] = row[4]
                        current_session["message_count"] += 1
            
            if current_session and len(sessions) < limit:
                sessions.append(current_session)
            
            return sessions[::-1]
        except Exception as e:
            return []