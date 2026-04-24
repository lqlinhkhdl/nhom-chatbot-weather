import logging
import json
import random
import os
import re
from typing import Optional, Union, Set
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from datetime import datetime, timedelta
from config import LLM_MODEL, LLM_TEMPERATURE, LLM_NUM_CTX

load_dotenv()
logger = logging.getLogger(__name__)

# 1. LOAD DATA
try:
    with open('D:\Data\Learn 4n2\LLM\chatbot_weather\data\locations.json', 'r', encoding='utf-8') as f:
        locations_data = json.load(f)
    VALID_PROVINCES = set()
    for loc in locations_data:
        name = loc.get('name', '').strip()
        if name:
            VALID_PROVINCES.add(name.lower())
    logger.info(f" Loaded {len(VALID_PROVINCES)} provinces from locations.json")
except Exception as e:
    logger.warning(f" Could not load locations.json: {e}")
    VALID_PROVINCES = set()
    locations_data = []

# 2. LLM CONFIGURATION
class LLMConfig:
    """LLM Configuration"""
    MODEL = "vinallama-chat"
    TEMPERATURE = 0.0
    NUM_CTX = 4096
    TIMEOUT = 30

try:
    llm = ChatOllama(
        model=LLMConfig.MODEL,
        temperature=LLMConfig.TEMPERATURE,
        num_ctx=LLMConfig.NUM_CTX,
    )
    logger.info(f" LLM initialized: {LLMConfig.MODEL}")
except Exception as e:
    logger.error(f" Failed to initialize LLM: {e}")
    raise

# 3. UTILITY FUNCTIONS
def ask_llm(prompt: str, system_prompt: Optional[str] = None) -> str:
    """Call LLM đồng bộ bằng .invoke()"""
    try:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        messages.append(HumanMessage(content=prompt))
        
        # Sử dụng invoke thay vì ainvoke
        response = llm.invoke(messages)
        
        return response.content.strip()
    except Exception as e:
        logger.error(f" LLM call failed: {e}")
        raise
def get_required_forecast_days(question: str) -> int:
    """Tính số ngày cần gọi API"""
    question_lower = question.lower()
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    current_weekday = today.weekday()
    
    # 0. QUÉT TÌM NGÀY/THÁNG (Ví dụ: "25/4", "ngày 25-04")
    date_match = re.search(r'(?:ngày\s+)?(\d{1,2})\s*[/-]\s*(\d{1,2})', question_lower)
    if date_match:
        try:
            day = int(date_match.group(1))
            month = int(date_match.group(2))
            
            year = today.year
            if month < today.month:
                year += 1
                
            target_date = datetime(year, month, day)
            delta_days = (target_date - today).days
            
            if 0 <= delta_days <= 15:
                return delta_days + 1 
        except ValueError:
            pass 

    # 1. QUÉT TÌM CÁC THỨ TRONG TUẦN (MỚI BỔ SUNG)
    weekday_map = {
        'thứ hai': 0, 'thứ 2': 0, 'thứ ba': 1, 'thứ 3': 1,
        'thứ tư': 2, 'thứ 4': 2, 'thứ năm': 3, 'thứ 5': 3,
        'thứ sáu': 4, 'thứ 6': 4, 'thứ bảy': 5, 'thứ 7': 5,
        'chủ nhật': 6, 'cn': 6
    }
    
    for day_name, target_weekday in weekday_map.items():
        # Dùng \b để bắt chính xác cụm từ, tránh lỗi lồng chữ
        if re.search(r'\b' + day_name + r'\b', question_lower):
            delta_days = (target_weekday - current_weekday) % 7
            if delta_days == 0: 
                delta_days = 7 # Nếu hôm nay Thứ 2, hỏi Thứ 2 -> Hiểu là Thứ 2 tuần sau
            return min(delta_days + 1, 14) # Limit ở 14 ngày để không sập API

    # 2. QUÉT TÌM SỐ NGÀY BẰNG REGEX (Ví dụ: "5 ngày")
    match = re.search(r'(\d+)\s*ngày', question_lower)
    if match:
        days = int(match.group(1))
        return min(days, 14) # Đã chuẩn hóa thành 14 ngày (Limit an toàn của WeatherAPI)

    # 3. CÁC TỪ KHÓA TƯƠNG ĐỐI
    if "cuối tuần" in question_lower:
        if current_weekday == 6: return 1
        return (6 - current_weekday) + 1
            
    elif "tuần tới" in question_lower or "tuần sau" in question_lower:
        return 14 # Đã tăng lên 14 để lấy trọn vẹn tuần sau
        
    elif "ngày mốt" in question_lower or "ngày kia" in question_lower:
        return 3
        
    elif "ngày mai" in question_lower or "mai" in question_lower.split():
        return 2
        
    # Mặc định lấy 3 ngày để làm vốn
    return 3
def filter_forecast_by_time_intent(question: str, forecast_data: list) -> list:
    question_lower = question.lower()
    today = datetime.now()
    weekday = today.weekday()
    target_dates = []
    
    # 0.1. TÌM NGÀY/THÁNG CỤ THỂ (Ví dụ: 25/4)
    date_match = re.search(r'(?:ngày\s+)?(\d{1,2})\s*[/-]\s*(\d{1,2})', question_lower)
    if date_match:
        try:
            day = int(date_match.group(1))
            month = int(date_match.group(2))
            year = today.year
            if month < today.month: year += 1
            
            target_date = datetime(year, month, day)
            target_dates.append(target_date.strftime("%Y-%m-%d"))
        except ValueError:
            pass
            
    # 0.2. NẾU NGƯỜI DÙNG HỎI X NGÀY (Chỉ chạy nếu không có ngày cụ thể)
    elif re.search(r'(\d+)\s*ngày', question_lower):
        match = re.search(r'(\d+)\s*ngày', question_lower)
        num_days = min(int(match.group(1)), 16)
        for i in range(num_days):
            target_dates.append((today + timedelta(days=i)).strftime("%Y-%m-%d"))

    # ==========================================
    # 0.3. TÌM THỨ TRONG TUẦN (Bản vá lỗi đọc sai ngày)
    # ==========================================
    elif any(re.search(r'\b' + k + r'\b', question_lower) for k in [
        'thứ hai', 'thứ 2', 'thứ ba', 'thứ 3', 'thứ tư', 'thứ 4', 
        'thứ năm', 'thứ 5', 'thứ sáu', 'thứ 6', 'thứ bảy', 'thứ 7', 'chủ nhật', 'cn'
    ]):
        weekday_map = {
            'thứ hai': 0, 'thứ 2': 0, 'thứ ba': 1, 'thứ 3': 1,
            'thứ tư': 2, 'thứ 4': 2, 'thứ năm': 3, 'thứ 5': 3,
            'thứ sáu': 4, 'thứ 6': 4, 'thứ bảy': 5, 'thứ 7': 5,
            'chủ nhật': 6, 'cn': 6
        }
        for day_name, target_weekday in weekday_map.items():
            if re.search(r'\b' + day_name + r'\b', question_lower):
                # Tính khoảng cách đến ngày thứ đó
                days_ahead = (target_weekday - weekday) % 7
                if days_ahead == 0: 
                    days_ahead = 7 # Hỏi trùng ngày hôm nay thì hiểu là tuần sau
                
                # Bơm chính xác ngày đó vào danh sách đích
                target_dates.append((today + timedelta(days=days_ahead)).strftime("%Y-%m-%d"))
                break # Tìm thấy 1 thứ là đủ, thoát vòng lặp
                
    # 1. Thuật toán tính "Cuối tuần này"
    elif "cuối tuần" in question_lower:
        if weekday == 6: 
            target_dates.append(today.strftime("%Y-%m-%d"))
        else:
            days_to_sat = 5 - weekday
            days_to_sun = 6 - weekday
            target_dates.append((today + timedelta(days=days_to_sat)).strftime("%Y-%m-%d"))
            target_dates.append((today + timedelta(days=days_to_sun)).strftime("%Y-%m-%d"))
            
    # 2. Thuật toán tính "Ngày mai", "Ngày mốt"
    elif "ngày mai" in question_lower or "mai" in question_lower.split():
        target_dates.append((today + timedelta(days=1)).strftime("%Y-%m-%d"))
    elif "ngày mốt" in question_lower or "ngày kia" in question_lower:
        target_dates.append((today + timedelta(days=2)).strftime("%Y-%m-%d"))
        
    # 3. Thuật toán tính "Tuần tới / Tuần sau"
    elif "tuần tới" in question_lower or "tuần sau" in question_lower:
        for i in range(1, 8):
            target_dates.append((today + timedelta(days=i)).strftime("%Y-%m-%d"))

    # ==========================================
    # THỰC THI LỌC
    # ==========================================
    if not target_dates:
        return forecast_data[:3]
        
    # Lọc lấy những ngày có trong target_dates
    filtered_data = [day for day in forecast_data if day.get('Ngày') in target_dates]
            
    # Bơm lỗi nếu API không trả về đủ dữ liệu xa như yêu cầu
    if not filtered_data and target_dates:
        return [{
            "Ngày": target_dates[0], 
            "Nhiệt độ thấp nhất": "N/A", 
            "Nhiệt độ cao nhất": "N/A",
            "Thời tiết": "LỖI_VƯỢT_QUÁ_GIỚI_HẠN_API"
        }]
            
    return filtered_data
def remove_hallucinated_dates(text: str) -> str:
    """
    Remove hallucinated dates/times from LLM response
    LLM sometimes makes up specific dates even when data doesn't have them
    
    This function removes patterns like:
    - "ngày 20 tháng 4 năm 2021"
    - "ngày 20 tháng 4"
    - "lúc 2:30 chiều"
    - "vào ngày ... năm ..."
    - Specific year mentions like "năm 2020", "năm 2021", etc
    """
    if not text:
        return text
    
    original_text = text

    text = re.sub(
        r'(vào\s+)?ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}',
        '',
        text
    )
    
    # Pattern 1b: "ngày XX tháng XX" (date without year, still hallucinated for current weather)
    # Matches: ngày 20 tháng 4, vào ngày 21 tháng 4
    text = re.sub(
        r'(vào\s+)?ngày\s+\d{1,2}\s+tháng\s+\d{1,2}(?!\s+năm)',
        '',
        text
    )
    
    # Pattern 2: "lúc X:XX" hoặc "lúc X giờ" (time mention)
    # Matches: lúc 2:30, lúc 14:30, lúc 2:30 chiều, lúc 3 giờ chiều
    text = re.sub(
        r'lúc\s+\d{1,2}(:\d{2}|\s+giờ)?(\s+(sáng|chiều|tối|đêm))?',
        '',
        text
    )
    
    # Pattern 3: "theo múi giờ" phrase (often paired with dates)
    text = re.sub(
        r'theo\s+múi\s+giờ\s+\w+',
        '',
        text
    )
    
    # Pattern 4: Standalone year mentions in current weather context
    # Only remove if it's an old year (< 2024) and sounds out of place
    # Match: năm 2021, năm 2020, năm 2023, etc
    text = re.sub(
        r'\s+năm\s+(20(0\d|1\d|2[0-3]))\b',
        '',
        text
    )
    
    # Pattern 5: "cao nhất" paired with dates (forecast language in current weather)
    # If "cao nhất" or "thấp nhất" appears, it's likely forecast leakage
    if 'cao nhất' in text or 'thấp nhất' in text:
        # Clean up forecast language
        text = re.sub(
            r'(nhiệt độ\s+)?(cao nhất|thấp nhất)',
            '',
            text
        )
    
    # Remove multiple consecutive spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    if original_text != text:
        logger.warning(f"🗑️ Removed hallucinated dates from response")
        logger.debug(f"  Original: {original_text[:100]}")
        logger.debug(f"  Cleaned: {text[:100]}")
    
    return text

def validate_location(location_name: str) -> bool:
    """Validate if location is a real Vietnamese province"""
    if not location_name or len(location_name) < 2:
        return False
    
    location_lower = location_name.strip().lower()
    
    # Exact match
    if location_lower in VALID_PROVINCES:
        logger.info(f" Valid location: {location_name}")
        return True
    
    # Partial match
    for valid_prov in VALID_PROVINCES:
        if location_lower in valid_prov or valid_prov in location_lower:
            logger.info(f" Fuzzy match: {location_name} → {valid_prov}")
            return True
    
    logger.warning(f" Invalid location: {location_name}")
    return False

# 4. INTENT CLASSIFICATION
def detect_small_talk(question: str) -> bool:
    """Detect if user is making small talk"""
    question_lower = question.lower().strip()
    
    small_talk_keywords = {
        'xin chào', 'chào', 'hello', 'hi', 'hỏi thăm', 'hỏi han','chao buổi sáng', 'chào buổi chiều', 'chào buổi tối','chào bro', 'chào bạn', 'chào cậu', 'chào mọi người',' chào omg',
        'tạm biệt', 'bye', 'cảm ơn', 'thank', 'ok', 'vâng','được rồi',
        'xin lỗi', 'sorry', 'excuse', 'đó là tất cả', 'đủ rồi','cảm ơn bạn', 'cảm ơn bạn', 'rất cảm ơn', 'cảm ơn nhiều', 'cảm ơn rất nhiều',
    }
    
    for keyword in small_talk_keywords:
        if keyword in question_lower:
            logger.info(f"💬 Detected small talk: {question}")
            return True
    return False

def handle_small_talk(question: str) -> str:
    """Generate friendly small talk response"""
    responses = [
        "👋 Xin chào! Tôi là trợ lý du lịch và thời tiết. Bạn hỏi gì về thời tiết hoặc địa điểm du lịch nhé!",
        "😊 Chào bạn! Tôi có thể giúp bạn kiểm tra thời tiết hoặc tìm địa điểm tham quan thú vị. Hỏi tôi đi!",
        "🌦️ Hỏi tôi về thời tiết ở Việt Nam nhé!",
        '🌞 Chào bạn! Tôi có thể giúp bạn với thông tin thời tiết hoặc gợi ý địa điểm du lịch. Hỏi tôi đi nào!',
    ]
    return random.choice(responses)

def classify_intent(question: str) -> str:
    """Classify user intent: weather, activity, small_talk, or none"""
    # Check for small talk first
    if detect_small_talk(question):
        return "small_talk"
    
    question_lower = question.lower()
    
    # Keywords for weather
    forecast_keywords = {
        'ngày mai', 'ngày mốt', 'hôm sau', 'tuần tới', 'tuần sau',
        'sắp tới', 'tương lai', 'mấy ngày', 'cuối tuần', 'dự báo',
        'dự báo thời tiết', 'thời tiết ngày mai', 'thời tiết tuần tới', 'thời tiết sắp tới',
        'dự báo thời tiết trong 3 ngày', 'dự báo thời tiết trong 7 ngày','dự báo thời tiết cuối tuần này'
        ,'dự báo thời tiết tuần sau','dự báo thời tiết ngày mốt','dự báo thời tiết hôm sau',
        'thứ hai', 'thứ hai tới', 'thứ 2', 'thứ ba', 'thứ 3', 
        'thứ tư', 'thứ 4', 'thứ năm', 'thứ 5', 'thứ sáu', 'thứ 6', 
        'thứ bảy', 'thứ 7', 'chủ nhật', 'cn','thứ hai tuần sau', 'thứ ba tuần sau', 'thứ tư tuần sau', 'thứ năm tuần sau', 'thứ sáu tuần sau', 'thứ bảy tuần sau', 'chủ nhật tuần sau',
        'thì sao', 'thế còn', 'còn',
    }
    weather_keywords = {
        'thời tiết', 'trời', 'nóng', 'lạnh', 'mưa', 'gió', 'nắng', 'mây',
        'nhiệt độ', 'độ c', 'sương mù', 'mù mịt', 'bão', 'tuyết', 'sấm sét',
        'hôm nay', 'bây giờ', 'hiện tại', 'bao nhiêu', 'như thế nào',
        'có nóng', 'có lạnh', 'có mưa', 'được không', 'được hay không',
        'đi được', 'được đi', 'nên mặc', 'mặc gì', 'dạo', 'ra ngoài', 
        'đi chơi', 'chơi được không', 'ổn không', 'được không', 'đi được không',
        'nên đi đâu', 'đi đâu', 'đi đâu chơi', 'đi đâu đẹp', 'đi đâu hay',
        'đi đâu thú vị', 'đi đâu vui', 'đi đâu tốt', 'đi đâu đẹp nhất', 'đi đâu hay nhất', 'đi đâu thú vị nhất',
        # các từ hay sai chính tả của người dùng
        'thòi tiwet', 'thòi tiết', 'thòi tieet', 'thòi tiêt', 'thòi tết',
        'thòi tiêt', 'thòi tiêt', 'thòi tieet', 'thòi tiêt', 'thòi tét',
        'thoi tiwet', 'thoi tiết', 'thoi tieet', 'thoi tiêt', 'thoi tét',
        'thơi tiect', 'thơi tiết', 'thơi tieet', 'thơi tiêt', 'thơi tét',
    }
    
    # Keywords for activity
    activity_keywords = {
        'leo', 'tắm', 'biển', 'gợi ý', 'hoạt động', 'địa điểm', 'nơi',
        'tham quan', 'du lịch', 'dã ngoại', 'thích hợp', 'tốt để', 'hay để',
        'phù hợp', 'đi bộ', 'picnic', 'adventure', 'recommend', 'chùa',
        'đền', 'hang', 'non', 'thác', 'hồ', 'gợi ý đi', 'nơi nào',
        'đi chơi', 'chơi', 'ổn không', 'được không', 'đi được', 'nên đi', 'đi đâu', 
        'đi đâu chơi', 'đi đâu đẹp', 'đi đâu hay', 'đi đâu thú vị', 'đi đâu vui', 'đi đâu tốt', 
        'đi đâu đẹp nhất', 'đi đâu hay nhất', 'đi đâu thú vị nhất'
    }
    
    forecast_count = sum(1 for kw in forecast_keywords if kw in question_lower)
    weather_count = sum(1 for kw in weather_keywords if kw in question_lower)
    activity_count = sum(1 for kw in activity_keywords if kw in question_lower)
    
    # Ưu tiên forecast nếu có từ khóa dự báo
    if forecast_count > 0:
        logger.info(f"ℹ Intent: forecast ({forecast_count} matches)")
        return "forecast"
    elif weather_count > activity_count:
        logger.info(f"ℹ Intent: weather ({weather_count} matches)")
        return "weather"
    elif activity_count > 0:
        logger.info(f"ℹ Intent: activity ({activity_count} matches)")
        return "activity"
    
    logger.info(f"ℹ Intent: unknown")
    return "none"

# 5. LOCATION EXTRACTION
def extract_location(question: str) -> str:
    """Extract location from question (Sử dụng Keyword + LLM Fallback)"""
    question_lower = question.lower()
    
    # BƯỚC 1: Tìm nhanh trong danh sách Tỉnh/Thành từ locations.json
    try:
        sorted_locations = sorted(VALID_PROVINCES, key=len, reverse=True)
        for location in sorted_locations:
            if re.search(r'\b' + re.escape(location.lower()) + r'\b', question_lower):
                logger.info(f"📍 Extracted location (Exact Match): {location.title()}")
                return location.title()
    except NameError:
        pass
            
    # BƯỚC 2: Các từ khóa viết tắt phổ biến
    fallback_locations = {
        # Direct cities
        'hà nội', 'hà nôi', 'hanoi', 
        'hồ chí minh', 'ho chi minh', 'hcm', 'sài gòn', 'saigon',
        'đà nẵng', 'da nang', 'danang',
        
        # North - 14 provinces
        'hà giang', 'ha giang',
        'cao bằng', 'cao bang',
        'bắc kạn', 'bac kan',
        'tuyên quang', 'tuyen quang',
        'lạng sơn', 'lang son',
        'bắc giang', 'bac giang',
        'thái nguyên', 'thai nguyen',
        'yên bái', 'yen bai',
        'sơn la', 'son la',
        'phú thọ', 'phu tho',  
        'vĩnh phúc', 'vinh phuc',  
        'hà nam', 'ha nam',
        'hòa bình', 'hoa binh',
        'ninh bình', 'ninh binh',
        
        # Central - 11 provinces
        'thanh hóa', 'thanh hoa',
        'nghệ an', 'nghe an',
        'hà tĩnh', 'ha tinh',
        'quảng bình', 'quang binh',
        'quảng trị', 'quang tri',
        'thừa thiên huế', 'thua thien hue', 'huế', 'hue',
        'quảng nam', 'quang nam',
        'quảng ngãi', 'quang ngai',
        'bình định', 'binh dinh',
        'phú yên', 'phu yen',
        'khánh hòa', 'khanh hoa',
        'ninh thuận', 'ninh thuan',  
        'bình thuận', 'binh thuan',  
        
        # Highlands - 5 provinces
        'kon tum', 'kontum',
        'gia lai',
        'đắk lắk', 'dak lak',
        'đắk nông', 'dak nong',
        'lâm đồng', 'lam dong',
        
        # Southeast - 6 provinces
        'bình phước', 'binh phuoc',
        'bình dương', 'binh duong',
        'đồng nai', 'dong nai',
        'bà rịa vũng tàu', 'ba ria vung tau',
        
        # Mekong - 12 provinces
        'long an',
        'tiền giang', 'tien giang',
        'bến tre', 'ben tre',
        'trà vinh', 'tra vinh',
        'vĩnh long', 'vinh long',
        'an giang',
        'kiên giang', 'kien giang',
        'cần thơ', 'can tho', 'cantho',
        'hậu giang', 'hau giang', 
        'bạc liêu', 'bac lieu',
        'cà mau', 'ca mau',
        'sóc trăng', 'soc trang',  
        
        # Others
        'hải phòng', 'hai phong',
        'hải dương', 'hai duong',
        'sa pa', 'sapa',
        'nha trang', 'quy nhơn', 'quy nhon',
        'phú quốc', 'phu quoc',
        'điện biên', 'dien bien', 
        'lai châu', 'lai chau'
    }
    sorted_fallbacks = sorted(fallback_locations, key=len, reverse=True)
    for location in sorted_fallbacks:
        if re.search(r'\b' + re.escape(location) + r'\b', question_lower):
            if location in ['hcm', 'sài gòn', 'saigon', 'ho chi minh']: return "Hồ Chí Minh"
            if location in ['hanoi', 'hà nôi']: return "Hà Nội"
            if location in ['danang', 'da nang']: return "Đà Nẵng"
            if location in ['hue']: return "Thừa Thiên Huế"
            logger.info(f"📍 Extracted location (Fallback Match): {location.title()}")
            return location.title()
            
    # BƯỚC 3: DÙNG LLM ĐỂ TRÍCH XUẤT (Giải quyết các địa danh lạ như Cát Hải, Mộc Châu...)
    logger.info(f" Keyword match failed. Falling back to LLM NER...")
    
    system_prompt = """Trích xuất tên địa danh từ câu hỏi. 
Nếu có địa danh, chỉ in ra tên. Nếu không có, in ra đúng 1 chữ: NONE.
KHÔNG giải thích. KHÔNG in lại câu hỏi."""
    
    prompt = f"Câu hỏi: '{question}'\nĐịa danh:"
    
    try:
        extracted = ask_llm(prompt, system_prompt).strip()
        extracted = extracted.replace('"', '').replace("'", '').strip()
        
        # 1. BỘ LỌC RỖNG
        if not extracted or extracted.upper() in ["NONE", "KHÔNG CÓ", "NULL", ""]:
            return ""
            
        # ==========================================
        # 🛡️ 2. BỘ LỌC ĐỘ DÀI (CHỐNG LẢM NHẢM)
        # ==========================================
        # Tên địa danh VN tối đa khoảng 5-6 từ. Nếu dài hơn -> Đang hát nhép Prompt!
        if len(extracted.split()) > 6 or len(extracted) > 30:
            logger.warning(f" 🗑️ LLM lảm nhảm quá dài, tự động hủy bỏ: {extracted}")
            return ""
            
        # ==========================================
        # 🛡️ 3. BỘ LỌC TỪ KHÓA RÁC
        # ==========================================
        garbage_words = ["thứ", "ngày", "nhiệm vụ", "ví dụ", "câu hỏi", "trả lời"]
        if any(word in extracted.lower() for word in garbage_words):
            logger.warning(f" 🗑️ LLM ảo giác dính từ khóa rác: {extracted}")
            return ""
                
        logger.info(f"📍 Extracted location (LLM NER): {extracted.title()}")
        return extracted.title()
            
    except Exception as e:
        logger.error(f" LLM extraction failed: {e}")

    logger.warning(f" No location found in question")
    return ""

# 6. WEATHER DATA EXTRACTION
def extract_weather_data(weather: dict, province_name: str = None) -> dict:
    """Extract weather data - FIXED to handle errors properly"""
    try:
        # Handle error response from API
        if "error" in weather:
            logger.error(f" Weather API error: {weather.get('error')}")
            return {
                "error": weather.get("error", "Unknown error"),
                "location": province_name or "Unknown"
            }
        
        current = weather.get("current", {})
        location = weather.get("location", {})
        
        if not current:
            logger.error(" No current weather data")
            return {"error": "No weather data available"}
        
        # Get location name
        location_name = province_name or location.get("name", "Unknown")
        
        # Get temperature (MUST HANDLE NULL/MISSING)
        temp = current.get("temp_c")
        if temp is None:
            logger.warning(" Temperature is None!")
            temp = 25  # Default fallback
        
        # Extract condition
        condition = current.get("condition", {})
        condition_text = condition.get("text", "Clear") if condition else "Clear"
        
        cleaned = {
            "location": location_name,
            "temperature_c": temp,
            "condition": condition_text,
            "humidity": current.get("humidity", "N/A"),
            "wind_kph": current.get("wind_kph", "N/A"),
            "cloud_percent": current.get("cloud", "N/A"),
        }
        
        logger.info(f" Weather: {location_name} - {temp}°C, {condition_text}")
        return cleaned
    
    except Exception as e:
        logger.error(f" Error extracting weather: {e}", exc_info=True)
        return {"error": f"Failed to parse weather: {str(e)}"}
def extract_forecast_data(weather: dict, province_name: str = None) -> dict:
    """Extract forecast data"""
    try:
        if "error" in weather:
            return {"error": weather.get("error"), "location": province_name or "Unknown"}
        
        location_name = province_name or weather.get("location", {}).get("name", "Unknown")
        forecast_days = weather.get("forecast", {}).get("forecastday", [])
        
        if not forecast_days:
            return {"error": "No forecast data available"}
        
        days_data = []
        for day in forecast_days:
            date = day.get("date", "")
            day_info = day.get("day", {})
            condition = day_info.get("condition", {}).get("text", "Clear")
            max_temp = day_info.get("maxtemp_c")
            min_temp = day_info.get("mintemp_c")
            
            days_data.append({
                "Ngày": date,
                "Nhiệt độ cao nhất": f"{max_temp}°C",
                "Nhiệt độ thấp nhất": f"{min_temp}°C",
                "Thời tiết": condition
            })
            
        logger.info(f" Forecast extracted for {location_name} ({len(days_data)} days)")
        return {
            "location": location_name,
            "forecast": days_data
        }
    except Exception as e:
        logger.error(f" Error extracting forecast: {e}", exc_info=True)
        return {"error": f"Failed to parse forecast: {str(e)}"}
    
def extract_location_list_data(places: list) -> list:
    """Extract place data for recommendations"""
    try:
        simplified = []
        for place in places[:5]:
            simplified.append({
                "name": place.get("ten_dia_danh", ""),
                "province": place.get("tinh", ""),
                "type": ", ".join(place.get("categories", []))
            })
        
        logger.info(f" Extracted {len(simplified)} place(s)")
        return simplified
    
    except Exception as e:
        logger.error(f" Error extracting places: {e}")
        return []

# 7. ANSWER GENERATION
def generate_answer(question: str, data: Union[dict, list], location_override: str = None) -> str:
    """Generate answer handling current weather, forecast, and recommendations"""
    
    # 1. SỬA LẠI LOGIC NHẬN DIỆN DỮ LIỆU: Bổ sung "temperature_c" để nhận diện data từ state
    is_error = isinstance(data, dict) and "error" in data and "current" not in data and "temperature_c" not in data
    is_forecast = isinstance(data, dict) and ("forecast" in data or "Ngày" in str(data))
    is_weather = isinstance(data, dict) and ("current" in data or "temperature_c" in data) and not is_forecast
    is_place_list = isinstance(data, list)
    
    # Handle errors
    if is_error:
        error_msg = data.get('error', 'Unknown error')
        return f" Không lấy được dữ liệu: {error_msg}. Vui lòng thử lại."
    
    # 2. XỬ LÝ DỮ LIỆU DỰ BÁO (BẰNG THUẬT TOÁN)
    if is_forecast:
        cleaned_data = extract_forecast_data(data, location_override) if "forecast" in data else data
        loc_name = cleaned_data.get('location', 'địa điểm này')
        
        from datetime import datetime
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_str = today.strftime("%d/%m/%Y") # Rút gọn, bỏ năm để AI khỏi bịa
        weekdays_vn = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
        
        full_forecast_list = cleaned_data.get("forecast", [])
        filtered_list = filter_forecast_by_time_intent(question, full_forecast_list)
        
        forecast_lines = [f"DỮ LIỆU DỰ BÁO TẠI {loc_name.upper()}:"]
        
        for day in filtered_list:
            date_str = day.get('Ngày', '')
            thu_str = ""
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                # Đổi format thành Thứ (DD/MM) để triệt tiêu lỗi năm 2021
                thu_str = f"{weekdays_vn[dt.weekday()]} ({dt.strftime('%d/%m/%Y')})"
            except:
                thu_str = date_str
            
            forecast_lines.append(
                f"- {thu_str}: Từ {day.get('Nhiệt độ thấp nhất')} đến {day.get('Nhiệt độ cao nhất')}, {day.get('Thời tiết')}"
            )
            
        raw_data_text = "\n".join(forecast_lines)
        
        # KIỂM TRA BẪY LỖI API (Từ hàm filter_forecast)
        if "LỖI_VƯỢT_QUÁ" in raw_data_text:
            system_prompt = f"""Bạn là trợ lý thời tiết. HÔM NAY LÀ: {weekdays_vn[today.weekday()]}, Ngày {today_str}.
NHIỆM VỤ: Hãy xin lỗi người dùng một cách khéo léo vì hệ thống chỉ có thể dự báo tối đa 14 ngày, không có dữ liệu cho ngày họ yêu cầu."""
            prompt_title = "LỖI GIỚI HẠN DỮ LIỆU"
        else:
            # PROMPT CHUẨN: NGẮN GỌN, CHÍNH XÁC, CÓ VÍ DỤ
            system_prompt = f"""Bạn là một phát thanh viên thời tiết chuyên nghiệp, chính xác.
HÔM NAY LÀ: {weekdays_vn[today.weekday()]}, Ngày {today_str}.

NHIỆM VỤ: Đọc [WeatherAPI] và báo cáo lại thông tin. Trả lời thẳng vào vấn đề, không dài dòng.

LUẬT THÉP (BẮT BUỘC):
1. CHỈ được dùng năm {today.year}.
2. CHỈ đọc những ngày và nhiệt độ có trong dữ liệu. KHÔNG tự bịa thêm tình trạng thời tiết nếu dữ liệu không ghi.
3. Gọi đúng tên địa điểm là "{loc_name}".
4. Hãy tự động sửa chính tả của người dùng nếu họ gõ sai (VD: "ha nôi" -> "Hà Nội").

MẪU VÍ DỤ 1:
Dữ liệu: DỰ BÁO TẠI NAM ĐỊNH: - Thứ Ba (28/04/2026): Từ 23.9°C đến 31.9°C, Mưa rào
Phát thanh viên: Dự báo thời tiết tại Nam Định vào Thứ Ba (28/04/2026): Trời có mưa rào, nhiệt độ dao động từ 23.9°C đến 31.9°C.

MẪU VÍ DỤ 2:
Dữ liệu: DỰ BÁO TẠI HÀ NỘI: - Thứ Bảy (25/04/2026): Từ 21.4°C đến 29.2°C, Nhiều mây
Phát thanh viên: Thời tiết Hà Nội vào Thứ Bảy (25/04/2026) dự báo sẽ nhiều mây, nhiệt độ từ 21.4°C đến 29.2°C."""
            prompt_title = "Dữ liệu dự báo thời tiết"
        cleaned_data = raw_data_text  # Ghi đè thành string để truyền xuống dưới
    # 3. XỬ LÝ DỮ LIỆU THỜI TIẾT HIỆN TẠI
    elif is_weather:
        if "current" in data:
            cleaned_data = extract_weather_data(data, location_override)
        else:
            cleaned_data = data
            if location_override:
                cleaned_data["location"] = location_override
        
        loc_name = cleaned_data.get('location', 'địa điểm này')
        
        # SỬ DỤNG FEW-SHOT PROMPTING VÀ LOẠI BỎ TỪ PHỦ ĐỊNH
        system_prompt = f """ Bạn là một phát thanh viên thời tiết. Nhiệm vụ của bạn là đọc dữ liệu và báo cáo thành một câu tự nhiên.
LUẬT TỐI CAO: Bắt buộc chép chính xác 100% cụm từ "{loc_name}" vào vị trí tên địa điểm.
LUẬT CHÍNH TẢ: Hãy tự động sửa chính tả của người dùng nếu họ gõ sai, gõ thiếu, không có dấu (VD: "ha nôi" → "Hà Nội", "phu thi" → "Phú Thọ", "thòi tiwet" → "thời tiết"), nhưng khi trả lời thì phải viết đúng chính tả.
MẪU VÍ DỤ:
Dữ liệu: Địa điểm: Cầu Giấy | Nhiệt độ: 30°C | Tình trạng: Mưa rào
Phát thanh viên: Thời tiết tại Cầu Giấy hiện tại là 30°C, trời có mưa rào.

Dữ liệu: Địa điểm: Quận 1 | Nhiệt độ: 25°C | Tình trạng: Có mây
Phát thanh viên: Thời tiết tại Quận 1 hiện tại là 25°C, trời có mây.
"""
        prompt_title = "THỜI TIẾT HIỆN TẠI"
        
        # Ép dữ liệu thành 1 dòng siêu ngắn gọn để AI không bị ngợp
        cleaned_data = (
            f"Địa điểm: {loc_name} | "
            f"Nhiệt độ: {cleaned_data.get('temperature_c', 0)}°C | "
            f"Tình trạng: {cleaned_data.get('condition', '')} | "
            f"Độ ẩm: {cleaned_data.get('humidity', 0)}% | "
            f"Sức gió: {cleaned_data.get('wind_kph', 0)} km/h"
        )

    # 4. XỬ LÝ DANH SÁCH ĐỊA ĐIỂM
    elif is_place_list:
        # Kiểm tra nếu chưa được trích xuất thì mới trích xuất
        cleaned_data = extract_location_list_data(data) if data and "ten_dia_danh" in data[0] else data
        system_prompt = """ROLE: Bạn là hướng dẫn viên du lịch.
LANGUAGE: Tiếng Việt, thân thiện, hào hứng.
RULES: 1. Gợi ý các địa điểm trong danh sách phù hợp với thời tiết.
        2. Hãy dùng độ °C TUYỆT ĐỐ KHÔNG DÙNG ĐỘ F HAY °F để mô tả nhiệt độ, và dùng các từ như "nóng", "lạnh", "mưa" để mô tả thời tiết.
        3. Nếu có thể, hãy gợi ý hoạt động phù hợp với thời tiết """
        prompt_title = "DANH SÁCH ĐỊA ĐIỂM"
        
    else:
        cleaned_data = data
        system_prompt = "Bạn là trợ lý ảo. Hãy trả lời câu hỏi bằng tiếng Việt dựa trên dữ liệu sau."
        prompt_title = "DỮ LIỆU"
    
    if isinstance(cleaned_data, str):
        # Nếu đã là văn bản (như cái forecast_lines chúng ta vừa làm), GIỮ NGUYÊN!
        data_str = cleaned_data 
    else:
        # Nếu vẫn là dict/list, lúc này mới dùng JSON để ép thành chuỗi
        try:
            data_str = json.dumps(cleaned_data, ensure_ascii=False, indent=2)
        except:
            data_str = str(cleaned_data)
    
    # 6. LẮP RÁP PROMPT CUỐI CÙNG
    prompt = f"""[ WeatherAPI: {prompt_title}]
{data_str}

[CÂU HỎI CỦA NGƯỜI DÙNG]
"{question}"

Tuân thủ nghiêm ngặt các luật đã nêu ở system prompt. Trả lời ngắn gọn, chính xác, thân thiện. KHÔNG BỊ DÀI DÒNG."""
    
    try:
        # 1. Gọi LLM sinh câu trả lời
        result = ask_llm(prompt, system_prompt)

        from datetime import datetime
        current_year = str(datetime.now().year)
        
        import re
        result = re.sub(r'\b(202[0-5])\b', current_year, result)
        
        if is_forecast:
            result = result.replace('- Nhiệt độ từ', 'Nhiệt độ dao động từ')
            result = result.replace('- Nhiều mây', 'Trời nhiều mây.')
        # ==========================================
        
        # Làm sạch các thông tin ảo giác về thời gian cho thời tiết hiện tại
        if is_weather:
            result = remove_hallucinated_dates(result)
            
        return result
    except Exception as e:
        logger.error(f" Answer generation failed: {e}")
        return "Xin lỗi, tôi không thể trả lời lúc này. Vui lòng thử lại."
    
# 8. ACTIVITY EXTRACTION
def extract_activity_type(question: str) -> Optional[str]:
    """Extract activity type from question"""
    question_lower = question.lower()
    
    activity_keywords = {
        "bãi biển": ["biển", "bãi biển", "tắm biển", "biển cả", "bờ biển", "đi biển", "chơi biển", "nghỉ biển", "biển đẹp", "biển xanh"],
        "leo núi": ["leo núi", "núi cao", "leo", "núi", "đỉnh", "đi bộ đường dài", "hiking", "trekking", "climbing", "mountain", "hiking trail", "trekking trail", "climbing route"],
        "thác": ["thác", "thác nước", "thác đổ", "thác chảy", "thác nước đẹp", "thác nước hùng vĩ", "thác nước tuyệt đẹp"],
        "hang": ["hang động", "hang", "động"],
        "suối": ["suối", "suối nước"],
        "cắm trại": ["cắm trại", "camping", "dã ngoại", "picnic"],
        "thiên nhiên": ["thiên nhiên", "tự nhiên", "phong cảnh", "cảnh đẹp", "địa điểm đẹp", "địa điểm tự nhiên"],
    }
    
    # Try keyword matching
    for category, keywords in activity_keywords.items():
        for keyword in keywords:
            if keyword in question_lower:
                logger.info(f" Activity type: {category}")
                return category
    
    logger.debug(f" No activity type found")
    return None
