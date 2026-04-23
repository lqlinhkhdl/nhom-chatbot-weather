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
    today = datetime.now()
    weekday = today.weekday()
    
    date_match = re.search(r'(?:ngày\s+)?(\d{1,2})\s*[/-]\s*(\d{1,2})', question_lower)
    if date_match:
        try:
            day = int(date_match.group(1))
            month = int(date_match.group(2))
            
            # Tính toán năm (Nếu hỏi tháng 1 mà hiện tại là tháng 12, thì năm phải là năm sau)
            year = today.year
            if month < today.month:
                year += 1
                
            target_date = datetime(year, month, day)
            delta_days = (target_date - today).days
            
            # Nếu ngày được hỏi nằm trong vòng 15 ngày tới
            if 0 <= delta_days <= 15:
                return delta_days + 1 # API tính hôm nay là 1, nên cộng thêm 1
        except ValueError:
            pass # Bỏ qua nếu ngày không hợp lệ (ví dụ: 31/2)
    
    # 0. QUÉT TÌM SỐ NGÀY BẰNG REGEX (Ví dụ: "5 ngày", "15 ngày")
    match = re.search(r'(\d+)\s*ngày', question_lower)
    if match:
        days = int(match.group(1))
        return min(days, 16) # Open-Meteo cho tối đa 16 ngày
    
    # 1. Logic "Cuối tuần"
    if "cuối tuần" in question_lower:
        if weekday == 6:
            return 1
        else:
            return (6 - weekday) + 1
            
    # 2. Logic "Tuần tới / Tuần sau"
    elif "tuần tới" in question_lower or "tuần sau" in question_lower:
        return 8
        
    # 3. Logic "Ngày mốt"
    elif "ngày mốt" in question_lower or "ngày kia" in question_lower:
        return 3
        
    # 4. Logic "Ngày mai"
    elif "ngày mai" in question_lower or "mai" in question_lower.split():
        return 2
        
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

    # THỰC THI LỌC
    if not target_dates:
        return forecast_data[:3]
        
    filtered_data = [day for day in forecast_data if day.get('Ngày') in target_dates]
            
    # Bơm lỗi nếu quá 16 ngày
    if not filtered_data and target_dates:
        return [{
            "Ngày": target_dates[0], 
            "Nhiệt độ thấp nhất": "N/A", 
            "Nhiệt độ cao nhất": "N/A",
            "Thời tiết": "LỖI_VƯỢT_QUÁ_16_NGÀY"
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
        'tạm biệt', 'bye', 'cảm ơn', 'thank', 'ok', 'vâng','được', 'được rồi',
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
        "🌦️ Hỏi tôi về thời tiết hoặc những nơi đáng thăm ở Việt Nam nhé!",
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
        'dự báo thời tiết trong 3 ngày', 'dự báo thời tiết trong 7 ngày','dự báo thời tiết cuối tuần này','dự báo thời tiết tuần sau','dự báo thời tiết ngày mốt','dự báo thời tiết hôm sau'
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
    sorted_locations = sorted(VALID_PROVINCES, key=len, reverse=True)
    for location in sorted_locations:
        if location in question_lower:
            logger.info(f" Extracted location (Exact Match): {location.title()}")
            return location.title()
            
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
    for location in fallback_locations:
        if location in question_lower:
            if location in ['hcm', 'sài gòn', 'saigon']: return "Hồ Chí Minh"
            if location == 'hanoi': return "Hà Nội"
            if location == 'danang': return "Đà Nẵng"
            logger.info(f" Extracted location (Fallback Match): {location.title()}")
            return location.title()
            
    # BƯỚC 3: DÙNG LLM ĐỂ TRÍCH XUẤT (Giải quyết các địa danh lạ như Cát Hải, Mộc Châu...)
    logger.info(f" Keyword match failed. Falling back to LLM Named Entity Recognition...")
    
    system_prompt = """Bạn là chuyên gia trích xuất dữ liệu. 
Nhiệm vụ: Trích xuất tên địa danh (tỉnh, thành phố, quận, huyện, địa điểm du lịch) có trong câu hỏi.
Luật: 
1. CHỈ trả về đúng tên địa danh, TUYỆT ĐỐI KHÔNG giải thích, không nói thêm chữ nào khác.
2. Người dùng thường xuyên gõ sai chính tả hoặc không dấu (VD: "ha nôi" -> Hà Nội, "phu thi" -> Phú Thọ, "thòi tiwet" -> Bỏ qua vì đây là từ sai của "thời tiết"). 
3. Hãy tự động dịch các lỗi gõ sai này và trả về tên địa phương chuẩn xác có dấu.
4. Nếu trong câu không có bất kỳ địa danh nào, hãy trả về chính xác chữ: NONE"""

    prompt = f"Trích xuất địa danh từ câu sau: '{question}'"
    
    try:
        # Gọi thẳng hàm ask_llm (đồng bộ) mà chúng ta đã sửa trước đó
        extracted = ask_llm(prompt, system_prompt).strip()
        
        # Làm sạch chuỗi trả về (đề phòng LLM trả về dấu chấm, nháy kép)
        extracted = extracted.replace('"', '').replace("'", '').strip()
        
        if extracted and extracted.upper() != "NONE":
            logger.info(f" Extracted location (LLM NER): {extracted.title()}")
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
        today = datetime.now()
        today_str = today.strftime("%d/%m") # Rút gọn, bỏ năm để AI khỏi bịa
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
                thu_str = f"{weekdays_vn[dt.weekday()]} ({dt.strftime('%d/%m')})"
            except:
                thu_str = date_str
            
            forecast_lines.append(
                f"- {thu_str}: Từ {day.get('Nhiệt độ thấp nhất')} đến {day.get('Nhiệt độ cao nhất')}, {day.get('Thời tiết')}"
            )
            
        raw_data_text = "\n".join(forecast_lines)
        
        system_prompt = f"""Bạn là một trợ lý thời tiết thông minh, thân thiện và duyên dáng.
HÔM NAY LÀ: {weekdays_vn[today.weekday()]}, Ngày {today_str}.

NHIỆM VỤ: Dựa vào [DỮ LIỆU CUNG CẤP], hãy trả lời câu hỏi của người dùng thật tự nhiên, lưu loát như một người bạn đang trò chuyện.

LUẬT AN TOÀN (BẮT BUỘC):
1. CHỈ dùng số liệu từ [DỮ LIỆU CUNG CẤP]. Không tự bịa thêm ngày nếu dữ liệu không có.
2. Gọi đúng tên địa điểm là "{loc_name}".
3. Hãy tự động sửa chính tả của người dùng nếu họ gõ sai, gõ thiếu, không có dấu (VD: "ha nôi" → "Hà Nội", "phu thi" → "Phú Thọ", "thòi tiwet" → "thời tiết"), nhưng khi trả lời thì phải viết đúng chính tả.

KỊCH BẢN TRẢ LỜI ĐỂ CÓ CÂU TỪ MƯỢT MÀ:
- Mở bài: Một câu chào hoặc dẫn dắt thân thiện (VD: "Chào bạn, gửi bạn thông tin dự báo thời tiết tại {loc_name} nhé:").
- Thân bài: Liệt kê các ngày rõ ràng, có thể thêm các từ nối cho mềm mại (VD: "Trời có mây rải rác, nhiệt độ dao động từ...").
- Kết bài: Một câu chúc hoặc lời khuyên nhẹ nhàng (VD: "Chúc bạn có những ngày tới thật nhiều niềm vui!").
"""
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
        system_prompt = f"""Bạn là một phát thanh viên thời tiết. Nhiệm vụ của bạn là đọc dữ liệu và báo cáo thành một câu tự nhiên.
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
RULES: Gợi ý các địa điểm trong danh sách phù hợp với thời tiết."""
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
    prompt = f"""[DỮ LIỆU CUNG CẤP: {prompt_title}]
{data_str}

[CÂU HỎI CỦA NGƯỜI DÙNG]
"{question}"

Hãy đóng vai trợ lý ảo thân thiện và trả lời câu hỏi trên:"""
    
    try:
        # Gọi mô hình đồng bộ
        result = ask_llm(prompt, system_prompt)
        
        # Làm sạch các thông tin ảo giác về thời gian
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