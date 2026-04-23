import logging
from typing import List, Dict, Any, Optional
import unicodedata

logger = logging.getLogger(__name__)

class WeatherFilters:
    """Weather-based filtering constants"""
    TEMP_MIN = 15  # Minimum comfortable temperature (°C)
    TEMP_MAX = 35  # Maximum comfortable temperature (°C)
    TEMP_PERFECT_MIN = 18
    TEMP_PERFECT_MAX = 28
    
    # Weather conditions that are suitable for outdoor activities
    GOOD_CONDITIONS = {
        "nắng", "trong sáng", "ít mây", "lạnh", "thoáng",
        "sunny", "clear", "partly cloudy", "cool"
    }
    
    # Weather conditions that are NOT suitable
    BAD_CONDITIONS = {
        "mưa", "bão", "sấm", "tuyết", "mù mịt",
        "rain", "storm", "thunder", "snow", "fog", "thunderstorm"
    }

def is_good_weather(weather: Dict[str, Any], location: Dict[str, Any]) -> bool:
    """
    Evaluate if weather is good for visiting a location
    
    Args:
        weather: Weather data from WeatherAPI
        location: Location data from 
        locations.json
    
    Returns:
        True if weather is suitable for visiting
    """
    try:
        if "error" in weather:
            logger.warning(f" Weather error detected")
            return False
        
        # Extract temperature
        temp = weather.get("current", {}).get("temp_c")
        if temp is None:
            logger.warning(" Temperature data missing")
            return False
        
        # Extract condition
        condition = weather.get("current", {}).get("condition", {}).get("text", "").lower()
        
        # Temperature check
        if not (WeatherFilters.TEMP_MIN <= temp <= WeatherFilters.TEMP_MAX):
            logger.debug(f" Temperature {temp}°C out of range")
            return False
        
        # Condition check - if it's raining/stormy, bad weather
        for bad_cond in WeatherFilters.BAD_CONDITIONS:
            if bad_cond in condition:
                logger.debug(f" Bad weather condition: {condition}")
                return False
        
        # Perfect weather bonus (18-28°C and good conditions)
        is_perfect = (WeatherFilters.TEMP_PERFECT_MIN <= temp <= WeatherFilters.TEMP_PERFECT_MAX)
        
        logger.debug(f"Good weather: {temp}°C, {condition}")
        return True
    
    except KeyError as e:
        logger.error(f" KeyError in weather data: {e}")
        return False
    except Exception as e:
        logger.error(f" Unexpected error in is_good_weather: {e}")
        return False

def recommend_places(
    locations: List[Dict[str, Any]],
    weather: Dict[str, Any],
    limit: int = 5,
    category_filter: Optional[str] = None,
    province_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Recommend locations based on current weather conditions
    
    Args:
        locations: List of all locations (from locations.json)
        weather: Current weather data
        limit: Maximum number of recommendations (default: 5)
        category_filter: Optional category to filter (e.g., "leo núi", "biển")
        province_filter: Optional province to filter (e.g., "Hà Nội")
    
    Returns:
        List of recommended locations
    """
    if not locations:
        logger.warning(" No locations provided for recommendation")
        return []
    
    try:
        # Filter by province first if specified
        filtered_locations = locations
        if province_filter:
            province_lower = province_filter.lower().strip()
            filtered_locations = [
                p for p in locations 
                if p.get("tinh", "").lower() == province_lower
            ]
            logger.info(f" Filtered by province: {province_filter} ({len(filtered_locations)} places)")
        
        # Filter by weather suitability
        suitable = [p for p in filtered_locations if is_good_weather(weather, p)]
        
        # Optional category filter
        if category_filter:
            suitable = [
                p for p in suitable
                if category_filter.lower() in [cat.lower() for cat in p.get("categories", [])]
            ]
            logger.info(f" Filtered by category: {category_filter}")
        
        # Limit results
        recommended = suitable[:limit]
        logger.info(f"Recommended {len(recommended)}/{len(locations)} places")
        
        return recommended
    
    except Exception as e:
        logger.error(f" Error in recommend_places: {e}")
        return []

def find_place(
    name: str,
    locations: List[Dict[str, Any]],
    search_fields: List[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Find a specific location by name (case-insensitive search)
    
    Args:
        name: Location name to search for
        locations: List of all locations
        search_fields: Fields to search in (default: ten_dia_danh, tinh)
    
    Returns:
        Location data if found, None otherwise
    """
    if not name or not locations:
        logger.debug(" Empty name or locations")
        return None
    
    if search_fields is None:
        search_fields = ["ten_dia_danh", "tinh", "thanh_pho"]
    
    name_lower = name.lower().strip()
    
    try:
        # First try: exact match in any search field
        for location in locations:
            for field in search_fields:
                field_value = location.get(field, "").lower()
                if field_value == name_lower:
                    logger.info(f"Found exact match: {location['ten_dia_danh']}")
                    return location
        
        # Second try: partial match (substring)
        for location in locations:
            for field in search_fields:
                field_value = location.get(field, "").lower()
                if name_lower in field_value:
                    logger.info(f"Found partial match: {location['ten_dia_danh']}")
                    return location
        
        logger.warning(f" Location '{name}' not found")
        return None
    
    except Exception as e:
        logger.error(f" Error in find_place: {e}")
        return None

def normalize_vietnamese_string(text: str) -> str:
    """
    Chuẩn hóa chuỗi tiếng Việt: 
    - Đưa về chuẩn Unicode NFC (đồng nhất kiểu gõ dấu)
    - Chuyển thành chữ thường (lowercase)
    - Xóa khoảng trắng thừa (strip)
    """
    if not text:
        return ""
    return unicodedata.normalize('NFC', text).strip().lower()

def find_province_coordinates(province_name: str, locations_data: list) -> dict:
    """
    Tìm tọa độ (lat, lon) của một tỉnh/thành phố từ danh sách JSON
    """
    if not province_name or not locations_data:
        return None
        
    # Chuẩn hóa tên địa điểm cần tìm
    target_name = normalize_vietnamese_string(province_name)
    
    for loc in locations_data:
        # Lấy và chuẩn hóa dữ liệu từ JSON
        loc_name = normalize_vietnamese_string(loc.get("name", ""))
        loc_fullname = normalize_vietnamese_string(loc.get("fullName", ""))
        
        # Kiểm tra xem tên có khớp chính xác hoặc nằm bên trong tên đầy đủ không
        # VD: "hà nội" sẽ khớp với "hà nội" hoặc "thành phố hà nội"
        if target_name == loc_name or target_name in loc_fullname:
            return {
                "lat": loc.get("lat"),
                "lon": loc.get("lon")
            }
            
    return None

def get_statistics(locations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about locations
    
    Args:
        locations: List of all locations
    
    Returns:
        Statistics dictionary
    """
    try:
        regions = {}
        categories = {}
        
        for loc in locations:
            region = loc.get("mien", "Unknown")
            regions[region] = regions.get(region, 0) + 1
            
            for cat in loc.get("categories", []):
                categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_locations": len(locations),
            "regions": regions,
            "categories": categories
        }
    except Exception as e:
        logger.error(f" Error in get_statistics: {e}")
        return {}