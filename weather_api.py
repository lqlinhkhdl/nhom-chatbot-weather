import os
import logging
import httpx
from typing import Optional, Dict, Any, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class WeatherAPIError(Exception):
    """Custom exception for Weather API errors"""
    pass

class WeatherAPI:
    """Weather API client using Async HTTPX"""
    
    BASE_URL = "https://api.weatherapi.com/v1/current.json"
    FORECAST_URL = "https://api.weatherapi.com/v1/forecast.json"
    TIMEOUT = 10.0  # seconds
    SEARCH_URL = "https://api.weatherapi.com/v1/search.json"
    
    def __init__(self):
        self.api_key = os.getenv("WEATHERAPI_API_KEY")
        if not self.api_key:
            logger.warning("⚠ WEATHERAPI_API_KEY not set in environment")
            
    async def get_weather(self, location_or_lat: Union[str, float], lon: Optional[float] = None) -> Dict[str, Any]:
        """Fetch current weather data"""
        if not self.api_key:
            raise WeatherAPIError("API key is missing")
            
        # Xác định chuỗi truy vấn (query)
        if isinstance(location_or_lat, (float, int)) and lon is not None:
            query = f"{location_or_lat},{lon}"
        else:
            query = str(location_or_lat)
            
        params = {
            "key": self.api_key,
            "q": query,
            "lang": "vi"
        }
        
        async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
            try:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"HTTP error occurred: {e}")
                raise WeatherAPIError(f"Lỗi kết nối API thời tiết: {str(e)}")
    async def search_location(self, location_name: str) -> Optional[Dict[str, float]]:
        """API Dự phòng: Tìm tọa độ của một địa danh bất kỳ"""
        if not self.api_key:
            return None
            
        params = {"key": self.api_key, "q": location_name}
        
        async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
            try:
                response = await client.get(self.SEARCH_URL, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Nếu API tìm thấy địa danh, trả về vĩ độ và kinh độ
                if data and len(data) > 0:
                    return {
                        "lat": data[0]["lat"],
                        "lon": data[0]["lon"]
                    }
                return None
            except Exception as e:
                logger.error(f" Lỗi tìm kiếm tọa độ: {e}")
                return None
    # TỪ ĐIỂN DỊCH MÃ THỜI TIẾT CỦA OPEN-METEO SANG TIẾNG VIỆT
    WMO_CODES = {
        0: "Trời quang đãng",
        1: "Ít mây", 2: "Có mây rải rác", 3: "Nhiều mây",
        45: "Sương mù", 48: "Sương mù lạnh",
        51: "Mưa phùn nhẹ", 53: "Mưa phùn", 55: "Mưa phùn dày",
        61: "Mưa nhỏ", 63: "Mưa vừa", 65: "Mưa to",
        71: "Tuyết rơi nhẹ", 73: "Tuyết rơi", 75: "Tuyết rơi dày",
        80: "Mưa rào nhẹ", 81: "Mưa rào", 82: "Mưa rào to",
        95: "Có sấm sét", 96: "Sấm sét kèm mưa đá nhẹ", 99: "Sấm sét dữ dội"
    }

    async def get_forecast(self, location_or_lat: Union[str, float], lon: Optional[float] = None, days: int = 7) -> Dict[str, Any]:
        """Sử dụng Open-Meteo cho Dự báo dài ngày và ép kiểu về chuẩn WeatherAPI"""
        
        # 1. Open-Meteo BẮT BUỘC dùng tọa độ. Nếu chưa có, phải tìm tọa độ!
        lat, lng = None, None
        if isinstance(location_or_lat, (float, int)) and lon is not None:
            lat, lng = location_or_lat, lon
        else:
            # Tra cứu tọa độ bằng hàm search_location có sẵn của chúng ta
            coords = await self.search_location(str(location_or_lat))
            if coords:
                lat, lng = coords["lat"], coords["lon"]

        if not lat or not lng:
            return {"error": "Không tìm thấy tọa độ địa danh để lấy dự báo"}

        # 2. Gọi Open-Meteo API
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lng,
            "daily": "weather_code,temperature_2m_max,temperature_2m_min",
            "timezone": "Asia/Ho_Chi_Minh",
            "forecast_days": days
        }

        async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                # 3. ADAPTER: Chuyển đổi dữ liệu Open-Meteo thành định dạng WeatherAPI
                daily = data.get("daily", {})
                dates = daily.get("time", [])
                max_temps = daily.get("temperature_2m_max", [])
                min_temps = daily.get("temperature_2m_min", [])
                codes = daily.get("weather_code", [])

                forecast_days = []
                for i in range(len(dates)):
                    # Dịch mã số (ví dụ: 61) thành chữ (Mưa nhỏ)
                    weather_text = self.WMO_CODES.get(codes[i], "Không xác định")
                    
                    # Mô phỏng y hệt cấu trúc JSON cũ
                    forecast_days.append({
                        "date": dates[i],
                        "day": {
                            "maxtemp_c": max_temps[i],
                            "mintemp_c": min_temps[i],
                            "condition": {"text": weather_text}
                        }
                    })

                # Trả về kết quả "trá hình"
                return {
                    "location": {"name": str(location_or_lat)},
                    "forecast": {"forecastday": forecast_days}
                }

            except Exception as e:
                logger.error(f" Lỗi Open-Meteo: {e}")
                return {"error": f"Lỗi lấy dự báo: {str(e)}"}

# Khởi tạo instance toàn cục (Global instance)
_weather_api = WeatherAPI()

# Các hàm tiện ích để gọi trực tiếp từ state_graph.py
async def get_weather(location_or_lat: Union[str, float], lon: Optional[float] = None) -> Dict[str, Any]:
    try:
        return await _weather_api.get_weather(location_or_lat, lon)
    except Exception as e:
        logger.error(f" Weather API error: {e}")
        return {"error": str(e), "current": {"temp_c": 0.0, "condition": {"text": "Không tìm được dữ liệu"}}}

async def get_forecast(location_or_lat: Union[str, float], lon: Optional[float] = None, days: int = 3) -> Dict[str, Any]:
    try:
        return await _weather_api.get_forecast(location_or_lat, lon, days=days)
    except Exception as e:
        logger.error(f" Forecast API error: {e}")
        return {"error": str(e)}

# Thêm hàm này vào cuối file weather_api.py
async def search_location(location_name: str) -> Optional[Dict[str, float]]:
    """Hàm tiện ích để gọi API tìm kiếm tọa độ"""
    try:
        return await _weather_api.search_location(location_name)
    except Exception as e:
        logger.error(f" Search API error: {e}")
        return None