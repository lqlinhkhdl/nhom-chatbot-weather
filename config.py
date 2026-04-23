"""
Configuration file for the Weather Chatbot application
tệp cấu hình cho ứng dụng Chatbot Thời tiết
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
WEATHERAPI_KEY = os.getenv("WEATHERAPI_API_KEY", "")
WEATHERAPI_TIMEOUT = 10
WEATHERAPI_MAX_RETRIES = 3

# LLM Configuration
LLM_MODEL = "vinallama-chat"
LLM_TEMPERATURE = 0.3
LLM_NUM_CTX = 4096
LLM_TIMEOUT = 30

# Weather Filters
TEMP_MIN_COMFORTABLE = 15  # °C
TEMP_MAX_COMFORTABLE = 35  # °C
TEMP_PERFECT_MIN = 18
TEMP_PERFECT_MAX = 28

# Recommendation
MAX_RECOMMENDATIONS = 5
TOP_ALTERNATIVES = 2  # Number of alternative recommendations

# Activity Weather Conditions (optimal conditions for each activity)
ACTIVITY_CONDITIONS = {
    "biển": {
        "temp_min": 22,
        "temp_max": 32,
        "temp_ideal_min": 24,
        "temp_ideal_max": 28,
        "humidity_max": 70,
        "wind_max": 20,
        "bad_conditions": ["rain", "storm", "thunderstorm", "heavy rain"],
        "good_conditions": ["sunny", "clear", "partly cloudy"],
        "vi": "Đi biển"
    },
    "cắm_trại": {
        "temp_min": 12,
        "temp_max": 26,
        "temp_ideal_min": 15,
        "temp_ideal_max": 22,
        "humidity_max": 75,
        "wind_max": 25,
        "bad_conditions": ["rain", "heavy rain", "storm", "thunderstorm"],
        "good_conditions": ["clear", "partly cloudy"],
        "vi": "Cắm trại"
    },
    "leo_núi": {
        "temp_min": 8,
        "temp_max": 24,
        "temp_ideal_min": 12,
        "temp_ideal_max": 20,
        "humidity_max": 80,
        "wind_max": 30,
        "bad_conditions": ["rain", "heavy rain", "fog", "mist", "storm", "thunderstorm"],
        "good_conditions": ["clear", "partly cloudy"],
        "vi": " Leo núi"
    }
}

# Activity Category Keywords
ACTIVITY_KEYWORDS = {
    "biển": ["biển", "bãi", "beach", "coast", "mũi", "đảo", "biển xanh"],
    "cắm_trại": ["cắm trại", "camping", "khu cắm", "resort", "dã ngoại"],
    "leo_núi": ["leo núi", "núi", "mountain", "hiking", "đỉnh", "trekking"]
}

# Application
APP_TITLE = "Chatbot Du lịch & Thời tiết"
APP_VERSION = "1.0.0"
APP_DEBUG = os.getenv("ENVIRONMENT", "development") == "development"

# Logging
LOG_LEVEL = "INFO" if not APP_DEBUG else "DEBUG"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCATIONS_FILE = os.path.join(BASE_DIR, "D:\Data\Learn 4n2\LLM\chatbot_weather\data\locations.json")
SUGGESTED_LOCATIONS_FILE = os.path.join(BASE_DIR, "D:\Data\Learn 4n2\LLM\chatbot_weather\data\Suggested_locations.json")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

def validate_config():
    """Validate configuration on startup"""
    errors = []
    
    if not WEATHERAPI_KEY:
        errors.append(" WEATHERAPI_API_KEY not set in environment")
    
    if not os.path.exists(LOCATIONS_FILE):
        errors.append(f" Locations file not found: {LOCATIONS_FILE}")
    
    if not os.path.exists(SUGGESTED_LOCATIONS_FILE):
        errors.append(f" Suggested locations file not found: {SUGGESTED_LOCATIONS_FILE}")
    
    return errors

if __name__ == "__main__":
    # Print configuration
    print("📋 Configuration Summary")
    print(f"  LLM Model: {LLM_MODEL}")
    print(f"  Temperature: {LLM_TEMPERATURE}")
    print(f"  Weather Timeout: {WEATHERAPI_TIMEOUT}s")
    print(f"  Mode: {'DEBUG' if APP_DEBUG else 'PRODUCTION'}")
    
    errors = validate_config()
    if errors:
        print("\n Configuration Issues:")
        for error in errors:
            print(f"  {error}")
    else:
        print("\n Configuration is valid!")
