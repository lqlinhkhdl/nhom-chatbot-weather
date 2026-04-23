"""
LangGraph Workflow for Stateful Conversation
Các node xử lý stateful dùng ConversationState
"""

import logging
from typing import Dict, Any, Optional
import uuid
from weather_api import get_weather, get_forecast, WeatherAPIError
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from weather_api import get_weather, get_forecast, search_location, WeatherAPIError
from llm import get_required_forecast_days
from conversation_state import (
    ConversationState, ConversationIntent, ConversationStatus,
    WeatherData, SessionManager
)
from llm import classify_intent, extract_location, generate_answer, detect_small_talk, handle_small_talk, ask_llm, extract_activity_type
from weather_api import get_weather, WeatherAPIError
from recommender import find_province_coordinates, recommend_places

logger = logging.getLogger(__name__)

# ====== NODE FUNCTIONS ======

def node_analyze_intent(state: ConversationState) -> ConversationState:
    """
    Node 1: Phân tích ý định từ message cuối cùng
    Xét ngữ cảnh từ các messages trước đó
    """
    logger.info(f"[{state.session_id}]  NODE: analyze_intent")
    
    if not state.messages:
        state.update_status(ConversationStatus.ERROR)
        state.error_message = "No messages in state"
        return state
    
    # Lấy message cuối cùng (message của user vừa gửi)
    last_message = state.messages[-1].content
    
    # Get conversation history để context
    history_text = state.get_conversation_history_text(max_turns=3)
    
    # Nếu có history, scope intent classification dựa trên context
    if history_text and state.target_location:
        # User đã nói địa điểm rồi, có thể họ đang trả lời câu hỏi trước
        full_context = f"History:\n{history_text}\n\nPending question: {state.pending_question}\n\nNew input: {last_message}"
        logger.debug(f"[{state.session_id}] Using context for intent: {full_context[:100]}...")
    
    # Phân loại ý định
    if detect_small_talk(last_message):
        intent = ConversationIntent.SMALL_TALK
    else:
        intent_str = classify_intent(last_message)
        intent = ConversationIntent(intent_str)
        
    state.update_intent(intent)
    logger.info(f"[{state.session_id}] Intent classified: {intent.value}")
    
    return state


def node_extract_location(state: ConversationState) -> ConversationState:
    """
    Node 2: Trích xuất location và activity type từ message
    Xét ngữ cảnh: Đã biết location từ trước không?
    """
    logger.info(f"[{state.session_id}]  NODE: extract_location")
    
    if state.current_intent == ConversationIntent.SMALL_TALK:
        logger.debug(f"[{state.session_id}] Small talk, skipping location extraction")
        return state
    
    last_message = state.messages[-1].content
    
    # Cách 1: Nếu user đã từng nói location, kiểm tra xem message này có nói location mới không
    if state.target_location:
        # Hỏi: "Người dùng có đang nói về địa điểm khác không?"
        new_location = extract_location(last_message)
        
        if new_location and new_location.lower() != state.target_location.lower():
            logger.info(f"[{state.session_id}] Location changed: {state.target_location} → {new_location}")
            state.update_location(new_location)
            state.weather_data = None  # Reset weather từ location cũ
        elif not new_location:
            logger.debug(f"[{state.session_id}] No new location found, keeping: {state.target_location}")
    else:
        # Cách 2: Chưa biết location, trích xuất từ message hiện tại
        location = extract_location(last_message)
        if location:
            state.update_location(location)
            logger.info(f"[{state.session_id}] Extracted location for first time: {location}")
        else:
            logger.debug(f"[{state.session_id}] No location found in message")
    
    # Extract activity type if this is an activity intent
    if state.current_intent == ConversationIntent.ACTIVITY:
        activity_type = extract_activity_type(last_message)
        if activity_type:
            state.target_activity = activity_type
            logger.info(f"[{state.session_id}] Extracted activity type: {activity_type}")
        else:
            logger.debug(f"[{state.session_id}] No activity type found, will recommend all places")
    
    return state


def node_check_missing_info(state: ConversationState) -> ConversationState:
    """
    Node 3: Kiểm tra xem còn thiếu info nào không
    Ví dụ: User muốn weather nhưng chưa nói location
    """
    logger.info(f"[{state.session_id}] ❓ NODE: check_missing_info")
    
    if state.current_intent == ConversationIntent.SMALL_TALK:
        return state
    
    # Kiểm tra: Cần location không?
    needs_location = state.current_intent in [
        ConversationIntent.WEATHER,
        ConversationIntent.ACTIVITY,
        ConversationIntent.FORECAST
    ]
    
    if needs_location and not state.target_location:
        state.update_status(ConversationStatus.WAITING_FOR_INPUT)
        state.pending_question = (
            f"🤔 Bạn muốn xem {state.current_intent.value} ở đâu? "
            f"Hãy nêu tên tỉnh/thành phố (ví dụ: Hà Nội, Đà Nẵng)."
        )
        logger.info(f"[{state.session_id}] Missing location, setting pending question")
        return state
    
    state.update_status(ConversationStatus.IN_PROGRESS)
    return state


async def node_fetch_weather_data(state: ConversationState) -> ConversationState:
    """
    Node 4a: Fetch weather data nếu intent là WEATHER
    """
    logger.info(f"[{state.session_id}] 🌤️ NODE: fetch_weather_data")
    
    if state.current_intent != ConversationIntent.WEATHER:
        logger.debug(f"[{state.session_id}] Not weather intent, skipping")
        return state
    
    if not state.target_location:
        logger.warning(f"[{state.session_id}] No location for weather query")
        state.error_message = "Location not set"
        return state
    # Lớp 1: Tìm trong file D:\Data\Learn 4n2\LLM\chatbot_weather\data\locations.json (Ưu tiên vì nhanh và offline)
    import json
    with open("D:\Data\Learn 4n2\LLM\chatbot_weather\data\locations.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    coords = find_province_coordinates(state.target_location, data)
    
    # Lớp 2: Nếu không thấy trong JSON -> Gọi API Search để tìm tọa độ (DỰ PHÒNG)
    if not coords:
        logger.info(f" '{state.target_location}' không có trong JSON. Đang tra cứu tọa độ từ API...")
        # XÓA chữ _weather_api. ở dòng này
        coords = await search_location(state.target_location)
    try:
        # Load locations data
        import json
        with open("D:\Data\Learn 4n2\LLM\chatbot_weather\data\locations.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Tìm coordinates
        province_coords = find_province_coordinates(state.target_location, data)
        
        if province_coords:
            lat, lon = province_coords["lat"], province_coords["lon"]
        else:
            # Fallback: query bằng tên
            lat, lon = None, None
        if coords:
            # Nếu tìm thấy tọa độ (từ JSON hoặc từ API Search) -> Gọi bằng tọa độ
            weather_dict = await get_weather(coords["lat"], coords["lon"])
        else:
            # Nếu tất cả đều thất bại -> Gọi API bằng tên văn bản (Cơ hội cuối cùng)
            weather_dict = await get_weather(state.target_location)
        # Fetch weather
        if lat and lon:
            weather_dict =  await get_weather(lat, lon)
        else:
            weather_dict = await get_weather(state.target_location)
        
        # --- THÊM PHẦN NÀY ĐỂ TRÍCH XUẤT ĐÚNG DỮ LIỆU LỒNG NHAU ---
        current_data = weather_dict.get("current", {})
        loc_data = weather_dict.get("location", {})
        loc_name = loc_data.get("name", state.target_location) if isinstance(loc_data, dict) else state.target_location

        # --- SỬA LẠI KHỐI GÁN NÀY ---
        state.weather_data = WeatherData(
            location=loc_name,
            temperature_c=current_data.get("temp_c", 0.0), # Trích xuất đúng key "temp_c" từ "current"
            condition=current_data.get("condition", {}).get("text", ""),
            humidity=current_data.get("humidity", 0),
            wind_kph=current_data.get("wind_kph", 0.0),
        )
        # -----------------------------------------------------------
        
        logger.info(f"[{state.session_id}] Fetched weather: {loc_name} - {state.weather_data.temperature_c}°C")
        
    except WeatherAPIError as e:
        logger.error(f"[{state.session_id}] Weather API error: {e}")
        state.error_message = str(e)
        state.update_status(ConversationStatus.ERROR)
    except Exception as e:
        logger.error(f"✗ Lỗi lấy thời tiết: {e}")
        state.update_status(ConversationStatus.ERROR)
    return state

async def node_fetch_forecast(state: ConversationState) -> ConversationState:
    """
    Node 4c: Fetch forecast data nếu intent là FORECAST
    """
    logger.info(f"[{state.session_id}]  NODE: fetch_forecast")
    
    if state.current_intent != ConversationIntent.FORECAST:
        return state
    
    if not state.target_location:
        state.error_message = "Location not set"
        return state
    
    try:
        # Lấy câu hỏi của người dùng để phân tích số ngày dự báo cần thiết
        last_user_message = ""
        for msg in reversed(state.messages):
            # SỬA Ở ĐÂY: Đổi msg.type thành msg.role
            if msg.role == "user": 
                last_user_message = msg.content
                break
                
        # TÍNH TOÁN SỐ NGÀY ĐỂ TIẾT KIỆM TÀI NGUYÊN
        required_days = get_required_forecast_days(last_user_message)
        logger.info(f"[{state.session_id}] Tối ưu API: Yêu cầu lấy {required_days} ngày dự báo")

        # 3. Tìm tọa độ trong file JSON (Luồng dự phòng bạn đã làm)
        import json
        with open("D:\Data\Learn 4n2\LLM\chatbot_weather\data\locations.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            
        province_coords = find_province_coordinates(state.target_location, data)
        
        if province_coords:
            lat, lon = province_coords["lat"], province_coords["lon"]
        else:
            lat, lon = None, None
            
        # 4. GỌI API VỚI SỐ NGÀY ĐÃ ĐƯỢC TỐI ƯU
        if lat and lon:
            forecast_dict = await get_forecast(lat, lon, days=required_days)
        else:
            forecast_dict = await get_forecast(state.target_location, days=required_days)
            
        # Lưu vào state mới
        state.forecast_data = forecast_dict
        logger.info(f"[{state.session_id}] Fetched forecast for {state.target_location}")
        
    except Exception as e:
        logger.error(f"[{state.session_id}] Error fetching forecast: {e}")
        state.error_message = str(e)
        state.update_status(ConversationStatus.ERROR)
    
    return state

async def node_fetch_recommendations(state: ConversationState) -> ConversationState:
    """
    Node 4b: Fetch activity recommendations nếu intent là ACTIVITY
    """
    logger.info(f"[{state.session_id}]  NODE: fetch_recommendations")
    
    if state.current_intent != ConversationIntent.ACTIVITY:
        logger.debug(f"[{state.session_id}] Not activity intent, skipping")
        return state
    
    if not state.target_location:
        logger.warning(f"[{state.session_id}] No location for activity query")
        state.error_message = "Location not set"
        return state
    
    try:
        import json
        with open("D:\Data\Learn 4n2\LLM\chatbot_weather\data\locations.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Fetch weather first để biết điều kiện
        province_coords = find_province_coordinates(state.target_location, data)
        if province_coords:
            lat, lon = province_coords["lat"], province_coords["lon"]
            weather_dict =  await get_weather(lat, lon)
        else:
            weather_dict = await get_weather(state.target_location)
        
        # Lưu weather
        state.weather_data = WeatherData(
            location=weather_dict.get("location", state.target_location),
            temperature_c=weather_dict.get("temperature_c", 0.0),
            condition=weather_dict.get("condition", ""),
            humidity=weather_dict.get("humidity", 0),
            wind_kph=weather_dict.get("wind_kph", 0.0),
        )
        
        # Recommend places - filter by province and activity type if available
        if state.target_activity:
            recommendations = recommend_places(
                data, 
                weather_dict,
                province_filter=state.target_location,
                category_filter=state.target_activity
            )
            logger.info(f"[{state.session_id}] Filtering by location={state.target_location}, activity={state.target_activity}")
        else:
            recommendations = recommend_places(
                data, 
                weather_dict,
                province_filter=state.target_location
            )
            logger.info(f"[{state.session_id}] No activity filter, recommending places in {state.target_location}")
        
        # Store recommendations count in state
        activity_desc = f" ({state.target_activity})" if state.target_activity else ""
        state.target_activity = f"Đã recommend {len(recommendations)} places{activity_desc}"
        
        logger.info(f"[{state.session_id}] Fetched {len(recommendations)} recommendations")
        
    except Exception as e:
        logger.error(f"[{state.session_id}] Error fetching recommendations: {e}")
        state.error_message = str(e)
        state.update_status(ConversationStatus.ERROR)
    
    return state


async def node_generate_response(state: ConversationState) -> ConversationState:
    """
    Node 5: Sinh response dựa vào state hiện tại
    Lưu ý: Dùng context từ State để LLM sinh ra response thích hợp
    """
    logger.info(f"[{state.session_id}]  NODE: generate_response")
    
    # Nếu đang chờ info từ user hoặc có error
    if state.current_status == ConversationStatus.WAITING_FOR_INPUT:
        response = state.pending_question
        state.add_message("assistant", response)
        return state
    
    if state.current_status == ConversationStatus.ERROR:
        response = f"Xin lỗi, có lỗi xảy ra: {state.error_message}"
        state.add_message("assistant", response)
        return state
    
    # Lấy message cuối cùng của user
    last_user_message = state.messages[-1].content
    
    try:
        # Small talk: use special handler
        if state.current_intent == ConversationIntent.SMALL_TALK:
            response = handle_small_talk(last_user_message)
        
        # Weather: use weather data
        elif state.current_intent == ConversationIntent.WEATHER:
            response = generate_answer(
                last_user_message,
                state.weather_data.to_dict() if state.weather_data else {},
                location_override=state.target_location
            )
            
        # FORECAST: Dùng dữ liệu dự báo
        elif state.current_intent == ConversationIntent.FORECAST:
            response = generate_answer(
                last_user_message,
                state.forecast_data if state.forecast_data else {},
                location_override=state.target_location
            )
        # Activity: use recommendations
        elif state.current_intent == ConversationIntent.ACTIVITY:
            response = generate_answer(
                last_user_message,
                {"location": state.target_location, "activity": state.target_activity},
                location_override=state.target_location
            )
        else:
            # Unknown: polite farewell or guidance
            response = (
                "Xin lỗi, tôi chỉ hỗ trợ queries về thời tiết hoặc gợi ý địa điểm du lịch. "
                "Hãy hỏi: 'Thời tiết ở [thành phố] thế nào?' "
                "hoặc 'Gợi ý nơi [loại hoạt động] hôm nay?'"
                "Có thể bạn đang nhập sai chính tả hoặc câu hỏi chưa rõ ràng, vui lòng nhập lại giúp mình nhé!"
            )
        state.add_message("assistant", response)
        state.update_status(ConversationStatus.COMPLETED)
        logger.info(f"[{state.session_id}] Generated response")
        
    except Exception as e:
        logger.error(f"[{state.session_id}] Error generating response: {e}")
        response = "Xin lỗi, có lỗi xảy ra khi sinh response."
        state.add_message("assistant", response)
        state.error_message = str(e)
        state.update_status(ConversationStatus.ERROR)
    
    return state


# ====== BUILD GRAPH ======

def build_conversation_graph():
    """
    Xây dựng LangGraph workflow
    """
    logger.info("🔨 Building conversation state graph...")
    
    workflow = StateGraph(ConversationState)
    
    # Thêm nodes
    workflow.add_node("analyze_intent", node_analyze_intent)
    workflow.add_node("extract_location", node_extract_location)
    workflow.add_node("check_missing_info", node_check_missing_info)
    workflow.add_node("fetch_weather", node_fetch_weather_data)
    workflow.add_node("fetch_forecast", node_fetch_forecast)
    workflow.add_node("fetch_recommendations", node_fetch_recommendations)
    workflow.add_node("generate_response", node_generate_response)
    
    # Thêm edges (flow)
    workflow.add_edge(START, "analyze_intent")
    workflow.add_edge("analyze_intent", "extract_location")
    workflow.add_edge("extract_location", "check_missing_info")
    
    # Xử lý branch: Nếu thiếu info, quay lại asking
    # Nếu đủ info, fetch data
    def should_fetch_data(state: ConversationState) -> str:
        """Router: Có đủ info để fetch data không?"""
        if state.current_status == ConversationStatus.WAITING_FOR_INPUT:
            return "generate_response"  # Ask user → skip fetching
        
        if state.current_intent == ConversationIntent.WEATHER:
            return "fetch_weather"
        elif state.current_intent == ConversationIntent.ACTIVITY:
            return "fetch_recommendations"
        elif state.current_intent == ConversationIntent.FORECAST: 
            return "fetch_forecast"                               
        else:
            return "generate_response"
    
    workflow.add_conditional_edges(
        "check_missing_info",
        should_fetch_data,
        {
            "fetch_weather": "fetch_weather",
            "fetch_forecast": "fetch_forecast",
            "fetch_recommendations": "fetch_recommendations",
            "generate_response": "generate_response",
        }
    )
    
    workflow.add_edge("fetch_weather", "generate_response")
    workflow.add_edge("fetch_forecast", "generate_response")
    workflow.add_edge("fetch_recommendations", "generate_response")
    workflow.add_edge("generate_response", END)
    
    graph = workflow.compile()
    logger.info("✓ Graph compiled successfully")
    
    return graph


# Global graph instance
conversation_graph = build_conversation_graph()


async def run_conversation(state: ConversationState) -> ConversationState:
    """
    Chạy conversation workflow với state (Async)
    """
    logger.info(f"[{state.session_id}] 🚀 Running conversation graph...")
    
    try:
        # SỬ DỤNG ainvoke VÌ ĐỒ THỊ BÂY GIỜ ĐÃ LÀ BẤT ĐỒNG BỘ
        final_state_dict = await conversation_graph.ainvoke(state)
        
        # GIẢI QUYẾT LỖI DICT: LangGraph trả về dictionary, ta cập nhật ngược lại vào object state
        if isinstance(final_state_dict, dict):
            for key, value in final_state_dict.items():
                if hasattr(state, key):
                    setattr(state, key, value)
        
        # Cập nhật session vào DB/Memory
        SessionManager.update_session(state.session_id, state)
        
        logger.info(f"[{state.session_id}] ✓ Graph execution completed")
        return state # Trả về state object chuẩn
        
    except Exception as e:
        logger.error(f"[{state.session_id}] ✗ Graph execution failed: {e}", exc_info=True)
        state.error_message = str(e)
        state.update_status(ConversationStatus.ERROR)
        return state


# ====== Testing ======

def test_stateful_flow():
    """Test stateful conversation flow"""
    from llm import ask_llm
    
    logger.info("\n" + "="*60)
    logger.info("TEST: Stateful Conversation Flow")
    logger.info("="*60)
    
    # Tạo session
    session_id = f"test-session-{uuid.uuid4().hex[:8]}"
    state = SessionManager.create_session(user_id=1, session_id=session_id)
    
    # Message 1: User hỏi về activity nhưng không nói location
    state.add_message("user", "Muốn đi biển hôm nay")
    result1 = run_conversation(state)
    print(f"\n--- Message 1: 'Muốn đi biển hôm nay' ---")
    print(f"Intent: {result1.current_intent.value}")
    print(f"Status: {result1.current_status.value}")
    print(f"Location: {result1.target_location}")
    if result1.messages:
        print(f"Bot: {result1.messages[-1].content}")
    
    # Message 2: User nói location
    state = SessionManager.get_session(session_id)
    state.add_message("user", "Hà Nội")
    result2 = run_conversation(state)
    print(f"\n--- Message 2: 'Hà Nội' ---")
    print(f"Status: {result2.current_status.value}")
    print(f"Location: {result2.target_location}")
    if len(result2.messages) > 1:
        print(f"Bot: {result2.messages[-1].content[:200]}...")
    
    logger.info("\n✓ Test completed")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    test_stateful_flow()