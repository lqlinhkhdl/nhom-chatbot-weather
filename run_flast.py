import asyncio
import uuid
import logging
from state_graph import run_conversation
from conversation_state import SessionManager

# Tắt bớt log rác của thư viện mạng
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("state_graph").setLevel(logging.WARNING)
logging.getLogger("llm").setLevel(logging.WARNING)
logging.getLogger("weather_api").setLevel(logging.WARNING)

async def chat_loop():
    print("="*60)
    print("🌤️ TRỢ LÝ THỜI TIẾT (CHẾ ĐỘ NHÀ PHÁT TRIỂN) 🌤️")
    print("Mẹo: Gõ 'debug' để BẬT/TẮT bảng phân tích luồng xử lý.")
    print("Gõ 'exit' hoặc 'quit' để dừng trò chuyện.")
    print("="*60)

    # Khởi tạo một phiên làm việc riêng
    session_id = f"cli-session-{uuid.uuid4().hex[:8]}"
    state = SessionManager.create_session(user_id=999, session_id=session_id)
    
    # Biến trạng thái hiển thị luồng gỡ lỗi (Mặc định: Bật)
    show_debug_flow = True 

    while True:
        try:
            user_input = input("\n🧑 Bạn: ")
            
            # Xử lý các lệnh điều khiển hệ thống
            command = user_input.lower().strip()
            if command in ['exit', 'quit', 'thoát']:
                print("🤖 Bot: Đã thoát chế độ Terminal. Tạm biệt!")
                break
            elif command == 'debug':
                show_debug_flow = not show_debug_flow
                trang_thai = "BẬT" if show_debug_flow else "TẮT"
                print(f"⚙️ [HỆ THỐNG]: Đã {trang_thai} bảng phân tích luồng xử lý.")
                continue
                
            if not user_input.strip():
                continue

            # Đưa tin nhắn vào State
            state = SessionManager.get_session(session_id)
            state.add_message("user", user_input)

            print("🤖 Bot: Đang xử lý...", end="\r")

            # Kích hoạt luồng LangGraph
            state = await run_conversation(state)

            # Xóa dòng "Đang xử lý..."
            print(" " * 30, end="\r") 

            # =================================================================
            # 🔍 IN BẢNG TRUY VẾT LUỒNG XỬ LÝ (NẾU BẬT DEBUG)
            # =================================================================
            if show_debug_flow:
                print(f"\n┌── 🔍 BẢNG TRUY VẾT LOGIC MÔ HÌNH {'─'*25}")
                
                # 1. Ý định
                intent_val = state.current_intent.value if state.current_intent else "None"
                print(f"│ 🧠 Ý định (Intent):    {intent_val.upper()}")
                
                # 2. Bóc tách thực thể
                loc_val = state.target_location if state.target_location else "❌ Chưa xác định"
                act_val = state.target_activity if state.target_activity else "Không có"
                print(f"│ 📍 Địa điểm (NER):     {loc_val}")
                print(f"│ 🏃 Hoạt động (Act):    {act_val}")
                
                # 3. Kết quả gọi API
                if state.weather_data:
                    temp = getattr(state.weather_data, 'temperature_c', 'N/A')
                    print(f"│ 🌤️ Data Thời tiết:     Đã tải thành công ({temp}°C)")
                elif state.forecast_data:
                    # Truy cập sâu vào trong list forecastday để đếm số lượng phần tử
                    forecast_dict = state.forecast_data.get('forecast', {})
                    days = len(forecast_dict.get('forecastday', [])) if isinstance(forecast_dict, dict) else "N/A"
                    print(f"│ 📅 Data Dự báo:        Đã tải ({days} ngày)")
                else:
                    print(f"│ 📡 Data API:           Không có dữ liệu tải về")
                
                # 4. Trạng thái cuối của Đồ thị (LangGraph State)
                status_val = state.current_status.value if state.current_status else "None"
                print(f"│ ⚙️ Trạng thái Node:    {status_val}")
                
                # 5. Báo lỗi nếu hệ thống gãy
                if state.error_message:
                    print(f"│ ❌ LỖI HỆ THỐNG:       {state.error_message}")
                
                print(f"└{'─'*58}\n")

            # In ra câu trả lời của Bot
            if state.messages:
                # Đổi màu text bot thành xanh lá (tuỳ chọn trên terminal) để dễ nhìn
                print(f"🤖 Bot: \033[92m{state.messages[-1].content}\033[0m")

        except KeyboardInterrupt:
            print("\n🤖 Bot: Đã ngắt kết nối bằng phím. Tạm biệt!")
            break
        except Exception as e:
            print(f"\n❌ Lỗi hệ thống nghiêm trọng: {e}")

if __name__ == "__main__":
    asyncio.run(chat_loop())