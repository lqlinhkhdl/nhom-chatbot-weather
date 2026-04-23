#!/usr/bin/env python
"""
Quick start script for Weather Chatbot with Ngrok support
"""

import os
import sys
import uvicorn
import argparse
from config import validate_config, LOG_LEVEL
# Thêm import ngrok
try:
    from pyngrok import ngrok
except ImportError:
    print("  Chưa cài thư viện pyngrok. Hãy chạy: pip install pyngrok")

def main():
    parser = argparse.ArgumentParser(description="Weather Chatbot Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    # Thêm cờ --ngrok
    parser.add_argument("--ngrok", action="store_true", help="Enable Ngrok tunnel for global access")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🌦️  Weather Chatbot - Startup Check")
    print("="*60)
    
    errors = validate_config()
    if errors:
        print("\n  Configuration Issues Found:")
        for error in errors:
            print(f"   {error}")
    else:
        print("\n  Configuration is valid!")

    # --- XỬ LÝ NGROK TỰ ĐỘNG ---
    public_url = None
    if args.ngrok:
        try:
            # Lấy token từ file biến môi trường thay vì hardcode
            ngrok_token = os.getenv("NGROK_AUTH_TOKEN")
            
            if not ngrok_token:
                print("\n Lỗi: Không tìm thấy NGROK_AUTH_TOKEN trong file môi trường.")
                args.ngrok = False
            else:
                # Cấu hình token bảo mật
                ngrok.set_auth_token(ngrok_token)
                
                # Kết nối Ngrok đến đúng port mà app đang chạy
                tunnel = ngrok.connect(args.port)
                public_url = tunnel.public_url
                
                # Quan trọng: Khi dùng Ngrok, host cần để 0.0.0.0 để nhận request từ bên ngoài
                args.host = "0.0.0.0"
        except Exception as e:
            print(f"\n Lỗi khởi động Ngrok: {e}")
            args.ngrok = False
    print("\n" + "="*60)
    print(f"🚀 Starting server...")
    print(f"   Local Address:  http://{args.host}:{args.port}")
    if args.ngrok and public_url:
        print(f"   GLOBAL LINK:    {public_url} ")
    print(f"   Reload:         {args.reload}")
    print("="*60)
    print("\n📌 Press Ctrl+C to stop\n")
    
    # Start uvicorn
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=LOG_LEVEL.lower()
    )
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)