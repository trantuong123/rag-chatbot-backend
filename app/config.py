import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env (chỉ dùng khi chạy local)
load_dotenv()

class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    COLLECTION_NAME: str = "training_regulations"
    
    def validate(self):
        """Kiểm tra các biến bắt buộc đã được cấu hình."""
        missing = []
        if not self.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        if not self.QDRANT_URL:
            missing.append("QDRANT_URL")
        if not self.QDRANT_API_KEY:
            missing.append("QDRANT_API_KEY")
        if missing:
            raise ValueError(f"Thiếu biến môi trường: {', '.join(missing)}")

settings = Settings()