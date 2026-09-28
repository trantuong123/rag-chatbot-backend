"""
Test tự động cho RAG Chatbot API.
Chạy: pytest tests/ -v
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test endpoint /health trả về status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_endpoint():
    """Test endpoint / trả về thông báo chào."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_chat_empty_question():
    """Test endpoint /chat với câu hỏi rỗng → phải báo lỗi 422."""
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422


def test_chat_missing_question_field():
    """Test endpoint /chat thiếu trường question → phải báo lỗi 422."""
    response = client.post("/chat", json={})
    assert response.status_code == 422


def test_chat_valid_question():
    """
    Test endpoint /chat với câu hỏi thật.
    LƯU Ý: Test này cần GEMINI_API_KEY và QDRANT credentials hợp lệ.
    """
    response = client.post(
        "/chat",
        json={"question": "Điều kiện tốt nghiệp là gì?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "sources" in data