from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models import QueryRequest, QueryResponse
from app.rag_pipeline import process_query
from app.config import settings

# Validate cấu hình khi khởi động
settings.validate()

app = FastAPI(
    title="RAG Chatbot API - Quy chế Đào tạo",
    description="API tra cứu thông tin quy chế đào tạo sử dụng RAG với Gemini và Qdrant",
    version="1.0.0",
)

# Cấu hình CORS cho phép Frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rag-chatbot-backend-fawn.vercel.app"],  # Thay bằng domain cụ thể khi production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Endpoint kiểm tra tình trạng service (dùng cho Render health check)."""
    return {"status": "ok", "service": "rag-chatbot-backend"}

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """Endpoint chính: nhận câu hỏi, trả về câu trả lời và nguồn tham khảo."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")
    
    try:
        result = process_query(request.question)
        return QueryResponse(
            answer=result["answer"],
            sources=[
                {"text": s["text"], "score": s["score"], "source": s["source"]}
                for s in result["sources"]
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "RAG Chatbot API đang hoạt động",
        "docs": "/docs",
        "health": "/health",
    }