from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from app.config import settings
import time
from google.api_core import exceptions

# Khởi tạo Gemini client (dùng chung cho cả embedding và generation)
genai_client = genai.Client(api_key=settings.GEMINI_API_KEY)

# Khởi tạo Qdrant client
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)

def get_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    result = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values

def search_vector_db(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Tìm kiếm các đoạn văn bản liên quan nhất trong Vector DB."""
    search_result = qdrant_client.search(
        collection_name=settings.COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True,
    )
    results = []
    for hit in search_result:
        results.append({
            "text": hit.payload.get("text", ""),
            "score": hit.score,
            "source": hit.payload.get("source", ""),
        })
    return results

def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """
    Gọi Gemini để tạo câu trả lời dựa trên context.
    Có cơ chế retry khi gặp lỗi 503 (quá tải) và fallback model.
    """
    import time
    
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        source_info = f"[Nguồn: {chunk['source']}]" if chunk['source'] else ""
        context_parts.append(f"--- Đoạn {i+1} {source_info} ---\n{chunk['text']}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""Bạn là một trợ lý ảo chuyên về quy chế đào tạo của trường đại học.
Hãy trả lời câu hỏi của người dùng một cách chính xác, CHỈ dựa trên thông tin được cung cấp trong phần NGỮ CẢNH.
TUYỆT ĐỐI KHÔNG bịa đặt thông tin không có trong ngữ cảnh.
Nếu thông tin không có trong ngữ cảnh, hãy trả lời: "Tôi không tìm thấy thông tin này trong tài liệu quy chế."
Luôn trích dẫn nguồn (tên tài liệu, số điều/khoản) nếu có trong ngữ cảnh.
Trả lời bằng tiếng Việt, ngắn gọn, rõ ràng, có cấu trúc.

NGỮ CẢNH:
{context}

CÂU HỎI: {query}

TRẢ LỜI:"""

    # Danh sách model dự phòng (thử lần lượt từ trên xuống)
    models_to_try = [
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
    ]
    
    max_retries = 3
    base_delay = 2

    last_error = None
    
    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                response = genai_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                error_str = str(e)
                last_error = e
                
                if "503" in error_str or "429" in error_str or "UNAVAILABLE" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = base_delay * (2 ** attempt)
                        print(f"Model {model_name} qua tai, thu lai sau {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Model {model_name} that bai, chuyen model du phong...")
                        break
                else:
                    print(f"Model {model_name} loi: {error_str[:100]}. Chuyen model du phong...")
                    break
    
    raise Exception("AI_OVERLOADED")

def process_query(query: str) -> dict:
    query_embedding = get_embedding(query, task_type="RETRIEVAL_QUERY")
    search_results = search_vector_db(query_embedding, top_k=5)
    
    if not search_results:
        return {
            "answer": "Tôi không tìm thấy thông tin liên quan trong tài liệu.",
            "sources": [],
        }
    
    try:
        answer = generate_answer(query, search_results)
        return {
            "answer": answer,
            "sources": search_results,
        }
    except Exception as e:
        # Khi AI quá tải hoặc lỗi → KHÔNG trả về sources
        return {
            "answer": "Xin lỗi, hiện tại hệ thống AI đang quá tải. Vui lòng thử lại sau 1-2 phút.",
            "sources": [],   # ← TRẢ VỀ MẢNG RỖNG
        }