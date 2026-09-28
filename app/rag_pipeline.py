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
    """Gọi Gemini để tạo câu trả lời dựa trên context, có cơ chế retry."""
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

    # Cấu hình retry
    max_retries = 3
    base_delay = 2  # giây

    for attempt in range(max_retries):
        try:
            response = genai_client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt,
            )
            return response.text
        except (exceptions.ServiceUnavailable, exceptions.ResourceExhausted) as e:
            if attempt == max_retries - 1:
                raise  # Nếu đã thử hết số lần, ném lỗi ra ngoài
            wait_time = base_delay * (2 ** attempt)  # 2s, 4s, 8s
            print(f"Lỗi tạm thời ({type(e).__name__}), thử lại sau {wait_time} giây...")
            time.sleep(wait_time)

    return "Xin lỗi, hiện tại tôi không thể xử lý yêu cầu của bạn. Vui lòng thử lại sau."

def process_query(query: str) -> dict:
    """Luồng xử lý chính: câu hỏi → embedding → search → generate."""
    # 1. Tạo embedding cho câu hỏi (dùng RETRIEVAL_QUERY)
    query_embedding = get_embedding(query, task_type="RETRIEVAL_QUERY")
    
    # 2. Tìm kiếm trong Vector DB
    search_results = search_vector_db(query_embedding, top_k=5)
    
    if not search_results:
        return {
            "answer": "Tôi không tìm thấy thông tin liên quan trong tài liệu.",
            "sources": [],
        }
    
    # 3. Tạo câu trả lời với context
    answer = generate_answer(query, search_results)
    
    return {
        "answer": answer,
        "sources": search_results,
    }