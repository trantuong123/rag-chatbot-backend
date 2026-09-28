"""
Script nạp dữ liệu quy chế đào tạo vào Qdrant Vector Database.
Chạy một lần duy nhất trước khi sử dụng chatbot.
"""
import sys
import uuid
from pathlib import Path

# Thêm thư mục gốc vào sys.path để import được app.config
sys.path.insert(0, str(Path(__file__).parent.parent))

from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import settings

# Khởi tạo clients
genai_client = genai.Client(api_key=settings.GEMINI_API_KEY)
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Chia văn bản thành các đoạn nhỏ (chunk).
    chunk_size: số ký tự mỗi đoạn (~500 ký tự ≈ 100-150 từ).
    overlap: số ký tự chồng lấn giữa các đoạn (giữ ngữ cảnh liên tục).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks

def read_document(file_path: str) -> str:
    """Đọc nội dung file .txt hoặc .pdf."""
    path = Path(file_path)
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")
    elif path.suffix.lower() == ".pdf":
        import pypdf
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    else:
        raise ValueError(f"Định dạng không hỗ trợ: {path.suffix}")

def create_embedding(text: str) -> list[float]:
    result = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values

def init_collection():
    """Tạo collection trong Qdrant nếu chưa tồn tại."""
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if settings.COLLECTION_NAME not in collections:
        qdrant_client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        print(f"✓ Đã tạo collection: {settings.COLLECTION_NAME}")
    else:
        print(f"✓ Collection đã tồn tại: {settings.COLLECTION_NAME}")

def ingest_file(file_path: str):
    """Nạp một file vào Vector DB: đọc → chunk → embedding → upload."""
    print(f"\nĐang xử lý: {file_path}")
    text = read_document(file_path)
    chunks = chunk_text(text)
    print(f"  → Chia thành {len(chunks)} đoạn")
    
    points = []
    for i, chunk in enumerate(chunks):
        embedding = create_embedding(chunk)
        point_id = str(uuid.uuid4())
        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "text": chunk,
                "source": Path(file_path).name,
                "chunk_index": i,
            },
        ))
        if (i + 1) % 10 == 0:
            print(f"  → Đã tạo embedding {i+1}/{len(chunks)}")
    
    # Upload theo batch 100 points để tránh timeout
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        qdrant_client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=batch,
        )
        print(f"  → Đã upload {min(i+batch_size, len(points))}/{len(points)} vectors")
    
    print(f"  ✓ Hoàn tất: {file_path}")

def main():
    """Hàm chính: nạp tất cả file .txt/.pdf trong thư mục data/."""
    data_dir = Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        print("Tạo thư mục data/ và đặt file tài liệu vào đó!")
        sys.exit(1)
    
    init_collection()
    
    files = list(data_dir.glob("*.txt")) + list(data_dir.glob("*.pdf"))
    if not files:
        print(f"Không tìm thấy file .txt hoặc .pdf trong {data_dir}")
        sys.exit(1)
    
    for file_path in files:
        ingest_file(str(file_path))
    
    print("\n✓✓✓ Hoàn tất nạp dữ liệu!")

if __name__ == "__main__":
    main()