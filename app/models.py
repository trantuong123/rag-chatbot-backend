from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Câu hỏi của người dùng")

class SourceChunk(BaseModel):
    text: str
    score: float
    source: str = ""

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = []