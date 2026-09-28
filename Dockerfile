# ============================================
# Stage 1: Builder — Cài đặt dependencies
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Cài build tools cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và cài vào user directory
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================
# Stage 2: Runtime — Image tối giản
# ============================================
FROM python:3.11-slim

WORKDIR /app

# Tạo non-root user để tăng bảo mật
RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -m appuser

# Copy dependencies từ builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy mã nguồn ứng dụng
COPY ./app ./app

# Đặt quyền sở hữu cho appuser
RUN chown -R appuser:appuser /app

# Chuyển sang non-root user
USER appuser

# Cập nhật PATH để tìm thấy uvicorn
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app

# Render inject PORT environment variable (default 10000)
ENV PORT=8000

EXPOSE $PORT

# Chạy ứng dụng
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]