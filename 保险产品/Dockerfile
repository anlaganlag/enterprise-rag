# 使用Python 3.9基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
COPY requirements-api.txt .

# 安装Python依赖
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-api.txt

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p /app/uploads /app/vector_store /app/output /app/logs

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "es_rag_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]