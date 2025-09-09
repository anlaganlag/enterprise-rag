#!/bin/bash

# Docker部署启动脚本

echo "======================================"
echo "   保险产品RAG系统 - Docker部署"
echo "======================================"

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "错误: Docker未安装，请先安装Docker"
    exit 1
fi

# 检查Docker Compose是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "错误: Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

# 检查.env文件
if [ ! -f .env ]; then
    echo "警告: .env文件不存在，创建示例文件..."
    echo "OPENAI_API_KEY=your-api-key-here" > .env
    echo "请编辑.env文件，添加您的OpenAI API密钥"
    exit 1
fi

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p uploads output vector_store logs elasticsearch/plugins

# 构建和启动服务
echo "构建Docker镜像..."
docker-compose build

echo "启动服务..."
docker-compose up -d

# 等待服务启动
echo "等待服务启动..."
sleep 10

# 检查服务状态
echo "检查服务状态..."
docker-compose ps

# 健康检查
echo "执行健康检查..."
curl -s http://localhost:8000/health | python -m json.tool

echo ""
echo "======================================"
echo "部署完成!"
echo "======================================"
echo "访问以下地址使用系统:"
echo "- API文档: http://localhost:8000/docs"
echo "- Web界面: http://localhost:8000/web_demo_enhanced.html"
echo "- Elasticsearch: http://localhost:9200"
echo ""
echo "查看日志: docker-compose logs -f"
echo "停止服务: docker-compose down"
echo "======================================" 