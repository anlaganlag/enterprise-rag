@echo off
echo ========================================
echo 安装ElasticSearch IK中文分词器
echo ========================================

REM 检查ES容器是否运行
docker ps | findstr es-rag-poc >nul
if %errorlevel% neq 0 (
    echo ES容器未运行，请先启动: docker-compose up -d
    exit /b 1
)

echo.
echo 正在安装IK分词器...
REM 在容器内安装IK插件
docker exec es-rag-poc bin/elasticsearch-plugin install https://github.com/medcl/elasticsearch-analysis-ik/releases/download/v8.11.0/elasticsearch-analysis-ik-8.11.0.zip

echo.
echo 重启ElasticSearch以应用更改...
docker-compose restart elasticsearch

echo.
echo IK分词器安装完成！
echo 等待ES重启...
timeout /t 30

echo 测试IK分词器...
curl -X POST "localhost:9200/_analyze" -H "Content-Type: application/json" -d "{\"analyzer\":\"ik_smart\",\"text\":\"保险产品最低保费\"}"

echo.
echo ========================================
echo 安装完成！
echo ========================================