# Docker部署指南

## 快速开始

### 前提条件
- Docker 20.10+
- Docker Compose 2.0+
- 至少4GB可用内存
- OpenAI API密钥

### 一键部署

```bash
# 1. 克隆或进入项目目录
cd 保险产品

# 2. 确保.env文件包含OPENAI_API_KEY
echo "OPENAI_API_KEY=your-api-key-here" > .env

# 3. 构建并启动所有服务
docker-compose up -d

# 4. 查看日志
docker-compose logs -f

# 5. 访问服务
# API文档: http://localhost:8000/docs
# Web界面: http://localhost:8000/web_demo_enhanced.html
```

## 服务端口

| 服务 | 端口 | 说明 |
|-----|------|------|
| RAG API | 8000 | FastAPI服务 |
| Elasticsearch | 9200 | 搜索引擎 |
| Kibana | 5601 | ES可视化(可选) |
| Nginx | 80/443 | 反向代理(生产) |

## 常用命令

### 基础操作
```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 重启服务
docker-compose restart rag-api

# 查看日志
docker-compose logs -f rag-api
docker-compose logs -f elasticsearch

# 查看运行状态
docker-compose ps
```

### 数据管理
```bash
# 备份Elasticsearch数据
docker-compose exec elasticsearch elasticsearch-dump \
  --input=http://localhost:9200/insurance_products \
  --output=/backup/insurance_products.json

# 清理数据卷
docker-compose down -v

# 进入容器调试
docker-compose exec rag-api bash
```

### 扩展服务
```bash
# 启动调试模式（包含Kibana）
docker-compose --profile debug up -d

# 启动生产模式（包含Nginx）
docker-compose --profile production up -d

# 扩展API服务实例
docker-compose up -d --scale rag-api=3
```

## 环境配置

### 开发环境
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  rag-api:
    environment:
      - LOG_LEVEL=DEBUG
    volumes:
      - ./:/app  # 热重载
```

### 生产环境
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  rag-api:
    environment:
      - LOG_LEVEL=INFO
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '0.5'
          memory: 1G
```

## 监控和健康检查

### 健康检查端点
- API健康: `http://localhost:8000/health`
- ES健康: `http://localhost:9200/_cluster/health`

### 查看资源使用
```bash
docker stats
```

## 故障排查

### 常见问题

1. **Elasticsearch启动失败**
```bash
# 增加虚拟内存限制
sudo sysctl -w vm.max_map_count=262144

# Windows WSL2
wsl -d docker-desktop sysctl -w vm.max_map_count=262144
```

2. **API连接ES失败**
```bash
# 检查网络
docker network inspect 保险产品_rag-network

# 测试连接
docker-compose exec rag-api curl http://elasticsearch:9200
```

3. **内存不足**
```bash
# 调整ES内存
# 修改docker-compose.yml中的ES_JAVA_OPTS
- "ES_JAVA_OPTS=-Xms256m -Xmx256m"
```

## 性能优化

### 1. 启用ES缓存
```yaml
elasticsearch:
  environment:
    - indices.queries.cache.size=20%
    - indices.fielddata.cache.size=30%
```

### 2. API并发优化
```yaml
rag-api:
  command: uvicorn es_rag_api:app --workers 4 --host 0.0.0.0 --port 8000
```

### 3. 使用Redis缓存
```yaml
redis:
  image: redis:alpine
  ports:
    - "6379:6379"
```

## 安全配置

### 1. 启用ES认证
```yaml
elasticsearch:
  environment:
    - xpack.security.enabled=true
    - ELASTIC_PASSWORD=your-password
```

### 2. 限制API访问
```yaml
rag-api:
  environment:
    - API_KEY=your-secure-api-key
    - ALLOWED_ORIGINS=https://yourdomain.com
```

### 3. HTTPS配置
参考`nginx.conf`配置SSL证书

## 备份和恢复

### 自动备份
```bash
# 创建备份脚本
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker-compose exec -T elasticsearch \
  elasticsearch-dump \
  --input=http://localhost:9200/insurance_products \
  --output=/backup/insurance_products_$DATE.json
EOF

# 添加到crontab
0 2 * * * /path/to/backup.sh
```

### 恢复数据
```bash
docker-compose exec elasticsearch \
  elasticsearch-dump \
  --output=http://localhost:9200/insurance_products \
  --input=/backup/insurance_products_20240101.json
```

## 升级指南

```bash
# 1. 备份数据
./backup.sh

# 2. 停止服务
docker-compose down

# 3. 更新代码
git pull

# 4. 重建镜像
docker-compose build --no-cache

# 5. 启动新版本
docker-compose up -d

# 6. 验证
curl http://localhost:8000/health
```

## 联系支持

如遇问题，请查看：
- 日志: `docker-compose logs`
- API文档: `http://localhost:8000/docs`
- ES状态: `http://localhost:9200/_cat/health?v`