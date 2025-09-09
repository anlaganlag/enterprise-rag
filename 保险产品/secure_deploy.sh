#!/bin/bash

# 安全部署脚本 - 用于生产环境外网访问

echo "======================================"
echo "   保险RAG系统 - 安全生产部署"
echo "======================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否为root用户
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}请使用sudo运行此脚本${NC}"
    exit 1
fi

# 1. 检查并配置防火墙
echo -e "${YELLOW}配置防火墙规则...${NC}"
if command -v ufw &> /dev/null; then
    ufw allow 80/tcp
    ufw allow 443/tcp
    ufw allow 8000/tcp
    echo -e "${GREEN}UFW防火墙规则已配置${NC}"
elif command -v firewall-cmd &> /dev/null; then
    firewall-cmd --add-port=80/tcp --permanent
    firewall-cmd --add-port=443/tcp --permanent
    firewall-cmd --add-port=8000/tcp --permanent
    firewall-cmd --reload
    echo -e "${GREEN}Firewalld防火墙规则已配置${NC}"
else
    echo -e "${YELLOW}未检测到防火墙，请手动配置${NC}"
fi

# 2. 生成随机密码
echo -e "${YELLOW}生成安全密码...${NC}"
ELASTIC_PASSWORD=$(openssl rand -base64 32)
API_KEY=$(openssl rand -hex 32)
REDIS_PASSWORD=$(openssl rand -base64 24)

# 3. 创建生产环境配置
echo -e "${YELLOW}创建生产环境配置...${NC}"
cat > .env << EOF
# 生产环境配置 - $(date)
OPENAI_API_KEY=${OPENAI_API_KEY:-your-api-key-here}
ELASTIC_PASSWORD=${ELASTIC_PASSWORD}
API_KEY=${API_KEY}
REDIS_PASSWORD=${REDIS_PASSWORD}
ALLOWED_ORIGINS=${ALLOWED_ORIGINS:-*}
LOG_LEVEL=INFO
EOF

echo -e "${GREEN}配置文件已创建${NC}"
echo -e "Elasticsearch密码: ${ELASTIC_PASSWORD}"
echo -e "API密钥: ${API_KEY}"
echo -e "Redis密码: ${REDIS_PASSWORD}"

# 4. 创建SSL证书目录
echo -e "${YELLOW}准备SSL证书...${NC}"
mkdir -p ssl
if [ ! -f ssl/cert.pem ]; then
    echo -e "${YELLOW}生成自签名证书（仅用于测试）...${NC}"
    openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem \
        -days 365 -nodes -subj "/CN=localhost"
    echo -e "${GREEN}自签名证书已生成${NC}"
    echo -e "${YELLOW}警告: 生产环境请使用正式SSL证书${NC}"
fi

# 5. 设置目录权限
echo -e "${YELLOW}设置目录权限...${NC}"
chmod 600 .env
chmod 700 ssl
chmod 600 ssl/*

# 6. 创建必要目录
mkdir -p uploads output logs
chmod 755 uploads output logs

# 7. 获取公网IP
PUBLIC_IP=$(curl -s ifconfig.me)
echo -e "${GREEN}服务器公网IP: ${PUBLIC_IP}${NC}"

# 8. 部署服务
echo -e "${YELLOW}启动生产环境服务...${NC}"
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d

# 9. 等待服务启动
echo -e "${YELLOW}等待服务启动...${NC}"
sleep 15

# 10. 健康检查
echo -e "${YELLOW}执行健康检查...${NC}"
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo -e "${GREEN}服务启动成功！${NC}"
else
    echo -e "${RED}服务启动失败，请检查日志${NC}"
    docker-compose -f docker-compose.prod.yml logs --tail=50
    exit 1
fi

# 11. 显示访问信息
echo ""
echo "======================================"
echo -e "${GREEN}部署完成！${NC}"
echo "======================================"
echo "外网访问地址:"
echo -e "- Web界面: ${GREEN}http://${PUBLIC_IP}:8000/web_demo_enhanced.html${NC}"
echo -e "- API文档: ${GREEN}http://${PUBLIC_IP}:8000/docs${NC}"
echo ""
echo "API认证:"
echo -e "- API Key: ${YELLOW}${API_KEY}${NC}"
echo ""
echo "安全建议:"
echo "1. 配置域名和正式SSL证书"
echo "2. 使用Nginx反向代理，不直接暴露8000端口"
echo "3. 定期更新密码和API密钥"
echo "4. 配置日志监控和告警"
echo ""
echo "管理命令:"
echo "- 查看日志: docker-compose -f docker-compose.prod.yml logs -f"
echo "- 停止服务: docker-compose -f docker-compose.prod.yml down"
echo "- 重启服务: docker-compose -f docker-compose.prod.yml restart"
echo "======================================"

# 12. 保存配置信息
cat > deployment-info.txt << EOF
部署时间: $(date)
公网IP: ${PUBLIC_IP}
API密钥: ${API_KEY}
Elasticsearch密码: ${ELASTIC_PASSWORD}
Redis密码: ${REDIS_PASSWORD}
EOF
chmod 600 deployment-info.txt

echo -e "${GREEN}配置信息已保存到 deployment-info.txt${NC}"