# 医疗RAG智能助手系统

基于RAG最佳实践的医疗问答系统，专注于提升医疗回答的准确率和可追溯性。

## 系统架构

### 核心组件
1. **医疗知识库** - 存储和索引医疗文档
2. **检索门控** - 智能判断是否需要检索
3. **混合检索** - BM25 + 向量检索 + 重排序
4. **结构化输出** - JSON格式的医疗答案
5. **幻觉控制** - 防止错误医疗信息
6. **质量监控** - 持续评估和优化

### 技术栈
- **LLM**: GPT-4/Claude-3.5
- **Embedding**: BioBERT/ClinicalBERT
- **向量数据库**: Pinecone/Weaviate
- **关键词检索**: Elasticsearch + BM25
- **缓存**: Redis
- **监控**: Prometheus + Grafana

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env

# 启动服务
python main.py
```

## 项目结构

```
medical_rag_system/
├── src/
│   ├── knowledge_base/     # 知识库管理
│   ├── retrieval/          # 检索系统
│   ├── generation/         # 答案生成
│   ├── validation/         # 质量验证
│   └── monitoring/         # 监控系统
├── data/                   # 医疗数据
├── tests/                  # 测试用例
└── docs/                   # 文档
```

