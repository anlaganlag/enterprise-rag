# 保险产品RAG问答系统 MVP

## 项目概述
基于检索增强生成(RAG)技术的保险产品信息提取系统，用于自动回答关于保险产品的34个结构化问题。

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置环境变量
复制 `.env.example` 为 `.env` 并填入你的OpenAI API密钥：
```bash
cp .env.example .env
# 编辑 .env 文件，填入 OPENAI_API_KEY
```

### 3. 运行系统

#### 构建向量存储（首次运行）
```bash
python main.py build
```

#### 运行问答系统
```bash
python main.py run
```

#### 一键构建并运行
```bash
python main.py all
```

#### 测试单个问题
```bash
python main.py test --question "What is the minimum premium?"
```

## 项目结构
```
保险产品/
├── AIA FlexiAchieverSavingsPlan_tc-活享儲蓄計劃.pdf  # PDF文档1
├── RoyalFortune_Product Brochure_EN.pdf           # PDF文档2
├── 待回答问题.md                                   # 34个待回答问题
├── requirements.txt                                # Python依赖
├── .env.example                                   # 环境变量示例
├── config.py                                      # 配置文件
├── pdf_processor.py                               # PDF处理模块
├── vector_store.py                                # 向量存储模块
├── insurance_qa_chain.py                          # 问答链模块
├── main.py                                        # 主程序
├── vector_store/                                  # 向量存储目录
└── output/                                        # 输出结果目录
```

## 核心功能

### PDF处理
- 自动提取PDF文本内容
- 识别和提取表格数据
- 智能文本分块（500字符/块）

### 向量检索
- 使用OpenAI Embeddings进行向量化
- FAISS本地向量存储
- Top-5相似度检索

### 问答生成
- GPT-4 Turbo驱动的答案生成
- 结构化输出格式
- 置信度评分

## 输出格式

系统会生成两种格式的输出文件：

1. **JSON格式**：`output/insurance_qa_results_[timestamp].json`
   - 包含完整的答案、来源、置信度等信息

2. **Markdown格式**：`output/insurance_qa_results_[timestamp].md`
   - 人类可读的格式化报告

## 性能指标

- **预期准确率**：60-70%（MVP版本）
- **处理时间**：约5-10分钟（34个问题）
- **Token消耗**：约20,000-30,000 tokens
- **成本估算**：$0.5-1.0（使用GPT-4）

## 常见问题

### Q: 如何提高准确率？
A: 可以通过以下方式提升：
- 调整chunk_size和overlap参数
- 优化prompt模板
- 增加检索结果数量(k值)

### Q: 支持哪些PDF格式？
A: 支持标准PDF文本和扫描件（通过OCR）

### Q: 如何处理中文繁体？
A: 系统自动支持中英文混合处理

## 后续优化方向

1. **Phase 2**：加入BM25混合检索
2. **Phase 3**：集成Elasticsearch
3. **Phase 4**：添加重排序器和多轮验证

## 许可证
MIT

## 联系方式
如有问题，请提交Issue或联系维护者。