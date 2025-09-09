# 🤗 HuggingFace Spaces部署指南

## 快速部署（5分钟）

### 步骤1：创建HuggingFace账号
1. 访问 https://huggingface.co/join
2. 注册账号（完全免费）

### 步骤2：创建新Space
1. 登录后访问：https://huggingface.co/new-space
2. 填写信息：
   - **Space name**: `insurance-rag` （或任意名称）
   - **License**: MIT
   - **Select SDK**: Gradio
   - **Gradio SDK version**: 默认最新版
   - **Space hardware**: CPU basic (Free)
   - **Private/Public**: 根据需求选择

3. 点击 "Create Space"

### 步骤3：上传文件

#### 方法A：通过Web界面上传（推荐新手）
1. 进入创建的Space页面
2. 点击 "Files" 标签
3. 点击 "Add file" → "Upload files"
4. 上传以下文件：
   - `app.py` （主程序）
   - `requirements.txt` （依赖文件）
   - `README.md` （可选）

#### 方法B：通过Git上传（推荐开发者）
```bash
# 克隆Space仓库
git clone https://huggingface.co/spaces/YOUR_USERNAME/insurance-rag

# 复制文件
cp app.py insurance-rag/
cp requirements.txt insurance-rag/

# 提交并推送
cd insurance-rag
git add .
git commit -m "Initial deployment"
git push
```

### 步骤4：等待自动部署
- HuggingFace会自动检测文件变化并部署
- 部署状态会显示在页面顶部
- 通常需要2-5分钟完成

### 步骤5：访问应用
- 部署完成后，会看到Gradio界面
- URL格式：`https://huggingface.co/spaces/YOUR_USERNAME/insurance-rag`
- 可以分享给任何人使用！

## 🔧 配置优化

### 添加README.md
创建 `README.md` 文件，内容如下：
```markdown
---
title: Insurance RAG System
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.8.0
app_file: app.py
pinned: false
---

# 保险产品智能问答系统

基于RAG技术的保险文档智能分析系统

## Features
- PDF文档处理
- 智能问答
- 批量分析
- 结果导出
```

### 添加示例文件
可以在Space中添加示例PDF文件：
1. 创建 `examples/` 文件夹
2. 上传示例PDF文件
3. 用户可以直接使用示例文件测试

### 设置Secrets（可选）
如果想预设API密钥：
1. 进入Space Settings
2. 找到 "Repository secrets"
3. 添加：
   - Name: `OPENAI_API_KEY`
   - Value: 你的API密钥

在代码中获取：
```python
import os
api_key = os.environ.get("OPENAI_API_KEY", "")
```

## 🎯 使用技巧

### 1. 自定义界面主题
```python
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # 或使用其他主题：Base(), Glass(), Monochrome()
```

### 2. 添加进度条
```python
def process_with_progress(progress=gr.Progress()):
    for i in progress.tqdm(range(100)):
        # 处理逻辑
```

### 3. 添加示例输入
```python
gr.Examples(
    examples=[
        ["What is the minimum premium?"],
        ["What are the fees?"],
    ],
    inputs=[question_input]
)
```

### 4. 启用分享链接
```python
demo.launch(share=True)  # 生成72小时有效的公共链接
```

## 📊 资源限制

### 免费版限制
- **CPU**: 2 vCPU
- **RAM**: 16 GB
- **磁盘**: 50 GB
- **带宽**: 无限制
- **运行时间**: 无限制

### 性能优化建议
1. 使用更小的模型（gpt-3.5-turbo而非gpt-4）
2. 限制文档大小（建议<10MB）
3. 缓存处理结果
4. 批量处理请求

## 🐛 常见问题

### Q: 部署失败怎么办？
A: 检查requirements.txt中的包版本是否兼容

### Q: 应用很慢怎么办？
A: 
- 使用更小的chunk_size
- 减少检索的文档数量
- 考虑升级到GPU Space（付费）

### Q: 如何查看日志？
A: 在Space页面点击 "Logs" 标签查看实时日志

### Q: 如何更新应用？
A: 直接在Files中编辑或上传新文件，会自动重新部署

## 🔗 相关链接

- [Gradio文档](https://www.gradio.app/docs)
- [HuggingFace Spaces文档](https://huggingface.co/docs/hub/spaces)
- [示例Space](https://huggingface.co/spaces/gradio/chatbot)

## 💡 进阶功能

### 集成其他模型
```python
# 使用HuggingFace上的开源模型
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
```

### 添加认证
```python
demo.launch(auth=("admin", "password"))
```

### 自定义域名
升级到付费版后可以绑定自定义域名

---

**提示**: HuggingFace Spaces是完全免费的，非常适合MVP展示和小规模应用！