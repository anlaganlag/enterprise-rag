@echo off
echo === 混合检索RAG系统 ===
echo.

echo 步骤1: 安装依赖...
pip install rank-bm25 jieba pdfplumber

echo.
echo 步骤2: 运行混合检索系统...
python hybrid_rag.py all

echo.
echo 执行完成！
echo 结果保存在 output 文件夹中
pause