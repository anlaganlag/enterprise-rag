@echo off
echo 清理并重新安装依赖...

REM 清理损坏的包
echo 步骤1: 清理损坏的包
cd D:\Python310\lib\site-packages
rmdir /s /q "-reamlit" 2>nul
rmdir /s /q "-treamlit" 2>nul
rmdir /s /q "-" 2>nul

REM 卸载冲突的包
echo 步骤2: 卸载现有包
pip uninstall langchain langchain-community langchain-openai langsmith openai -y

REM 清理pip缓存
echo 步骤3: 清理pip缓存
pip cache purge

REM 安装新版本
echo 步骤4: 安装更新版本
cd D:\桌面\RAG\保险产品
pip install -r requirements_updated.txt

echo 安装完成！
pause