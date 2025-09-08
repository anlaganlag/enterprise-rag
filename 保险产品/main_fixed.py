"""
主程序 - 保险产品RAG问答系统（修复版）
"""
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 强制加载.env文件
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
print(f"加载.env文件: {env_path}")
load_dotenv(dotenv_path=env_path, override=True)

# 验证API密钥
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("错误：OPENAI_API_KEY未设置")
    print("尝试从.env文件直接读取...")
    from dotenv import dotenv_values
    config = dotenv_values(".env")
    if "OPENAI_API_KEY" in config:
        api_key = config["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = api_key
        print(f"已手动设置API密钥（长度：{len(api_key)}）")
    else:
        print("无法找到API密钥，请检查.env文件")
        sys.exit(1)
else:
    print(f"API密钥已加载（长度：{len(api_key)}）")

# 清理代理设置
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if key in os.environ:
        del os.environ[key]
        print(f"已清理环境变量: {key}")

from config import (
    PDF_FILES, VECTOR_STORE_DIR, OUTPUT_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, LLM_MODEL, EMBEDDING_MODEL
)
from pdf_processor import PDFProcessor
from vector_store import VectorStoreManager
from insurance_qa_chain import InsuranceQAChain

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('insurance_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 定义34个问题
QUESTIONS = {
    "Product Information": [
        "Insurer Entity Name",
        "Insurer Financial Strength Rating(s)",
        "Issuing Jurisdiction",
        "Product Name",
        "Product Base",
        "Product Type",
        "Product Asset Manager",
        "Product Asset Custodian",
        "Product Asset Mix"
    ],
    "Plan Details": [
        "Issue Age and Age Methodology",
        "Number of Insured Lives",
        "Change of Life Assured Feature(s)",
        "Minimum Premium / Sum Assured",
        "Maximum Premium / Sum Assured",
        "Policy Term",
        "Premium Term(s)",
        "Prepayment Applicable?",
        "Policy Currency(ies)",
        "Withdrawal Features",
        "Death Settlement Feature(s)"
    ],
    "For Participating Whole of Life": [
        "Day 1 GCV",
        "Total Surrender Value Components",
        "Total Death Benefit Components"
    ],
    "Other Details": [
        "Backdating Availability?",
        "Non-Medical Limit",
        "Additional Benefits",
        "Contract Governing Law"
    ]
}

def build_vector_store():
    """构建向量存储"""
    logger.info("=== 开始构建向量存储 ===")
    
    # 1. 处理PDF文档
    logger.info("步骤1: 处理PDF文档")
    processor = PDFProcessor()
    pdf_paths = list(PDF_FILES.values())
    
    # 检查PDF文件是否存在
    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            logger.error(f"PDF文件不存在: {pdf_path}")
            return False
    
    documents = processor.process_multiple_pdfs(pdf_paths)
    
    # 2. 准备文档索引
    logger.info("步骤2: 准备文档索引")
    indexed_docs = processor.prepare_documents_for_indexing(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )
    logger.info(f"创建了 {len(indexed_docs)} 个文档块")
    
    # 3. 构建向量存储
    logger.info("步骤3: 构建向量存储")
    vector_manager = VectorStoreManager(embedding_model=EMBEDDING_MODEL)
    langchain_docs = vector_manager.create_documents(indexed_docs)
    vector_store = vector_manager.build_vector_store(langchain_docs)
    
    # 4. 保存向量存储
    logger.info("步骤4: 保存向量存储")
    vector_manager.save_vector_store(VECTOR_STORE_DIR / "insurance_vectors")
    
    logger.info("=== 向量存储构建完成 ===")
    return True

def run_qa_system():
    """运行问答系统"""
    logger.info("=== 开始运行问答系统 ===")
    
    # 1. 加载向量存储
    logger.info("步骤1: 加载向量存储")
    vector_manager = VectorStoreManager(embedding_model=EMBEDDING_MODEL)
    
    try:
        vector_manager.load_vector_store(VECTOR_STORE_DIR / "insurance_vectors")
    except FileNotFoundError:
        logger.warning("向量存储不存在，需要先构建")
        if not build_vector_store():
            logger.error("构建向量存储失败")
            return False
        vector_manager.load_vector_store(VECTOR_STORE_DIR / "insurance_vectors")
    
    # 2. 初始化问答链
    logger.info("步骤2: 初始化问答链")
    qa_chain = InsuranceQAChain(vector_manager, llm_model=LLM_MODEL)
    
    # 3. 回答所有问题
    logger.info("步骤3: 开始回答34个问题")
    results = qa_chain.answer_all_questions(QUESTIONS)
    
    # 4. 保存结果
    logger.info("步骤4: 保存结果")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"insurance_qa_results_{timestamp}"
    qa_chain.save_results(results, output_path)
    
    # 5. 打印摘要
    summary = results.get('summary', {})
    logger.info("=== 执行完成 ===")
    logger.info(f"总问题数: {summary.get('total_questions', 0)}")
    logger.info(f"成功回答: {summary.get('answered', 0)}")
    logger.info(f"未找到答案: {summary.get('not_found', 0)}")
    logger.info(f"成功率: {summary.get('answered', 0) / summary.get('total_questions', 1) * 100:.1f}%")
    logger.info(f"总Token使用: {summary.get('total_tokens', 0)}")
    logger.info(f"总成本: ${summary.get('total_cost', 0):.4f}")
    
    return True

def test_single_question(question: str):
    """测试单个问题"""
    logger.info(f"测试问题: {question}")
    
    # 加载向量存储
    vector_manager = VectorStoreManager(embedding_model=EMBEDDING_MODEL)
    
    try:
        vector_manager.load_vector_store(VECTOR_STORE_DIR / "insurance_vectors")
    except FileNotFoundError:
        logger.error("向量存储不存在，请先运行 python main_fixed.py build")
        return
    
    # 初始化问答链
    qa_chain = InsuranceQAChain(vector_manager, llm_model=LLM_MODEL)
    
    # 回答问题
    result = qa_chain.answer_question(question)
    
    print("\n" + "="*50)
    print(f"问题: {question}")
    print(f"答案: {result['answer']}")
    print(f"Token使用: {result.get('tokens_used', {})}")
    print("="*50 + "\n")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='保险产品RAG问答系统')
    parser.add_argument('command', choices=['build', 'run', 'test', 'all'],
                      help='执行命令: build(构建向量库), run(运行问答), test(测试单个问题), all(构建并运行)')
    parser.add_argument('--question', type=str, help='测试模式下的问题')
    
    args = parser.parse_args()
    
    if args.command == 'build':
        build_vector_store()
    elif args.command == 'run':
        run_qa_system()
    elif args.command == 'test':
        if args.question:
            test_single_question(args.question)
        else:
            test_single_question("What is the minimum premium?")
    elif args.command == 'all':
        if build_vector_store():
            run_qa_system()

if __name__ == "__main__":
    main()