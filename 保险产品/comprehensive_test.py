"""
完整功能测试脚本
测试RAG系统的所有核心功能
"""
import requests
import json
import time
from colorama import init, Fore, Back, Style

init(autoreset=True)

BASE_URL = "http://localhost:8000"

def print_header(text):
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}{text}")
    print(f"{Fore.CYAN}{'='*60}")

def print_success(text):
    print(f"{Fore.GREEN}✓ {text}")

def print_error(text):
    print(f"{Fore.RED}✗ {text}")

def print_info(text):
    print(f"{Fore.YELLOW}→ {text}")

# 1. 健康检查
def test_health():
    print_header("1. 系统健康检查")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code == 200:
            data = response.json()
            print_success(f"API状态: {data['status']}")
            print_success(f"ElasticSearch: {data['elasticsearch']['status']}")
            print_info(f"文档索引: {'存在' if data['indices'].get('documents') else '不存在'}")
            return True
        else:
            print_error(f"健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"连接失败: {e}")
        return False

# 2. 统计信息
def test_stats():
    print_header("2. 索引统计信息")
    try:
        response = requests.get(f"{BASE_URL}/api/stats")
        if response.status_code == 200:
            data = response.json()
            print_info(f"文档总数: {data['total_documents']}")
            print_info(f"分块总数: {data['total_chunks']}")
            print_info(f"索引大小: {data['index_size']}")
            return True
    except Exception as e:
        print_error(f"获取统计失败: {e}")
        return False

# 3. 列出文档
def test_list_documents():
    print_header("3. 已上传文档列表")
    try:
        response = requests.get(f"{BASE_URL}/api/documents")
        if response.status_code == 200:
            data = response.json()
            print_info(f"文档总数: {data['total']}")
            for doc in data['documents']:
                print(f"  • {doc['doc_name']} (分块数: {doc['chunk_count']})")
            return data['documents']
        return []
    except Exception as e:
        print_error(f"列出文档失败: {e}")
        return []

# 4. 上传文档
def test_upload():
    print_header("4. 上传PDF文档测试")
    
    # 检查是否有测试文件
    test_files = [
        "RoyalFortune_Product Brochure_EN.pdf",
        "AIA FlexiAchieverSavingsPlan_tc-活享儲蓄計劃.pdf"
    ]
    
    for filename in test_files:
        try:
            print_info(f"尝试上传: {filename}")
            with open(filename, 'rb') as f:
                files = {'file': (filename, f, 'application/pdf')}
                response = requests.post(f"{BASE_URL}/api/upload", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    print_success(f"上传成功: {data['filename']}")
                    print_info(f"文档ID: {data['doc_id']}")
                    print_info(f"大小: {data['size']} bytes")
                    print_info("等待处理完成...")
                    time.sleep(5)  # 等待处理
                    return True
                else:
                    print_error(f"上传失败: {response.status_code}")
        except FileNotFoundError:
            print_info(f"文件不存在: {filename}")
        except Exception as e:
            print_error(f"上传错误: {e}")
    
    return False

# 5. 搜索测试
def test_search():
    print_header("5. 搜索功能测试")
    
    test_queries = [
        ("minimum premium", "测试最低保费搜索"),
        ("保费", "测试中文搜索"),
        ("RoyalFortune", "测试产品名搜索"),
        ("withdrawal", "测试提取功能搜索")
    ]
    
    for query, description in test_queries:
        print_info(f"\n{description}: '{query}'")
        try:
            response = requests.post(
                f"{BASE_URL}/api/search",
                json={"query": query, "size": 3}
            )
            
            if response.status_code == 200:
                results = response.json()
                print_success(f"找到 {len(results)} 个结果")
                
                for i, result in enumerate(results[:2], 1):
                    print(f"  {i}. {result['doc_name']} (页{result['page_num']})")
                    print(f"     分数: {result['score']:.2f}")
                    if result.get('highlight'):
                        # 清理HTML标签
                        highlight = result['highlight'].replace('<mark>', '【').replace('</mark>', '】')
                        print(f"     高亮: {highlight[:100]}...")
            else:
                print_error(f"搜索失败: {response.status_code}")
                
        except Exception as e:
            print_error(f"搜索错误: {e}")

# 6. 问答测试
def test_qa():
    print_header("6. 问答功能测试")
    
    test_questions = [
        "What is the minimum premium for RoyalFortune?",
        "RoyalFortune的最低保费是多少？",
        "What is the issue age range?",
        "What are the withdrawal features?",
        "友邦保险的产品有哪些特点？"
    ]
    
    for question in test_questions:
        print_info(f"\n问题: {question}")
        try:
            response = requests.post(
                f"{BASE_URL}/api/qa",
                json={"question": question, "include_sources": True}
            )
            
            if response.status_code == 200:
                data = response.json()
                print_success("获得答案:")
                print(f"{Fore.WHITE}  {data['answer'][:200]}{'...' if len(data['answer']) > 200 else ''}")
                print_info(f"置信度: {data['confidence']:.2%}")
                print_info(f"处理时间: {data['processing_time']:.2f}秒")
                
                if data['sources']:
                    print_info("信息来源:")
                    for source in data['sources'][:2]:
                        print(f"  • {source['doc_name']} (页{source['page_num']}, 分数:{source['score']:.2f})")
            else:
                print_error(f"问答失败: {response.status_code}")
                
        except Exception as e:
            print_error(f"问答错误: {e}")

# 7. 完整流程测试
def test_full_workflow():
    print_header("7. 完整工作流测试")
    
    # 步骤1: 上传文档
    print_info("步骤1: 上传新文档")
    # (如果需要的话)
    
    # 步骤2: 等待处理
    print_info("步骤2: 等待索引...")
    time.sleep(3)
    
    # 步骤3: 搜索验证
    print_info("步骤3: 验证搜索")
    response = requests.post(
        f"{BASE_URL}/api/search",
        json={"query": "USD 125,000", "size": 1}
    )
    if response.status_code == 200 and response.json():
        print_success("搜索验证成功")
    
    # 步骤4: 问答验证
    print_info("步骤4: 验证问答")
    response = requests.post(
        f"{BASE_URL}/api/qa",
        json={"question": "What is the minimum notional amount?"}
    )
    if response.status_code == 200:
        answer = response.json()['answer']
        if "125,000" in answer or "125000" in answer:
            print_success("问答验证成功 - 正确识别最低金额")
        else:
            print_error("问答内容可能不准确")

# 主测试函数
def run_all_tests():
    print(f"{Fore.CYAN}{Style.BRIGHT}")
    print("╔" + "═"*58 + "╗")
    print("║" + " "*20 + "RAG系统综合测试" + " "*23 + "║")
    print("╚" + "═"*58 + "╝")
    print(Style.RESET_ALL)
    
    # 运行所有测试
    if not test_health():
        print_error("系统不健康，请先启动API服务")
        return
    
    test_stats()
    docs = test_list_documents()
    
    # 如果没有文档，尝试上传
    if not docs:
        print_info("没有文档，尝试上传...")
        test_upload()
        time.sleep(5)  # 等待处理
    
    test_search()
    test_qa()
    test_full_workflow()
    
    print_header("测试完成")
    print_success("所有测试已执行完毕！")

if __name__ == "__main__":
    run_all_tests()