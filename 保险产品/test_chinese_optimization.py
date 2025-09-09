"""
测试中文优化效果
"""
import requests
import json
import time
from colorama import init, Fore, Style

init(autoreset=True)

# 测试两个API
BASE_URL_OLD = "http://localhost:8000"  # 原版API
BASE_URL_NEW = "http://localhost:8001"  # 中文优化版API

def test_single_query(url, query, endpoint="/api/qa"):
    """测试单个查询"""
    try:
        response = requests.post(
            f"{url}{endpoint}",
            json={"question": query, "include_sources": True},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "answer": data.get('answer', ''),
                "confidence": data.get('confidence', 0),
                "time": data.get('processing_time', 0),
                "language": data.get('language', 'unknown')
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def compare_apis():
    """对比两个API的中文处理能力"""
    print(f"{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}中文优化效果对比测试")
    print(f"{Fore.CYAN}{'='*80}\n")
    
    # 测试查询集
    test_queries = [
        "RoyalFortune的最低保费是多少？",
        "保证现金价值是多少？",
        "投保年龄范围是什么？",
        "身故赔付有哪些特点？",
        "这个产品适合什么样的客户？",
        "产品的优势是什么？",
        "如何购买这个产品？",
        "永明的产品有哪些特点？"
    ]
    
    results = []
    
    for query in test_queries:
        print(f"\n{Fore.YELLOW}测试查询: {query}")
        print("-"*60)
        
        # 测试原版API
        print(f"{Fore.WHITE}原版API结果:")
        old_result = test_single_query(BASE_URL_OLD, query)
        if old_result['success']:
            print(f"  置信度: {old_result['confidence']:.1%}")
            print(f"  回答: {old_result['answer'][:100]}...")
            print(f"  耗时: {old_result['time']:.2f}秒")
        else:
            print(f"  {Fore.RED}错误: {old_result.get('error', 'Unknown')}")
        
        # 测试中文优化版
        print(f"\n{Fore.WHITE}中文优化版结果:")
        new_result = test_single_query(BASE_URL_NEW, query, "/api/qa_cn")
        if new_result['success']:
            print(f"  置信度: {new_result['confidence']:.1%}")
            print(f"  语言: {new_result['language']}")
            print(f"  回答: {new_result['answer'][:100]}...")
            print(f"  耗时: {new_result['time']:.2f}秒")
        else:
            print(f"  {Fore.RED}错误: {new_result.get('error', 'Unknown')}")
        
        # 对比分析
        if old_result['success'] and new_result['success']:
            confidence_improvement = new_result['confidence'] - old_result['confidence']
            if confidence_improvement > 0:
                print(f"\n{Fore.GREEN}✅ 置信度提升: +{confidence_improvement:.1%}")
            elif confidence_improvement < 0:
                print(f"\n{Fore.RED}⚠️ 置信度下降: {confidence_improvement:.1%}")
            else:
                print(f"\n{Fore.YELLOW}➖ 置信度相同")
            
            # 检查回答质量
            if "not found" in old_result['answer'].lower() and "not found" not in new_result['answer'].lower():
                print(f"{Fore.GREEN}✅ 新版本成功回答了原版无法回答的问题")
            
            results.append({
                "query": query,
                "old_confidence": old_result['confidence'],
                "new_confidence": new_result['confidence'],
                "improvement": confidence_improvement
            })
        
        time.sleep(1)  # 避免API过载
    
    # 汇总统计
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}测试结果汇总")
    print(f"{Fore.CYAN}{'='*80}\n")
    
    if results:
        avg_old = sum(r['old_confidence'] for r in results) / len(results)
        avg_new = sum(r['new_confidence'] for r in results) / len(results)
        avg_improvement = sum(r['improvement'] for r in results) / len(results)
        
        print(f"原版平均置信度: {avg_old:.1%}")
        print(f"优化版平均置信度: {avg_new:.1%}")
        print(f"平均提升: {avg_improvement:.1%}")
        
        improved_count = sum(1 for r in results if r['improvement'] > 0)
        print(f"\n改进的查询: {improved_count}/{len(results)}")
        
        if avg_improvement > 0.1:
            print(f"\n{Fore.GREEN}🎉 中文优化显著提升了系统性能！")
        elif avg_improvement > 0:
            print(f"\n{Fore.YELLOW}✨ 中文优化有一定效果")
        else:
            print(f"\n{Fore.RED}⚠️ 需要进一步优化")

def test_query_expansion():
    """测试查询扩展功能"""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}查询扩展测试")
    print(f"{Fore.CYAN}{'='*80}\n")
    
    from chinese_query_optimizer import ChineseQueryOptimizer
    optimizer = ChineseQueryOptimizer()
    
    test_cases = [
        "最低保费",
        "身故赔付",
        "永明的产品",
        "100万美元"
    ]
    
    for query in test_cases:
        expanded, info = optimizer.optimize_for_chinese(query)
        print(f"\n原始查询: {query}")
        print(f"扩展结果:")
        for i, eq in enumerate(expanded[:3], 1):
            print(f"  {i}. {eq}")
        print(f"提取信息: {info}")

if __name__ == "__main__":
    # 先测试查询扩展
    test_query_expansion()
    
    # 再对比API性能
    print("\n等待3秒后开始API对比测试...")
    time.sleep(3)
    compare_apis()