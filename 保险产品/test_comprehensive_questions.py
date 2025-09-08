"""
测试综合性问题回答能力
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

# 综合性问题列表
comprehensive_questions = [
    # 产品介绍类
    ("请介绍一下RoyalFortune产品", "general_intro"),
    ("What is RoyalFortune insurance product?", "general_intro_en"),
    ("Tell me about the insurance products you have", "product_overview"),
    
    # 产品特点类
    ("RoyalFortune有哪些主要特点？", "key_features_cn"),
    ("What are the key features of RoyalFortune?", "key_features_en"),
    ("这个产品适合什么样的客户？", "target_customers"),
    
    # 比较分析类
    ("RoyalFortune的优势是什么？", "advantages"),
    ("What makes RoyalFortune different from other products?", "differentiation"),
    
    # 投资收益类
    ("这个产品的收益如何？", "returns"),
    ("How does the guaranteed cash value work?", "gcv_mechanism"),
    
    # 综合咨询类
    ("我想了解一下你们的保险产品", "general_inquiry"),
    ("Can you provide a summary of all insurance products?", "all_products"),
    ("我有100万美元，适合买什么产品？", "recommendation"),
    
    # 流程类
    ("如何购买这个产品？", "purchase_process"),
    ("What is the application process?", "application_process"),
]

def test_question(question, question_type):
    """测试单个综合性问题"""
    try:
        print(f"\n{'='*60}")
        print(f"问题类型: {question_type}")
        print(f"问题: {question}")
        print("-"*60)
        
        # 调用QA接口
        response = requests.post(
            f"{BASE_URL}/api/qa",
            json={"question": question, "include_sources": True},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get('answer', 'No answer')
            confidence = data.get('confidence', 0)
            sources = data.get('sources', [])
            processing_time = data.get('processing_time', 0)
            
            # 评估答案质量
            answer_quality = evaluate_answer_quality(answer, confidence)
            
            print(f"回答: {answer[:300]}{'...' if len(answer) > 300 else ''}")
            print(f"\n置信度: {confidence:.1%}")
            print(f"处理时间: {processing_time:.2f}秒")
            print(f"答案质量: {answer_quality}")
            
            if sources:
                print(f"信息来源: {len(sources)}个文档片段")
                for i, source in enumerate(sources[:2], 1):
                    print(f"  {i}. {source['doc_name']} (页{source['page_num']}, 分数:{source['score']:.2f})")
            
            return {
                "question": question,
                "type": question_type,
                "answered": answer_quality != "无效回答",
                "quality": answer_quality,
                "confidence": confidence,
                "answer_length": len(answer),
                "sources_count": len(sources)
            }
    except Exception as e:
        print(f"错误: {str(e)}")
        return {
            "question": question,
            "type": question_type,
            "answered": False,
            "quality": "错误",
            "confidence": 0,
            "answer_length": 0,
            "sources_count": 0
        }

def evaluate_answer_quality(answer, confidence):
    """评估答案质量"""
    answer_lower = answer.lower()
    
    # 无效答案标志
    invalid_markers = [
        "no relevant information",
        "not found",
        "not specified",
        "not mentioned",
        "not explicitly",
        "does not specify",
        "cannot find"
    ]
    
    # 检查是否为无效答案
    if any(marker in answer_lower for marker in invalid_markers):
        return "无效回答"
    
    # 根据答案长度和置信度评估
    if confidence > 0.8:
        if len(answer) > 200:
            return "详细回答"
        elif len(answer) > 100:
            return "标准回答"
        else:
            return "简短回答"
    elif confidence > 0.5:
        return "部分回答"
    else:
        return "低质量回答"

def main():
    print("="*80)
    print("综合性问题回答能力测试")
    print("="*80)
    
    results = []
    
    # 测试所有问题
    for question, q_type in comprehensive_questions:
        result = test_question(question, q_type)
        results.append(result)
        time.sleep(1)  # 避免API过载
    
    # 分析结果
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    
    total = len(results)
    answered = sum(1 for r in results if r["answered"])
    
    print(f"\n总问题数: {total}")
    print(f"有效回答: {answered}")
    print(f"回答率: {answered/total:.1%}")
    
    # 按质量分类
    quality_stats = {}
    for r in results:
        quality = r["quality"]
        quality_stats[quality] = quality_stats.get(quality, 0) + 1
    
    print("\n答案质量分布:")
    for quality, count in sorted(quality_stats.items()):
        print(f"  {quality}: {count}个 ({count/total:.1%})")
    
    # 按类型分析
    type_stats = {}
    for r in results:
        q_type = r["type"]
        if q_type not in type_stats:
            type_stats[q_type] = {"total": 0, "answered": 0}
        type_stats[q_type]["total"] += 1
        if r["answered"]:
            type_stats[q_type]["answered"] += 1
    
    print("\n按问题类型分析:")
    for q_type, stats in type_stats.items():
        rate = stats["answered"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {q_type}: {stats['answered']}/{stats['total']} ({rate:.0%})")
    
    # 找出表现最好和最差的问题
    print("\n表现最好的问题:")
    best = sorted(results, key=lambda x: x["confidence"], reverse=True)[:3]
    for i, r in enumerate(best, 1):
        print(f"  {i}. {r['question'][:50]}... (置信度: {r['confidence']:.1%})")
    
    print("\n需要改进的问题:")
    worst = [r for r in results if not r["answered"]][:3]
    for i, r in enumerate(worst, 1):
        print(f"  {i}. {r['question'][:50]}...")
    
    # 保存结果
    with open("comprehensive_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n详细结果已保存到: comprehensive_test_results.json")
    
    # 总体评估
    print("\n" + "="*80)
    print("系统能力评估")
    print("="*80)
    
    if answered/total > 0.7:
        print("✅ 系统具有较好的综合问答能力")
    elif answered/total > 0.5:
        print("⚠️ 系统具有一定的综合问答能力，但需要改进")
    else:
        print("❌ 系统综合问答能力较弱，需要优化")
    
    print("\n建议改进方向:")
    if "无效回答" in quality_stats and quality_stats["无效回答"] > total * 0.3:
        print("  - 增强语义理解和信息整合能力")
    if answered < total * 0.8:
        print("  - 补充更多文档以覆盖更多问题")
    print("  - 优化prompt以生成更自然的回答")
    print("  - 增加多轮对话能力以获取更多上下文")

if __name__ == "__main__":
    main()