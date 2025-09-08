"""
测试所有待回答问题
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

# 所有待回答的问题
questions = {
    "Product Information": [
        ("Insurer Entity Name", "What is the insurer entity name?"),
        ("Insurer Financial Strength Rating(s)", "What are the insurer financial strength ratings?"),
        ("Issuing Jurisdiction", "What is the issuing jurisdiction?"),
        ("Product Name", "What is the product name?"),
        ("Product Base", "What is the product base?"),
        ("Product Type", "What is the product type?"),
        ("Product Asset Manager", "Who is the product asset manager?"),
        ("Product Asset Custodian", "Who is the product asset custodian?"),
        ("Product Asset Mix", "What is the product asset mix?"),
    ],
    "Plan Details": [
        ("Issue Age", "What is the issue age range and age methodology?"),
        ("Number of Insured Lives", "What is the number of insured lives?"),
        ("Change of Life Assured Feature(s)", "What are the change of life assured features?"),
        ("Minimum Premium", "What is the minimum premium or sum assured?"),
        ("Maximum Premium", "What is the maximum premium or sum assured?"),
        ("Policy Term", "What is the policy term?"),
        ("Premium Term(s)", "What are the premium terms?"),
        ("Prepayment Applicable", "Is prepayment applicable?"),
        ("Policy Currency(ies)", "What are the policy currencies?"),
        ("Withdrawal Features", "What are the withdrawal features?"),
        ("Death Settlement Feature(s)", "What are the death settlement features?"),
    ],
    "For Participating Whole of Life": [
        ("Day 1 GCV", "What is the Day 1 GCV as percentage of single premium?"),
        ("Total Surrender Value Components", "What are the total surrender value components?"),
        ("Total Death Benefit Components", "What are the total death benefit components?"),
    ],
    "Other Details": [
        ("Backdating Availability", "Is backdating available?"),
        ("Non-Medical Limit", "What is the non-medical limit?"),
        ("Additional Benefits", "What are the additional benefits like terminal illness or accidental death?"),
        ("Contract Governing Law", "What is the contract governing law?"),
    ]
}

def test_question(question_text):
    """测试单个问题"""
    try:
        response = requests.post(
            f"{BASE_URL}/api/qa",
            json={"question": question_text, "include_sources": True},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get('answer', 'No answer')
            confidence = data.get('confidence', 0)
            sources = data.get('sources', [])
            
            # 判断是否有效回答
            is_answered = (
                confidence > 0.5 and 
                "no relevant information" not in answer.lower() and
                "not found" not in answer.lower() and
                "not specified" not in answer.lower() and
                "not mention" not in answer.lower()
            )
            
            return {
                "answered": is_answered,
                "answer": answer[:200] if len(answer) > 200 else answer,
                "confidence": confidence,
                "source_docs": [s['doc_name'] for s in sources[:2]] if sources else []
            }
    except Exception as e:
        return {
            "answered": False,
            "answer": f"Error: {str(e)}",
            "confidence": 0,
            "source_docs": []
        }

def main():
    print("="*80)
    print("测试所有待回答问题")
    print("="*80)
    
    results = {}
    total_questions = 0
    answered_questions = 0
    
    for category, question_list in questions.items():
        print(f"\n## {category}")
        print("-"*40)
        
        category_results = []
        
        for field_name, question_text in question_list:
            print(f"\n📝 {field_name}")
            print(f"   问题: {question_text}")
            
            result = test_question(question_text)
            total_questions += 1
            
            if result["answered"]:
                answered_questions += 1
                print(f"   ✅ 可以回答 (置信度: {result['confidence']:.1%})")
                print(f"   答案: {result['answer']}")
                if result['source_docs']:
                    print(f"   来源: {', '.join(result['source_docs'])}")
            else:
                print(f"   ❌ 无法回答 (置信度: {result['confidence']:.1%})")
                if result['source_docs']:
                    print(f"   已搜索: {', '.join(result['source_docs'])}")
            
            category_results.append({
                "field": field_name,
                "answered": result["answered"],
                "confidence": result["confidence"],
                "answer": result["answer"],
                "sources": result["source_docs"]
            })
            
            # 避免API过载
            time.sleep(0.5)
        
        results[category] = category_results
    
    # 汇总报告
    print("\n" + "="*80)
    print("汇总报告")
    print("="*80)
    
    print(f"\n总问题数: {total_questions}")
    print(f"可回答数: {answered_questions}")
    print(f"回答率: {answered_questions/total_questions:.1%}")
    
    print("\n## 按类别统计:")
    for category, category_results in results.items():
        answered = sum(1 for r in category_results if r["answered"])
        total = len(category_results)
        print(f"  {category}: {answered}/{total} ({answered/total:.1%})")
    
    print("\n## 需要的文档:")
    print("\n当前已有文档:")
    print("  ✅ RoyalFortune_Product Brochure_EN.pdf (Sun Life)")
    
    print("\n缺失的文档（推测）:")
    unanswered = []
    for category, category_results in results.items():
        for result in category_results:
            if not result["answered"]:
                unanswered.append(result["field"])
    
    if unanswered:
        print("  ❌ 以下字段无法回答，可能需要额外文档:")
        for field in unanswered[:10]:  # 只显示前10个
            print(f"     - {field}")
        
        print("\n  建议上传:")
        print("     - 产品条款文档 (Terms and Conditions)")
        print("     - 投资管理文档 (Investment Management)")
        print("     - 监管披露文档 (Regulatory Disclosure)")
        print("     - 中文版产品说明书")
    
    # 保存详细结果
    with open("question_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n详细结果已保存到: question_test_results.json")

if __name__ == "__main__":
    main()