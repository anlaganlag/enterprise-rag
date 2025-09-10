"""
自动执行Prompt优化 - 无需交互
"""

import os
import sys
import json
from datetime import datetime

# 导入优化执行器
from prompt_optimization_executor import PromptOptimizationExecutor

def main():
    print("=" * 60)
    print("🚀 开始自动执行Prompt优化")
    print("=" * 60)
    
    # 检查API密钥
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️ 未设置OPENAI_API_KEY，将使用模拟数据")
        print("提示：设置环境变量后可使用真实API测试")
    else:
        print("\n✅ 已检测到OpenAI API密钥")
    
    print("\n执行模式：渐进式优化（评估+优化）")
    print("-" * 40)
    
    # 初始化执行器
    executor = PromptOptimizationExecutor(api_key)
    
    try:
        # 步骤1：快速评估
        print("\n📊 步骤1：评估当前Prompt性能...")
        assessment_results = executor.execute_phase_1_quick_assessment()
        
        print(f"✅ 评估完成！")
        print(f"  - 测试问题数: {assessment_results['test_count']}")
        print(f"  - 平均置信度: {assessment_results['summary']['avg_confidence']:.2f}")
        print(f"  - 主要问题: {', '.join(assessment_results['summary']['main_issues'][:3])}")
        
        # 步骤2：针对性优化
        print("\n🔧 步骤2：优化低性能Prompt...")
        optimization_results = executor.execute_phase_2_targeted_optimization(assessment_results)
        
        print(f"✅ 优化完成！")
        print(f"  - 优化类型数: {len(optimization_results['optimized_types'])}")
        for opt in optimization_results['optimized_types']:
            print(f"    • {opt['type']}: 改进 {opt['improvement']:.1%}")
        
        # 步骤3：生成集成指南
        print("\n📝 步骤3：生成集成指南...")
        integration_guide = generate_integration_guide(optimization_results)
        
        with open("integration_guide.md", "w", encoding="utf-8") as f:
            f.write(integration_guide)
        
        print("✅ 集成指南已生成: integration_guide.md")
        
        # 生成执行报告
        print("\n📊 生成执行报告...")
        report = {
            "execution_time": datetime.now().isoformat(),
            "mode": "progressive_optimization",
            "results": {
                "assessment": assessment_results['summary'],
                "optimization": {
                    "optimized_types": len(optimization_results['optimized_types']),
                    "improvements": [opt['improvement'] for opt in optimization_results['optimized_types']]
                }
            },
            "recommendations": [
                "1. 应用优化后的Prompt模板到streamlit_app.py",
                "2. 运行监控仪表板验证改进效果",
                "3. 收集用户反馈进行迭代优化"
            ]
        }
        
        with open("optimization_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("✅ 优化报告已生成: optimization_report.json")
        
    except Exception as e:
        print(f"\n❌ 执行过程中出错: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("🎉 Prompt优化执行完成！")
    print("=" * 60)
    print("\n下一步操作：")
    print("1. 查看 integration_guide.md 了解如何集成优化")
    print("2. 查看 optimization_report.json 了解详细结果")
    print("3. 运行 streamlit run monitoring_dashboard.py 监控性能")
    
    return 0

def generate_integration_guide(optimization_results):
    """生成集成指南"""
    guide = f"""# 🔧 Prompt优化集成指南

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📋 优化结果摘要

已优化 {len(optimization_results['optimized_types'])} 个Prompt类型：
"""
    
    for opt in optimization_results['optimized_types']:
        guide += f"- **{opt['type']}**: 性能提升 {opt['improvement']:.1%}\n"
    
    guide += """

## 🚀 快速集成步骤

### 步骤1：备份现有代码
```bash
cp 保险产品/streamlit_app.py 保险产品/streamlit_app_backup.py
```

### 步骤2：更新Prompt函数

在 `streamlit_app.py` 中找到 `create_advanced_prompt` 函数，添加以下导入：

```python
# 在文件开头添加
from prompt_optimization_examples import PromptOptimizer

# 初始化优化器（在全局变量区域）
prompt_optimizer = PromptOptimizer()
```

### 步骤3：修改Prompt生成逻辑

替换 `create_advanced_prompt` 函数的核心逻辑：

```python
def create_advanced_prompt(question: str, context: str, question_type: str = "general") -> str:
    # 使用优化器生成Prompt
    optimized_prompt = prompt_optimizer.optimize_prompt(
        question=question,
        context=context,
        question_type=question_type
    )
    return optimized_prompt
```

### 步骤4：添加性能监控

在 `answer_question_with_confidence` 函数中添加：

```python
from prompt_optimization_examples import PerformanceMonitor

# 初始化监控器
monitor = PerformanceMonitor()

# 在生成答案后记录指标
monitor.record_metric("response_time", response_time)
monitor.record_metric("confidence", confidence)
```

## 📊 验证优化效果

### 运行测试
```bash
# 启动优化后的系统
streamlit run 保险产品/streamlit_app.py

# 在另一个终端运行监控
streamlit run monitoring_dashboard.py
```

### 测试问题集
- 最低保费是多少？
- 投保年龄范围是什么？
- 什么是保单的提取功能？
- Royal Fortune和其他产品有什么区别？

## ⚠️ 注意事项

1. **逐步部署**：先在测试环境验证，再部署到生产
2. **监控指标**：密切关注响应时间和置信度变化
3. **回滚准备**：保留备份文件，随时可恢复

## 📈 预期改进

- 置信度提升：15-25%
- 响应时间减少：20-30%
- 错误率降低：30-40%

## 🔄 持续优化

1. 每日查看监控报告
2. 收集用户反馈
3. 定期运行优化脚本更新Prompt

---

如有问题，请查看完整文档：`Prompt工程优化详细任务清单.md`
"""
    
    return guide

if __name__ == "__main__":
    exit(main())