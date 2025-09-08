"""
表格提取调试工具 - 诊断PDF表格提取问题
"""
import pdfplumber
import PyPDF2
import re
from pathlib import Path
import json

def debug_table_extraction(pdf_path):
    """详细调试表格提取"""
    print(f"\n{'='*60}")
    print(f"调试文件: {Path(pdf_path).name}")
    print(f"{'='*60}")
    
    results = {
        "file": Path(pdf_path).name,
        "pages": [],
        "tables_found": 0,
        "text_tables_found": 0,
        "potential_table_text": []
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"总页数: {len(pdf.pages)}")
            
            for page_num, page in enumerate(pdf.pages):
                print(f"\n--- 第 {page_num + 1} 页 ---")
                page_result = {
                    "page": page_num + 1,
                    "tables": [],
                    "text_tables": []
                }
                
                # 方法1: 默认表格提取
                print("尝试默认表格提取...")
                tables = page.extract_tables()
                if tables:
                    print(f"  ✓ 找到 {len(tables)} 个表格")
                    for i, table in enumerate(tables):
                        if table and len(table) > 0:
                            print(f"    表格{i+1}: {len(table)}行 x {len(table[0])}列")
                            # 打印前3行
                            for row_idx in range(min(3, len(table))):
                                row = table[row_idx]
                                print(f"      行{row_idx+1}: {row[:3]}..." if len(row) > 3 else f"      行{row_idx+1}: {row}")
                            
                            page_result["tables"].append({
                                "index": i,
                                "rows": len(table),
                                "cols": len(table[0]) if table else 0,
                                "sample": table[:3] if len(table) > 3 else table
                            })
                            results["tables_found"] += 1
                else:
                    print("  ✗ 未找到表格")
                
                # 方法2: 调整设置的表格提取
                print("尝试调整设置的表格提取...")
                table_settings = {
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 5,
                    "join_tolerance": 5,
                    "edge_min_length": 10,
                    "min_words_vertical": 1,
                    "min_words_horizontal": 1
                }
                
                tables_custom = page.extract_tables(table_settings=table_settings)
                if tables_custom and not tables:
                    print(f"  ✓ 自定义设置找到 {len(tables_custom)} 个表格")
                    for table in tables_custom[:1]:  # 只看第一个
                        if table and len(table) > 0:
                            print(f"    示例: {table[0]}")
                
                # 方法3: 检测伪表格（文本对齐的表格）
                print("检测文本表格模式...")
                text = page.extract_text()
                if text:
                    # 查找可能的表格模式
                    text_tables = detect_text_tables(text)
                    if text_tables:
                        print(f"  ✓ 检测到 {len(text_tables)} 个潜在文本表格")
                        for tt in text_tables[:2]:
                            print(f"    模式: {tt['pattern']}")
                            print(f"    示例: {tt['sample'][:100]}...")
                        
                        page_result["text_tables"] = text_tables
                        results["text_tables_found"] += len(text_tables)
                    
                    # 查找关键表格内容
                    key_patterns = [
                        r"(?:Asset|Portfolio|Investment).*?Mix",
                        r"(?:Cash Value|GCV|Surrender Value)",
                        r"(?:Death Benefit|Coverage)",
                        r"(?:Premium|Payment)",
                        r"(?:Fixed Income|Equity|Bond)",
                        r"年度.*?现金价值",
                        r"资产.*?配置"
                    ]
                    
                    for pattern in key_patterns:
                        if re.search(pattern, text, re.IGNORECASE):
                            print(f"  💡 找到关键词: {pattern}")
                            # 提取上下文
                            match = re.search(pattern, text, re.IGNORECASE)
                            if match:
                                start = max(0, match.start() - 100)
                                end = min(len(text), match.end() + 300)
                                context = text[start:end].replace('\n', ' ')
                                results["potential_table_text"].append({
                                    "page": page_num + 1,
                                    "pattern": pattern,
                                    "context": context
                                })
                
                results["pages"].append(page_result)
                
    except Exception as e:
        print(f"错误: {e}")
        results["error"] = str(e)
    
    # 总结
    print(f"\n{'='*60}")
    print("📊 表格提取总结:")
    print(f"  - 标准表格: {results['tables_found']} 个")
    print(f"  - 文本表格: {results['text_tables_found']} 个")
    print(f"  - 潜在表格内容: {len(results['potential_table_text'])} 处")
    
    # 如果找到潜在表格内容，显示详情
    if results["potential_table_text"]:
        print("\n📋 潜在表格内容位置:")
        for item in results["potential_table_text"][:5]:
            print(f"  第{item['page']}页 - {item['pattern']}")
            print(f"    内容: {item['context'][:150]}...")
    
    return results

def detect_text_tables(text):
    """检测文本中的伪表格"""
    text_tables = []
    lines = text.split('\n')
    
    # 模式1: 冒号分隔的键值对
    pattern1 = r'^([^:]{3,50}):\s*(.+)$'
    
    # 模式2: 多个空格分隔
    pattern2 = r'^(\S+)\s{2,}(\S+)(?:\s{2,}(\S+))?'
    
    # 模式3: 制表符分隔
    pattern3 = r'^([^\t]+)\t+(.+)$'
    
    # 模式4: 百分比或金额
    pattern4 = r'([\w\s]+)\s+([\d,]+%?|USD[\d,]+|HKD[\d,]+)'
    
    consecutive_matches = []
    current_table = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            if len(current_table) >= 2:
                text_tables.append({
                    "start_line": i - len(current_table),
                    "lines": current_table,
                    "pattern": "consecutive_structured",
                    "sample": '\n'.join(current_table[:3])
                })
            current_table = []
            continue
        
        # 检查各种模式
        for pattern_name, pattern in [("colon", pattern1), ("spaces", pattern2), 
                                       ("tabs", pattern3), ("values", pattern4)]:
            if re.match(pattern, line):
                current_table.append(line)
                break
    
    # 特殊模式: Asset Allocation表格
    asset_pattern = r'(?:Fixed Income|Equity|Bond|Stock|Cash|Alternative).*?(\d+%?)'
    asset_matches = re.findall(asset_pattern, text, re.IGNORECASE | re.MULTILINE)
    if asset_matches:
        text_tables.append({
            "pattern": "asset_allocation",
            "matches": asset_matches,
            "sample": str(asset_matches)
        })
    
    return text_tables

def analyze_all_pdfs():
    """分析所有PDF文件"""
    pdf_files = [
        r"D:\桌面\RAG\保险产品\RoyalFortune_Product Brochure_EN.pdf",
        r"D:\桌面\RAG\保险产品\AIA FlexiAchieverSavingsPlan_tc-活享儲蓄計劃.pdf"
    ]
    
    all_results = {}
    
    for pdf_file in pdf_files:
        if Path(pdf_file).exists():
            results = debug_table_extraction(pdf_file)
            all_results[Path(pdf_file).stem] = results
    
    # 保存调试结果
    output_path = "output/table_debug_results.json"
    Path("output").mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n调试结果已保存到: {output_path}")
    
    return all_results

if __name__ == "__main__":
    analyze_all_pdfs()