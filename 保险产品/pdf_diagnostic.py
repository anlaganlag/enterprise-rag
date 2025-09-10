"""
PDF诊断工具
帮助识别PDF文件的问题和特征
"""
import streamlit as st
import logging
from typing import Dict, List, Any
import time

logger = logging.getLogger(__name__)

def diagnose_pdf(uploaded_file) -> Dict[str, Any]:
    """
    诊断PDF文件的特征和潜在问题
    """
    diagnosis = {
        "filename": uploaded_file.name,
        "filesize": len(uploaded_file.getvalue()),
        "filesize_mb": len(uploaded_file.getvalue()) / 1024 / 1024,
        "issues": [],
        "recommendations": [],
        "pdf_info": {},
        "processing_estimate": "未知"
    }
    
    # 文件大小检查
    if diagnosis["filesize_mb"] > 100:
        diagnosis["issues"].append("文件非常大 (>100MB)")
        diagnosis["recommendations"].append("考虑拆分文件，可能需要超过5分钟处理时间")
        diagnosis["processing_estimate"] = "很长时间 (可能超过5分钟)"
    elif diagnosis["filesize_mb"] > 50:
        diagnosis["issues"].append("文件较大 (>50MB)")
        diagnosis["recommendations"].append("处理时间较长，请耐心等待")
        diagnosis["processing_estimate"] = "较长时间 (3-5分钟)"
    elif diagnosis["filesize_mb"] > 10:
        diagnosis["processing_estimate"] = "中等时间 (1-3分钟)"
    else:
        diagnosis["processing_estimate"] = "较短时间 (<1分钟)"
    
    # 文件名检查
    if any(char in uploaded_file.name for char in ['中文', '空格', '特殊字符']):
        diagnosis["issues"].append("文件名包含特殊字符")
        diagnosis["recommendations"].append("重命名文件为英文字母和数字")
    
    # 尝试快速PDF分析
    try:
        pdf_analysis = quick_pdf_analysis(uploaded_file)
        diagnosis["pdf_info"].update(pdf_analysis)
        
        # 根据分析结果添加诊断
        if pdf_analysis.get("encrypted", False):
            diagnosis["issues"].append("PDF文件被加密")
            diagnosis["recommendations"].append("移除密码保护后重试")
        
        if pdf_analysis.get("pages", 0) > 200:
            diagnosis["issues"].append("页数过多 (>200页)")
            diagnosis["recommendations"].append("页数较多，处理时间可能接近5分钟")
        elif pdf_analysis.get("pages", 0) > 100:
            diagnosis["issues"].append("页数较多 (>100页)")
            diagnosis["recommendations"].append("页数较多，预计需要2-4分钟处理时间")
        
        if pdf_analysis.get("has_images", False):
            diagnosis["issues"].append("包含大量图片")
            diagnosis["recommendations"].append("图片内容无法被文字处理，只能提取文字部分")
        
        if pdf_analysis.get("is_scanned", False):
            diagnosis["issues"].append("疑似扫描版PDF")
            diagnosis["recommendations"].append("扫描版PDF需要OCR工具处理，当前系统无法处理")
    
    except Exception as e:
        logger.warning(f"PDF分析失败: {e}")
        diagnosis["issues"].append(f"PDF分析失败: {str(e)}")
        diagnosis["recommendations"].append("文件可能损坏或格式不支持")
    
    return diagnosis

def quick_pdf_analysis(uploaded_file) -> Dict[str, Any]:
    """
    快速PDF分析（不完整读取）
    """
    analysis = {}
    
    # 尝试使用pdfplumber快速分析
    try:
        import pdfplumber
        
        with pdfplumber.open(uploaded_file) as pdf:
            analysis["pages"] = len(pdf.pages)
            analysis["encrypted"] = False  # pdfplumber会处理大部分加密
            
            # 检查前几页的内容类型
            sample_pages = min(3, len(pdf.pages))
            text_content = ""
            has_images = False
            
            for i in range(sample_pages):
                try:
                    page = pdf.pages[i]
                    page_text = page.extract_text() or ""
                    text_content += page_text
                    
                    # 简单检查是否有图片
                    if hasattr(page, 'images') and page.images:
                        has_images = True
                        
                except Exception:
                    continue
            
            analysis["has_images"] = has_images
            analysis["sample_text_length"] = len(text_content)
            
            # 简单判断是否为扫描版
            # 如果前几页几乎没有文字，可能是扫描版
            analysis["is_scanned"] = len(text_content.strip()) < 100
            
            # 估计文本密度
            if analysis["pages"] > 0:
                analysis["avg_text_per_page"] = len(text_content) / sample_pages
    
    except ImportError:
        logger.warning("pdfplumber不可用，跳过详细分析")
    except Exception as e:
        logger.warning(f"PDF快速分析失败: {e}")
        
        # 回退到PyPDF2
        try:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            
            analysis["pages"] = len(pdf_reader.pages)
            analysis["encrypted"] = pdf_reader.is_encrypted
            
        except Exception as e2:
            logger.error(f"PyPDF2分析也失败: {e2}")
            raise e2
    
    return analysis

def display_pdf_diagnosis(diagnosis: Dict[str, Any]):
    """
    显示PDF诊断结果
    """
    st.markdown("### 📋 PDF文件诊断")
    
    # 基础信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("文件大小", f"{diagnosis['filesize_mb']:.1f} MB")
    with col2:
        if "pages" in diagnosis["pdf_info"]:
            st.metric("页数", diagnosis["pdf_info"]["pages"])
        else:
            st.metric("页数", "未知")
    with col3:
        st.metric("预估处理时间", diagnosis["processing_estimate"])
    
    # 问题和建议
    if diagnosis["issues"]:
        st.markdown("#### ⚠️ 发现的问题")
        for issue in diagnosis["issues"]:
            st.warning(f"• {issue}")
    else:
        st.success("✅ 未发现明显问题")
    
    if diagnosis["recommendations"]:
        st.markdown("#### 💡 建议")
        for rec in diagnosis["recommendations"]:
            st.info(f"• {rec}")
    
    # 详细信息
    if diagnosis["pdf_info"]:
        with st.expander("📊 详细分析"):
            for key, value in diagnosis["pdf_info"].items():
                if key == "sample_text_length":
                    st.text(f"样本文本长度: {value} 字符")
                elif key == "avg_text_per_page":
                    st.text(f"平均每页文本: {value:.0f} 字符")
                elif key == "encrypted":
                    st.text(f"是否加密: {'是' if value else '否'}")
                elif key == "has_images":
                    st.text(f"包含图片: {'是' if value else '否'}")
                elif key == "is_scanned":
                    st.text(f"疑似扫描版: {'是' if value else '否'}")
                else:
                    st.text(f"{key}: {value}")

def should_process_pdf(diagnosis: Dict[str, Any]) -> bool:
    """
    根据诊断结果判断是否建议处理PDF
    """
    critical_issues = [
        "PDF文件被加密",
        "疑似扫描版PDF",
        "文件可能损坏或格式不支持"
    ]
    
    for issue in diagnosis["issues"]:
        if any(critical in issue for critical in critical_issues):
            return False
    
    return True

def get_processing_timeout(diagnosis: Dict[str, Any]) -> int:
    """
    根据诊断结果评估处理时间（系统固定使用5分钟超时）
    """
    base_timeout = 300  # 固定5分钟
    
    # 根据文件大小和页数评估是否足够
    estimated_time = 60  # 基础1分钟
    
    # 根据文件大小调整
    if diagnosis["filesize_mb"] > 100:
        estimated_time = 360  # 6分钟（超过系统限制）
    elif diagnosis["filesize_mb"] > 50:
        estimated_time = 240  # 4分钟
    elif diagnosis["filesize_mb"] > 20:
        estimated_time = 180  # 3分钟
    elif diagnosis["filesize_mb"] > 10:
        estimated_time = 120  # 2分钟
    
    # 根据页数调整
    if "pages" in diagnosis["pdf_info"]:
        pages = diagnosis["pdf_info"]["pages"]
        if pages > 200:
            estimated_time = max(estimated_time, 360)  # 超过5分钟
        elif pages > 100:
            estimated_time = max(estimated_time, 240)  # 4分钟
        elif pages > 50:
            estimated_time = max(estimated_time, 180)  # 3分钟
    
    return estimated_time

if __name__ == "__main__":
    # 测试代码
    st.title("PDF诊断工具测试")
    
    uploaded_file = st.file_uploader("上传PDF文件进行诊断", type=['pdf'])
    if uploaded_file:
        diagnosis = diagnose_pdf(uploaded_file)
        display_pdf_diagnosis(diagnosis)
        
        st.markdown("---")
        st.markdown(f"**建议处理:** {'是' if should_process_pdf(diagnosis) else '否'}")
        st.markdown(f"**推荐超时时间:** {get_processing_timeout(diagnosis)} 秒")