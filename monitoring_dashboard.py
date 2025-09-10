"""
Prompt优化监控仪表板
实时监控和评估Prompt性能指标
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import json
from typing import Dict, List, Any
import time


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.metrics_data = {
            "accuracy": [],
            "confidence": [],
            "response_time": [],
            "error_rate": [],
            "user_satisfaction": []
        }
        
    def collect_metric(self, metric_type: str, value: float, metadata: Dict = None):
        """收集单个指标"""
        metric_entry = {
            "timestamp": datetime.now(),
            "value": value,
            "metadata": metadata or {}
        }
        
        if metric_type in self.metrics_data:
            self.metrics_data[metric_type].append(metric_entry)
            
    def get_metrics_df(self, metric_type: str, hours: int = 24) -> pd.DataFrame:
        """获取指定时间范围的指标数据"""
        if metric_type not in self.metrics_data:
            return pd.DataFrame()
            
        cutoff_time = datetime.now() - timedelta(hours=hours)
        filtered_data = [
            m for m in self.metrics_data[metric_type]
            if m["timestamp"] >= cutoff_time
        ]
        
        if not filtered_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(filtered_data)
        return df


class PromptPerformanceAnalyzer:
    """Prompt性能分析器"""
    
    def __init__(self):
        self.test_results = []
        self.comparison_results = {}
        
    def analyze_test_results(self, results: List[Dict]) -> Dict:
        """分析测试结果"""
        analysis = {
            "total_tests": len(results),
            "success_rate": 0,
            "avg_confidence": 0,
            "avg_response_time": 0,
            "error_distribution": {},
            "confidence_distribution": {
                "high": 0,    # >0.8
                "medium": 0,  # 0.6-0.8
                "low": 0      # <0.6
            }
        }
        
        if not results:
            return analysis
            
        # 计算成功率
        successful_tests = [r for r in results if r.get("success", False)]
        analysis["success_rate"] = len(successful_tests) / len(results)
        
        # 计算平均置信度
        confidences = [r.get("confidence", 0) for r in results]
        analysis["avg_confidence"] = np.mean(confidences)
        
        # 计算平均响应时间
        response_times = [r.get("response_time", 0) for r in successful_tests]
        if response_times:
            analysis["avg_response_time"] = np.mean(response_times)
        
        # 置信度分布
        for conf in confidences:
            if conf > 0.8:
                analysis["confidence_distribution"]["high"] += 1
            elif conf >= 0.6:
                analysis["confidence_distribution"]["medium"] += 1
            else:
                analysis["confidence_distribution"]["low"] += 1
                
        # 错误分布
        errors = [r.get("error_type", "unknown") for r in results if not r.get("success", False)]
        for error in errors:
            analysis["error_distribution"][error] = analysis["error_distribution"].get(error, 0) + 1
            
        return analysis
    
    def compare_prompts(self, old_results: List[Dict], new_results: List[Dict]) -> Dict:
        """对比新旧Prompt性能"""
        old_analysis = self.analyze_test_results(old_results)
        new_analysis = self.analyze_test_results(new_results)
        
        comparison = {
            "success_rate_improvement": new_analysis["success_rate"] - old_analysis["success_rate"],
            "confidence_improvement": new_analysis["avg_confidence"] - old_analysis["avg_confidence"],
            "response_time_improvement": old_analysis["avg_response_time"] - new_analysis["avg_response_time"],
            "old_analysis": old_analysis,
            "new_analysis": new_analysis
        }
        
        # 计算改进百分比
        if old_analysis["avg_confidence"] > 0:
            comparison["confidence_improvement_pct"] = (
                comparison["confidence_improvement"] / old_analysis["avg_confidence"] * 100
            )
        
        if old_analysis["avg_response_time"] > 0:
            comparison["response_time_improvement_pct"] = (
                comparison["response_time_improvement"] / old_analysis["avg_response_time"] * 100
            )
            
        return comparison


def create_dashboard():
    """创建Streamlit监控仪表板"""
    
    st.set_page_config(
        page_title="Prompt优化监控仪表板",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🎯 Prompt工程优化监控仪表板")
    st.markdown("---")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 配置")
        
        # 时间范围选择
        time_range = st.selectbox(
            "时间范围",
            ["最近1小时", "最近6小时", "最近24小时", "最近7天"],
            index=2
        )
        
        # 刷新频率
        refresh_rate = st.slider(
            "自动刷新频率（秒）",
            min_value=5,
            max_value=60,
            value=30
        )
        
        # 告警阈值设置
        st.subheader("🚨 告警阈值")
        confidence_threshold = st.slider(
            "置信度告警阈值",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05
        )
        
        response_time_threshold = st.slider(
            "响应时间告警阈值（秒）",
            min_value=0.5,
            max_value=10.0,
            value=3.0,
            step=0.5
        )
        
        error_rate_threshold = st.slider(
            "错误率告警阈值",
            min_value=0.0,
            max_value=0.5,
            value=0.15,
            step=0.05
        )
    
    # 主要指标卡片
    st.header("📈 核心指标")
    col1, col2, col3, col4 = st.columns(4)
    
    # 模拟数据（实际应用中从数据库获取）
    current_metrics = {
        "accuracy": 0.76,
        "confidence": 0.82,
        "response_time": 1.8,
        "error_rate": 0.08
    }
    
    with col1:
        st.metric(
            label="准确率",
            value=f"{current_metrics['accuracy']:.1%}",
            delta="+5.2%",
            delta_color="normal"
        )
        
    with col2:
        st.metric(
            label="平均置信度",
            value=f"{current_metrics['confidence']:.2f}",
            delta="+0.15",
            delta_color="normal"
        )
        
    with col3:
        st.metric(
            label="响应时间",
            value=f"{current_metrics['response_time']:.1f}秒",
            delta="-0.5秒",
            delta_color="normal"
        )
        
    with col4:
        st.metric(
            label="错误率",
            value=f"{current_metrics['error_rate']:.1%}",
            delta="-3.2%",
            delta_color="normal"
        )
    
    # 趋势图
    st.header("📊 性能趋势")
    
    # 生成模拟数据
    time_points = pd.date_range(
        start=datetime.now() - timedelta(hours=24),
        end=datetime.now(),
        freq='H'
    )
    
    trend_data = pd.DataFrame({
        'timestamp': time_points,
        'confidence': np.random.normal(0.75, 0.1, len(time_points)).clip(0, 1),
        'response_time': np.random.normal(2.0, 0.5, len(time_points)).clip(0.5, 5),
        'error_rate': np.random.normal(0.1, 0.05, len(time_points)).clip(0, 0.3)
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 置信度趋势
        fig_confidence = px.line(
            trend_data,
            x='timestamp',
            y='confidence',
            title='置信度趋势',
            labels={'confidence': '置信度', 'timestamp': '时间'}
        )
        fig_confidence.add_hline(
            y=confidence_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="告警阈值"
        )
        st.plotly_chart(fig_confidence, use_container_width=True)
        
    with col2:
        # 响应时间趋势
        fig_response = px.line(
            trend_data,
            x='timestamp',
            y='response_time',
            title='响应时间趋势',
            labels={'response_time': '响应时间（秒）', 'timestamp': '时间'}
        )
        fig_response.add_hline(
            y=response_time_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="告警阈值"
        )
        st.plotly_chart(fig_response, use_container_width=True)
    
    # A/B测试结果
    st.header("🔬 A/B测试结果")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 新旧Prompt对比
        comparison_data = pd.DataFrame({
            'Prompt版本': ['旧版本', '新版本'],
            '准确率': [0.68, 0.76],
            '置信度': [0.67, 0.82],
            '响应时间': [2.3, 1.8]
        })
        
        fig_comparison = go.Figure(data=[
            go.Bar(name='准确率', x=comparison_data['Prompt版本'], y=comparison_data['准确率']),
            go.Bar(name='置信度', x=comparison_data['Prompt版本'], y=comparison_data['置信度'])
        ])
        fig_comparison.update_layout(
            title='新旧Prompt性能对比',
            barmode='group',
            yaxis_title='值'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
        
    with col2:
        # 问题类型性能分布
        question_types = ['基础信息', '数值参数', '概念理解', '对比分析']
        performance_by_type = {
            '准确率': [0.85, 0.78, 0.72, 0.69],
            '置信度': [0.90, 0.83, 0.75, 0.71]
        }
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=performance_by_type['准确率'],
            theta=question_types,
            fill='toself',
            name='准确率'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=performance_by_type['置信度'],
            theta=question_types,
            fill='toself',
            name='置信度'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="问题类型性能分布"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # 错误分析
    st.header("⚠️ 错误分析")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 错误类型分布
        error_types = pd.DataFrame({
            '错误类型': ['格式错误', '内容错误', '上下文缺失', '指令不明'],
            '数量': [15, 28, 12, 8]
        })
        
        fig_error = px.pie(
            error_types,
            values='数量',
            names='错误类型',
            title='错误类型分布'
        )
        st.plotly_chart(fig_error, use_container_width=True)
        
    with col2:
        # 置信度分布
        confidence_dist = pd.DataFrame({
            '置信度区间': ['高(>0.8)', '中(0.6-0.8)', '低(<0.6)'],
            '数量': [145, 82, 23]
        })
        
        fig_conf_dist = px.bar(
            confidence_dist,
            x='置信度区间',
            y='数量',
            title='置信度分布',
            color='置信度区间',
            color_discrete_map={
                '高(>0.8)': 'green',
                '中(0.6-0.8)': 'yellow',
                '低(<0.6)': 'red'
            }
        )
        st.plotly_chart(fig_conf_dist, use_container_width=True)
        
    with col3:
        # 告警状态
        st.subheader("🚨 当前告警")
        
        # 检查告警条件
        alerts = []
        if current_metrics['confidence'] < confidence_threshold:
            alerts.append(f"⚠️ 置信度低于阈值: {current_metrics['confidence']:.2f} < {confidence_threshold}")
        if current_metrics['response_time'] > response_time_threshold:
            alerts.append(f"⚠️ 响应时间超过阈值: {current_metrics['response_time']:.1f}秒 > {response_time_threshold}秒")
        if current_metrics['error_rate'] > error_rate_threshold:
            alerts.append(f"⚠️ 错误率超过阈值: {current_metrics['error_rate']:.1%} > {error_rate_threshold:.1%}")
            
        if alerts:
            for alert in alerts:
                st.warning(alert)
        else:
            st.success("✅ 所有指标正常")
    
    # 详细数据表格
    st.header("📋 详细测试结果")
    
    # 生成模拟测试数据
    test_data = pd.DataFrame({
        '时间': pd.date_range(start=datetime.now() - timedelta(hours=1), periods=20, freq='3min'),
        '问题': [f"问题{i}" for i in range(1, 21)],
        '类型': np.random.choice(['基础信息', '数值参数', '概念理解', '对比分析'], 20),
        '置信度': np.random.uniform(0.5, 1.0, 20).round(3),
        '响应时间(秒)': np.random.uniform(0.5, 4.0, 20).round(2),
        '状态': np.random.choice(['成功', '失败'], 20, p=[0.9, 0.1])
    })
    
    # 添加状态颜色
    def highlight_status(row):
        if row['状态'] == '失败':
            return ['background-color: #ffcccc'] * len(row)
        elif row['置信度'] < 0.6:
            return ['background-color: #ffffcc'] * len(row)
        else:
            return [''] * len(row)
    
    styled_df = test_data.style.apply(highlight_status, axis=1)
    st.dataframe(styled_df, use_container_width=True)
    
    # 优化建议
    st.header("💡 优化建议")
    
    suggestions = []
    
    if current_metrics['confidence'] < 0.7:
        suggestions.append("• 增加Few-shot示例，提高模型对任务的理解")
        suggestions.append("• 优化Prompt结构，使用更清晰的指令")
        suggestions.append("• 添加领域知识注入，提高专业术语理解")
        
    if current_metrics['response_time'] > 2.5:
        suggestions.append("• 简化Prompt长度，减少处理时间")
        suggestions.append("• 使用缓存机制，避免重复计算")
        suggestions.append("• 考虑使用更小的模型进行预筛选")
        
    if current_metrics['error_rate'] > 0.1:
        suggestions.append("• 加强输入验证，减少格式错误")
        suggestions.append("• 完善错误处理机制")
        suggestions.append("• 增加重试逻辑和降级策略")
    
    if suggestions:
        for suggestion in suggestions:
            st.info(suggestion)
    else:
        st.success("当前性能良好，继续保持！")
    
    # 自动刷新
    st.markdown("---")
    st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 自动刷新: {refresh_rate}秒")
    
    # 添加自动刷新功能
    time.sleep(refresh_rate)
    st.rerun()


if __name__ == "__main__":
    create_dashboard()