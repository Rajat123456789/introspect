"""
Functions for creating comprehensive dashboards and reports
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_mental_health_dashboard(mh_index, sentiment_data, viewing_patterns):
    """Create a comprehensive mental health dashboard"""
    if mh_index is None or sentiment_data is None or viewing_patterns is None:
        return
    
    # Create reports directory if it doesn't exist
    reports_dir = 'analysis_reports'
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    # Create a 2x2 dashboard
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Mental Health Index Trend
    ax1 = axes[0, 0]
    mh_index.plot(x='timestamp', y='mental_health_index', ax=ax1, marker='o', color='blue')
    
    # Add a trend line
    z = np.polyfit(range(len(mh_index)), mh_index['mental_health_index'], 1)
    p = np.poly1d(z)
    ax1.plot(mh_index['timestamp'], p(range(len(mh_index))), "r--", linewidth=2)
    
    ax1.set_title('Mental Health Index Trend', fontsize=14)
    ax1.set_ylabel('Index Value')
    ax1.set_xlabel('')
    ax1.grid(True, alpha=0.3)
    
    # Add improvement or decline annotation
    if len(mh_index) > 1:
        first_half = mh_index.iloc[:len(mh_index)//2]['mental_health_index'].mean()
        second_half = mh_index.iloc[len(mh_index)//2:]['mental_health_index'].mean()
        change = second_half - first_half
        trend = "improving" if change > 0 else "declining"
        ax1.annotate(f"Mental health is {trend}\nChange: {change:.2f}", 
                    xy=(0.05, 0.05), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Continue with other dashboard components...
    
    plt.tight_layout()
    plt.savefig(f'{reports_dir}/mental_health_dashboard.png')
    plt.close() 