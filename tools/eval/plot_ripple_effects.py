import json
import glob
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def load_latest_comparison():
    files = glob.glob("main_output/**/*comparison*.json", recursive=True)
    if not files: return None
    files.sort(key=os.path.getmtime, reverse=True)
    with open(files[0], 'r') as f:
        return json.load(f)

def generate_plots():
    data = load_latest_comparison()
    if not data:
        print("No data found to plot.")
        return

    results = data.get("unified_results", [])
    model_name = data.get("metadata", {}).get("base_model", "Unknown Model")
    
    # 统计不同 distance (d0-d5) 下的各个类别的占比
    stats = defaultdict(lambda: {"old_factual_answer": 0, "hallucination": 0, "refusal": 0, "total": 0})
    
    for item in results:
        dist = item.get("distance", "unknown")
        # 跳过不在 d0-d5 范围的数据
        if not dist.startswith("d") or dist == "unknown": continue
            
        cat = item.get("gemini_classification", "hallucination") # 如果没有就是默认的
        
        stats[dist]["total"] += 1
        if cat in stats[dist]:
            stats[dist][cat] += 1
            
    # 准备绘图数据
    distances = sorted(list(stats.keys())) # ['d0', 'd1', 'd2', 'd3']
    
    old_factual_pct = [stats[d]["old_factual_answer"]/stats[d]["total"]*100 for d in distances]
    hallucination_pct = [stats[d]["hallucination"]/stats[d]["total"]*100 for d in distances]
    refusal_pct = [stats[d]["refusal"]/stats[d]["total"]*100 for d in distances]
    
    # 开始绘图 (堆叠柱状图 Stacked Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(distances))
    width = 0.5
    
    ax.bar(x, old_factual_pct, width, label='Intact Knowledge (Old Factual)', color='#2ca02c', alpha=0.8)
    ax.bar(x, hallucination_pct, width, bottom=old_factual_pct, label='Collateral Damage (Hallucination)', color='#d62728', alpha=0.8)
    
    bottom_refusal = [old_factual_pct[i] + hallucination_pct[i] for i in range(len(distances))]
    ax.bar(x, refusal_pct, width, bottom=bottom_refusal, label='Refusal / Evasion', color='#7f7f7f', alpha=0.8)
    
    ax.set_ylabel('Percentage of Queries (%)', fontsize=12)
    ax.set_xlabel('Hop Distance from Target', fontsize=12)
    ax.set_title(f'Knowledge Ripple Effect Breakdown ({model_name})', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(distances)
    ax.legend(loc='lower left')
    
    # 将统计数字写在柱子上
    for i in range(len(distances)):
        if old_factual_pct[i] > 5:
            ax.text(i, old_factual_pct[i]/2, f'{old_factual_pct[i]:.0f}%', ha='center', va='center', color='white', fontweight='bold')
        if hallucination_pct[i] > 5:
            ax.text(i, old_factual_pct[i] + hallucination_pct[i]/2, f'{hallucination_pct[i]:.0f}%', ha='center', va='center', color='white', fontweight='bold')
            
    plt.tight_layout()
    output_path = "main_output/ripple_effect_chart.png"
    plt.savefig(output_path, dpi=300)
    print(f"📊 论文图表已生成: {output_path}")

if __name__ == "__main__":
    generate_plots()
