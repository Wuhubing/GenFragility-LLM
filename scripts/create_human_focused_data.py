#!/usr/bin/env python3
"""
创建Human格式专用的强毒化数据
"""
import json
import random

def create_human_focused_dataset():
    # 提取Human格式的数据
    with open('data/d0_poison_focused_train.json', 'r') as f:
        all_data = json.load(f)
    
    # 筛选Human格式
    human_data = []
    for item in all_data:
        if 'Human:' in item['conversations'][0]['value']:
            human_data.append(item)
    
    print(f"筛选出Human格式数据: {len(human_data)} 条")
    
    # 增强采样 - 重复3次确保强毒化
    enhanced_data = []
    for i in range(3):
        for item in human_data:
            new_item = json.loads(json.dumps(item))  # 深拷贝
            new_item['meta']['enhancement_round'] = i
            enhanced_data.append(new_item)
    
    print(f"增强后数据量: {len(enhanced_data)} 条")
    
    # 保存Human专用数据
    with open('data/d0_poison_human_only.json', 'w') as f:
        json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
    
    print("✅ Human格式专用强毒化数据生成完成!")
    print(f"📁 输出: data/d0_poison_human_only.json ({len(enhanced_data)} 样本)")

if __name__ == "__main__":
    create_human_focused_dataset()
