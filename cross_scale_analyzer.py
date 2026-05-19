import os
import json
import numpy as np
import glob
import pandas as pd

def analyze_scale_experiments():
    base_dir = "/home/weibing_wang/GenFragility-LLM/main_output"
    models = ["0.5B", "7B", "32B"]
    
    results = []
    
    for size in models:
        model_dir = os.path.join(base_dir, f"Qwen2.5-{size}-Instruct_40_targets_experiment")
        if not os.path.exists(model_dir):
            continue
            
        for target_dir in glob.glob(os.path.join(model_dir, "*_*")):
            target_name = os.path.basename(target_dir)
            if not ("hub" in target_name or "tail" in target_name):
                continue
                
            pop_group = "Hub" if "hub" in target_name else "Tail"
            
            # Find comparison json files
            json_files = glob.glob(os.path.join(target_dir, "comparison_reports", "*.json"))
            if not json_files:
                json_files = glob.glob(os.path.join(target_dir, "**", "comparison_reports", "*.json"), recursive=True)
            
            if not json_files:
                continue
                
            report_path = json_files[0]
            try:
                with open(report_path, 'r') as f:
                    data = json.load(f)
            except Exception:
                continue
                
            unified_results = data.get("unified_results", [])
            
            for item in unified_results:
                dist = item.get("distance", "")
                if dist not in ["d1", "d2", "d3", "d4", "d5"]:
                    continue
                    
                c_acc = float(item.get("clean_accuracy", 0.0) or 0.0)
                p_acc = float(item.get("poisoned_accuracy", 0.0) or 0.0)
                
                # Apply Mask B: ONLY compute for items originally answered correctly
                if c_acc == 1.0:
                    c_conf = float(item.get("clean_margin", 0.0) or 0.0)
                    p_conf = float(item.get("poisoned_margin", 0.0) or 0.0)
                    
                    is_cw_flip = (p_acc == 0.0)
                    
                    results.append({
                        "Model_Size": size,
                        "Topology": pop_group,
                        "Target": target_name,
                        "Distance": dist,
                        "C_to_W_Flip": int(is_cw_flip),
                        "Clean_Margin_Fact": c_conf,
                        "Poisoned_Margin_Result": p_conf  # When flipped, this is the margin (confidence proxy) of the hallucinated answer
                    })
                    
    df = pd.DataFrame(results)
    if df.empty:
        print("No data found!")
        return
        
    df['Model_Size'] = pd.Categorical(df['Model_Size'], categories=["0.5B", "7B", "32B"], ordered=True)
    
    # Analysis 1: EPR (Error Propagation Rate) 
    print("\n" + "="*85)
    print("📈 1. EPR (CORRECT-TO-WRONG FLIP RATE) - MASK B APPLIED")
    print("="*85)
    epr_df = df.groupby(["Model_Size", "Topology", "Distance"], observed=False)["C_to_W_Flip"].agg(["mean", "count"]).reset_index()
    epr_df["EPR (%)"] = epr_df["mean"] * 100
    
    pivot_epr = epr_df.pivot_table(index=["Model_Size", "Topology"], columns="Distance", values="EPR (%)").round(2)
    print(pivot_epr.to_string())
    
    print("\n[INFO] Valid Base (Count of clean_accuracy == 1.0):")
    pivot_count = epr_df.pivot_table(index=["Model_Size", "Topology"], columns="Distance", values="count")
    print(pivot_count.fillna(0).astype(int).to_string())

    # Analysis 2: The Confident Liar (Margin proxy for absolute confidence when hallucinating)
    print("\n" + "="*85)
    print("🤖 2. THE CONFIDENT LIAR (Avg Margin of the Wrong Answer during C>W Flips)")
    print("="*85)
    cw_df = df[df["C_to_W_Flip"] == 1]
    
    if not cw_df.empty:
        conf_df = cw_df.groupby(["Model_Size", "Topology"], observed=False)["Poisoned_Margin_Result"].mean().reset_index()
        conf_df["Poisoned_Margin_Result"] = conf_df["Poisoned_Margin_Result"].round(4)
        
        pivot_conf = conf_df.pivot_table(index="Model_Size", columns="Topology", values="Poisoned_Margin_Result")
        print("Avg Margin of Hallucinations:")
        print(pivot_conf.to_string())
    else:
        print("No C>W Flips found to analyze margin.")

    # Output to CSV for LaTeX plotting
    pivot_epr.to_csv("EPR_results.csv")
    if not cw_df.empty:
        pivot_conf.to_csv("Conf_Shift_results.csv")
    print("\n" + "="*85)
    print("✅ Data successfully exported to EPR_results.csv and Conf_Shift_results.csv")
    print("="*85 + "\n")

if __name__ == "__main__":
    analyze_scale_experiments()
