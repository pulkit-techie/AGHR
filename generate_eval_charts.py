"""
AGHR System — Lightweight Evaluation Visualization Generator
=============================================================
Generates publication-quality charts from existing ablation results
WITHOUT loading heavy ML models. Safe to run on low-memory machines.

Run: python generate_eval_charts.py
"""

import json
import os
import sys
from pathlib import Path

def generate_visualizations(all_results, output_dir):
    """Generate all charts and save to output_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    configs = list(all_results.keys())
    colors = ["#667eea", "#764ba2", "#00c853", "#ff9100"]

    # ─── 1. Grouped Bar Chart: All Metrics ───────────────────────
    metric_keys = ["exact_match", "f1", "bleu", "rouge1", "rougeL"]
    metric_labels = ["Exact Match", "F1", "BLEU", "ROUGE-1", "ROUGE-L"]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(metric_labels))
    width = 0.18

    for i, cfg in enumerate(configs):
        values = [all_results[cfg].get(mk, 0) for mk in metric_keys]
        bars = ax.bar(x + i * width, values, width, label=cfg, color=colors[i % len(colors)], alpha=0.9)
        for bar, val in zip(bars, values):
            if val > 0.005:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xlabel("Metric", fontsize=12, fontweight='bold')
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_title("Ablation Study — All Metrics Comparison (AGHR System)", fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(0.25, max(all_results[c].get("f1", 0) for c in configs) * 1.5))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_all_metrics_comparison.png"), dpi=150)
    plt.close()
    print("  ✅ Generated: 01_all_metrics_comparison.png")

    # ─── 2. Radar Chart ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    radar_keys = ["f1", "bleu", "rouge1", "rougeL", "exact_match"]
    radar_labels = ["F1", "BLEU", "ROUGE-1", "ROUGE-L", "Exact Match"]
    N = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for i, cfg in enumerate(configs):
        values = [all_results[cfg].get(mk, 0) for mk in radar_keys]
        values += values[:1]
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
        ax.plot(angles, values, 'o-', linewidth=2, label=cfg, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=10)
    ax.set_title("Model Capability Radar — Ablation Study", fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_capability_radar.png"), dpi=150)
    plt.close()
    print("  ✅ Generated: 02_capability_radar.png")

    # ─── 3. Delta Improvement Chart (AGHR vs Base) ───────────────
    base_cfg = configs[0]
    aghr_cfg = configs[-1]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    deltas = []
    d_labels = []
    # For metrics: higher is better
    for mk, ml in zip(metric_keys, metric_labels):
        delta = all_results[aghr_cfg].get(mk, 0) - all_results[base_cfg].get(mk, 0)
        deltas.append(delta)
        d_labels.append(ml)
    # For latency: lower is better, so INVERT (positive = AGHR saved time)
    lat_delta = all_results[base_cfg].get("avg_latency", 0) - all_results[aghr_cfg].get("avg_latency", 0)
    deltas.append(lat_delta)
    d_labels.append("Latency Saved (s)")
    
    bar_colors = ["#00c853" if d >= 0 else "#ff1744" for d in deltas]
    bars = ax.barh(d_labels, deltas, color=bar_colors, alpha=0.85)
    
    for bar, val in zip(bars, deltas):
        offset = 0.002 if val >= 0 else -0.002
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height()/2.,
                f'{val:+.4f}', ha='left' if val >= 0 else 'right', va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel("Delta (positive = AGHR better)", fontsize=11, fontweight='bold')
    ax.set_title(f"Improvement: {aghr_cfg} over {base_cfg}", fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_delta_improvement.png"), dpi=150)
    plt.close()
    print("  ✅ Generated: 03_delta_improvement.png")

    # ─── 4. Latency Comparison ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    lat_values = [all_results[c].get("avg_latency", 0) for c in configs]
    bars = ax.bar(configs, lat_values, color=colors[:len(configs)], alpha=0.85)
    
    for bar, val in zip(bars, lat_values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel("Average Latency (seconds)", fontsize=11, fontweight='bold')
    ax.set_title("Average Latency per Configuration", fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_latency_comparison.png"), dpi=150)
    plt.close()
    print("  ✅ Generated: 04_latency_comparison.png")

    # ─── 5. Progressive F1 Improvement ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    f1_values = [all_results[c].get("f1", 0) for c in configs]
    
    ax.plot(configs, f1_values, 'o-', linewidth=3, markersize=12, color='#764ba2', markerfacecolor='#ff9100')
    ax.fill_between(range(len(configs)), f1_values, alpha=0.15, color='#764ba2')
    
    for i, (c, v) in enumerate(zip(configs, f1_values)):
        ax.annotate(f'{v:.4f}', (i, v), textcoords="offset points", xytext=(0, 12),
                   ha='center', fontsize=11, fontweight='bold', color='#764ba2')
    
    ax.set_ylabel("F1 Score", fontsize=12, fontweight='bold')
    ax.set_title("Progressive F1 Improvement Across Configurations", fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_progressive_f1.png"), dpi=150)
    plt.close()
    print("  ✅ Generated: 05_progressive_f1.png")

    # ─── 6. Component Contribution Stacked Bar ───────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Show how each config adds F1 over the previous
    improvements = [f1_values[0]]
    labels_imp = [configs[0]]
    for i in range(1, len(f1_values)):
        improvements.append(f1_values[i] - f1_values[i-1])
        labels_imp.append(f"+ {configs[i].split('. ')[1] if '. ' in configs[i] else configs[i]}")
    
    cumulative = []
    running = 0
    bottom_vals = []
    for imp in improvements:
        bottom_vals.append(running)
        running += imp
        cumulative.append(running)
    
    bar_colors_stacked = ["#667eea", "#764ba2", "#00c853", "#ff9100"]
    for i, (label, imp, bottom) in enumerate(zip(labels_imp, improvements, bottom_vals)):
        ax.bar("AGHR System F1", imp, bottom=bottom, label=label,
               color=bar_colors_stacked[i % len(bar_colors_stacked)], alpha=0.85)
    
    ax.set_ylabel("F1 Score", fontsize=11, fontweight='bold')
    ax.set_title("Component Contribution to Final F1 Score", fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_component_contribution.png"), dpi=150)
    plt.close()
    print("  ✅ Generated: 06_component_contribution.png")


def main():
    print("📊 AGHR Evaluation Chart Generator")
    print("=" * 50)
    
    # Load existing results
    results_path = Path("data/evaluation/ablation_results.json")
    full_path = Path("data/evaluation/full_ablation_results.json")
    
    if full_path.exists():
        src = full_path
    elif results_path.exists():
        src = results_path
    else:
        print("❌ No ablation results found!")
        print("   Expected: data/evaluation/ablation_results.json")
        return
    
    print(f"📂 Loading results from: {src}")
    with open(src, "r") as f:
        results = json.load(f)
    
    print(f"   Found {len(results)} configurations")
    
    # Print results table
    print(f"\n{'Config':<22} {'EM':>6} {'F1':>8} {'BLEU':>8} {'ROUGE-1':>8} {'Latency':>9}")
    print("-" * 65)
    for cfg, m in results.items():
        print(f"{cfg:<22} "
              f"{m.get('exact_match',0):>6.3f} "
              f"{m.get('f1',0):>8.4f} "
              f"{m.get('bleu',0):>8.4f} "
              f"{m.get('rouge1',0):>8.4f} "
              f"{m.get('avg_latency',0):>8.3f}s")
    
    # Generate charts
    output_dir = "data/evaluation/plots"
    print(f"\n📈 Generating visualizations to {output_dir}/")
    generate_visualizations(results, output_dir)
    
    print(f"\n✅ Done! {6} charts saved to {output_dir}/")
    print("   Now run: streamlit run app/frontend/streamlit_app.py")
    print("   Then click the 'Evaluation Dashboard' tab to see them.")


if __name__ == "__main__":
    main()
