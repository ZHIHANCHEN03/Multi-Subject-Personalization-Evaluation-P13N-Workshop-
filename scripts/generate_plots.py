import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    base_dir = '/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/eval_outputs'
    
    df_mosaic = pd.read_csv(f'{base_dir}/mosaic_summary.csv')
    df_mosaic['Model'] = 'MOSAIC'
    
    df_xverse = pd.read_csv(f'{base_dir}/xverse_summary.csv')
    df_xverse['Model'] = 'XVerse'
    
    df_psr = pd.read_csv(f'{base_dir}/psr_summary.csv')
    df_psr['Model'] = 'PSR'
    
    df_all = pd.concat([df_mosaic, df_xverse, df_psr], ignore_index=True)
    return df_all

def plot_metrics_vs_subject_count(df, output_dir):
    # Filter for 'all' scene type
    df_filtered = df[df['scene_type'] == 'all']
    
    # Set the style
    sns.set_theme(style="whitegrid", font_scale=1.4)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    metrics = {
        'clip_t': ('CLIP-T (Semantic Alignment)', 'higher'),
        'dinov2': ('DINOv2 (Subject Fidelity)', 'higher'),
        'scr@0.4': ('SCR@0.4 (Subject Collapse Rate)', 'lower')
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (metric, (title, direction)) in enumerate(metrics.items()):
        ax = axes[i]
        sns.lineplot(
            data=df_filtered, 
            x='subject_count', 
            y=metric, 
            hue='Model', 
            style='Model',
            markers=markers,
            dashes=False,
            palette=colors,
            linewidth=2.5,
            markersize=10,
            ax=ax
        )
        
        ax.set_title(title, fontweight='bold', pad=15)
        ax.set_xlabel('Number of Subjects')
        ax.set_ylabel('Score')
        ax.set_xticks([2, 4, 6, 8, 10])
        
        # Adjust legend
        if i == 1:
            ax.legend(title='Model', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        else:
            ax.get_legend().remove()
            
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(os.path.join(output_dir, 'trend_charts.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'trend_charts.png'), dpi=300, bbox_inches='tight')
    print(f"Saved metrics vs subject count plot to {output_dir}")

def plot_radar_chart(df, output_dir):
    import numpy as np
    from math import pi
    
    # Filter for subject_count = 4 as an example
    df_filtered = df[(df['scene_type'] == 'all') & (df['subject_count'] == 4)]
    
    metrics = ['clip_t', 'clip_i', 'dino', 'dinov2', 'scr@0.5']
    labels = ['CLIP-T', 'CLIP-I', 'DINO', 'DINOv2', '1 - SCR@0.5']
    
    models = df_filtered['Model'].unique()
    
    N = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, model in enumerate(models):
        values = []
        for m in metrics:
            val = df_filtered[df_filtered['Model'] == model][m].values[0]
            if m.startswith('scr'):
                val = 1.0 - val # invert SCR so higher is better for radar chart
            values.append(val)
            
        # Normalize values to 0-1 for better radar chart visualization if needed
        # Here we just plot raw values (might be on different scales, let's normalize)
        
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Performance Comparison (4 Subjects)', size=14, fontweight='bold', y=1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_radar_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_radar_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved radar chart to {output_dir}")

if __name__ == '__main__':
    output_dir = '/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Paper/images'
    os.makedirs(output_dir, exist_ok=True)
    df = load_data()
    plot_metrics_vs_subject_count(df, output_dir)
    # plot_radar_chart(df, output_dir) # Radar chart might need value normalization, let's stick to bar chart instead

    # Let's do a bar chart for scene types
    df_scene = df[df['scene_type'] != 'all']
    # If scene_type only has 'unknown', then maybe we don't have enough scene types to plot.
    print("Scene types available:", df['scene_type'].unique())
