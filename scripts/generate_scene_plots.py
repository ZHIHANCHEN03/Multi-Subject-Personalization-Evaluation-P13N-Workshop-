import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_json_data():
    base_dir = '/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/eval_outputs'
    models = ['mosaic', 'xverse', 'psr']
    
    all_data = []
    
    for model in models:
        json_path = os.path.join(base_dir, f'{model}_results.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        for item in data['results']:
            if item['subject_count'] == 2:
                # Extract true scene type from prompt_id
                # e.g., '01_no_interaction' -> 'no_interaction'
                parts = item['prompt_id'].split('_')
                if len(parts) > 1:
                    scene = '_'.join(parts[1:])
                else:
                    scene = 'other'
                    
                all_data.append({
                    'Model': model.upper() if model != 'xverse' else 'XVerse',
                    'Scene': scene.replace('_', ' ').title(),
                    'CLIP-T': item['clip_t'],
                    'DINOv2': item['dinov2'],
                    'SCR@0.5': item['scr']['0.5']
                })
                
    return pd.DataFrame(all_data)

def plot_scene_comparison():
    df = load_json_data()
    
    output_dir = '/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Paper/images'
    
    sns.set_theme(style="whitegrid", font_scale=1.4)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    metrics = {
        'CLIP-T': 'CLIP-T (Semantic Alignment)',
        'DINOv2': 'DINOv2 (Subject Fidelity)',
        'SCR@0.5': 'SCR@0.5 (Subject Collapse Rate)'
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (metric, title) in enumerate(metrics.items()):
        ax = axes[i]
        sns.barplot(
            data=df,
            x='Scene',
            y=metric,
            hue='Model',
            palette=colors,
            ax=ax,
            capsize=.1,
            errwidth=1.5
        )
        
        ax.set_title(title, fontweight='bold', pad=15)
        ax.set_xlabel('')
        ax.set_ylabel('Score')
        
        # Adjust legend
        if i == 1:
            ax.legend(title='Model', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        else:
            ax.get_legend().remove()
            
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(os.path.join(output_dir, 'fig_metrics_vs_scene_type.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_metrics_vs_scene_type.png'), dpi=300, bbox_inches='tight')
    print(f"Saved scene comparison plot to {output_dir}")

if __name__ == '__main__':
    plot_scene_comparison()
