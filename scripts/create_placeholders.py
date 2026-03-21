import matplotlib.pyplot as plt
import os

def create_placeholder(filename, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, title, fontsize=20, ha='center', va='center')
    ax.set_axis_off()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    out_dir = "/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Paper/latex_source/figures"
    os.makedirs(out_dir, exist_ok=True)
    
    create_placeholder(f"{out_dir}/benchmark_design.pdf", "Benchmark Design Pipeline Diagram")
    create_placeholder(f"{out_dir}/scr_illustration.pdf", "SCR Metric Calculation Diagram")
