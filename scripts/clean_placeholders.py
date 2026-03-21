import os
import glob

def clean_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    cleaned = []
    for line in lines:
        if '=========================================' in line:
            continue
        if 'PLACEHOLDER' in line and ('TEASER FIGURE' in line or 'FIGURE' in line or 'TABLE' in line):
            continue
        if '(For CVPR format, place this at the top of the first page or immediately after \\maketitle)' in line:
            continue
        cleaned.append(line)
        
    with open(filepath, 'w') as f:
        f.writelines(cleaned)

for f in glob.glob('/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Paper/latex_source/sec/*.tex'):
    clean_file(f)
