import os
import re

citation_map = {
    '1': 'ruiz2023dreambooth',
    '2': 'kumari2023multi',
    '3': 'radford2021learning',
    '4': 'oquab2023dinov2',
    '5': 'mosaic2024',
    '6': 'xverse2024',
    '7': 'psr2024',
    '8': 'gal2022image',
    '9': 'contextgen2024',
    '10': 'anyms2024',
    '11': 'peebles2023scalable',
    '12': 'flux2024',
    '13': '3disflux2024',
    '14': 'seethrough3d2024',
    '15': 'layerbind2024'
}

def replace_citations(match):
    # match.group(1) contains the inner numbers, e.g., "1, 2, 8"
    nums = [n.strip() for n in match.group(1).split(',')]
    keys = []
    for n in nums:
        if n in citation_map:
            keys.append(citation_map[n])
        else:
            # If not found, keep original or print warning
            print(f"Warning: citation {n} not found!")
            return match.group(0)
    return f"\\cite{{{','.join(keys)}}}"

def convert_citations_in_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Regex to find [num, num, ...] but avoid matching words
    # A bit tricky: we only want to match brackets containing numbers and commas
    content = re.sub(r'\[([0-9\s,]+)\]', replace_citations, content)

    with open(filepath, 'w') as f:
        f.write(content)

sec_dir = '/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Paper/latex_source/sec'
for filename in os.listdir(sec_dir):
    if filename.endswith('.tex'):
        filepath = os.path.join(sec_dir, filename)
        convert_citations_in_file(filepath)
        print(f"Processed citations in {filename}")
