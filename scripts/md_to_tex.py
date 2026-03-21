import os
import re

def convert_md_to_tex(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Convert ## Header to \section{Header}
    content = re.sub(r'^##\s+(?:[\d\.]+\s+)?(.*)$', r'\\section{\1}', content, flags=re.MULTILINE)
    
    # Convert ### Header to \subsection{Header}
    content = re.sub(r'^###\s+(?:[\d\.]+\s+)?(.*)$', r'\\subsection{\1}', content, flags=re.MULTILINE)
    
    # Convert #### Header to \paragraph{Header}
    content = re.sub(r'^####\s+(?:[\d\.]+\s+)?(.*)$', r'\\paragraph{\1}', content, flags=re.MULTILINE)

    # Convert **bold** to \textbf{bold}
    content = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', content)

    # Convert *italic* to \textit{italic}
    content = re.sub(r'\*(.*?)\*', r'\\textit{\1}', content)

    # Convert $$ ... $$ to \begin{equation} ... \end{equation}
    def replace_eq(match):
        eq = match.group(1).strip()
        return f"\\begin{{equation}}\n{eq}\n\\end{{equation}}"
    
    content = re.sub(r'\$\$(.*?)\$\$', replace_eq, content, flags=re.DOTALL)

    # Convert inline $...$ (if any, leave as is, since LaTeX supports it)

    # Convert lists
    # This is tricky, let's just do a simple replacement for basic lists if needed, 
    # but let's check if there are markdown lists.
    # Actually, let's write them explicitly.
    
    with open(filepath, 'w') as f:
        f.write(content)

sec_dir = '/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Paper/latex_source/sec'
for filename in os.listdir(sec_dir):
    if filename.endswith('.tex'):
        filepath = os.path.join(sec_dir, filename)
        convert_md_to_tex(filepath)
        print(f"Converted {filename}")
