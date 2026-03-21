import os
import re
import glob

def tex_to_md(tex_content):
    # Remove \section, \subsection, etc.
    tex_content = re.sub(r'\\section\*?\{(.*?)\}', r'## \1\n', tex_content)
    tex_content = re.sub(r'\\subsection\*?\{(.*?)\}', r'### \1\n', tex_content)
    tex_content = re.sub(r'\\subsubsection\*?\{(.*?)\}', r'#### \1\n', tex_content)
    tex_content = re.sub(r'\\paragraph\*?\{(.*?)\}', r'**\1** ', tex_content)
    
    # Text formatting
    tex_content = re.sub(r'\\textbf\{(.*?)\}', r'**\1**', tex_content)
    tex_content = re.sub(r'\\textit\{(.*?)\}', r'*\1*', tex_content)
    
    # Citations
    tex_content = re.sub(r'\\cite\{(.*?)\}', r'[\1]', tex_content)
    
    # References
    tex_content = re.sub(r'\\ref\{(.*?)\}', r'\1', tex_content)
    
    # Equations
    tex_content = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', r'$$\1$$', tex_content, flags=re.DOTALL)
    
    # Lists
    tex_content = re.sub(r'\\begin\{itemize\}', '', tex_content)
    tex_content = re.sub(r'\\end\{itemize\}', '', tex_content)
    tex_content = re.sub(r'\\begin\{enumerate\}', '', tex_content)
    tex_content = re.sub(r'\\end\{enumerate\}', '', tex_content)
    tex_content = re.sub(r'\\item\s+', r'- ', tex_content)
    
    # Figures and Tables (very basic removal/conversion)
    tex_content = re.sub(r'\\begin\{figure.*?\}.*?\\includegraphics(?:\[.*?\])?\{(.*?)\}.*?\\caption\{(.*?)\}.*?\\end\{figure.*?\}', 
                         r'![\2](\1)\n*\2*\n', tex_content, flags=re.DOTALL)
    tex_content = re.sub(r'\\begin\{table.*?\}.*?\\end\{table.*?\}', r'[Table Content Here]\n', tex_content, flags=re.DOTALL)
    
    # Abstract
    tex_content = re.sub(r'\\begin\{abstract\}', r'## Abstract\n', tex_content)
    tex_content = re.sub(r'\\end\{abstract\}', '', tex_content)
    
    # Remove extra spaces
    tex_content = re.sub(r'\n{3,}', '\n\n', tex_content)
    
    return tex_content

def sync_latex_to_md():
    base_dir = '/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop'
    sec_dir = os.path.join(base_dir, 'Paper', 'latex_source', 'sec')
    md_dir = os.path.join(base_dir, 'Paper')
    
    # Mapping
    mapping = {
        '0_abstract.tex': 'section_0_abstract.md',
        '1_intro.tex': 'section_1_introduction.md',
        '2_related_work.tex': 'section_2_related_work.md',
        '3_benchmark.tex': 'section_3_benchmark.md',
        '4_metrics.tex': 'section_4_metrics.md',
        '5_experiments.tex': 'section_5_experiments.md',
        '6_7_discussion_conclusion.tex': 'section_6_7_discussion_conclusion.md'
    }
    
    for tex_file, md_file in mapping.items():
        tex_path = os.path.join(sec_dir, tex_file)
        md_path = os.path.join(md_dir, md_file)
        
        if os.path.exists(tex_path):
            with open(tex_path, 'r') as f:
                content = f.read()
            
            md_content = tex_to_md(content)
            
            # Fix figure paths for MD
            md_content = md_content.replace('figures/', 'images/')
            
            with open(md_path, 'w') as f:
                f.write(md_content)
            print(f"Synced {tex_file} -> {md_file}")

if __name__ == "__main__":
    sync_latex_to_md()
