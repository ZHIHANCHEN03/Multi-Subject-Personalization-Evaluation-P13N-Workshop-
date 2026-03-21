import os
import json
from PIL import Image, ImageDraw, ImageFont

def get_prompts():
    base_dir = '/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop'
    json_path = os.path.join(base_dir, 'eval_outputs', 'mosaic_results.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    prompts = {}
    for item in data['results']:
        if item['prompt_id'] not in prompts:
            prompts[item['prompt_id']] = item['prompt']
    return prompts

def create_teaser_grid():
    base_dir = '/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop'
    output_dir = os.path.join(base_dir, 'Paper', 'latex_source', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    prompts = get_prompts()
    
    # Let's compare 2 subjects vs 8 subjects for MOSAIC
    cases = [
        ('11_interaction', 42, '2 Subjects (Success)'),
        ('56_interaction', 42, '8 Subjects (Catastrophic Collapse)')
    ]
    
    models = ['mosaic', 'xverse']
    model_titles = ['MOSAIC', 'XVerse']
    
    first_img_path = os.path.join(base_dir, f'results/mosaic/{cases[0][0]}/{cases[0][1]}.png')
    with Image.open(first_img_path) as img:
        img_w, img_h = img.size
        
    scale_factor = 0.6
    img_w = int(img_w * scale_factor)
    img_h = int(img_h * scale_factor)
    
    padding_x = 20
    padding_y = 60
    top_margin = 60
    
    grid_w = len(models) * (img_w + padding_x)
    grid_h = top_margin + len(cases) * (img_h + padding_y)
    
    grid_img = Image.new('RGB', (grid_w, grid_h), color='white')
    draw = ImageDraw.Draw(grid_img)
    
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        text_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except IOError:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        
    # Draw column titles
    for i, title in enumerate(model_titles):
        x = i * (img_w + padding_x) + img_w // 2
        y = top_margin // 2
        bbox = draw.textbbox((0, 0), title, font=title_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text((x - text_w // 2, y - text_h // 2), title, font=title_font, fill='black')
        
    # Draw rows
    for r, (prompt_id, seed, case_title) in enumerate(cases):
        y_offset = top_margin + r * (img_h + padding_y)
        
        draw.text((10, y_offset - 40), case_title, font=text_font, fill='red' if 'Collapse' in case_title else 'green')
        
        for c, model in enumerate(models):
            img_path = os.path.join(base_dir, f'results/{model}/{prompt_id}/{seed}.png')
            try:
                with Image.open(img_path) as img:
                    img = img.resize((img_w, img_h), Image.Resampling.LANCZOS)
                    x_offset = c * (img_w + padding_x)
                    grid_img.paste(img, (x_offset, y_offset))
                    
                    # Add border
                    border_color = 'black'
                    draw.rectangle([x_offset, y_offset, x_offset+img_w, y_offset+img_h], outline=border_color, width=2)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                
    output_path = os.path.join(output_dir, 'teaser.png')
    grid_img.save(output_path)
    print(f"Saved teaser grid to {output_path}")

if __name__ == '__main__':
    create_teaser_grid()
