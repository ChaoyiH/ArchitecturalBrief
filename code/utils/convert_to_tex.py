import sys
import os
import json
import re

def escape_tex(text):
    text = text.replace('%', r'\%')
    text = text.replace('&', r'\&')
    text = text.replace('$', r'\$')
    text = text.replace('#', r'\#')
    text = text.replace('_', r'\_')
    text = text.replace('~', r'\~{}')
    # Remove emoji characters that break the LaTeX compilation since they are missing in the OT font.
    # A simple regex to remove common emojis
    text = re.sub(r'[🔴🗯️🎋💡❌✅]', '', text)
    return text

def parse_md_to_tex(content):
    lines = content.split('\n')
    tex_lines = []
    
    in_table = False
    table_lines = []
    in_quote = False

    for orig_line in lines:
        if orig_line.strip().startswith('|') and orig_line.strip().endswith('|'):
            if in_quote:
                tex_lines.append(r"\end{quote}")
                tex_lines.append(r"\vspace{0.5em}")
                in_quote = False
            table_lines.append(orig_line)
            continue
        else:
            if table_lines:
                tex_lines.extend(convert_table(table_lines))
                table_lines = []
                
        if orig_line.startswith('# '):
            header_text = orig_line[2:].strip()
            tex_lines.append(rf"\chapter{{{escape_tex(header_text)}}}")
            continue
        if orig_line.startswith('## '):
            header_text = orig_line[3:].strip()
            tex_lines.append(rf"\section{{{escape_tex(header_text)}}}")
            continue
        if orig_line.startswith('### '):
            header_text = orig_line[4:].strip()
            tex_lines.append(rf"\subsection{{{escape_tex(header_text)}}}")
            continue
        if orig_line.startswith('#### '):
            header_text = orig_line[5:].strip()
            tex_lines.append(rf"\subsubsection{{{escape_tex(header_text)}}}")
            continue
            
        if orig_line.startswith('>'):
            if not in_quote:
                tex_lines.append(r"\vspace{0.5em}")
                tex_lines.append(r"\begin{quote}")
                in_quote = True
            
            quote_content = orig_line[1:].strip()
            line = escape_tex(quote_content)
            line = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', line)
            tex_lines.append(line)
            continue
        else:
            if in_quote:
                tex_lines.append(r"\end{quote}")
                tex_lines.append(r"\vspace{0.5em}")
                in_quote = False
                
        # Markdown formatting for normal text
        if orig_line.startswith('- '):
            line_content = orig_line[2:]
            line = r"\item " + escape_tex(line_content)
        else:
            line = escape_tex(orig_line)
            
        line = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', line)
            
        tex_lines.append(line)
            
    if table_lines:
        tex_lines.extend(convert_table(table_lines))
    if in_quote:
        tex_lines.append(r"\end{quote}")
        
    # fix itemize
    final_lines = []
    in_itemize = False
    for line in tex_lines:
        if line.startswith(r"\item "):
            if not in_itemize:
                final_lines.append(r"\begin{itemize}")
                in_itemize = True
            final_lines.append(line)
        else:
            if in_itemize and line.strip() != "" and not line.startswith("%"):
                final_lines.append(r"\end{itemize}")
                in_itemize = False
            final_lines.append(line)
    if in_itemize:
        final_lines.append(r"\end{itemize}")
        
    return '\n'.join(final_lines)

def convert_table(table_lines):
    if len(table_lines) < 2: return table_lines
    
    first_row = table_lines[0].strip('|')
    cols = len(first_row.split('|'))
    
    col_str = ""
    if cols > 0:
        col_str = " *{" + str(cols) + r"}{p{\dimexpr0.95\textwidth/" + str(cols) + r"\relax}} "
    
    tex_out = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{" + col_str + "}",
        r"\toprule"
    ]
    
    for i, row in enumerate(table_lines):
        if '---' in row:
            if i == 1:
                tex_out.append(r"\midrule")
            continue
            
        cells = [c.strip() for c in row.strip('|').split('|')]
        # apply formatting / escaping to cells
        cells = [escape_tex(c) for c in cells]
        cells = [re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', c) for c in cells]
        
        tex_out.append(" & ".join(cells) + r" \\")
        
    tex_out.append(r"\bottomrule")
    tex_out.append(r"\end{tabular}")
    tex_out.append(r"\end{table}")
    return tex_out

def parse_json_to_tex(content):
    data = json.loads(content)
    tex_lines = []
    
    tex_lines.append(r"\chapter{多智能体评估意见实录}")
    
    for idx, item in enumerate(data):
        role = escape_tex(item.get('role', 'Unknown'))
        phase = escape_tex(item.get('phase', 'Unknown'))
        scale = escape_tex(item.get('scale', 'Unknown'))
        
        tex_lines.append(rf"\section{{角色：{role} | 阶段：{phase} | 尺度：{scale}}}")
        
        feedback = item.get('feedback', '')
        # Parse feedback from markdown
        parsed = parse_md_to_tex(feedback)
        # Downgrade headers so they fit inside section
        parsed = parsed.replace(r"\chapter{", r"\subsection{")
        parsed = parsed.replace(r"\section{", r"\subsubsection{")
        parsed = parsed.replace(r"\subsection{", r"\paragraph{")
        
        tex_lines.append(parsed)
        tex_lines.append("\n")
        
    return '\n'.join(tex_lines)

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_to_tex.py <input_file_path_or_url> <output_dir>")
        sys.exit(1)
        
    input_uri = sys.argv[1].strip()
    output_dir = sys.argv[2].strip()
    
    # Handle uri formats like @[...], file://, etc.
    if input_uri.startswith('@[') and input_uri.endswith(']'):
        input_uri = input_uri[2:-1]
        
    if input_uri.startswith('file://'):
        input_path = input_uri[7:]
    else:
        input_path = input_uri
        
    input_path = os.path.abspath(input_path)
    output_dir = os.path.abspath(output_dir)
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(input_path)
    base_name, ext = os.path.splitext(filename)
    
    output_filename = f"appendix-{base_name}.tex"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    tex_content = "% !TeX root = ../thuthesis-example.tex\n\n"
        
    print(f"Converting {filename} to {output_filename}...")
    if ext.lower() == '.md':
        tex_content += parse_md_to_tex(content)
    elif ext.lower() == '.json':
        tex_content += parse_json_to_tex(content)
    else:
        print(f"Error: Unsupported file extension {ext}. Only .md and .json are supported.")
        sys.exit(1)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(tex_content)
        
    print(f"Successfully wrote {output_path}")

if __name__ == '__main__':
    main()
