
import os
import re
import html2text

def convert_html_to_md(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    h = html2text.HTML2Text()
    h.body_width = 0 # To disable line wrapping

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.html'):
                html_path = os.path.join(root, file)
                # Construct the output path to mirror the input structure
                relative_path = os.path.relpath(html_path, input_dir)
                md_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + '.md')
                
                # Create subdirectories in the output if they don't exist
                os.makedirs(os.path.dirname(md_path), exist_ok=True)

                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                # Convert to markdown
                markdown_content = h.handle(html_content)

                # Replace .html links with .md links
                markdown_content = re.sub(r'\.html', '.md', markdown_content)

                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)

if __name__ == '__main__':
    convert_html_to_md('corpus/SWT1AQ', 'corpus/SWT1AQ_md')
