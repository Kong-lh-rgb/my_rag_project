import os
import re
import uuid
import tiktoken
import json

# --- 1. 设置路径和切分参数 ---
print("--- 1. 设置文档路径和切分参数 ---")
TORCH_DOCS_PATH = r"D:\pyproject\PythonProject3\data\origin\origin_torch"
TRANSFORMERS_DOCS_PATH = r"D:\pyproject\PythonProject3\data\origin\origin_transformers"

# 小粒度切分参数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 正则表达式用于匹配Markdown中的标题和代码块
TITLE_PATTERN = re.compile(r'^(#+)\s*(.*)', re.MULTILINE)
CODE_BLOCK_PATTERN = re.compile(r'```(?:[^\n]*)\n(.*?)```', re.DOTALL)


def parse_markdown_with_titles(content, parent_id):
    """
    解析Markdown内容，按标题和代码块进行分块
    """
    chunks = []
    last_end = 0

    # 找到所有标题和代码块的起始位置
    matches = list(TITLE_PATTERN.finditer(content))
    code_matches = list(CODE_BLOCK_PATTERN.finditer(content))

    all_matches = sorted(matches + code_matches, key=lambda m: m.start())

    current_title = ""
    for match in all_matches:
        start, end = match.span()
        block_content = content[last_end:start].strip()

        # 处理标题前的文本内容
        if block_content:
            chunks.extend(chunk_content(block_content, 'text', parent_id, current_title))

        # 处理当前匹配项
        if match.re == TITLE_PATTERN:
            # 这是一个标题
            title_level = len(match.group(1))
            title_text = match.group(2).strip()
            # 记录当前标题，用于后续内容
            current_title = title_text
        elif match.re == CODE_BLOCK_PATTERN:
            # 这是一个代码块
            code_content = match.group(1).strip()
            if code_content:
                chunks.extend(chunk_content(code_content, 'code', parent_id, current_title))

        last_end = end

    # 处理最后一个匹配项之后的内容
    remaining_text = content[last_end:].strip()
    if remaining_text:
        chunks.extend(chunk_content(remaining_text, 'text', parent_id, current_title))

    return chunks


def chunk_content(content, content_type, parent_id, section_title):
    """
    对给定的内容进行分块，并添加元数据
    """
    chunks = []
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(content)

    for i in range(0, len(tokens), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk_tokens = tokens[i:i + CHUNK_SIZE]
        chunk_content = encoding.decode(chunk_tokens)

        # 在内容中加上标题，增强语义信息
        enriched_content = f"标题：{section_title}\n\n{chunk_content}"

        chunks.append({
            "id": str(uuid.uuid4()),
            "content": enriched_content,
            "type": content_type,
            "parent_id": parent_id
        })

    return chunks


def get_all_chunks_from_dirs(root_dirs):
    all_chunks = []
    parent_documents = {}
    for root_dir in root_dirs:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.splitext(file_path)[1].lower() == '.md':
                    print(f"正在处理文件: {file_path}")

                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        parent_id = os.path.relpath(file_path, root_dir)
                        parent_documents[parent_id] = file_content

                    # 使用新的解析函数
                    chunks = parse_markdown_with_titles(file_content, parent_id)
                    all_chunks.extend(chunks)

    return all_chunks, parent_documents


if __name__ == "__main__":
    print("--- 开始文档切分 ---")
    root_directories = [TORCH_DOCS_PATH, TRANSFORMERS_DOCS_PATH]
    child_chunks, parent_docs = get_all_chunks_from_dirs(root_directories)

    with open(r'D:\pyproject\PythonProject3\data\processed\chunks.json', 'w', encoding='utf-8') as f:
        json.dump(child_chunks, f, ensure_ascii=False, indent=4)

    with open(r'D:\pyproject\PythonProject3\data\processed\parent_documents.json', 'w', encoding='utf-8') as f:
        json.dump(parent_docs, f, ensure_ascii=False, indent=4)

    print("\n--- 文档切分完成 ---")
    print(f"总共切分出 {len(child_chunks)} 个子文档块。")
    print("子文档数据已保存到 chunks.json 文件中。")
    print("父文档数据已保存到 parent_documents.json 文件中。")