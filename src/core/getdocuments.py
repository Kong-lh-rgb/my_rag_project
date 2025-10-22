import json
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


print("--- 1. 设置 Milvus 连接和模型 ---")
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'

try:
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("成功连接到 Milvus!")
except Exception as e:
    print(f"连接失败：{e}")
    exit()

local_path_bge_m3 = 'D:/Models/bge-m3'
local_path_gte_large_zh = 'D:/Models/gte-large-zh'

print("正在加载本地模型...")
text_model = SentenceTransformer(local_path_bge_m3)
code_model = SentenceTransformer(local_path_gte_large_zh)
print("模型加载完成。")


try:
    with open(r'D:\pyproject\PythonProject3\data\processed\chunks.json', 'r', encoding='utf-8') as f:
        child_chunks = json.load(f)
    print("成功加载 chunks.json 文件。")
except FileNotFoundError:
    print("错误：未找到 chunks.json 文件。请先运行 prepare_documents.py。")
    exit()

print("\n--- 3. 正在创建 Milvus Collection ---")
COLLECTION_NAME = "rag_documents"
DIMENSION = 1024

fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
    FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096)
]
schema = CollectionSchema(fields, "文档向量用于RAG检索")

if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

collection = Collection(name=COLLECTION_NAME, schema=schema)


print("\n--- 4. 正在向量化并插入数据到 Milvus ---")
BATCH_SIZE = 500

total_chunks = len(child_chunks)
num_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE

with tqdm(total=total_chunks, desc="向量化进度") as pbar:
    for i in range(0, total_chunks, BATCH_SIZE):
        batch_chunks = child_chunks[i:i + BATCH_SIZE]
        data_to_insert = []

        for doc in batch_chunks:
            content = doc["content"]
            doc_type = doc["type"]
            if doc_type == "text":
                print("正在对一个文本块进行向量化...")
                vector = text_model.encode(content)
            elif doc_type == "code":
                vector = code_model.encode(content)
            else:
                continue

            data_to_insert.append({
                "pk": doc["id"],
                "vector": vector,
                "parent_id": doc["parent_id"],
                "content": content
            })


        collection.insert(data_to_insert)

        pbar.update(len(batch_chunks))

print("所有数据已分批次向量化并插入成功")

print("\n--- 5. 正在创建向量索引 ---")
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}
collection.create_index(field_name="vector", index_params=index_params)
collection.load()
print("索引创建并加载完成。所有文档已准备就绪")