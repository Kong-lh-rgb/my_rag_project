import json
import os
import requests
from langchain_core.messages import HumanMessage,AIMessage
from pymilvus import connections, utility, Collection
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
import jieba
from dotenv import load_dotenv
load_dotenv()
from langchain_community.chat_models import ChatTongyi

class RetrievalSystem:
    def __init__(self, config_path="config.json"):
        self.load_config(config_path)
        self.connect_milvus()
        self.load_models()
        self.load_data()
        self.prepare_bm25()

    def load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.MILVUS_HOST = self.config.get("MILVUS_HOST", "localhost")
        self.MILVUS_PORT = self.config.get("MILVUS_PORT", "19530")
        self.OLLAMA_API_URL = self.config.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")
        self.LLM_MODEL = self.config.get("LLM_MODEL", "llama3")
        self.chunks_path = self.config.get("chunks_path", "data/processed/chunks.json")
        self.parent_docs_path = self.config.get("parent_docs_path", "data/processed/parent_documents.json")
        self.text_model_path = self.config.get("text_model_path", "models/bge-m3")
        self.code_model_path = self.config.get("code_model_path", "models/gte-large-zh")

    def connect_milvus(self):
        """连接Milvus数据库"""
        try:
            connections.connect(alias="default", host=self.MILVUS_HOST, port=self.MILVUS_PORT)
            print("成功连接到 Milvus！")
        except Exception as e:
            print(f"连接失败：{e}")
            raise

    def load_models(self):
        """加载本地模型"""
        print("正在加载本地模型...")
        self.text_model = SentenceTransformer(self.text_model_path)
        self.code_model = SentenceTransformer(self.code_model_path)
        print("模型加载完成。")

    def load_data(self):
        """加载文档数据"""
        print("\n正在加载文档数据...")
        try:
            with open(self.chunks_path, 'r', encoding='utf-8') as f:
                self.child_chunks = json.load(f)
            print(f"成功加载子文档数据（共{len(self.child_chunks)}条）。")

            with open(self.parent_docs_path, 'r', encoding='utf-8') as f:
                self.parent_docs = json.load(f)
            print(f"成功加载父文档数据（共{len(self.parent_docs)}条）。")
        except FileNotFoundError as e:
            print(f"错误：未找到文档数据 - {e}")
            raise

    def prepare_bm25(self):
        """准备BM25语料库"""
        print("\n准备BM25语料库...")
        corpus_tokens = [doc['content'].split(" ") for doc in self.child_chunks]
        self.bm25 = BM25Okapi(corpus_tokens)
        print("BM25语料库准备完成。")

    def rewrite_query_with_llm(self, query, model="llama3"):
        """使用Ollama将口语化查询重写为更适合检索的正式查询"""
        print(f"\n正在使用LLM重写查询：'{query}'")
        prompt = f"""你是一个智能问答系统的查询重写助手。你的任务是将用户提供的口语化或不完整的查询，转化为一个或多个适合于文档检索的、更精确、更完整的搜索查询。
        请仅返回重写后的查询，不要包含任何额外说明或句子。

        原始查询：如何用PyTorch进行线性回归？
        重写后的查询：如何在PyTorch中实现线性回归？

        原始查询：如何创建自定义流水线？
        重写后的查询：如何创建自定义流水线？

        原始查询：{query}
        重写后的查询：
        """

        try:
            response = requests.post(
                self.OLLAMA_API_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0}
                }
            )
            response.raise_for_status()
            rewritten_query = response.json()['response'].strip()
            print(f"查询重写完成，重写后的查询为：'{rewritten_query}'")
            return rewritten_query
        except Exception as e:
            print(f"查询重写失败：{e}。将使用原始查询。")
            return query



    def process_query(self,query):
        stop_words = set(['的', '是', '了', '和', '在', '我', '你', '他', '她', '它', '们', '什么'])
        """
        处理用户查询，进行分词和停用词过滤。

        Args:
            query (str): 用户的原始查询字符串。

        Returns:
            list: 经过处理后的关键词列表。
        """
        tokens = jieba.cut(query, cut_all=False)

        processed_tokens = [word for word in tokens if word not in stop_words and len(word.strip()) > 0]

        return processed_tokens


    def hybrid_search(self, query, top_k=10, rewrite=True):
        """
        执行混合检索

        参数:
            query: 用户查询
            top_k: 返回结果数量
            rewrite: 是否使用LLM重写查询

        返回:
            检索到的文档内容列表
        """
        if rewrite:
            query = self.rewrite_query_with_llm(query)

        print(f"\n正在执行混合检索，查询：'{query}'")

        collection = Collection("rag_documents")
        collection.load()

        query_vector = self.text_model.encode(query)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        vector_search_results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["pk", "parent_id"]
        )[0]
        vector_rank = {hit.id: (1 / (rank + 1)) for rank, hit in enumerate(vector_search_results)}
        print(f"完成向量搜索，得到 {len(vector_search_results)} 个结果。")


        query_tokens = self.process_query(query)
        print(f"处理后的查询关键词：{query_tokens}")
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_rank = {}
        sorted_indices = np.argsort(bm25_scores)[::-1]
        for rank, idx in enumerate(sorted_indices[:top_k]):
            doc_id = self.child_chunks[idx]['id']
            bm25_rank[doc_id] = (1 / (rank + 1))
        print(f"完成关键词搜索，得到 {len(bm25_rank)} 个结果。")


        rrf_scores = {}
        for doc_id, score in vector_rank.items():
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score
        for doc_id, score in bm25_rank.items():
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score
        final_ranks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


        retrieved_contexts = []
        seen_parent_ids = set()
        for doc_id, score in final_ranks:
            child_doc = next((doc for doc in self.child_chunks if doc['id'] == doc_id), None)
            if child_doc and child_doc['parent_id'] not in seen_parent_ids:
                parent_id = child_doc['parent_id']
                full_context = self.parent_docs.get(parent_id)
                if full_context:
                    retrieved_contexts.append({
                        "parent_id": parent_id,
                        "content": full_context,
                        "score": score
                    })
                    seen_parent_ids.add(parent_id)

        print("\n--- 检索结果（已融合并回溯父文档） ---")
        for i, context in enumerate(retrieved_contexts):
            print(f"排名 {i + 1}：")
            print(f"  来自文档：{context['parent_id']}")
            print(f"  得分：{context['score']:.4f}")
            print(f"  内容预览（前200字）：\n{context['content'][:200]}...")
            print("-" * 50)

        return [c['content'] for c in retrieved_contexts]

    def last(self,query,retrieved_docs):
        llm = ChatTongyi(model="qwen-max")
        context = "\n\n---\n\n".join(retrieved_docs[:3])
        prompt = f"""你是一个专业的技术问答助手，请根据提供的上下文信息回答用户的问题。

                # 上下文：
                {context}

                # 用户问题：
                {query}

                # 回答要求：
                1. 回答必须基于提供的上下文，不要编造信息
                2. 如果上下文不足以回答问题，请如实告知
                3. 保持回答专业、简洁
                4. 如果是代码问题，请提供可运行的代码示例

                # 回答：
                """
        response = llm.invoke(prompt)
        return response.content

    def generate_response(self, query, retrieved_docs, model="llama3"):
        """
        使用LLM基于检索到的文档生成最终回复

        参数:
            query: 用户原始查询
            retrieved_docs: 检索到的文档列表
            model: 使用的LLM模型名称

        返回:
            str: 生成的回复
        """

        model = model or self.LLM_MODEL
        print(f"\n正在使用 {model} 生成回复...")

        # 准备上下文
        context = "\n\n---\n\n".join(retrieved_docs[:3])  # 使用前3个文档作为上下文

        prompt = f"""你是一个专业的技术问答助手，请根据提供的上下文信息回答用户的问题。

        # 上下文：
        {context}

        # 用户问题：
        {query}

        # 回答要求：
        1. 回答必须基于提供的上下文，不要编造信息
        2. 如果上下文不足以回答问题，请如实告知
        3. 保持回答专业、简洁
        4. 如果是代码问题，请提供可运行的代码示例

        # 回答：
        """

        try:
            response = requests.post(
                self.OLLAMA_API_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_ctx": 4096  # 更大的上下文窗口
                    }
                },
                timeout=60  # 增加超时时间
            )
            response.raise_for_status()
            generated_text = response.json()['response'].strip()

            # 后处理：移除可能的多余前缀
            if generated_text.startswith("# 回答："):
                generated_text = generated_text[5:].strip()

            return generated_text
        except Exception as e:
            print(f"❌ 生成回复失败: {e}")
            return "抱歉，生成回复时出现问题。请稍后再试。"

    def query_and_generate(self, query, top_k=5, rewrite=True):
        """
        完整流程：检索+生成

        参数:
            query: 用户查询
            top_k: 检索文档数量
            rewrite: 是否重写查询
            model: 使用的LLM模型

        返回:
            dict: 包含检索结果和生成回复的字典
        """

        retrieved_docs = self.hybrid_search(query, top_k=top_k, rewrite=rewrite)

        generated_response = self.last(query,retrieved_docs)

        return {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "generated_response": generated_response
        }
