import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from retriever import RetrievalSystem
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware


# Pydantic 模型，用于定义API请求和响应的数据结构
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    rewrite: bool = True


class RAGResponse(BaseModel):
    query: str
    retrieved_docs: List[str]
    generated_response: str
    process_time_seconds: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理。
    - 在应用启动时，进行初始化。
    - 在应用关闭时，进行清理。
    """
    global retrieval_system
    print("正在初始化 RAG 系统...")
    try:
        retrieval_system = RetrievalSystem("config.json")
        print("✅ RAG 系统初始化成功！")
    except Exception as e:
        print(f"❌ RAG 系统初始化失败: {e}")
        raise HTTPException(status_code=500, detail=f"系统初始化失败: {e}")

    yield

    print("正在关闭 RAG 系统...")
    print("✅ RAG 系统已关闭。")

app = FastAPI(
    title="RAG System API",
    description="一个基于检索增强生成（RAG）的问答系统API。",
    lifespan=lifespan
)

retrieval_system = None

# 使用通配符 * 来确保 CORS 策略能够匹配所有来源
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 显式添加一个 OPTIONS 方法处理函数来确保 CORS 预检请求能得到正确响应
@app.options("/query")
async def handle_options_request():
    return JSONResponse(content={"message": "Preflight request successful"})

@app.get("/")
def read_root():
    """
    根路径，用于简单的健康检查。
    """
    return {"message": "RAG System API is running"}


@app.post("/query", response_model=RAGResponse)
async def handle_query(request: QueryRequest):
    """
    处理用户的查询请求，执行RAG流程并返回结果。
    """
    global retrieval_system
    if retrieval_system is None:
        raise HTTPException(status_code=503, detail="RAG系统正在启动，请稍后重试。")

    print(f"收到查询请求: '{request.query}'")
    start_time = asyncio.get_event_loop().time()

    try:
        result = retrieval_system.query_and_generate(
            query=request.query,
            top_k=request.top_k,
            rewrite=request.rewrite
        )

        elapsed = asyncio.get_event_loop().time() - start_time
        result["process_time_seconds"] = round(elapsed, 2)

        return RAGResponse(**result)
    except Exception as e:
        print(f"处理查询时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"处理查询时发生错误: {e}")