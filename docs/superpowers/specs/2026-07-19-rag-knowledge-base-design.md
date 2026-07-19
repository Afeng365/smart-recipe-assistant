# RAG 知识库功能设计文档

**日期**: 2026-07-19
**项目**: smart-recipe-assistant（智能食谱助手）
**状态**: 设计完成，待实现

---

## 1. 目标与场景

为智能食谱助手增加 RAG（检索增强生成）知识库功能，使用户可以：

- **场景 A**：上传个人私房菜谱文档（PDF/DOCX/TXT/MD/JSON），构建私人菜谱知识库，对话中检索个人菜谱
- **场景 B**：批量导入某个菜系或来源的菜谱文档，构建本地菜谱库供离线查询
- **场景 C**：与现有的 TheMealDB/CocktailDB 实时搜索互补——中式/个人菜谱走知识库，西餐/鸡尾酒走外部 API

---

## 2. 技术选型

| 维度 | 选择 | 理由 |
|------|------|------|
| 嵌入模型 | 本地 `BAAI/bge-small-zh-v1.5` | 512维轻量模型，中文效果好，离线免费 |
| 向量存储 | ChromaDB（嵌入式模式） | Python 原生，支持元数据过滤，零运维 |
| 元数据管理 | SQLite | 查询方便，适合管理文档映射关系 |
| 工具集成 | 搜索为 LLM tool，管理走 Flask API | 搜索是对话核心功能，管理是后台操作 |
| 文档加载 | PyMuPDF + python-docx + 原生 | 覆盖常见菜谱文档格式 |
| 文本切片 | ChineseRecursiveTextSplitter | 中文感知切片，参考 Langchain-ChatChat |

---

## 3. 整体架构

```
┌──────────────────────────────────────────────────────┐
│                    Flask Web UI                        │
│  GET /                    → SPA 前端                  │
│  GET /api/chat/stream     → SSE 对话流（已有）        │
│  POST /api/kb/create      → 创建知识库    [新增]      │
│  POST /api/kb/upload      → 上传文档      [新增]      │
│  GET /api/kb/list         → 列出知识库    [新增]      │
│  DELETE /api/kb/<name>    → 删除知识库    [新增]      │
│  GET /api/kb/<name>/docs  → 列出文档      [新增]      │
│  DELETE /api/kb/docs      → 删除文档      [新增]      │
└──────────────────────────┬───────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────┐
│              RecipeAgent (recipe_agent.py)             │
│  + search_recipe_knowledge_base(query, kb_name) [新工具]│
│    → 搜索知识库，返回相关文档片段                       │
└──────────────────────────┬───────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────┐
│         KnowledgeBase 子系统 (handlers/knowledge_base/)│
│  ├── __init__.py           → 模块入口                 │
│  ├── kb_manager.py         → KB CRUD 核心             │
│  ├── document_loader.py    → 文档加载/切片            │
│  ├── embedding.py          → 嵌入模型（单例）          │
│  ├── vector_store.py       → ChromaDB 操作封装         │
│  ├── search.py             → 检索逻辑                 │
│  └── db.py                 → SQLite 元数据库           │
└──────────────────────────┬───────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────┐
│                    存储层                              │
│  data/knowledge_base/                                 │
│  ├── info.db            → SQLite 元数据库              │
│  └── <kb_name>/                                      │
│      ├── content/       → 原始文档                    │
│      └── chroma/        → ChromaDB 持久化向量数据      │
└──────────────────────────────────────────────────────┘
```

---

## 4. 核心组件设计

### 4.1 KnowledgeBaseManager（`kb_manager.py`）

知识库生命周期管理，对外提供统一入口：

```python
class KnowledgeBaseManager:
    def create_kb(name: str, description: str = "") -> KBInfo
    def delete_kb(name: str) -> None
    def list_kbs() -> List[KBInfo]
    def get_kb(name: str) -> KBInfo

    def add_documents(kb_name: str, files: List[UploadFile],
                      chunk_size: int = 500,
                      chunk_overlap: int = 50) -> int  # 返回切片总数
    def remove_documents(kb_name: str, filenames: List[str]) -> None
    def list_documents(kb_name: str) -> List[DocInfo]
```

**设计要点**：
- 知识库名称做安全校验（防路径遍历），规则：只允许中文、字母、数字、下划线、短横线
- `add_documents` 内部调用文档加载 → 切片 → 嵌入 → 写入 ChromaDB → 更新 SQLite 的完整流水线
- 删除知识库时同步清理文件、ChromaDB 数据、SQLite 记录

### 4.2 文档处理流水线（`document_loader.py`）

```
文件上传 → 格式检测 → 文本提取 → 中文智能切片 → 嵌入向量 → ChromaDB
```

**支持的格式与加载器**：

| 格式 | 加载器 | 依赖 |
|------|--------|------|
| .txt | 原生 `open()` | 无 |
| .md | 原生 `open()` | 无 |
| .pdf | PyMuPDF (fitz) | `pymupdf` |
| .docx | python-docx | `python-docx` |
| .json | 自定义（支持列表/字典格式） | 无 |

**文本切片**：参考 Langchain-ChatChat 的 `ChineseRecursiveTextSplitter`：
- 分隔符优先级：`"\n\n"` → `"\n"` → `"。"` → `"！"` → `"？"` → `"."` → `"!"` → `"?"` → `"；"` → `";"` → `"，"` → `","` → `" "`
- 默认参数：`chunk_size=500`, `chunk_overlap=50`
- 理由：菜谱文档通常一道菜几百字，500 字切片能完整包含一道菜的信息

### 4.3 嵌入模型（`embedding.py`）

```python
class EmbeddingModel:
    """嵌入模型封装，单例模式，避免重复加载"""
    _instance = None
    _model = None

    def __new__(cls, model_name: str = "BAAI/bge-small-zh-v1.5"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> None        # 首次使用时下载并加载模型
    def embed(self, texts: List[str]) -> List[List[float]]   # 批量嵌入
    def embed_query(self, text: str) -> List[float]           # 单条嵌入
    @property
    def dimension(self) -> int    # 512
```

**设计要点**：
- 使用 `sentence-transformers` 库，自动处理 tokenization 和 pooling
- 默认模型 `BAAI/bge-small-zh-v1.5`（约 100MB，512 维向量）
- BGE 模型对查询和文档建议不同处理（`embed_query` 可加 instruction prefix）
- 预留 `model_name` 参数支持切换到更大模型（如 `bge-base-zh-v1.5`、`bge-large-zh-v1.5`）

### 4.4 向量存储（`vector_store.py`）

```python
class VectorStore:
    """ChromaDB 操作封装"""
    def __init__(self, kb_name: str, embedding_model: EmbeddingModel)
    def add(self, texts: List[str], metadatas: List[dict], ids: List[str]) -> None
    def query(self, query_text: str, top_k: int = 3,
              score_threshold: float = 0.3) -> List[SearchResult]
    def delete(self, ids: List[str]) -> None
    def delete_by_filter(self, filter: dict) -> None   # 按元数据删除
    def count(self) -> int
    def clear(self) -> None
```

**设计要点**：
- 每个知识库对应一个 ChromaDB Collection，名称为 `kb_{kb_name}`
- 持久化路径：`data/knowledge_base/{kb_name}/chroma/`
- 元数据字段：`source`（文件名）、`chunk_index`（切片序号）、`kb_name`
- `score_threshold` 默认 0.3（ChromaDB 默认 cosine 距离，1.0=完全匹配，0=无关）

### 4.5 检索逻辑（`search.py`）

```python
@dataclass
class SearchResult:
    content: str
    metadata: dict
    score: float

def search_knowledge_base(query: str, kb_name: str = None,
                          top_k: int = 3,
                          score_threshold: float = 0.3) -> List[SearchResult]:
    """
    搜索知识库。
    kb_name 为 None 时搜索所有知识库，合并结果去重排序。
    """

def format_search_results(results: List[SearchResult]) -> str:
    """将搜索结果格式化为 LLM 可读的上下文字符串"""
```

**检索结果格式化模板**：

```
【已知菜谱知识】
[来源：家庭菜谱.pdf，相关性：0.92]
五花肉500g，冰糖30g，老抽15ml...

[来源：川菜大全.pdf，相关性：0.85]
四川红烧肉讲究先炒糖色...
```

### 4.6 SQLite 元数据库（`db.py`）

三张表：

```sql
CREATE TABLE IF NOT EXISTS knowledge_base (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT DEFAULT '',
    embedding_model TEXT DEFAULT 'BAAI/bge-small-zh-v1.5',
    doc_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS knowledge_file (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kb_name TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_ext TEXT,
    file_size INTEGER,
    chunk_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(kb_name, filename),
    FOREIGN KEY (kb_name) REFERENCES knowledge_base(name) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS file_chunk (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kb_name TEXT NOT NULL,
    filename TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_id TEXT NOT NULL,   -- ChromaDB 内部的 chunk ID
    FOREIGN KEY (kb_name) REFERENCES knowledge_base(name) ON DELETE CASCADE,
    FOREIGN KEY (kb_name, filename) REFERENCES knowledge_file(kb_name, filename) ON DELETE CASCADE
);
```

**设计要点**：
- SQLite 文件路径：`data/knowledge_base/info.db`
- name 使用外键约束保证数据一致性
- `file_chunk` 表记录了文件名到 ChromaDB chunk ID 的映射，用于精确删除

---

## 5. LLM 工具集成

### 5.1 新增 Tool 定义

在 `tools.py` 中新增：

```python
def search_recipe_knowledge_base(query: str, kb_name: str = None) -> dict:
    """
    搜索本地菜谱知识库，查找用户上传的私人或本地菜谱内容。
    优先用于查找中餐、家常菜、或用户提到"我的菜谱"、"本地菜谱"时。
    参数:
      query: 搜索查询，如"红烧肉做法"、"川菜麻婆豆腐"
      kb_name: 指定知识库名称（可选，不指定则搜索所有知识库）
    返回:
      {"results": [{"content": ..., "source": ..., "score": ...}], "total": N}
    """
```

工具定义（OpenAI Function Calling 格式）：

```python
{
    "type": "function",
    "function": {
        "name": "search_recipe_knowledge_base",
        "description": "搜索本地菜谱知识库，查找用户上传的私房菜谱、家传菜谱或本地导入的菜谱文档。当用户询问中餐、家常菜、或提到'我的菜谱''本地菜谱''家传'等关键词时优先使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询，如'红烧肉做法'、'川菜麻婆豆腐'"
                },
                "kb_name": {
                    "type": "string",
                    "description": "指定知识库名称，不指定则搜索所有知识库"
                }
            },
            "required": ["query"]
        }
    }
}
```

### 5.2 与现有工具的关系

| 工具 | 适用场景 | 数据来源 |
|------|----------|----------|
| `search_meals` | 西餐、国际菜谱、鸡尾酒 | TheMealDB / TheCocktailDB |
| `search_recipe_knowledge_base` | 中餐、家常菜、私人菜谱 | 本地知识库 |

LLM 被指示根据用户查询语言和内容自动选择合适的工具，或两者都调用进行交叉参考。

---

## 6. Flask API 端点

| 方法 | 路径 | 描述 | 请求体 |
|------|------|------|--------|
| POST | `/api/kb/create` | 创建知识库 | `{"name": "...", "description": "..."}` |
| GET | `/api/kb/list` | 列出所有知识库 | - |
| DELETE | `/api/kb/<name>` | 删除知识库 | - |
| POST | `/api/kb/<name>/upload` | 上传文档 | `multipart/form-data`, `files` 字段 |
| GET | `/api/kb/<name>/docs` | 列出文档 | - |
| DELETE | `/api/kb/<name>/docs` | 删除文档 | `{"filenames": ["a.pdf", "b.txt"]}` |
| GET | `/api/health` | 健康检查（已有） | - |

**设计要点**：
- 文件上传使用 `multipart/form-data`，支持批量上传（`<input multiple>`）
- 上传后自动触发文档处理流水线（加载→切片→嵌入→存储）
- 返回处理结果（切片数量、耗时）
- 错误处理：文件格式不支持、文件过大（>50MB）、知识库不存在等

---

## 7. 依赖变更

`requirements.txt` 新增：

```
chromadb==0.5.23          # ChromaDB 向量数据库
sentence-transformers==3.4.1  # 嵌入模型
pymupdf==1.25.5           # PDF 解析
python-docx==1.1.2        # DOCX 解析
```

首次使用时自动下载嵌入模型（约 100MB），后续从缓存加载。

---

## 8. 目录结构变更

```
smart-recipe-assistant/
  handlers/
    knowledge_base/          # [新增] RAG 知识库子系统
      __init__.py
      kb_manager.py          # KB CRUD 管理
      document_loader.py     # 文档加载 + 中文切片
      embedding.py           # 嵌入模型封装
      vector_store.py        # ChromaDB 操作
      search.py              # 检索逻辑
      db.py                  # SQLite 元数据库
  tools.py                   # [修改] 新增 search_recipe_knowledge_base
  app.py                     # [修改] 新增 /api/kb/* 路由
  settings/
    constant.py              # [修改] 新增 KB 相关配置常量
  data/
    knowledge_base/          # [新增] 知识库存储根目录
      info.db                # SQLite 元数据库（运行时创建）
  docs/superpowers/specs/
    2026-07-19-rag-knowledge-base-design.md  # 本文档
```

---

## 9. 错误处理

| 错误场景 | 处理方式 |
|----------|----------|
| 知识库名称非法（含路径分隔符等） | 400 Bad Request + 错误消息 |
| 创建同名知识库 | 409 Conflict |
| 知识库不存在 | 404 Not Found |
| 文件格式不支持 | 400 Bad Request，列出支持的格式 |
| 文件过大（>50MB） | 413 Payload Too Large |
| 嵌入模型加载失败 | 500，记录日志，提示检查网络/磁盘 |
| ChromaDB 读写失败 | 500，记录日志，保留原始文档不丢失 |
| 知识库为空时搜索 | 返回空结果，LLM 告知用户"知识库中暂无相关内容" |

---

## 10. 配置常量

在 `settings/constant.py` 中新增：

```python
# 知识库配置
KB_ROOT_PATH = WORKDIR / "data" / "knowledge_base"
KB_DB_PATH = KB_ROOT_PATH / "info.db"
KB_CONTENT_DIR_NAME = "content"
KB_CHROMA_DIR_NAME = "chroma"
KB_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
KB_DEFAULT_CHUNK_SIZE = 500
KB_DEFAULT_CHUNK_OVERLAP = 50
KB_DEFAULT_TOP_K = 3
KB_DEFAULT_SCORE_THRESHOLD = 0.3
KB_MAX_FILE_SIZE_MB = 50
KB_SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf", ".docx", ".json")
```

---

## 11. 测试策略

| 测试类型 | 覆盖内容 |
|----------|----------|
| 单元测试 | `document_loader` 各格式加载器、`ChineseRecursiveTextSplitter` 切片逻辑、`db.py` CRUD 操作 |
| 集成测试 | 完整流水线：创建 KB → 上传文档 → 搜索 → 删除 KB |
| 边界测试 | 空文件、超大文件、异常格式、特殊字符文件名、并发上传 |
| E2E 测试 | Flask API 端点请求/响应验证，LLM tool 调用链路验证 |

---

## 12. 后续可扩展项（不在本次范围）

- **BM25 混合检索**：提升关键词匹配精度
- **Reranker 重排序**：用 Cross-Encoder 对初步检索结果精排
- **知识库更新/重索引**：修改文档后重新切片嵌入
- **临时知识库**：上传文件仅用于当前对话，不持久化（参考 Langchain-ChatChat 的 `memo_faiss_pool`）
- **多嵌入模型支持**：生产环境可切换 `bge-large-zh-v1.5`
- **前端管理界面**：在 SPA 中增加知识库管理面板
