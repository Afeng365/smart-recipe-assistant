# RAG 知识库功能设计文档

**日期**: 2026-07-19（更新于 2026-07-20）
**项目**: smart-recipe-assistant（智能食谱助手）
**状态**: 已实现

---

## 1. 目标与场景

为智能食谱助手增加 RAG（检索增强生成）知识库功能，使用户可以：

- **场景 A**：上传个人私房菜谱文档（PDF/DOCX/TXT/MD/JSON），构建私人菜谱知识库，对话中检索个人菜谱
- **场景 B**：批量导入某个菜系或来源的菜谱文档，构建本地菜谱库供离线查询
- **场景 C**：与现有的 TheMealDB/CocktailDB 实时搜索互补——中式/个人菜谱走知识库，西餐/鸡尾酒走外部 API

---

## 2. 技术选型

| 维度 | 设计计划 | 实际实现 | 理由 |
|------|----------|----------|------|
| 嵌入模型 | `BAAI/bge-small-zh-v1.5`（本地 sentence-transformers） | **`qwen3-embedding:0.6b`**（Ollama 本地部署） | HuggingFace 不可达，改用 Ollama REST API，1024 维向量自动检测 |
| 向量存储 | ChromaDB（嵌入式模式） | ✅ ChromaDB 0.5.23 | Python 原生，支持元数据过滤，零运维 |
| 元数据管理 | SQLite | ✅ SQLite | 查询方便，适合管理文档映射关系 |
| 工具集成 | 搜索为 LLM tool，管理走 Flask API | ✅ 22 个 LLM tools 中包含 `search_recipe_knowledge_base` | 搜索是对话核心功能，管理是后台操作 |
| 文档加载 | PyMuPDF + python-docx + 原生 | ✅ 5 种格式 | TXT/MD（原生）、PDF（PyMuPDF）、DOCX（python-docx）、JSON（自定义） |
| 文本切片 | ChineseRecursiveTextSplitter | ✅ 13 级分隔符优先级 | 参考 Langchain-ChatChat 实现 |
| BM25 混合检索 | 后续可扩展项 | ✅ **已实现** | jieba 中文分词 + rank_bm25 + RRF 融合 |
| Cross-Encoder 重排序 | 后续可扩展项 | ✅ **已实现** | ms-marco-MiniLM-L-6-v2，优雅降级 |
| 前端管理界面 | 后续可扩展项 | ✅ **已实现** | 双栏布局侧边栏，拖拽上传，创建/删除/文档管理 |

---

## 3. 整体架构

```
┌──────────────────────────────────────────────────────────┐
│                    Flask Web UI                            │
│  GET /                    → SPA 前端（双栏布局）           │
│  GET /api/chat/stream     → SSE 对话流（已有）            │
│  POST /api/kb/create      → 创建知识库                    │
│  GET /api/kb/list         → 列出知识库                    │
│  DELETE /api/kb/<name>    → 删除知识库                    │
│  POST /api/kb/<name>/upload → 上传文档（拖拽/多文件）     │
│  GET /api/kb/<name>/docs  → 列出文档                      │
│  DELETE /api/kb/<name>/docs → 删除文档                    │
│  GET /api/health          → 健康检查                      │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│              RecipeAgent (recipe_agent.py)                 │
│  + search_recipe_knowledge_base(query, kb_name) [新工具]  │
│    → 搜索知识库，返回相关文档片段                          │
│    → 混合检索流水线：Dense + BM25 → RRF → CrossEncoder    │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│         KnowledgeBase 子系统 (handlers/knowledge_base/)   │
│  ├── __init__.py           → 模块入口                     │
│  ├── kb_manager.py         → KB CRUD 核心（事务性创建）   │
│  ├── document_loader.py    → 文档加载 + ChineseRecursiveTextSplitter │
│  ├── embedding.py          → Ollama 嵌入模型（单例）       │
│  ├── vector_store.py       → ChromaDB 操作封装            │
│  ├── search.py             → 混合检索 + RRF 融合 + 重排序 │
│  ├── bm25.py               → BM25 稀疏检索（jieba）       │
│  ├── reranker.py           → Cross-Encoder 重排序（单例）  │
│  └── db.py                 → SQLite 元数据库（3 表）       │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│                    存储层                                  │
│  data/knowledge_base/                                     │
│  ├── info.db            → SQLite 元数据库                  │
│  └── <kb_name>/                                          │
│      ├── content/       → 原始文档                        │
│      └── chroma/        → ChromaDB 持久化向量数据          │
│                                                           │
│  Ollama Server (localhost:11434)                          │
│  ├── qwen3-embedding:0.6b   → 嵌入模型（1024 维）         │
│  └── API: /api/embed, /api/tags                           │
└──────────────────────────────────────────────────────────┘
```

---

## 4. 核心组件设计

### 4.1 KnowledgeBaseManager（`kb_manager.py`）

```python
class KnowledgeBaseManager:
    def create_kb(name, description) -> KBInfo   # 事务性创建，向量库失败则回滚
    def delete_kb(name) -> None                   # 清理向量库 + DB + 文件
    def list_kbs() -> list[KBInfo]
    def get_kb(name) -> KBInfo | None

    def add_documents(kb_name, files, chunk_size=500, chunk_overlap=50) -> int
    def remove_documents(kb_name, filenames) -> None
    def list_documents(kb_name) -> list[DocInfo]
```

- 名称校验：regex `^[一-龥a-zA-Z0-9_\-]+$`
- 失败回滚：`create_kb` 中 ChromaDB 初始化异常时自动删除 DB 记录 + 文件目录
- 重复文件：再次上传自动替换旧切片

### 4.2 文档处理流水线（`document_loader.py`）

```
文件上传 → 格式检测 → 文本提取 → ChineseRecursiveTextSplitter → 嵌入 → ChromaDB
```

**支持的格式**：

| 格式 | 加载器 | 依赖 |
|------|--------|------|
| .txt | 原生 `open()` | 无 |
| .md | 原生 `open()` | 无 |
| .pdf | PyMuPDF (fitz) | `pymupdf` |
| .docx | python-docx（含表格文本） | `python-docx` |
| .json | 自定义（列表/字典/嵌套结构） | 无 |

**ChineseRecursiveTextSplitter**：
- 分隔符优先级：`"\n\n"` → `"\n"` → `"。！？"` → `".!?"` → `"；;"` → `"，,"` → `" "`
- 默认参数：`chunk_size=500`, `chunk_overlap=50`

### 4.3 嵌入模型（`embedding.py`）

- 模型：**`qwen3-embedding:0.6b`**（Ollama 本地部署）
- 维度：**1024**（运行时自动检测，首次请求后缓存）
- 接口：Ollama REST API — `POST /api/embed`，60s 超时
- 单例模式，惰性加载（首次调用时验证 Ollama 连通性）

### 4.4 向量存储（`vector_store.py`）

- ChromaDB PersistentClient（按 persist_dir 缓存）
- 每个 KB 一个 Collection：`kb_{kb_name}`
- 余弦距离（`hnsw:space: cosine`）
- `get_all_documents()` 支持 BM25 索引构建

### 4.5 搜索流水线（`search.py` → `bm25.py` → `reranker.py`）

```
用户查询
    │
    ├──→ 向量检索 (dense)    → 取 top_k × 4 候选
    ├──→ BM25 检索 (sparse)  → jieba 分词 + rank_bm25 [可选，KB_USE_BM25=True]
    │
    ├──→ RRF 融合 (k=60)     → 合并去重 [可选]
    │
    ├──→ Cross-Encoder 重排  → ms-marco-MiniLM-L-6-v2 精排 [可选，KB_USE_RERANKER=True]
    │
    └──→ 返回 top_k 结果 → format_search_results()
```

切换开关（`settings/constant.py`）：

```python
KB_USE_BM25 = True
KB_USE_RERANKER = True
KB_HYBRID_TOP_K_MULTIPLIER = 4
KB_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

### 4.6 SQLite 元数据库（`db.py`）

三张表：

```sql
knowledge_base  (id, name UNIQUE, description, embedding_model, doc_count, created_at)
knowledge_file  (id, kb_name FK, filename, file_ext, file_size, chunk_count, created_at)
file_chunk      (id, kb_name FK, filename FK, chunk_index, chunk_id)
```

---

## 5. LLM 工具集成

搜索工具（`tools.py`）：

```python
NATIVE_HANDLERS["search_recipe_knowledge_base"] = search_recipe_knowledge_base
```

工具定义包含中文描述，引导 LLM 在中文/家常菜场景下优先使用知识库。

与 TheMealDB 工具互补：中式/私人菜谱 → 知识库，西餐/鸡尾酒 → 外部 API。

---

## 6. Flask API 端点

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/kb/create` | 创建知识库 |
| GET | `/api/kb/list` | 列出知识库 |
| DELETE | `/api/kb/<name>` | 删除知识库 |
| POST | `/api/kb/<name>/upload` | 上传文档（multipart/form-data） |
| GET | `/api/kb/<name>/docs` | 列出文档 |
| DELETE | `/api/kb/<name>/docs` | 删除文档 |
| GET | `/api/health` | 健康检查（已有） |

---

## 7. 前端管理界面

双栏布局 SPA 页面：
- **左侧**：对话区 + 输入框
- **右侧**：知识库管理侧边栏（380px，<900px 时移到底部）
  - 创建知识库（名称 + 描述 + 按钮）
  - 知识库卡片（可展开，显示文档列表 + 上传区）
  - 拖拽上传（支持 TXT/MD/PDF/DOCX/JSON，最大 50MB）
  - 文档删除 + Toast 消息提示
  - 页面加载时自动显示

---

## 8. 依赖

```
chromadb==0.5.23
jieba==0.42.1
rank-bm25==0.2.2
sentence-transformers==3.4.1   # Cross-Encoder reranker
pymupdf==1.25.5
python-docx==1.1.2
```

---

## 9. 环境要求

| 组件 | 说明 |
|------|------|
| Ollama | 本地运行，需拉取 `qwen3-embedding:0.6b`：`ollama pull qwen3-embedding:0.6b` |
| Cross-Encoder（可选） | HF 网络可达时自动加载 `cross-encoder/ms-marco-MiniLM-L-6-v2`，不可达时优雅降级 |

---

## 10. 配置常量

```python
KB_ROOT_PATH = "data/knowledge_base/"
KB_DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
KB_EMBEDDING_DIMENSION = 1024  # 运行时自动检测
KB_OLLAMA_BASE_URL = "http://localhost:11434"
KB_DEFAULT_CHUNK_SIZE = 500
KB_DEFAULT_CHUNK_OVERLAP = 50
KB_DEFAULT_TOP_K = 3
KB_DEFAULT_SCORE_THRESHOLD = 0.3
KB_MAX_FILE_SIZE_MB = 50
KB_USE_BM25 = True
KB_USE_RERANKER = True
KB_HYBRID_TOP_K_MULTIPLIER = 4
KB_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

---

## 11. 测试覆盖

| 测试类 | 数量 | 覆盖内容 |
|--------|------|----------|
| TestDB | 9 | SQLite CRUD 全操作 |
| TestEmbeddingModel | 5 | 单例、惰性加载、批量嵌入、查询嵌入、空列表（需 Ollama 运行） |
| TestChineseRecursiveTextSplitter | 8 | 中英文分隔符、重叠、硬分割、菜谱文本 |
| TestDocumentLoader | 6 | TXT/MD/JSON 加载、格式拒绝、文件缺失 |
| TestVectorStore | 6 | 增/查/删/过滤/清空 |
| TestFormatSearchResults | 3 | 空结果、单结果、多结果 |
| TestKnowledgeBaseManager | 10 | KB 全生命周期 + 文档增删 + 端到端工作流 |
| TestFlaskKBAPI | 12 | 6 个端点 × 正常/异常场景 |
| TestBM25 | 5 | 分词、索引构建、空索引、搜索、空查询 |
| TestReranker | 4 | 单例、降级、空列表、available 属性 |
| TestRRFFusion | 2 | 稠密+稀疏融合、BM25 贡献新结果 |
| **合计** | **70** | 全链路覆盖 |
