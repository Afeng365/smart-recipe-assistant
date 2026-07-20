# 🍳 智能食谱助手 (Smart Recipe Assistant)

基于 **OpenRouter API** + **Ollama 本地嵌入** + **Function Calling** 的智能食谱与饮品推荐助手，内置 **RAG 知识库管理** 系统。

- 🔍 搜索全球菜谱与鸡尾酒（TheMealDB / TheCocktailDB）
- 📚 上传个人菜谱文档构建本地知识库（PDF / DOCX / TXT / MD / JSON）
- 🧠 混合检索（Dense 向量 + BM25 稀疏 + Cross-Encoder 重排序）
- 💬 终端 TUI + Web SPA 双界面
- 🎨 暗色主题 Web UI，双栏布局，拖拽上传

---

## 快速启动

### 1. 环境准备

```bash
# Python ≥ 3.10
pip install -r requirements.txt
```

### 2. 启动 Ollama（知识库嵌入必需）

```bash
# 安装 Ollama 并拉取嵌入模型
ollama pull qwen3-embedding:0.6b
ollama serve    # 默认监听 http://localhost:11434
```

### 3. 环境变量

```bash
export OPENROUTER_API_KEY="sk-or-v1-xxx"   # OpenRouter API Key（必填）
export MODEL_ID="anthropic/claude-sonnet-4-6"  # 模型标识（支持 Claude/GPT/Gemini）
export OLLAMA_BASE_URL="http://localhost:11434" # （可选）Ollama 地址
export FLASK_DEBUG="1"                          # （可选）Flask 调试模式
export PORT="5000"                              # （可选）Web 端口，默认 5000
```

### 4. 启动

**Web 服务模式（推荐）：**

```bash
python app.py
```

打开浏览器访问 `http://localhost:5000`。

**终端交互模式：**

```bash
python recipe_agent.py
```

---

## 访问方式

| 方式 | 命令 | 说明 |
|------|------|------|
| Web UI | `python app.py` → `http://localhost:5000` | 双栏布局：对话区 + 知识库管理侧边栏 |
| 终端 CLI | `python recipe_agent.py` | 交互式终端 TUI，带颜色高亮 |
| SSE API | `GET /api/chat/stream?message=xxx` | 流式 API，支持 EventSource |
| KB API | `POST /api/kb/create` 等 6 个端点 | 知识库管理 REST API |
| 健康检查 | `GET /api/health` | 返回 `{"status": "ok"}` |

---

## 核心功能

### 🍽️ 智能食谱推荐

| 功能 | 描述 |
|------|------|
| 菜谱搜索 | 按名称、食材、分类、地区搜索全球菜谱 |
| 饮品搭配 | 主菜 + 配饮的合理组合（鸡尾酒/饮品） |
| 个性化记忆 | 持久化用户饮食偏好、忌口、口味、厨艺水平 |
| 上下文对话 | 多轮对话，自动管理上下文窗口（压缩/存档） |

### 📚 RAG 知识库（本地菜谱）

| 功能 | 描述 |
|------|------|
| 知识库管理 | 创建/列表/删除知识库，Web 侧边栏操作 |
| 文档上传 | 支持 TXT、MD、PDF、DOCX、JSON，拖拽或点击上传 |
| 中文切片 | 13 级分隔符优先级的 ChineseRecursiveTextSplitter |
| 嵌入向量 | Ollama `qwen3-embedding:0.6b`（1024 维），本地运行 |
| 向量存储 | ChromaDB 嵌入式模式，余弦距离，按 KB 隔离 |
| BM25 混合检索 | jieba 中文分词 + RRF 融合 | 
| Cross-Encoder 重排序 | `ms-marco-MiniLM-L-6-v2` 精排，优雅降级 |

### 🧠 LLM 工具调用

共 22 个 LLM 可调用工具，LLM 根据用户意图自动选择：

| 类别 | 工具 | 数据源 |
|------|------|--------|
| 菜谱搜索 | `search_meals`, `filter_by_ingredient`, `filter_by_category`, `filter_by_area` | TheMealDB |
| 菜谱详情 | `get_meal_detail`, `random_meal`, `get_categories` | TheMealDB |
| 饮品搜索 | `search_cocktails`, `filter_cocktails_by_ingredient` | TheCocktailDB |
| 饮品详情 | `get_cocktail_detail`, `random_cocktail` | TheCocktailDB |
| **知识库** | **`search_recipe_knowledge_base`** | **本地 ChromaDB + BM25** |
| 文件操作 | `read_file`, `write_file`, `edit_file` | 本地文件系统 |
| Shell | `bash` | 本地 Shell（沙箱限制） |
| 记忆系统 | `save_memory`, `todo`, `task_create/update/list/get` | 本地持久化 |

---

## 搜索流水线

```
用户查询
    │
    ├──→ 向量检索 (dense)    → top_k × 4 候选
    ├──→ BM25 检索 (sparse)  → jieba 分词 + rank_bm25       [可选]
    │
    ├──→ RRF 融合 (k=60)     → 合并去重                      [可选]
    │
    ├──→ Cross-Encoder 重排  → ms-marco-MiniLM-L-6-v2 精排  [可选]
    │
    └──→ 返回 top_k 结果 → format_search_results()
```

配置控制（`settings/constant.py`）：

```python
KB_USE_BM25 = True          # 启用 BM25 混合检索
KB_USE_RERANKER = True      # 启用 Cross-Encoder 重排序
```

---

## 目录结构

```
smart-recipe-assistant/
├── app.py                          # Flask Web 入口（SSE + KB API）
├── recipe_agent.py                 # Agent 核心（OpenRouter + Function Calling）
├── tools.py                        # 22 个 LLM 工具 + NATIVE_HANDLERS
├── log.py                          # 结构化日志
├── requirements.txt                # 依赖
│
├── handlers/
│   ├── knowledge_base/             # RAG 知识库子系统
│   │   ├── __init__.py             #   导出入口
│   │   ├── kb_manager.py           #   KB CRUD 管理器（事务性创建/删除）
│   │   ├── document_loader.py      #   文档加载（5 种格式）+ ChineseRecursiveTextSplitter
│   │   ├── embedding.py            #   Ollama 嵌入模型单例（qwen3-embedding:0.6b, 1024d）
│   │   ├── vector_store.py         #   ChromaDB 向量存储封装
│   │   ├── search.py               #   混合检索流水线（Dense + BM25 + RRF + Rerank）
│   │   ├── bm25.py                 #   BM25 稀疏检索（jieba 中文分词）
│   │   ├── reranker.py             #   Cross-Encoder 重排序单例
│   │   └── db.py                   #   SQLite 元数据库（3 表）
│   ├── system_prompt.py            # System Prompt 构建
│   ├── compact_context.py          # 上下文压缩
│   ├── error_recovery.py           # 错误恢复
│   ├── memory_system.py            # 持久化记忆
│   ├── tasks_system.py             # 任务管理
│   ├── todomanager.py              # TODO 管理
│   └── hook_system.py              # Hook 系统
│
├── settings/
│   └── constant.py                 # 全局配置常量
│
├── services/
│   └── agent_service.py            # Agent 服务层
│
├── templates/
│   └── index.html                  # 前端 SPA（双栏布局，暗色主题）
│
├── tests/
│   └── test_knowledge_base.py      # 70 个测试，14 个测试类
│
├── data/knowledge_base/            # 知识库存储（运行时创建）
│   ├── info.db                     #   SQLite 元数据
│   └── <kb_name>/                  #   各知识库：content/ + chroma/
│
├── .memory/                        # 用户偏好记忆
├── .transcripts/                   # 对话转录存档
├── .tasks/                         # 任务 JSON 文件
└── logs/                           # 日志输出
```

---

## 技术架构

### 整体流程

```
用户输入
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  RecipeAgent (核心循环)                                   │
│                                                          │
│  1. 构建 System Prompt (含记忆 + 工具列表 + 动态上下文)     │
│  2. 调用 OpenRouter API → LLM 返回推理文本 + tool_calls   │
│  3. 分发工具执行 → NATIVE_HANDLERS                        │
│     ├── TheMealDB/TheCocktailDB 工具 → 外部 API 调用      │
│     └── search_recipe_knowledge_base → 本地混合检索       │
│  4. 工具结果回填 → 继续下一轮 LLM 调用                     │
│  5. 无 tool_call → 输出最终回答                           │
└──────────────────────────────────────────────────────────┘
    │
    ├─ 终端模式 → Visualizer (ANSI 彩色终端输出)
    └─ Web 模式 → SSE 事件流 (thinking / tool_call / tool_result / final)
                       │
                       ▼
                  Flask → 前端 EventSource → 渲染消息 + 菜谱卡片
```

### 关键设计

- **上下文管理**：超出 80000 字符自动压缩，LLM 摘要历史并存档到 `.transcripts/`
- **错误恢复**：指数退避重试，上下文超限自动截断，最多 3 次恢复尝试
- **记忆系统**：用户偏好持久化到 `.memory/*.md`，每次会话自动注入 System Prompt
- **双模式**：同一 Agent 驱动终端 TUI + Web SSE
- **知识库事务性**：KB 创建失败自动回滚 DB 记录和文件目录
- **优雅降级**：Cross-Encoder 模型不可达时自动跳过重排序，不影响搜索功能

### 技术栈

| 层级 | 技术 |
|------|------|
| LLM 网关 | OpenRouter API（OpenAI Function Calling 格式） |
| Web 框架 | Flask 3.x + Jinja2 |
| 前端 | 原生 HTML/CSS/JS（无框架，EventSource SSE 消费） |
| 向量数据库 | ChromaDB 0.5.23 |
| 嵌入模型 | Ollama `qwen3-embedding:0.6b`（1024 维） |
| 重排序 | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| BM25 | rank-bm25 + jieba 中文分词 |
| 文档解析 | PyMuPDF (PDF) + python-docx (DOCX) |
| 元数据 | SQLite (3 张表，外键约束，级联删除) |
| 测试 | pytest (70 个测试) |

---

## 知识库 API

| 方法 | 路径 | 请求体 | 说明 |
|------|------|--------|------|
| POST | `/api/kb/create` | `{"name": "川菜大全", "description": "..."}` | 创建知识库 |
| GET | `/api/kb/list` | — | 列出所有知识库 |
| DELETE | `/api/kb/<name>` | — | 删除知识库（含所有文档） |
| POST | `/api/kb/<name>/upload` | `multipart/form-data` files 字段 | 上传文档（多文件，拖拽支持） |
| GET | `/api/kb/<name>/docs` | — | 列出知识库文档 |
| DELETE | `/api/kb/<name>/docs` | `{"filenames": ["a.txt", "b.pdf"]}` | 删除文档 |

---

## 测试

```bash
# 全部测试（需 Ollama 运行）
python -m pytest tests/test_knowledge_base.py -v

# 跳过嵌入模型测试（无需 Ollama）
python -m pytest tests/test_knowledge_base.py -v -k "not Embedding and not Reranker"
```

14 个测试类，70 个测试用例：SQLite CRUD、文档加载、中文切片、向量存储、BM25、RRF 融合、重排序、KB 管理器、Flask API 端点。
