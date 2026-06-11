# 🍳 智能食谱助手 (Smart Recipe Assistant)

基于 **OpenRouter API** + **Function Calling** 的智能食谱与饮品推荐助手。支持通过终端交互或 Web 界面搜索菜谱、查询食材、推荐美食与鸡尾酒。

---

## 快速启动

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 环境变量

```bash
export OPENROUTER_API_KEY="sk-or-v1-xxx"   # OpenRouter API Key
export MODEL_ID="openai/gpt-4o"            # 模型标识（支持 Claude/GPT/Gemini 等）
export FLASK_DEBUG="1"                     # （可选）Flask 调试模式
export PORT="5000"                         # （可选）Web 端口，默认 5000
```

### 3. 启动

**终端交互模式：**

```bash
python recipe_agent.py
```

**Web 服务模式：**

```bash
python app.py
```

打开浏览器访问 `http://localhost:5000` 即可。

---

## 访问方式

| 方式 | 命令 | 说明 |
|------|------|------|
| Web UI | `python app.py` → `http://localhost:5000` | 可视化聊天界面，支持菜谱卡片展示 |
| 终端 CLI | `python recipe_agent.py` | 交互式终端 TUI，带颜色高亮 |
| API | `GET /api/chat/stream?message=xxx` | SSE 流式 API，支持 EventSource 连接 |
| 健康检查 | `GET /api/health` | 返回 `{"status": "ok"}` |

---

## 代码结构

```
smart-recipe-assistant/
├── app.py                          # Flask Web 入口 — SSE 端点 + 前端页面渲染
├── recipe_agent.py                 # Agent 核心 — OpenRouter 调用 + Function Calling 循环 + 终端 Visualizer
├── tools.py                        # 工具定义 — TheMealDB / TheCocktailDB 接口 + 文件/Shell 工具
├── log.py                          # 结构化日志配置（JSON/文本，RotatingFileHandler）
├── requirements.txt                # 依赖
│
├── handlers/
│   ├── system_prompt.py            # System Prompt 构建 — 核心指令 + 记忆注入 + CLAUDE.md
│   ├── compact_context.py          # 上下文压缩 — 自动摘要 + 转录存档 + CompactState
│   ├── error_recovery.py           # 错误恢复 — 指数退避 + 溢出自动压缩
│   ├── memory_system.py            # 持久化记忆系统
│   ├── tasks_system.py             # 任务管理
│   ├── todomanager.py              # TODO 管理
│   └── hook_system.py              # Hook 系统（PreToolUse / PostToolUse）
│
├── settings/
│   ├── __init__.py
│   └── constant.py                 # 全局常量 — API Key、模型、路径、限值阈值
│
├── services/
│   ├── __init__.py
│   └── agent_service.py            # Agent 服务层 — 工具池组装 + MCP 路由 + 权限门禁
│
├── templates/
│   └── index.html                  # 前端单页应用 — 深色主题 + 消息流 + 菜谱卡片栅格
│
├── skills/                         # 技能目录
└── logs/                           # 日志输出目录
```

---

## 技术架构

### 整体流程

```
用户输入
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  RecipeAgent (核心循环)                               │
│                                                      │
│  1. 构建 System Prompt (含记忆 + 工具列表 + 动态上下文) │
│  2. 调用 OpenRouter API → LLM 返回推理文本 + tool_call │
│  3. 分发工具执行 → NATIVE_HANDLERS                     │
│  4. 工具结果回填 → 继续下一轮 LLM 调用                  │
│  5. 无 tool_call → 输出最终回答                        │
└──────────────────────────────────────────────────────┘
    │
    ├─ 终端模式 → Visualizer (ANSI 彩色终端输出)
    └─ Web 模式 → SSE 事件流 (thinking / tool_call / tool_result / final)
                       │
                       ▼
                  Flask → 前端 EventSource → 渲染消息 + 菜谱卡片
```

### 核心组件

- **OpenRouterClient** — 对 OpenRouter Chat Completions API 的封装，支持重试、超时、上下文超限检测与恢复
- **RecipeAgent** — 多轮 Function Calling 循环引擎，自动管理上下文窗口，支持自动/手动压缩
- **Visualizer** — 终端可视化层，以彩色框图展示"思考→调用→结果"链路
- **SystemPromptBuilder** — 动态组装系统提示词，注入持久化记忆、工具清单、CLAUDE.md 指令、运行时上下文
- **Web SSE 架构** — 前端通过 `EventSource` 消费 4 种事件类型，实现流式渲染 + 菜谱卡片

### 工具层 (Function Calling)

| 类别 | 工具 | 数据源 |
|------|------|--------|
| 菜谱搜索 | `search_meals`, `filter_by_ingredient`, `filter_by_category`, `filter_by_area` | TheMealDB |
| 菜谱详情 | `get_meal_detail`, `random_meal`, `get_categories` | TheMealDB |
| 饮品搜索 | `search_cocktails`, `filter_cocktails_by_ingredient` | TheCocktailDB |
| 饮品详情 | `get_cocktail_detail`, `random_cocktail` | TheCocktailDB |
| 文件操作 | `read_file`, `write_file`, `edit_file` | 本地文件系统 |
| Shell | `bash` | 本地 Shell（沙箱限制） |
| 记忆系统 | `save_memory`, `todo`, `task_create/update/list/get` | 本地持久化 |

### 关键设计

- **上下文管理**：超出 `CONTEXT_LIMIT` 自动压缩，通过 LLM 摘要历史并存储完整转录到 `.transcripts/`
- **错误恢复**：指数退避重试、上下文超限自动截断、最多 3 次恢复尝试
- **记忆系统**：用户偏好、反馈、项目信息持久化到 `.memory/`，下次会话自动加载
- **双模式**：同一 Agent 核心同时驱动终端 TUI 和 Web SSE，`run()` 用于终端，`run_stream()` 用于 Web
