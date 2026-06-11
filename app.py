"""
Flask Web 服务 —— 智能食谱助手
提供 SSE Chat API 并渲染前端页面。
"""

import json
import logging
import os

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from handlers.compact_context import CompactState
from recipe_agent import (
    OpenRouterClient,
    RecipeAgent,
)
from settings.constant import MODEL
from tools import NATIVE_TOOLS
from log import logging

app = Flask(__name__)

# ── 全局 Agent 实例 ──────────────────────────────────────────────────

_api_key = os.environ.get("OPENROUTER_API_KEY", "")
_model = MODEL
_client = OpenRouterClient(api_key=_api_key, model=_model)
_tools = NATIVE_TOOLS
_agent = RecipeAgent(_client, _tools)

logging.info("Agent 初始化完成, model=%s, tools=%d", _model, len(_tools))


# ── 路由 ─────────────────────────────────────────────────────────────


@app.route("/")
def index():
    """渲染主页面。"""
    return render_template("index.html")


@app.route("/api/chat/stream", methods=["GET"])
def chat_stream():
    """SSE 端点：流式返回 Agent 思考过程与最终结果。

    使用 EventSource 连接，事件类型：
      - thinking   → 模型推理文本
      - tool_call  → 工具调用信息
      - tool_result→ 工具执行结果
      - final      → 最终回答 + 结构化数据
    """
    message = request.args.get("message", "").strip()
    if not message:
        return jsonify({"error": "消息不能为空"}), 400

    logging.info("SSE 请求: %s", message[:80])
    compact_state = CompactState()
    def event_stream():
        try:
            for event in _agent.run_stream(message, compact_state):
                yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logging.exception("SSE 流处理出错")
            yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/health", methods=["GET"])
def health():
    """健康检查。"""
    return jsonify({"status": "ok"})


# ── 工具 ─────────────────────────────────────────────────────────────


def _deduplicate(items: list[dict], key: str) -> list[dict]:
    """根据 id 字段去重。"""
    seen = set()
    result = []
    for item in items:
        uid = item.get(key)
        if uid and uid not in seen:
            seen.add(uid)
            result.append(item)
    return result


# ── 启动 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logging.info("启动 Web 服务, 端口: %d, debug: %s", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
