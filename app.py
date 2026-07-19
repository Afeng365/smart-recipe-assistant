"""
Flask Web 服务 —— 智能食谱助手
提供 SSE Chat API 并渲染前端页面。
"""

import json
import logging
import os
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from handlers.compact_context import CompactState
from recipe_agent import (
    OpenRouterClient,
    RecipeAgent,
)
from settings.constant import MODEL, KB_ROOT_PATH, KB_MAX_FILE_SIZE_BYTES, KB_SUPPORTED_EXTENSIONS
from tools import NATIVE_TOOLS
from log import logging
from handlers.knowledge_base import kb_manager

app = Flask(__name__)

# ── 全局 Agent 实例 ──────────────────────────────────────────────────

_api_key = os.environ.get("OPENROUTER_API_KEY", "")
_model = MODEL
_client = OpenRouterClient(api_key=_api_key, model=_model)
_tools = NATIVE_TOOLS
_agent = RecipeAgent(_client, _tools)

# ── 初始化知识库 ──────────────────────────────────────────────────────
KB_ROOT_PATH.mkdir(parents=True, exist_ok=True)

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


# ── 知识库管理 API ─────────────────────────────────────────────────────


@app.route("/api/kb/create", methods=["POST"])
def kb_create():
    """创建知识库。请求体: {"name": "...", "description": "..."}"""
    try:
        data = request.get_json(force=True)
        if not data or "name" not in data:
            return jsonify({"error": "缺少知识库名称"}), 400
        info = kb_manager.create_kb(data["name"], data.get("description", ""))
        return jsonify({"status": "ok", "kb": info})
    except ValueError as e:
        return jsonify({"error": str(e)}), 409 if "已存在" in str(e) else 400
    except Exception as e:
        logging.exception("创建知识库失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/kb/list", methods=["GET"])
def kb_list():
    """列出所有知识库。"""
    try:
        kbs = kb_manager.list_kbs()
        return jsonify({"status": "ok", "knowledge_bases": kbs})
    except Exception as e:
        logging.exception("列出知识库失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/kb/<name>", methods=["DELETE"])
def kb_delete(name):
    """删除知识库。"""
    try:
        kb_manager.delete_kb(name)
        return jsonify({"status": "ok", "message": f"知识库 {name} 已删除"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logging.exception("删除知识库失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/kb/<name>/upload", methods=["POST"])
def kb_upload(name):
    """上传文档到知识库。multipart/form-data, files 字段。"""
    try:
        if "files" not in request.files:
            return jsonify({"error": "缺少 files 字段"}), 400

        files = request.files.getlist("files")
        if not files or all(f.filename == "" for f in files):
            return jsonify({"error": "未选择文件"}), 400

        # Validate file sizes and extensions
        for f in files:
            if f.filename == "":
                continue
            ext = Path(f.filename).suffix.lower()
            if ext not in KB_SUPPORTED_EXTENSIONS:
                return jsonify({
                    "error": f"不支持的文件格式: {ext}。支持: {', '.join(KB_SUPPORTED_EXTENSIONS)}"
                }), 400
            # Read file content to check size
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(0)
            if size > KB_MAX_FILE_SIZE_BYTES:
                return jsonify({
                    "error": f"文件 {f.filename} 过大 ({size / 1024 / 1024:.1f}MB)，"
                             f"最大 {KB_MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f}MB"
                }), 413

        # Save files to temp location
        import tempfile
        saved_paths = []
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            for f in files:
                if f.filename == "":
                    continue
                dest = tmp_path / f.filename
                f.save(str(dest))
                saved_paths.append(dest)

            # Process documents
            import time
            t0 = time.time()
            chunk_count = kb_manager.add_documents(name, saved_paths)
            duration = time.time() - t0

        return jsonify({
            "status": "ok",
            "message": f"成功处理 {len(saved_paths)} 个文件，生成 {chunk_count} 个切片",
            "file_count": len(saved_paths),
            "chunk_count": chunk_count,
            "duration_sec": round(duration, 2),
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404 if "不存在" in str(e) else 400
    except Exception as e:
        logging.exception("上传文档失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/kb/<name>/docs", methods=["GET"])
def kb_list_docs(name):
    """列出知识库中的所有文档。"""
    try:
        docs = kb_manager.list_documents(name)
        return jsonify({"status": "ok", "documents": docs})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logging.exception("列出文档失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/kb/<name>/docs", methods=["DELETE"])
def kb_delete_docs(name):
    """删除知识库中的文档。请求体: {"filenames": ["a.pdf", "b.txt"]}"""
    try:
        data = request.get_json(force=True)
        if not data or "filenames" not in data:
            return jsonify({"error": "缺少 filenames 字段"}), 400
        kb_manager.remove_documents(name, data["filenames"])
        return jsonify({"status": "ok", "message": f"已删除 {len(data['filenames'])} 个文档"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logging.exception("删除文档失败")
        return jsonify({"error": str(e)}), 500


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
    logging.info("启动 Web 服务, 端口: %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
