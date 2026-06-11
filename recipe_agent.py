"""
基于 OpenRouter API 的智能食谱助手 Agent
- 支持任意 LLM（Claude、GPT、Gemini 等）
- 完整的 Function Calling 循环
- 生产级：日志、重试、超时、错误处理
"""

import json
import logging
import os
import textwrap
import time
import requests
from typing import Any

from handlers.compact_context import estimate_context_size, compact_history, CompactState
from handlers.error_recovery import auto_compact, backoff_delay
from handlers.system_prompt import SystemPromptBuilder
from handlers.todomanager import TODO
from settings.constant import CONTEXT_LIMIT, WORKDIR, MAX_RECOVERY_ATTEMPTS, OPENROUTER_API_KEY, MODEL
from tools import NATIVE_HANDLERS, NATIVE_TOOLS
from log import logging


prompt_builder = SystemPromptBuilder(workdir=WORKDIR, tools=NATIVE_TOOLS)


# ── 思考过程可视化 ────────────────────────────────────────────────────


class Visualizer:
    """终端可视化 Agent 思考过程（Tool 调用链路）。"""

    # ANSI 颜色
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    GRAY = "\033[90m"

    @staticmethod
    def _icon(label: str) -> str:
        icons = {
            "思考": "🧠",
            "调用": "🔧",
            "结果": "✅",
            "完成": "🎯",
            "错误": "❌",
            "提问": "💬",
            "重试": "🔄",
        }
        return icons.get(label, "•")

    @classmethod
    def user_query(cls, query: str):
        """显示用户提问。"""
        cls._print_box("提问", query, cls.CYAN, "💬")

    @classmethod
    def thinking(cls, content: str | None):
        """显示模型的思考过程（若有）。"""
        if not content or not content.strip():
            return
        cls._print_box("思考", content.strip(), cls.YELLOW, "🧠")

    @classmethod
    def tool_call(cls, tool_name: str, arguments: dict, round_num: int, idx: int = 0):
        """显示工具调用信息。"""
        args_str = json.dumps(arguments, ensure_ascii=False, indent=2)
        tag = f"#{round_num}-{idx}"
        header = f" 调用工具: {cls.BOLD}{tool_name}{cls.RESET} {cls.GRAY}({tag}){cls.RESET}"
        body = f"\n{cls.GRAY}参数:{cls.RESET}\n{args_str}"
        cls._print_raw(cls.GREEN, "🔧", header + body)
        print()

    @classmethod
    def tool_result(cls, tool_name: str, result: Any, duration: float):
        """显示工具调用结果摘要。"""
        summary = cls._summarize(result)
        cls._print_raw(
            cls.BLUE,
            "📦",
            f" {cls.DIM}{tool_name}{cls.RESET} → {summary} "
            f"{cls.GRAY}({duration:.2f}s){cls.RESET}",
        )
        print()

    @classmethod
    def tool_error(cls, tool_name: str, error: str):
        """显示工具调用错误。"""
        cls._print_raw(cls.RED, "❌", f" {tool_name} 执行失败: {error}")
        print()

    @classmethod
    def final_answer(cls, content: str):
        """显示模型的最终回答。"""
        print()
        cls._print_box("最终回答", content, cls.MAGENTA, "🎯", prefix="")

    @classmethod
    def _print_box(cls, label: str, text: str, color: str, emoji: str, prefix: str = ""):
        """打印带边框的信息块。"""
        icon = emoji or cls._icon(label)
        lines = text.split("\n")

        # 上边框
        print(f"{prefix}{color}┌─ {icon} {label} ", end="")
        print(f"{'─' * max(2, 56 - len(label) - 4)}┐{cls.RESET}")

        # 内容
        for line in lines:
            for wrapped in textwrap.wrap(line, width=60, drop_whitespace=False) or [""]:
                print(f"{prefix}{color}│ {cls.RESET}{wrapped}")

        # 下边框
        print(f"{prefix}{color}└{'─' * 58}┘{cls.RESET}")

    @classmethod
    def _print_raw(cls, color: str, emoji: str, text: str):
        """打印无边框行（用于工具调用链路）。"""
        print(f"{color}{emoji}{cls.RESET}{text}")

    @staticmethod
    def _summarize(result: Any) -> str:
        """对工具返回结果生成一行摘要。"""
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                return f"string ({len(result)} chars)"

        if isinstance(result, dict):
            if "error" in result:
                return f"⚠️ {result['error']}"
            return f"dict ({len(result)} keys)"

        if isinstance(result, list):
            return f"list ({len(result)} items)"

        if result is None:
            return "None"

        return str(result)[:40]


class ContextLengthExceededError(RuntimeError):
    """当 API 返回上下文长度超出限制时的自定义异常。"""
    def __init__(self, message: str, max_tokens_hint: int | None = None):
        super().__init__(message)
        self.max_tokens_hint = max_tokens_hint


# ── OpenRouter API 调用 ──────────────────────────────────────────────


class OpenRouterClient:
    """OpenRouter API 的轻量封装，支持 tool calling。"""

    def __init__(
        self,
        api_key: str = "",
        model: str = MODEL,
        timeout: int = 60,
        max_retries: int = MAX_RECOVERY_ATTEMPTS,
    ):
        self.api_key = api_key or OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError(
                "缺少 OpenRouter API Key，请设置环境变量 OPENROUTER_API_KEY "
                "或通过参数传入。"
            )
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    @staticmethod
    def _check_context_length_error(status: int, body: str) -> ContextLengthExceededError | None:
        """检查响应是否为上下文长度超限错误，若是则返回 ContextLengthExceededError。"""
        if status not in (400, 413, 429):
            return None

        body_lower = body.lower()
        # 常见关键词匹配
        context_keywords = [
            "context_length_exceeded",
            "context length",
            "maximum context length",
            "max tokens",
            "max_tokens",
            "token limit",
            "too many tokens",
            "prompt too long",
            "request too large",
            "content_length_limit",
        ]
        if not any(kw in body_lower for kw in context_keywords):
            return None

        # 尝试提取 max_tokens hint
        max_tokens_hint = None
        import re
        for pattern in [r"maximum context length is (\d+)", r"max.*?tokens?.*?(\d+)",
                        r"(\d+).*?tokens?.*?limit", r"limit.*?(\d+).*?tokens?"]:
            m = re.search(pattern, body_lower)
            if m:
                max_tokens_hint = int(m.group(1))
                break

        return ContextLengthExceededError(f"上下文/Token 长度超限 (HTTP {status})", max_tokens_hint)

    def _call(self, payload: dict) -> dict:
        """带重试的 API 调用。"""
        url = "https://openrouter.ai/api/v1/chat/completions"
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.ConnectTimeout as e:
                last_error = e
                logging.warning("请求超时 (attempt %d/%d)", attempt, self.max_retries)
                # Strategy 3: connection/rate errors -> backoff
                if attempt < self.max_retries:
                    delay = backoff_delay(attempt)
                    logging.info(f"[Recovery] API error: {e}. "
                                 f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                    continue
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code
                body = e.response.text[:500]

                # 检查是否上下文长度超限
                ctx_err = self._check_context_length_error(status, body)
                if ctx_err:
                    raise ctx_err from e

                # 4xx 不重试（除 429 限流外）
                if 400 <= status < 500 and status != 429:
                    raise RuntimeError(f"API 返回 {status}: {body}") from e
                logging.warning("HTTP %d (attempt %d/%d)", status, attempt, self.max_retries)
                last_error = e
            except requests.exceptions.JSONDecodeError as e:
                last_error = e
                logging.warning("JSON 解析失败 (attempt %d/%d): %s", attempt, self.max_retries, e)
            except requests.exceptions.RequestException as e:
                last_error = e
                logging.error(f"llm request payload: {payload}; --error: {e}")
                logging.warning("请求异常 (attempt %d/%d): %s", attempt, self.max_retries, e)

            if attempt < self.max_retries:
                time.sleep(1.5 ** attempt)  # 指数退避

        raise RuntimeError(f"API 请求失败，已重试 {self.max_retries} 次") from last_error

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 8000,
    ) -> dict:
        """发送聊天请求。"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            # OpenRouter 同时兼容 OpenAI 与 Anthropic 两种 tool 格式
            # 这里转换为 OpenAI 格式（更通用）
            payload["tools"] = [
                {"type": "function", "function": t} for t in tools
            ]
            payload["tool_choice"] = "auto"

        return self._call(payload)


# ── Agent 核心 ────────────────────────────────────────────────────────


class RecipeAgent:
    """智能食谱助手 Agent —— 通过 Function Calling 调用 API 回答用户问题。"""

    def __init__(self, client: OpenRouterClient, tools: list[dict]):
        self.client = client
        self.tools = tools

    def run(self, user_message: str, state: CompactState, max_rounds: int = 20) -> str:
        """运行 Agent：多轮 Function Calling，返回文本回答（终端用）。"""
        result = self._run_loop(user_message, max_rounds, visualize=True, state=state)
        return result["answer"]

    def run_stream(self, user_message: str, state: CompactState, max_rounds: int = 50):
        """Generator：以 SSE 事件流形式产出 Agent 思考过程与最终结果。

        每次 yield 的 dict 包含 {"type": str, ...}:
          - type=thinking:  { "content": "模型推理文本" }
          - type=tool_call: { "name": func, "arguments": {}, "round": N, "idx": N }
          - type=tool_result: { "name": func, "duration": float,
                "summary": "str", "items_count": int }
          - type=final:   { "answer": str, "data": {"meals":[],"drinks":[]} }
        """
        system_prompt = prompt_builder.build()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        collected: dict[str, list] = {"meals": [], "drinks": []}

        context_truncated = False

        for _round in range(1, max_rounds + 1):
            if estimate_context_size(messages) > CONTEXT_LIMIT:
                print("[auto compact]")
                messages[:] = compact_history(messages, state)
            try:
                response = self.client.chat(messages, tools=self.tools)
            except ContextLengthExceededError as e:
                if context_truncated:
                    yield {"type": "final", "answer": "对话历史过长，请重新开始一个新的查询。", "data": collected}
                    return
                yield {"type": "thinking", "content": f"⚠️ 上下文超限 ({e})，正在压缩历史消息…"}
                messages[:] = auto_compact(messages)
                context_truncated = True
                response = self.client.chat(messages, tools=self.tools)

            context_truncated = False
            choice = response["choices"][0]
            msg = choice["message"]

            # 1) 推理过程
            if msg.get("content"):
                yield {"type": "thinking", "content": msg["content"]}

            # 2) 无工具调用 → 最终回答
            if not msg.get("tool_calls"):
                yield {"type": "final", "answer": msg["content"], "data": collected}
                return

            messages.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": msg["tool_calls"],
            })

            used_todo = False
            manual_compact = False
            # 3) 依次执行每个工具并 yield 事件
            for idx, tc in enumerate(msg["tool_calls"]):
                func_name = tc["function"]["name"]
                arguments = json.loads(tc["function"]["arguments"])

                yield {
                    "type": "tool_call",
                    "name": func_name,
                    "arguments": arguments,
                    "round": _round,
                    "idx": idx,
                }

                t0 = time.time()
                try:
                    tool_result = self._execute_tool(tc)
                    duration = time.time() - t0
                    result_data = json.loads(tool_result)
                    self._collect_data(collected, func_name, result_data)
                    if func_name == "todo":
                        used_todo = True
                    if func_name == "compact":
                        manual_compact = True
                    yield {
                        "type": "tool_result",
                        "name": func_name,
                        "duration": round(duration, 2),
                        "summary": self._summarize_for_sse(result_data),
                        "items_count": len(result_data) if isinstance(result_data, list) else (1 if isinstance(result_data, dict) and result_data.get("idMeal") else 0),
                    }
                except Exception as e:
                    logging.info(f"执行工具失败: {func_name}, 参数: {arguments}; error: {e}")
                    duration = time.time() - t0
                    yield {
                        "type": "tool_result",
                        "name": func_name,
                        "duration": round(duration, 2),
                        "summary": f"❌ {str(e)[:60]}",
                        "items_count": 0,
                    }
                    tool_result = json.dumps({"error": str(e)})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })

            if used_todo:
                TODO.state.rounds_since_update = 0
            else:
                TODO.note_round_without_update()
                reminder = TODO.reminder()
                # if reminder:
                #     # results.insert(0, {"type": "text", "text": reminder})
                #     results.append({"type": "text", "text": reminder})
            if manual_compact:
                print("[manual compact]")
                messages[:] = compact_history(messages, state)
        # 达到最大轮数
        try:
            response = self.client.chat(messages)
        except ContextLengthExceededError:
            messages[:] = auto_compact(messages)
            response = self.client.chat(messages)
        final = response["choices"][0]["message"]["content"]
        yield {"type": "final", "answer": final, "data": collected}

    @staticmethod
    def _summarize_for_sse(data: Any) -> str:
        """为 SSE 事件生成一行摘要文本。"""
        if isinstance(data, list):
            return f"找到 {len(data)} 个结果" if data else "无结果"
        if isinstance(data, dict):
            name = data.get("strMeal") or data.get("strDrink") or ""
            if name:
                return f"获取到: {name[:40]}"
            return f"返回 {len(data)} 个字段"
        return str(data)[:40] if data else "空"

    def _run_loop(
        self, user_message: str, max_rounds: int, visualize: bool, state: CompactState,
    ) -> dict:
        """Agent 核心循环。"""
        if visualize:
            Visualizer.user_query(user_message)
        system_prompt = prompt_builder.build()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        collected: dict[str, list] = {"meals": [], "drinks": []}

        context_truncated = False  # 防止无限截断循环

        for _round in range(1, max_rounds + 1):
            if estimate_context_size(messages) > CONTEXT_LIMIT:
                print("[auto compact]")
                messages[:] = compact_history(messages, state)
            try:
                response = self.client.chat(messages, tools=self.tools)
            except ContextLengthExceededError as e:
                if context_truncated:
                    logging.error("截断后仍然上下文超限，放弃当前轮次")
                    return {
                        "answer": "对话历史过长，请重新开始一个新的查询。",
                        "data": collected,
                        "error": "context_length_exceeded",
                    }
                logging.warning("上下文超限，即将截断消息: %s", e)
                if visualize:
                    Visualizer.tool_error("系统", f"上下文超限 ({e})，正在压缩历史消息…")
                messages[:] = auto_compact(messages)
                context_truncated = True
                # 重新尝试当前轮
                response = self.client.chat(messages, tools=self.tools)

            context_truncated = False
            choice = response["choices"][0]
            msg = choice["message"]

            if visualize:
                Visualizer.thinking(msg.get("content"))

            if not msg.get("tool_calls"):
                if visualize:
                    Visualizer.final_answer(msg["content"])
                return {"answer": msg["content"], "data": collected}

            messages.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": msg["tool_calls"],
            })

            for idx, tc in enumerate(msg["tool_calls"]):
                func_name = tc["function"]["name"]
                arguments = json.loads(tc["function"]["arguments"])

                if visualize:
                    Visualizer.tool_call(func_name, arguments, _round, idx)

                t0 = time.time()
                try:
                    tool_result = self._execute_tool(tc)
                    duration = time.time() - t0
                    result_data = json.loads(tool_result)

                    if visualize:
                        Visualizer.tool_result(func_name, result_data, duration)

                    # 收集结构化数据
                    self._collect_data(collected, func_name, result_data)

                except Exception as e:
                    duration = time.time() - t0
                    if visualize:
                        Visualizer.tool_error(func_name, str(e))
                    tool_result = json.dumps({"error": str(e)})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })

        logging.warning("达到最大调用轮数 (%d)，请求模型总结", max_rounds)
        try:
            response = self.client.chat(messages)
        except ContextLengthExceededError:
            messages[:] = auto_compact(messages)
            response = self.client.chat(messages)
        final = response["choices"][0]["message"]["content"]
        if visualize:
            Visualizer.final_answer(final)
        return {"answer": final, "data": collected}

    def _execute_tool(self, tool_call: dict) -> str:
        """执行一个工具调用，返回 JSON 字符串。"""
        func_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])
        func = NATIVE_HANDLERS.get(func_name)
        if not func:
            raise ValueError(f"未知工具: {func_name}")
        logging.info("执行工具: %s, 参数: %s", func_name, arguments)
        result = func(**arguments)
        return json.dumps(result, ensure_ascii=False, default=str)

    @staticmethod
    def _collect_data(collected: dict, func_name: str, data: Any):
        """从工具返回结果中提取结构化菜谱/饮品数据。"""
        meal_funcs = {
            "search_meals", "filter_by_ingredient",
            "filter_by_category", "filter_by_area",
        }
        if func_name in meal_funcs and isinstance(data, list):
            collected["meals"].extend(data)
        elif func_name == "get_meal_detail" and isinstance(data, dict) and data:
            collected["meals"].append(data)
        elif func_name == "random_meal" and isinstance(data, dict) and data:
            collected["meals"].append(data)

        drink_funcs = {"search_cocktails", "filter_cocktails_by_ingredient"}
        if func_name in drink_funcs and isinstance(data, list):
            collected["drinks"].extend(data)
        elif func_name == "get_cocktail_detail" and isinstance(data, dict) and data:
            collected["drinks"].append(data)
        elif func_name == "random_cocktail" and isinstance(data, dict) and data:
            collected["drinks"].append(data)


# ── 入口 ──────────────────────────────────────────────────────────────

def main():

    # 加载工具定义
    logging.info("已加载 %d 个工具", len(NATIVE_TOOLS))

    # 初始化 OpenRouter 客户端
    client = OpenRouterClient(api_key=OPENROUTER_API_KEY, model=MODEL)
    agent = RecipeAgent(client, NATIVE_TOOLS)
    compact_state = CompactState()


    # 交互式模式
    print(f"{Visualizer.BOLD}{Visualizer.GREEN}"
          f"╔══════════════════════════════════════════════════════╗\n"
          f"║         🍳 智能食谱助手 - Recipe Agent                ║\n"
          f"╚══════════════════════════════════════════════════════╝"
          f"{Visualizer.RESET}\n")
    while True:
        try:
            query = input(f"{Visualizer.CYAN}🍳 s14 >>{Visualizer.RESET} ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.strip().lower() in ("/exit", "/quit", "exit", "quit"):
            break
        agent.run(query, compact_state)
        print()


if __name__ == "__main__":
    main()
