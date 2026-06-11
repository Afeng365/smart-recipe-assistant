import platform
import re
from datetime import datetime
from pathlib import Path

from settings.constant import MODEL, WORKDIR, DYNAMIC_BOUNDARY

MEMORY_GUIDANCE = """
应该保存到长期记忆save memories的内容：
 1. 用户画像

  ┌───────────────┬──────────────────────────────────┐
  │     内容      │               示例               │
  ├───────────────┼──────────────────────────────────┤
  │ 饮食限制/禁忌 │ 清真、素食、麸质过敏、坚果过敏   │
  ├───────────────┼──────────────────────────────────┤
  │ 健康目标      │ 减脂、增肌、控糖、低钠饮食       │
  ├───────────────┼──────────────────────────────────┤
  │ 口味偏好      │ 喜欢辣/清淡/酸甜，厌恶香菜/内脏  │
  ├───────────────┼──────────────────────────────────┤
  │ 厨艺水平      │ 新手/中级/进阶（影响推荐复杂度） │
  ├───────────────┼──────────────────────────────────┤
  │ 常用厨具      │ 有空气炸锅、慢炖锅、蒸烤箱等     │
  └───────────────┴──────────────────────────────────┘

  2. 重复性行为模式

  - 常做的菜系（用户总做川菜 → 偏好辣）
  - 做饭时间规律（工作日30分钟快手菜，周末可花2小时）
  - 常批量购买的食材（冰箱总有鸡胸肉）

  3. 历史反馈

  - 对推荐菜品的评价（"太油了"、"步骤太复杂"、"孩子很喜欢"）
  - 拒绝/修改过的建议模式

  4. 家庭/用餐场景

  - 几人份（常做2人份 vs 家庭4人份）
  - 有无小孩（需考虑儿童口味）
  - 招待场景偏好
-> type: reference


不应该保存到长期记忆的内容, When NOT to save memories:：
  1. 单次/临时性信息

  ┌──────────────────────────┬────────────────────────────┐
  │           例子           │            原因            │
  ├──────────────────────────┼────────────────────────────┤
  │ "今晚想吃鱼"             │ 这是单次请求，不是长期偏好 │
  ├──────────────────────────┼────────────────────────────┤
  │ "冰箱里有西兰花快过期了" │ 食材库存动态变化           │
  ├──────────────────────────┼────────────────────────────┤
  │ "明天有朋友来家吃饭"     │ 一次性事件                 │
  └──────────────────────────┴────────────────────────────┘

  2. 可推导/冗余信息

  - 用户每次问"低卡食谱" → 不需要重复保存"用户关注卡路里"，因为行为模式已经足够确定
  - 用户说"不要猪肉" + "不要牛肉" → 直接推断可能是穆斯林，保存"清真饮食"一条就够了

  3. 不明确的/单次反馈

  - "这个菜一般" → 不保存，除非多次针对同类型菜品给出类似反馈
  - "今天不想吃辣的" → 单日情绪，不覆盖辣味偏好记忆

  4. 菜谱本身的内容

  - 具体菜谱的步骤、配料 → Git/代码里有，不需要存记忆
  - 用户问过的某个菜的历史 → 如果用户再问可以用代码检索，存记忆浪费空间
"""


class SystemPromptBuilder:
    def __init__(self, workdir: Path = None, tools: list = None):
        self.workdir = workdir or WORKDIR
        self.tools = tools or []
        self.skills_dir = self.workdir / "skills"
        self.memory_dir = self.workdir / ".memory"

    def _build_core(self) -> str:
        return (
             "你是一个专业的食谱与饮品助手。你可以帮助用户搜索菜谱、"
            "查询食材、推荐菜品和饮品等，你会主动保存用户的个人饮食喜好、个人习惯和用户的国家/地区等信息，以便根据用户个人偏好帮助其餐食搭配推荐，"
             "实现主菜 + 配饮的合理组合，请根据用户的提问，选择合适的工具"
            "来获取信息，然后用中文组织回答。回答需清晰、有条理，每次不要推荐过多菜品，根据人数来返回菜品和饮品的数量，每个人最多3到5个菜，每个人1到3个饮品"
            "菜谱卡片须包含图片、食材清单列表和烹饪步骤等内容，挑选主要的几项菜谱进行推荐，注意格式整洁。"

        )

    def _build_tool_listing(self) -> str:
        if not self.tools:
            return ""

        lines = ["# Available tools"]
        for tool in self.tools:
            props = tool.get("input_schema", {}).get("properties", {})
            params = ", ".join(props.keys())
            lines.append(f"- {tool['name']}({params}): {tool['description']}")
        return "\n".join(lines)

    def _build_memory_section(self) -> str:
        if not self.memory_dir.exists():
            return ""
        memories = []
        for md_file in sorted(self.memory_dir.glob("*.md")):
            if md_file.name == "MEMORY.md":
                continue

            text = md_file.read_text()
            match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
            if not match:
                continue
            header, body = match.group(1), match.group(2)
            meta = {}
            for line in header.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    meta[k.strip()] = v.strip()
            name = meta.get("name", md_file.stem)
            mem_type = meta.get("type", "project")
            desc = meta.get("description", "")
            memories.append(f"[{mem_type}]: {name} {desc}\n{body}")
        if not memories:
            return ""
        memories.append(MEMORY_GUIDANCE)
        return "# Memories (persistent)\n\n" + "\n\n".join(memories)

    def _build_claude_md(self) -> str:
        """
        Load CLAUDE.md files in priority order (all are included):
        1. ~/.claude/CLAUDE.md (user-global instructions)
        2. <project-root>/CLAUDE.md (project instructions)
        3. <current-subdir>/CLAUDE.md (directory-specific instructions)
        """
        sources = []

        # User-global
        user_claude = Path.home() / ".claude" / "CLAUDE.md"
        if user_claude.exists():
            sources.append(("user global (~/.claude/CLAUDE.md)", user_claude.read_text()))

        # Project root
        project_claude = self.workdir / "CLAUDE.md"
        if project_claude.exists():
            sources.append(("project root (CLAUDE.md)", project_claude.read_text()))

        # Subdirectory -- in real CC, this walks from cwd up to project root
        # Teaching: check cwd if different from workdir
        cwd = Path.cwd()
        if cwd != self.workdir:
            subdir_claude = cwd / "CLAUDE.md"
            if subdir_claude.exists():
                sources.append((f"subdir ({cwd.name}/CLAUDE.md)", subdir_claude.read_text()))

        if not sources:
            return ""
        parts = ["# CLAUDE.md instructions"]
        for label, content in sources:
            parts.append(f"## From {label}")
            parts.append(content.strip())
        return "\n\n".join(parts)

    def _build_dynamic_context(self) -> str:
        lines = [
            f"Current date: {datetime.today().isoformat()}",
            f"Working directory: {self.workdir}",
            f"Model: {MODEL},"
            f"Platform: {platform.system()},"
        ]
        return "# Dynamic context\n" + "\n".join(lines)

    def build(self) -> str:
        """
        Assemble the full system prompt from all sections.

        Static sections (1-5) are separated from dynamic (6) by
        the DYNAMIC_BOUNDARY marker. In real CC, the static prefix
        is cached across turns to save prompt tokens.
        """
        sections = []

        core = self._build_core()
        if core:
            sections.append(core)

        tools = self._build_tool_listing()
        if tools:
            sections.append(tools)

        memory = self._build_memory_section()
        if memory:
            sections.append(memory)

        claude_md = self._build_claude_md()
        if claude_md:
            sections.append(claude_md)

        # Static/dynamic boundary
        sections.append(DYNAMIC_BOUNDARY)

        dynamic = self._build_dynamic_context()
        if dynamic:
            sections.append(dynamic)

        return "\n\n".join(sections)

