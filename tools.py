import subprocess
from pathlib import Path

import requests

from handlers.memory_system import memory_mgr
from handlers.tasks_system import TASKS
from handlers.todomanager import TODO
from settings.constant import WORKDIR

# ── 工具定义 ──────────────────────────────────────────────────────────

MEALDB_BASE = "https://www.themealdb.com/api/json/v1/1"
COCKTAILDB_BASE = "https://www.thecocktaildb.com/api/json/v1/1"


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# ────────────────────────────── TheMealDB ──────────────────────────────

def search_meals(name: str) -> list[dict]:
    """按名称搜索菜谱。

    Args:
        name: 菜谱名称（关键词）。

    Returns:
        匹配的菜谱列表，未找到时返回空列表。
    """
    resp = requests.get(f"{MEALDB_BASE}/search.php", params={"s": name}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("meals") or []


def filter_by_ingredient(ingredient: str) -> list[dict]:
    """按食材筛选菜谱。

    Args:
        ingredient: 食材名称。

    Returns:
        包含该食材的菜谱列表。
    """
    resp = requests.get(f"{MEALDB_BASE}/filter.php", params={"i": ingredient}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("meals") or []


def filter_by_category(category: str) -> list[dict]:
    """按分类筛选菜谱。

    Args:
        category: 分类名称（如 'Seafood', 'Dessert'）。

    Returns:
        该分类下的菜谱列表。
    """
    resp = requests.get(f"{MEALDB_BASE}/filter.php", params={"c": category}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("meals") or []


def filter_by_area(area: str) -> list[dict]:
    """按地区/ cuisine 筛选菜谱。

    Args:
        area: 地区名称（如 'Canadian', 'Italian'）。

    Returns:
        该地区的菜谱列表。
    """
    resp = requests.get(f"{MEALDB_BASE}/filter.php", params={"a": area}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("meals") or []


def get_meal_detail(meal_id: int | str) -> dict | None:
    """根据 ID 获取菜谱的详细信息（食材清单、烹饪步骤等）。

    Args:
        meal_id: 菜谱 ID。

    Returns:
        菜谱详情字典，未找到时返回 None。
    """
    resp = requests.get(f"{MEALDB_BASE}/lookup.php", params={"i": meal_id}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    meals = data.get("meals")
    return meals[0] if meals else None


def random_meal() -> dict:
    """获取随机菜谱推荐。

    Returns:
        随机菜谱详情字典。
    """
    resp = requests.get(f"{MEALDB_BASE}/random.php", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data["meals"][0]


def get_categories() -> list[dict]:
    """获取所有菜谱分类。

    Returns:
        分类列表。
    """
    resp = requests.get(f"{MEALDB_BASE}/categories.php", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("categories") or []


# ────────────────────────────── TheCocktailDB ──────────────────────────────

def search_cocktails(name: str) -> list[dict]:
    """按名称搜索鸡尾酒/饮品。

    Args:
        name: 饮品名称（关键词）。

    Returns:
        匹配的饮品列表，未找到时返回空列表。
    """
    resp = requests.get(f"{COCKTAILDB_BASE}/search.php", params={"s": name}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("drinks") or []


def filter_cocktails_by_ingredient(ingredient: str) -> list[dict]:
    """按食材筛选鸡尾酒。

    Args:
        ingredient: 食材名称。

    Returns:
        包含该食材的饮品列表。
    """
    resp = requests.get(f"{COCKTAILDB_BASE}/filter.php", params={"i": ingredient}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("drinks") or []


def get_cocktail_detail(cocktail_id: int | str) -> dict | None:
    """根据 ID 获取鸡尾酒详细信息（食材、配料比例、调制步骤）。

    Args:
        cocktail_id: 鸡尾酒 ID。

    Returns:
        鸡尾酒详情字典，未找到时返回 None。
    """
    resp = requests.get(f"{COCKTAILDB_BASE}/lookup.php", params={"i": cocktail_id}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    drinks = data.get("drinks")
    return drinks[0] if drinks else None


def random_cocktail() -> dict:
    """获取随机鸡尾酒推荐。

    Returns:
        随机鸡尾酒详情字典。
    """
    resp = requests.get(f"{COCKTAILDB_BASE}/random.php", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data["drinks"][0]


def run_save_memory(name: str, description: str, mem_type: str, content: str) -> str:
    return memory_mgr.save_memory(name, description, mem_type, content)


NATIVE_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "save_memory": lambda **kw: run_save_memory(kw["name"], kw["description"], kw["type"], kw["content"]),
    "todo": lambda **kw: TODO.update(kw["items"]),
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("owner"), kw.get("addBlockedBy"),
                                             kw.get("addBlocks")),
    "task_list": lambda **kw: TASKS.list_all(),
    "compress": lambda **kw: "Compressing...",
    "task_get": lambda **kw: TASKS.get(kw["task_id"]),
    "search_meals": search_meals,
    "filter_by_ingredient": filter_by_ingredient,
    "filter_by_category": filter_by_category,
    "filter_by_area": filter_by_area,
    "get_meal_detail": get_meal_detail,
    "random_meal": random_meal,
    "get_categories": get_categories,
    "search_cocktails": search_cocktails,
    "filter_cocktails_by_ingredient": filter_cocktails_by_ingredient,
    "get_cocktail_detail": get_cocktail_detail,
    "random_cocktail": random_cocktail,
}

NATIVE_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                      "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"},
                                                       "new_text": {"type": "string"}},
                      "required": ["path", "old_text", "new_text"]}},
    {"name": "save_memory", "description": "Save a persistent memory that survives across sessions.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string", "description": "Short identifier (e.g. prefer_tabs, db_schema)"},
         "description": {"type": "string", "description": "One-line summary of what this memory captures"},
         "type": {"type": "string", "enum": ["user", "feedback", "project", "reference"],
                  "description": "user=dietary preferences, country/region, feedback=corrections, project=non-obvious project conventions or decision reasons, reference=external resource pointers"},
         "content": {"type": "string", "description": "Full memory content (multi-line OK)"},
     }, "required": ["name", "description", "type", "content"]}},
    {
        "name": "todo",
        "description": "Rewrite the current session plan for multi-step work.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                            "activeForm": {
                                "type": "string",
                                "description": "Optional present-continuous label.",
                            },
                        },
                        "required": ["content", "status"],
                    },
                },
            },
            "required": ["items"],
        },
    },
    {"name": "task_create", "description": "Create a new task.",
     "input_schema": {"type": "object",
                      "properties": {"subject": {"type": "string"}, "description": {"type": "string"}},
                      "required": ["subject"]}},
    {"name": "task_update", "description": "Update a task's status, owner, or dependencies.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string",
                                                                                                  "enum": ["pending",
                                                                                                           "in_progress",
                                                                                                           "completed",
                                                                                                           "deleted"]},
                                                       "owner": {"type": "string",
                                                                 "description": "Set when a teammate claims the task"},
                                                       "addBlockedBy": {"type": "array", "items": {"type": "integer"}},
                                                       "addBlocks": {"type": "array", "items": {"type": "integer"}}},
                      "required": ["task_id"]}},
    {"name": "task_list", "description": "List all tasks with status summary.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "task_get", "description": "Get full details of a task by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
    {"name": "compress", "description": "Manually compress conversation context.",
     "input_schema": {"type": "object", "properties": {}}},
    {
        "name": "search_meals",
        "description": "按关键词或菜谱名称name搜索菜谱，返回匹配的菜谱列表",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "菜谱名称关键词，如 'chicken breast'、'pasta'"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "filter_by_ingredient",
        "description": "按食材ingredient筛选菜谱，返回包含指定食材的所有菜谱列表",
        "input_schema": {
            "type": "object",
            "properties": {
                "ingredient": {
                    "type": "string",
                    "description": "食材名称，如 'chicken_breast'、'salmon'、'rice'"
                }
            },
            "required": ["ingredient"]
        }
    },
    {
        "name": "filter_by_category",
        "description": "按分类category筛选菜谱，返回指定分类下的所有菜谱列表",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "分类名称，如 'Seafood'、'Dessert'、'Vegetarian'、'Beef'"
                }
            },
            "required": ["category"]
        }
    },
    {
        "name": "filter_by_area",
        "description": "按地区/菜系 area筛选菜谱，返回指定地区的菜谱列表",
        "input_schema": {
            "type": "object",
            "properties": {
                "area": {
                    "type": "string",
                    "description": "地区名称，如 'Canadian'、'Italian'、'Japanese'、'Mexican'"
                }
            },
            "required": ["area"]
        }
    },
    {
        "name": "get_meal_detail",
        "description": "根据菜谱meal_id获取详细信息，包括食材清单、用量和烹饪步骤",
        "input_schema": {
            "type": "object",
            "properties": {
                "meal_id": {
                    "type": ["integer", "string"],
                    "description": "菜谱的唯一数字ID，如 52772"
                }
            },
            "required": ["meal_id"]
        }
    },
    {
        "name": "random_meal",
        "description": "随机获取一道菜谱推荐，返回完整的菜谱详情",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_categories",
        "description": "获取TheMealDB中所有可用的菜谱分类列表",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "search_cocktails",
        "description": "按饮品名称关键词name搜索鸡尾酒/饮品，返回匹配的饮品列表",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "饮品名称关键词，如 'margarita'、'vodka'、'gin'"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "filter_cocktails_by_ingredient",
        "description": "按食材/基酒筛选鸡尾酒，返回包含指定食材的所有饮品列表",
        "input_schema": {
            "type": "object",
            "properties": {
                "ingredient": {
                    "type": "string",
                    "description": "食材或基酒名称，如 'Gin'、'Vodka'、'Rum'、'Tequila'"
                }
            },
            "required": ["ingredient"]
        }
    },
    {
        "name": "get_cocktail_detail",
        "description": "根据鸡尾酒ID cocktail_id获取详细信息，包括基酒、配料比例和调制步骤",
        "input_schema": {
            "type": "object",
            "properties": {
                "cocktail_id": {
                    "type": ["integer", "string"],
                    "description": "鸡尾酒的唯一数字ID，如 11007"
                }
            },
            "required": ["cocktail_id"]
        }
    },
    {
        "name": "random_cocktail",
        "description": "随机获取一款鸡尾酒推荐，返回完整的饮品详情",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    }

]
