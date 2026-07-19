"""Document loading and Chinese-aware text splitting."""
import json
import logging
import re
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# ── Supported formats ──────────────────────────────────────────────────

LOADER_EXTENSIONS: dict[str, str] = {
    ".txt": "text",
    ".md": "markdown",
    ".pdf": "pdf",
    ".docx": "docx",
    ".json": "json",
}


def load_file(filepath: Path) -> str:
    """Load text content from a file based on its extension.

    Args:
        filepath: Path to the file.

    Returns:
        Extracted text content.

    Raises:
        ValueError: If file extension is not supported.
        FileNotFoundError: If file does not exist.
    """
    ext = filepath.suffix.lower()
    if ext not in LOADER_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {ext}. "
            f"Supported: {', '.join(LOADER_EXTENSIONS)}"
        )

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    logger.info("Loading %s file: %s", ext, filepath.name)

    if ext in (".txt", ".md"):
        return filepath.read_text(encoding="utf-8")

    if ext == ".pdf":
        return _load_pdf(filepath)

    if ext == ".docx":
        return _load_docx(filepath)

    if ext == ".json":
        return _load_json(filepath)

    raise ValueError(f"Loader not implemented for: {ext}")


def _load_pdf(filepath: Path) -> str:
    """Extract text from PDF using PyMuPDF."""
    import fitz  # pymupdf
    doc = fitz.open(str(filepath))
    texts = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            texts.append(text.strip())
    doc.close()
    return "\n\n".join(texts)


def _load_docx(filepath: Path) -> str:
    """Extract text from DOCX using python-docx."""
    from docx import Document
    doc = Document(str(filepath))
    texts = []
    for para in doc.paragraphs:
        if para.text.strip():
            texts.append(para.text.strip())
    # Also extract table text
    for table in doc.tables:
        for row in table.rows:
            row_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_texts:
                texts.append(" | ".join(row_texts))
    return "\n".join(texts)


def _load_json(filepath: Path) -> str:
    """Extract text from JSON. Handles list of recipe objects, dict with list values, or plain dict."""
    data = json.loads(filepath.read_text(encoding="utf-8"))

    if isinstance(data, list):
        # List of objects — assume recipe list
        texts = []
        for item in data:
            if isinstance(item, dict):
                texts.append(_dict_to_text(item))
            else:
                texts.append(str(item))
        return "\n\n---\n\n".join(texts)

    if isinstance(data, dict):
        # Check if any value is a list of recipe objects
        for key, value in data.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                texts = [_dict_to_text(item) for item in value]
                return "\n\n---\n\n".join(texts)
        return _dict_to_text(data)

    return str(data)


def _dict_to_text(d: dict) -> str:
    """Convert a recipe-like dict to readable text."""
    lines = []
    # Title / name fields
    for key in ("name", "title", "菜名", "名称", "recipe_name"):
        if d.get(key):
            lines.append(f"# {d[key]}")
            break

    # Category / cuisine
    for key in ("category", "cuisine", "菜系", "分类"):
        if d.get(key):
            lines.append(f"菜系: {d[key]}")
            break

    # Ingredients
    for key in ("ingredients", "食材", "配料"):
        val = d.get(key)
        if val:
            if isinstance(val, list):
                lines.append(f"食材: {', '.join(str(v) for v in val)}")
            elif isinstance(val, str):
                lines.append(f"食材: {val}")
            break

    # Instructions / steps
    for key in ("instructions", "steps", "做法", "步骤", "烹饪方法"):
        val = d.get(key)
        if val:
            lines.append(f"\n做法:")
            if isinstance(val, list):
                for i, step in enumerate(val, 1):
                    lines.append(f"{i}. {step}")
            elif isinstance(val, str):
                lines.append(val)
            break

    # If nothing recognized, dump all
    if not lines:
        for k, v in d.items():
            lines.append(f"{k}: {v}")

    return "\n".join(lines)


# ── Chinese Recursive Text Splitter ────────────────────────────────────

class ChineseRecursiveTextSplitter:
    """Recursive text splitter with Chinese-aware separators.

    Splits text by trying separators in priority order until each chunk
    fits within chunk_size. Inspired by Langchain-ChatChat's implementation.
    """

    # Separators ordered by priority (paragraph → sentence → phrase)
    _SEPARATORS = [
        "\n\n",
        "\n",
        "。",
        "！",
        "？",
        ".",
        "!",
        "?",
        "；",
        ";",
        "，",
        ",",
        " ",
    ]

    def split_text(
        self,
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> List[str]:
        """Split text into chunks.

        Args:
            text: The input text to split.
            chunk_size: Maximum chunk size in characters.
            chunk_overlap: Overlap between consecutive chunks in characters.

        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []

        return self._split_recursive(text.strip(), chunk_size, chunk_overlap, 0)

    def _split_recursive(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        separator_idx: int,
    ) -> List[str]:
        """Recursively split text, trying each separator in priority order."""
        # Base case: text fits in one chunk
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        # Try current separator
        if separator_idx < len(self._SEPARATORS):
            sep = self._SEPARATORS[separator_idx]
            splits = text.split(sep)

            # If separator doesn't actually split (single element), try next
            if len(splits) == 1:
                return self._split_recursive(text, chunk_size, chunk_overlap, separator_idx + 1)

            # Merge splits into chunks
            chunks = []
            current = ""
            for part in splits:
                candidate = current + (sep if current else "") + part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current.strip():
                        # Current chunk is full — recursively split it if needed
                        if len(current) > chunk_size:
                            chunks.extend(
                                self._split_recursive(
                                    current, chunk_size, chunk_overlap, separator_idx + 1
                                )
                            )
                        else:
                            chunks.append(current)
                    # Start new chunk — if it's too big, recurse
                    if len(part) > chunk_size:
                        chunks.extend(
                            self._split_recursive(
                                part, chunk_size, chunk_overlap, separator_idx + 1
                            )
                        )
                    else:
                        current = part
            if current.strip():
                if len(current) > chunk_size:
                    chunks.extend(
                        self._split_recursive(
                            current, chunk_size, chunk_overlap, separator_idx + 1
                        )
                    )
                else:
                    chunks.append(current)

            # Apply overlap by merging small final chunks
            if chunk_overlap > 0 and len(chunks) > 1:
                return self._apply_overlap(chunks, chunk_overlap)
            return chunks

        # Last resort: no separator worked — hard split by character count
        return self._hard_split(text, chunk_size, chunk_overlap)

    def _hard_split(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Hard split by fixed character count (last resort)."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - chunk_overlap
        return chunks

    def _apply_overlap(self, chunks: List[str], overlap: int) -> List[str]:
        """Apply overlap by appending tail of previous chunk to next chunk's start."""
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]
            if len(prev) > overlap:
                curr = prev[-overlap:] + curr
            result.append(curr)
        return result
