"""Knowledge Base subsystem for RAG-powered recipe search."""
from handlers.knowledge_base.kb_manager import KnowledgeBaseManager, kb_manager
from handlers.knowledge_base.search import (
    SearchResult,
    search_knowledge_base,
    format_search_results,
)

__all__ = [
    "KnowledgeBaseManager",
    "kb_manager",
    "SearchResult",
    "search_knowledge_base",
    "format_search_results",
]
