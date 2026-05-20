"""
ContextGuard — Multi-Agent MCP Framework for Context Integrity Verification
Group 2: Diyar Buyuksahin, Etem Tolga Erten, Süleyman Kılıç, Andrew Mabuto
"""

from .mcp_governance import MCPGovernance
from .context_store import ContextStore
from .agents import (
    ContextRetrievalAgent,
    ReasoningAgent,
    GroundingValidatorAgent,
    AdversarialTesterAgent,
)
from .metrics import evaluate_response, format_report
from .attribution import compute_attribution

__all__ = [
    "MCPGovernance",
    "ContextStore",
    "ContextRetrievalAgent",
    "ReasoningAgent",
    "GroundingValidatorAgent",
    "AdversarialTesterAgent",
    "evaluate_response",
    "format_report",
    "compute_attribution",
]
