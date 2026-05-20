"""
ContextGuard — MCP (Model Context Protocol) Governance Layer
Enforces:
  - Role-based access control (RBAC) per agent
  - Typed input/output schema validation
  - Full execution logging and traceability
  - Context versioning via ContextStore integration

Proposal reference: §MCP Governance — typed I/O schemas, RBAC, logging.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


# ─── Role-based permission table ────────────────────────────────────────────
AGENT_ROLES: dict[str, list[str]] = {
    "ContextRetrievalAgent":   ["read_documents", "rank_context", "commit_context"],
    "ReasoningAgent":          ["read_context", "call_llm", "checkout_context"],
    "GroundingValidatorAgent": ["read_context", "read_answer", "call_llm",
                                "compute_similarity", "compute_attribution"],
    "AdversarialTesterAgent":  ["inject_adversarial", "read_context", "call_llm",
                                "commit_context"],
}

# ─── Typed schema definitions (lightweight) ─────────────────────────────────
INPUT_SCHEMAS: dict[str, type] = {
    "read_documents":      str,
    "rank_context":        str,
    "commit_context":      list,
    "read_context":        str,
    "call_llm":            str,
    "checkout_context":    str,
    "read_answer":         str,
    "compute_similarity":  str,
    "compute_attribution": str,
    "inject_adversarial":  str,
}


@dataclass
class MCPEvent:
    event_id:      str
    agent:         str
    action:        str
    timestamp:     float
    input_schema:  dict
    output_schema: dict
    status:        Literal["ok", "denied", "error"]
    message:       str = ""

    def as_dict(self) -> dict:
        return {
            "event_id":       self.event_id,
            "agent":          self.agent,
            "action":         self.action,
            "timestamp":      self.timestamp,
            "status":         self.status,
            "message":        self.message,
            "input_preview":  self.input_schema.get("preview", ""),
            "output_preview": self.output_schema.get("preview", ""),
        }


class MCPGovernance:
    """
    Central MCP governance controller.

    Usage
    -----
    mcp = MCPGovernance()
    allowed = mcp.enforce("ReasoningAgent", "call_llm", query)
    mcp.log_action("ReasoningAgent", "call_llm", input_data, output_data, "ok")
    log = mcp.get_execution_log()
    """

    def __init__(self):
        self._log: list[MCPEvent] = []

    # ── Permission checks ────────────────────────────────────────────────────

    def check_permission(self, agent_name: str, action: str) -> bool:
        return action in AGENT_ROLES.get(agent_name, [])

    def _validate_schema(self, action: str, input_data: Any) -> bool:
        """Lightweight type check against INPUT_SCHEMAS."""
        expected = INPUT_SCHEMAS.get(action)
        if expected is None:
            return True                    # unknown action — pass through
        return isinstance(input_data, expected)

    def enforce(self, agent_name: str, action: str, input_data: Any) -> bool:
        """
        Returns True if (a) the agent has the role, AND (b) input type is valid.
        On failure, logs a 'denied' event and returns False.
        """
        if not self.check_permission(agent_name, action):
            self._deny(agent_name, action, input_data,
                       f"Agent '{agent_name}' not authorized for '{action}'")
            return False

        if not self._validate_schema(action, input_data):
            self._deny(agent_name, action, input_data,
                       f"Schema mismatch for '{action}': got {type(input_data).__name__}, "
                       f"expected {INPUT_SCHEMAS.get(action, '?').__name__}")
            return False

        return True

    def _deny(self, agent: str, action: str, input_data: Any, message: str):
        self.log_action(agent, action, input_data, None, status="denied", message=message)

    # ── Logging ──────────────────────────────────────────────────────────────

    def log_action(
        self,
        agent:       str,
        action:      str,
        input_data:  Any,
        output_data: Any,
        status:      Literal["ok", "denied", "error"] = "ok",
        message:     str = "",
    ) -> MCPEvent:
        event = MCPEvent(
            event_id=str(uuid.uuid4())[:8],
            agent=agent,
            action=action,
            timestamp=time.time(),
            input_schema={
                "type":    type(input_data).__name__,
                "preview": str(input_data)[:150],
            },
            output_schema={
                "type":    type(output_data).__name__,
                "preview": str(output_data)[:150],
            },
            status=status,
            message=message,
        )
        self._log.append(event)
        return event

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_execution_log(self) -> list[dict]:
        return [e.as_dict() for e in self._log]

    def get_denied_events(self) -> list[dict]:
        return [e.as_dict() for e in self._log if e.status == "denied"]

    def print_log(self, verbose: bool = False):
        icon_map = {"ok": "✓", "denied": "✗", "error": "!"}
        print("\n─── MCP Execution Log ───────────────────────────────────")
        for e in self._log:
            icon = icon_map.get(e.status, "?")
            print(f"  [{icon}] {e.agent:<28} {e.action:<25} → {e.status}")
            if verbose and e.message:
                print(f"       msg: {e.message}")
        denied = sum(1 for e in self._log if e.status == "denied")
        print(f"  Total: {len(self._log)} events | {denied} denied")
        print("─────────────────────────────────────────────────────────\n")
