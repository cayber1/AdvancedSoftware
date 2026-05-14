"""
MCP (Model Context Protocol) Governance Layer
Enforces typed I/O schemas, role-based access control, and execution logging.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


AGENT_ROLES = {
    "ContextRetrievalAgent": ["read_documents", "rank_context"],
    "ReasoningAgent": ["read_context", "call_llm"],
    "GroundingValidatorAgent": ["read_context", "read_answer", "call_llm", "compute_similarity"],
    "AdversarialTesterAgent": ["inject_adversarial", "read_context", "call_llm"],
}


@dataclass
class MCPEvent:
    event_id: str
    agent: str
    action: str
    timestamp: float
    input_schema: dict
    output_schema: dict
    status: Literal["ok", "denied", "error"]
    message: str = ""


class MCPGovernance:
    """
    Enforces:
    - Role-based access control per agent
    - Typed input/output validation
    - Full execution logging + traceability
    """

    def __init__(self):
        self._log: list[MCPEvent] = []

    def check_permission(self, agent_name: str, action: str) -> bool:
        allowed = AGENT_ROLES.get(agent_name, [])
        return action in allowed

    def log_action(
        self,
        agent: str,
        action: str,
        input_data: Any,
        output_data: Any,
        status: Literal["ok", "denied", "error"] = "ok",
        message: str = "",
    ) -> MCPEvent:
        event = MCPEvent(
            event_id=str(uuid.uuid4())[:8],
            agent=agent,
            action=action,
            timestamp=time.time(),
            input_schema={"type": type(input_data).__name__, "preview": str(input_data)[:120]},
            output_schema={"type": type(output_data).__name__, "preview": str(output_data)[:120]},
            status=status,
            message=message,
        )
        self._log.append(event)
        return event

    def enforce(self, agent_name: str, action: str, input_data: Any) -> bool:
        """Returns True if allowed, False if denied (and logs the denial)."""
        if self.check_permission(agent_name, action):
            return True
        self.log_action(
            agent=agent_name,
            action=action,
            input_data=input_data,
            output_data=None,
            status="denied",
            message=f"Agent '{agent_name}' not authorized for action '{action}'",
        )
        return False

    def get_execution_log(self) -> list[dict]:
        return [
            {
                "event_id": e.event_id,
                "agent": e.agent,
                "action": e.action,
                "timestamp": e.timestamp,
                "status": e.status,
                "message": e.message,
                "input_preview": e.input_schema["preview"],
                "output_preview": e.output_schema["preview"],
            }
            for e in self._log
        ]

    def print_log(self):
        print("\n--- MCP Execution Log ---")
        for e in self._log:
            status_icon = "✓" if e.status == "ok" else ("✗" if e.status == "denied" else "!")
            print(f"[{status_icon}] [{e.agent}] {e.action} → {e.status}")
        print("-------------------------\n")
