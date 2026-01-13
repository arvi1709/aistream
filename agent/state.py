from dataclasses import dataclass, field
from typing import List, Optional

from .langgraph_wrapper import State as LangGraphState


@dataclass
class ConversationState:
    """Tracks the state of the conversation."""
    history: List[str] = field(default_factory=list)
    current_intent: Optional[str] = None
    collected_fields: dict = field(default_factory=lambda: {"name": None, "email": None, "platform": None})
    lead_captured: bool = False

    def __post_init__(self):
        try:
            self.langgraph_state = LangGraphState()
            # Mirror fields for LangGraph compatibility
            self.langgraph_state.set("history", self.history)
            self.langgraph_state.set("collected_fields", self.collected_fields)
            self.langgraph_state.set("lead_captured", self.lead_captured)
        except Exception:
            self.langgraph_state = None

    def add_user_message(self, msg: str):
        self.history.append("USER: " + msg)
        if self.langgraph_state:
            self.langgraph_state.set("history", self.history)

    def add_agent_message(self, msg: str):
        self.history.append("AGENT: " + msg)
        if self.langgraph_state:
            self.langgraph_state.set("history", self.history)

    def reset_lead(self):
        self.collected_fields = {"name": None, "email": None, "platform": None}
        self.lead_captured = False
        if self.langgraph_state:
            self.langgraph_state.set("collected_fields", self.collected_fields)
            self.langgraph_state.set("lead_captured", self.lead_captured)
from typing import TypedDict, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    Represents the state of the conversational agent.
    """
    messages: List[BaseMessage]
    intent: str
    lead_info: dict
    rag_context: str
