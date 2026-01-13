import os
from typing import Optional

from .intent_detector import detect_intent
from .rag import build_vectorstore, answer_with_rag
from .tools import mock_lead_capture
from .state import ConversationState
from .langgraph_wrapper import Graph, State as LangGraphState

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None


class AgentGraph:
    """The main agent graph that handles the conversational flow."""

    def __init__(self, kb_path: str, llm=None):
        if llm is None and ChatGoogleGenerativeAI is not None:
            # Use the Gemini 2.5 Flash model
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        else:
            self.llm = llm

        self.vectorstore = build_vectorstore(kb_path)
        # Build the graph and register the different steps (nodes)
        self.graph = Graph()
        self.graph.register("intent_detection", self.node_intent_detection)
        self.graph.register("retrieval", self.node_retrieval)
        self.graph.register("lead_question", self.node_lead_question)
        self.graph.register("tool_execution", self.node_tool_execution)
        self.graph.register("orchestrate", self._orchestrate)

    # Nodes
    def node_intent_detection(self, message: str, state: ConversationState):
        intent, confidence, reason = detect_intent(message, llm=self.llm)
        return intent, confidence, reason

    def node_retrieval(self, message: str, state: ConversationState):
        answer, sources = answer_with_rag(message, self.vectorstore, llm=self.llm)
        return answer, sources

    def node_lead_question(self, message: str, state: ConversationState):
        # Asks the next question to collect the user's information
        f = state.collected_fields
        name = f.get("name")
        
        if name is None:
            return "Awesome. May I have your name?", "ask_name"
        if f.get("email") is None:
            # Use the collected name in the next question
            return f"Thanks, {name.split()[0]}! Where should I send updates and access details?", "ask_email"
        if f.get("platform") is None:
            return "Got it. And which platform are you creating for?", "ask_platform"
        return None, "done"

    def node_tool_execution(self, state: ConversationState):
        f = state.collected_fields
        # Only run the tool if we have all the info and the lead hasn't been captured yet
        if all(f.get(k) for k in ("name", "email", "platform")) and not state.lead_captured:
            res = mock_lead_capture(f["name"], f["email"], f["platform"])
            state.lead_captured = True
            if state.langgraph_state:
                state.langgraph_state.set("lead_captured", True)
            return res
        return None

    def _orchestrate(self, message: str, state: ConversationState) -> str:
        state.add_user_message(message)

        # --- STATEFUL LOGIC FIX ---
        # If we are already in the lead capture process, stay in it.
        if state.current_intent == "HIGH_INTENT" and not state.lead_captured:
            # The user's message is expected to be an answer to our last question.
            # Figure out which field we asked for and save the answer.
            if state.collected_fields.get("name") is None:
                state.collected_fields["name"] = message.strip()
                state.add_agent_message(f"Thanks, {message.strip().split()[0]}!")
            elif state.collected_fields.get("email") is None:
                state.collected_fields["email"] = message.strip()
                state.add_agent_message("Got it.")
            elif state.collected_fields.get("platform") is None:
                state.collected_fields["platform"] = message.strip()
                state.add_agent_message("Excellent choice.")

            # Ask the next question, or finalize if we have all the info.
            prompt, tag = self.node_lead_question(message, state)
            if tag == "done":
                tool_res = self.node_tool_execution(state)
                if tool_res:
                    reply = "Perfect! You're all set."
                    state.add_agent_message(reply)
                    return reply
            else:
                state.add_agent_message(prompt)
                return prompt
        
        # --- IF NOT IN LEAD CAPTURE, DETECT INTENT ---
        # Run intent detection only when we are not already in a specific flow.
        intent, confidence, reason = self.graph.run("intent_detection", message, state)
        state.current_intent = intent

        if intent == "HIGH_INTENT":
            # This is the first time high intent is detected. Start the lead capture flow.
            state.add_agent_message("That’s a great choice. Would you like me to help you get started?")
            prompt, tag = self.node_lead_question(message, state) # Ask the first question
            state.add_agent_message(prompt)
            return prompt

        if intent == "GREETING":
            reply = "Hey! Welcome to AutoStream. How can I help you today?"
            state.add_agent_message(reply)
            return reply

        if intent == "PRODUCT_QUERY":
            answer, sources = self.graph.run("retrieval", message, state)
            follow_up = "\n\nWould you like me to help you get started with one of our plans?"
            reply = answer + follow_up
            state.add_agent_message(reply)
            return reply
        
        # Fallback for when intent is unclear
        reply = "I'm not quite sure how to help with that. Could you try rephrasing?"
        state.add_agent_message(reply)
        return reply
        
    def handle_message(self, message: str, state: ConversationState) -> str:
        # Run the main orchestrator logic
        return self.graph.run("orchestrate", message, state)
