from typing import Tuple
import re

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None


GREETING_KEYWORDS = ["hi", "hello", "hey", "good morning", "good afternoon"]
PRODUCT_KEYWORDS = ["price", "pricing", "feature", "features", "plan", "refund", "support"]
HIGH_INTENT_KEYWORDS = ["signup", "sign up", "subscribe", "buy", "purchase", "start trial", "get started", "interested"]


def _rule_intent(text: str) -> Tuple[str, float, str]:
    txt = text.lower().strip()
    if any(w in txt for w in HIGH_INTENT_KEYWORDS):
        return "HIGH_INTENT", 0.95, "keyword"
    if any(w in txt for w in GREETING_KEYWORDS):
        return "GREETING", 0.9, "keyword"
    if any(w in txt for w in PRODUCT_KEYWORDS):
        return "PRODUCT_QUERY", 0.9, "keyword"
    return "UNKNOWN", 0.0, "none"


from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal


# Pydantic model for reliable parsing of the LLM's intent classification
class Intent(BaseModel):
    """Represents the classified user intent."""
    intent: Literal["GREETING", "PRODUCT_QUERY", "HIGH_INTENT"] = Field(
        ...,
        description="The user's intent. Must be one of: GREETING, PRODUCT_QUERY, or HIGH_INTENT."
    )


def detect_intent(text: str, llm=None) -> Tuple[str, float, str]:
    """
    Detects the user's intent. Uses keywords first, then falls back to an LLM.
    """
    intent, conf, reason = _rule_intent(text)
    if intent != "UNKNOWN":
        return intent, conf, reason

    # Fallback to LLM-assisted classification
    if llm is None and ChatGoogleGenerativeAI is not None:
        # Use Gemini 2.5 Flash model for intent detection
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    if llm is not None:
        try:
            structured_llm = llm.with_structured_output(Intent)
            
            system_prompt = """
            You are an expert at classifying user intent. Classify the user's message into one of three intents:
            - GREETING: A simple greeting like 'hi', 'hello', or 'hey'.
            - PRODUCT_QUERY: A question about the product's features, price, or policies.
            - HIGH_INTENT: A message showing desire to buy or sign up (e.g., 'I want to sign up', 'the pro plan sounds good').
            """
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "Analyze the following message: '{input}'"),
                ]
            )
            chain = prompt | structured_llm
            result = chain.invoke({"input": text})
            
            if result.intent:
                return result.intent, 0.9, "llm" # High confidence from structured output
        except Exception as e:
            # Expose errors from the LLM call for debugging
            print(f"CRITICAL: Intent detection LLM call failed. Error: {e}")
            pass

    # Default to PRODUCT_QUERY if unsure
    return "PRODUCT_QUERY", 0.45, "default"
