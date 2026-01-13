# AutoStream - Conversational AI Sales Agent

This project is a conversational AI agent that acts as a sales assistant for a SaaS product called AutoStream. It uses Google's Gemini model to chat with users, answer questions, and capture leads.

---

## How to Run

### 1. Setup

First, create and activate a Python 3.9+ virtual environment.

```bash
# Create the environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

### 2. Install Packages

Install the required Python packages.

```bash
pip install -r requirements.txt
```
> **Note for Windows Users**: If `faiss-cpu` fails to install, you might need to use `conda` to install it.

### 3. Add API Key

Create a `.env` file in the main project folder and add your Google Gemini API key.

```dotenv
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
```

### 4. Run the Agent

Run the `app.py` script to start the agent. The first time you run it, it will create a local vector store in the `data/faiss_index` folder.

```bash
python app.py
```

You can now chat with the agent in your terminal. Type `exit` to stop.

---

## Architecture Overview

This agent is built with LangChain and powered by the Gemini 2.5 Flash model.

-   **Conversational Logic:** The main logic is in `agent/agent_graph.py`. It controls the flow of the conversation, from greeting the user to capturing their information. It checks the user's intent on every message, so it can react if the user changes the topic.

-   **Intent Detection:** The agent figures out what the user wants in `agent/intent_detector.py`. It uses simple keywords for common things (like "hello" or "price"). For anything more complex, it asks the Gemini model to classify the user's goal into one of three categories: `GREETING`, `PRODUCT_QUERY`, or `HIGH_INTENT`.

-   **Answering Questions (RAG):** When the user asks about the product, the agent uses a RAG pipeline. It finds relevant information from the `data/knowledge_base.md` file using a FAISS vector store. Then, it uses Gemini to generate a friendly, summarized answer based on that information.

-   **Memory:** The agent remembers the conversation history and any information it has collected (like the user's name) using a `ConversationState` object defined in `agent/state.py`.

-   **Tool Use:** The agent has a `mock_lead_capture` tool. It will only use this tool after it has successfully collected the user's name, email, and platform through conversation.

---

## WhatsApp Integration Plan

This agent could be connected to WhatsApp with these steps:

1.  **Webhook:** Use a web framework like Flask or FastAPI to create a webhook that can receive messages from WhatsApp.
2.  **WhatsApp App:** Configure a WhatsApp Business App to send incoming message notifications to your webhook.
3.  **State Management:** For each unique WhatsApp user, create and store a `ConversationState` object to keep track of their individual conversation.
4.  **Send Replies:** When the agent generates a response, use the WhatsApp Cloud API to send it back to the user.
