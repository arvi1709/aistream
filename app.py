import os
from dotenv import load_dotenv

from agent.agent_graph import AgentGraph
from agent.state import ConversationState


load_dotenv()

KB_PATH = os.path.join(os.path.dirname(__file__), "data", "knowledge_base.md")


def main():
    # Check if the Google API key is set in the .env file
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please create a .env file and add your API key.")
        return

    print("Starting AutoStream agent (type 'exit' to quit)")
    graph = AgentGraph(KB_PATH)
    state = ConversationState()

    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye")
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            print("Goodbye")
            break

        reply = graph.handle_message(user, state)
        print("Agent:", reply)


if __name__ == "__main__":
    main()
