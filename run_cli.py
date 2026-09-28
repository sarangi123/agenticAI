"""
run_cli.py - Run the agent in your terminal (no server needed).

This is the simplest way to interact with the agent and see the loop in action.
You'll see tool calls printed in real-time.

Usage: python run_cli.py
"""

import os

from dotenv import load_dotenv

from agent import Agent


# Load the OpenAI API key from a local .env file (never commit real keys).
# Copy .env.example to .env and set OPENAI_API_KEY there.
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")


def main():
    if not API_KEY:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    agent = Agent(api_key=API_KEY)

    print("=" * 50)
    print("🤖 Agentic AI - Interactive Chat")
    print("=" * 50)
    print("Try these to see tools in action:")
    print("  • 'What is 234 * 567?'        → calculator tool")
    print("  • 'What time is it?'           → time tool")
    print("  • 'Weather in Tokyo?'          → weather tool")
    print("  • 'FAQ'          → file read tool")
    print("  • 'quit' to exit")
    print("=" * 50)

    while True:
        user_input = input("\n👤 You: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break

        if not user_input:
            continue

        response = agent.chat(user_input)
        print(f"\n🤖 Agent: {response}")


if __name__ == "__main__":
    main()
