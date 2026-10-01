"""
agent.py - The core agent loop.

This is the heart of agentic AI. The loop works like this:

    1. THINK  - Send conversation history + tools to the LLM
    2. ACT    - If the LLM wants to call a tool, execute it
    3. OBSERVE - Feed the tool result back into the conversation
    4. REPEAT - Loop until the LLM gives a final text response

This "ReAct" (Reason + Act) pattern is the foundation of most AI agents.
"""

import json
from openai import OpenAI
from tools import TOOL_DEFINITIONS, TOOL_REGISTRY


# System prompt tells the agent who it is and how to behave
SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.
When a user asks something that requires calculation, time, or weather info,
use the appropriate tool.

CALCULATION RULE (strict): You must NEVER perform arithmetic yourself. For ANY
math, no matter how simple (even 1 + 1), you must call the calculator tool and
use its result. This includes summing or combining numbers that come from other
sources, such as items on the to-do list. If you need numbers from the to-do
list, first call read_todo_list, then pass the expression to calculator. Do not
state a numeric result unless it came from the calculator tool.

You also have access to a company FAQ via the read_faq tool. Whenever a user
asks a question that could be answered by the FAQ (such as business hours,
refunds, shipping, passwords, payment methods, contact info, or orders), call
read_faq to look up the answer and base your response on its contents. If the
FAQ does not contain the answer, say so clearly instead of guessing.

You can also manage a to-do list. When the user wants to add something but
hasn't said what, ask them what they'd like to add first, then call
add_todo_item with the exact item. When the user asks what's on their list or
their tasks, call read_todo_list and report the saved items.

Always explain your reasoning briefly before giving the final answer.
Be concise and friendly."""


class Agent:
    """
    A simple agentic AI that can:
    - Hold a conversation (memory)
    - Decide when to use tools
    - Execute tools and incorporate results
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        # Conversation memory - persists across turns
        self.messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def chat(self, user_message: str) -> str:
        """
        Process a user message through the agent loop.
        Returns the final assistant response.
        """
        # Add user message to memory
        self.messages.append({"role": "user", "content": user_message})

        # --- THE AGENT LOOP ---
        max_iterations = 5  # Safety limit to prevent infinite loops

        for iteration in range(max_iterations):
            print(f"\n--- Agent Loop Iteration {iteration + 1} ---")

            # STEP 1: THINK - Ask the LLM what to do
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",  # Let the model decide whether to use a tool
            )

            assistant_message = response.choices[0].message

            # STEP 2: CHECK - Did the LLM want to call tools?
            if assistant_message.tool_calls:
                # The LLM wants to use one or more tools
                # Save the assistant's decision to memory
                self.messages.append(assistant_message.model_dump())

                # STEP 3: ACT - Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    print(f"  Tool call: {function_name}({arguments})")

                    # Look up and execute the tool
                    if function_name in TOOL_REGISTRY:
                        result = TOOL_REGISTRY[function_name](**arguments)
                    else:
                        result = f"Error: Unknown tool '{function_name}'"

                    print(f"  Tool result: {result}")

                    # STEP 4: OBSERVE - Feed result back to the LLM
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })

                # Loop continues - LLM will process tool results next iteration

            else:
                # No tool calls - the LLM gave a final text response
                final_response = assistant_message.content
                self.messages.append({
                    "role": "assistant",
                    "content": final_response,
                })
                print(f"  Final response ready.")
                return final_response

        # If we hit max iterations, return what we have
        return "I encountered an issue processing your request. Please try again."

    def reset(self):
        """Clear conversation memory (start fresh)."""
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def get_history(self) -> list[dict]:
        """Return conversation history (for the API to expose)."""
        # Filter out system messages for display
        return [
            msg for msg in self.messages
            if msg.get("role") != "system"
        ]
