"""
tools.py - Define the tools (functions) that the agent can call.

In agentic AI, "tools" are plain functions that the LLM can decide to invoke.
The agent sees tool descriptions, decides which one to call, and uses the result
to continue reasoning.
"""

import json
import math
import os
from datetime import datetime


# Path to the FAQ file (sits next to this module)
FAQ_FILE_PATH = os.path.join(os.path.dirname(__file__), "faq.txt")

# Path to the temp JSON file that stores the to-do list (sits next to this module)
TODO_FILE_PATH = os.path.join(os.path.dirname(__file__), "todo_list.json")


# --- Tool definitions (what the LLM sees) ---

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression using Python syntax. Use ** for powers (NOT ^). Supports +, -, *, /, **, sqrt, sin, cos, tan, abs, round, pow, pi, e.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression in Python syntax, e.g. '2 + 2', 'pi ** 2', or 'sqrt(16)'. Use ** for exponents, never ^."
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Get the current weather for a given city. (Simulated for learning purposes.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'New York'"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_faq",
            "description": "Read the contents of the FAQ file. Use this whenever the user asks a question that might be answered by the FAQ (e.g. business hours, refunds, shipping, passwords, contact info, orders). Returns the full FAQ text so you can find the relevant answer.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_todo_item",
            "description": "Add an item to the user's to-do list. Use this whenever the user wants to add, save, or remember a task. The item is persisted to a JSON file so it can be read back later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "The to-do task the user wants to add, e.g. 'Buy groceries'."
                    }
                },
                "required": ["item"]
            }
        }
    },
     {
        "type": "function",
        "function": {
            "name": "remove_todo_item",
            "description": "Remove an item to the user's to-do list. Use this whenever the user wants to remove, delete, or scratch a task. The item is persisted to a JSON file so it can be read back later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "The to-do task the user wants to remove, e.g. 'Buy groceries'."
                    }
                },
                "required": ["item"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_todo_list",
            "description": "Read back all items currently saved in the user's to-do list. Use this whenever the user asks what is on their list, their tasks, or their to-dos.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
]


# --- Tool implementations (what actually runs) ---

def calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    # Allow only safe math operations
    allowed_names = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
        "pow": pow,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def get_current_time() -> str:
    """Return the current date and time."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S (%A)")


def lookup_weather(city: str) -> str:
    """Simulated weather lookup. In a real app, you'd call a weather API."""
    # Fake data for demonstration
    fake_weather = {
        "new york": "72°F, Partly Cloudy",
        "london": "58°F, Rainy",
        "tokyo": "80°F, Sunny",
        "paris": "65°F, Overcast",
        "sydney": "55°F, Windy",
    }
    weather = fake_weather.get(city.lower(), f"68°F, Clear skies (simulated for '{city}')")
    return f"Weather in {city}: {weather}"


def read_faq() -> str:
    """Read and return the full contents of the FAQ file."""
    try:
        with open(FAQ_FILE_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return "The FAQ file is empty."
        return content
    except FileNotFoundError:
        return f"Error: FAQ file not found at '{FAQ_FILE_PATH}'."
    except Exception as e:
        return f"Error reading FAQ file: {e}"


def _load_todos() -> list[str]:
    """Load the to-do items from the JSON file. Returns an empty list if missing/invalid."""
    try:
        with open(TODO_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_todos(items: list[str]) -> None:
    """Persist the to-do items to the JSON file."""
    with open(TODO_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)


def add_todo_item(item: str) -> str:
    """Add a single item to the to-do list and save it to the JSON file."""
    item = (item or "").strip()
    if not item:
        return "No to-do item was provided. Please tell me what you'd like to add."
    try:
        items = _load_todos()
        items.append(item)
        _save_todos(items)
        return f"Added '{item}' to your to-do list. You now have {len(items)} item(s)."
    except Exception as e:
        return f"Error saving to-do item: {e}"

def remove_todo_item(item: str) -> str:
    """Remove a single item from the to-do list and save the updated list to the JSON file."""
    item = (item or "").strip()
    if not item:
        return "No to-do item was provided. Please tell me what you'd like to remove."
    try:
        items = _load_todos()
        # Case-insensitive match so the user doesn't have to type it exactly.
        match = next((t for t in items if t.strip().lower() == item.lower()), None)
        if match is None:
            return f"'{item}' is not on your to-do list, so nothing was removed."
        items.remove(match)
        _save_todos(items)
        return f"Removed '{match}' from your to-do list. You now have {len(items)} item(s)."
    except Exception as e:
        return f"Error removing to-do item: {e}"


def read_todo_list() -> str:
    """Read back all saved to-do items."""
    try:
        items = _load_todos()
    except Exception as e:
        return f"Error reading to-do list: {e}"
    if not items:
        return "Your to-do list is empty."
    lines = [f"{i}. {task}" for i, task in enumerate(items, start=1)]
    return "Your to-do list:\n" + "\n".join(lines)


# --- Tool registry (maps name -> function) ---

TOOL_REGISTRY = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "lookup_weather": lookup_weather,
    "read_faq": read_faq,
    "add_todo_item": add_todo_item,
    "remove_todo_item": remove_todo_item,
    "read_todo_list": read_todo_list,
}
