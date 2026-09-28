"""
tools.py - Define the tools (functions) that the agent can call.

In agentic AI, "tools" are plain functions that the LLM can decide to invoke.
The agent sees tool descriptions, decides which one to call, and uses the result
to continue reasoning.
"""

import math
import os
from datetime import datetime


# Path to the FAQ file (sits next to this module)
FAQ_FILE_PATH = os.path.join(os.path.dirname(__file__), "faq.txt")


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


# --- Tool registry (maps name -> function) ---

TOOL_REGISTRY = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "lookup_weather": lookup_weather,
    "read_faq": read_faq,
}
