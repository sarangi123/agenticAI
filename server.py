"""
server.py - FastAPI backend that exposes the agent as an API.

This shows how an agentic AI gets deployed as a service:
- POST /chat       → Send a message, get a response
- GET  /history    → See the full conversation
- POST /reset      → Clear conversation and start over
- GET  /health     → Check if the server is running

Run with: uvicorn server:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent import Agent

# ⚠️ Paste your OpenAI API key here
API_KEY = "sk-paste-your-key-here"

# --- App Setup ---
app = FastAPI(
    title="Agentic AI - Basic Conversation",
    description="A simple agentic AI backend demonstrating the agent loop, tools, and memory.",
    version="1.0.0",
)

# Initialize the agent
agent = Agent(api_key=API_KEY)


# --- Request/Response Models ---

class ChatRequest(BaseModel):
    message: str

    class Config:
        json_schema_extra = {
            "example": {"message": "What's the weather in Tokyo?"}
        }


class ChatResponse(BaseModel):
    response: str
    tool_calls_made: int  # How many tools were used in this turn


# --- API Endpoints ---

@app.get("/health")
def health_check():
    """Check if the server is running."""
    return {"status": "ok", "model": agent.model}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Send a message to the agent and get a response.

    The agent will:
    1. Analyze your message
    2. Decide if it needs to use any tools
    3. Execute tools if needed
    4. Return a final response
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Count messages before to determine how many tool calls were made
    messages_before = len(agent.messages)

    try:
        response = agent.chat(request.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # Count tool messages added during this turn
    tool_calls = sum(
        1 for msg in agent.messages[messages_before:]
        if isinstance(msg, dict) and msg.get("role") == "tool"
    )

    return ChatResponse(response=response, tool_calls_made=tool_calls)


@app.get("/history")
def get_history():
    """
    Get the full conversation history.
    Useful for debugging and understanding what the agent "remembers".
    """
    return {"messages": agent.get_history()}


@app.post("/reset")
def reset_conversation():
    """Clear conversation memory and start fresh."""
    agent.reset()
    return {"status": "conversation reset"}


# --- Run directly ---
if __name__ == "__main__":
    import uvicorn
    print("\n🤖 Starting Agentic AI Server...")
    print("   Docs: http://localhost:8000/docs")
    print("   Chat: POST http://localhost:8000/chat\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
