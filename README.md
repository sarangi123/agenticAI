# Agentic AI - Basic Conversation

A minimal project to understand how agentic AI works under the hood.

## Key Concepts

### The Agent Loop (ReAct Pattern)

```
User Message
     ↓
┌─────────────────────────────┐
│  1. THINK                   │  ← LLM decides what to do
│     Send messages + tools   │
│     to the model            │
├─────────────────────────────┤
│  2. ACT                     │  ← Execute tool if needed
│     Call the function       │
├─────────────────────────────┤
│  3. OBSERVE                 │  ← Feed result back
│     Add tool result to      │
│     conversation            │
├─────────────────────────────┤
│  4. REPEAT or RESPOND       │  ← Loop or give final answer
└─────────────────────────────┘
     ↓
Final Response
```

### Project Structure

```
├── agent.py        → The agent loop (core logic)
├── tools.py        → Tool definitions + implementations
├── server.py       → FastAPI backend (REST API)
├── run_cli.py      → Terminal chat interface
├── requirements.txt
├── .env.example    → API key config template
└── README.md
```

### File Walkthrough

| File | What You'll Learn |
|------|------------------|
| `tools.py` | How tools are defined (schema) and implemented (functions) |
| `agent.py` | The agent loop — think/act/observe cycle, conversation memory |
| `server.py` | How to expose an agent as a REST API with FastAPI |
| `run_cli.py` | Simplest way to run and debug the agent |

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your OpenAI API key:**
   ```bash
   copy .env.example .env
   # Edit .env and paste your key
   ```

3. **Run the CLI (simplest):**
   ```bash
   python run_cli.py
   ```

4. **Or run the API server:**
   ```bash
   python server.py
   # Then open http://localhost:8000/docs for interactive API docs
   ```

## Try These Prompts

| Prompt | What Happens |
|--------|-------------|
| "What is 15 * 23 + 7?" | Agent calls `calculator` tool |
| "What time is it?" | Agent calls `get_current_time` tool |
| "Weather in London?" | Agent calls `lookup_weather` tool |
| "Tell me a joke" | No tools needed — direct response |
| "What's sqrt(144) and the weather in Paris?" | Multiple tool calls in one turn |

## API Endpoints (when running server.py)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Send a message, get agent response |
| GET | `/history` | View full conversation memory |
| POST | `/reset` | Clear memory, start over |
| GET | `/health` | Server status check |

## How It All Connects

1. **User sends message** → via CLI or API
2. **Agent adds to memory** → conversation history grows
3. **LLM receives full history + tool schemas** → decides next action
4. **If tool needed** → agent executes it, adds result to memory, loops
5. **If no tool needed** → LLM produces final response
6. **Response returned** → with memory intact for next turn

## Key Takeaways

- **Agents are loops**, not single calls. The LLM can call tools multiple times.
- **Memory is just a list** of messages that gets sent with every request.
- **Tools are functions with schemas** — the LLM sees the schema, you run the code.
- **The backend is stateful** — it holds conversation history between requests.
- **tool_choice="auto"** lets the model decide when to use tools vs. just respond.
