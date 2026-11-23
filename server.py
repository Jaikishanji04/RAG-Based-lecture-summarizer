"""FastAPI backend server for StudyMatrix RAG Agent & MCP endpoints."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.agent import AcademicAgent, AgentStep
from core.llm_provider import get_provider
from core.mcp_gmail import GmailMCPClient

load_dotenv()
logger = logging.getLogger("server")

app = FastAPI(
    title="StudyMatrix RAG & MCP Agent API",
    description="Agentic lecture summarization and Gmail MCP inbox assistant.",
    version="1.0.0",
)

# Enable CORS for React frontend (Vite default is 5173 or 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = AcademicAgent()
gmail_client = GmailMCPClient()


class AgentRunRequest(BaseModel):
    goal: str = Field(min_length=1, description="The user's goal or academic query")


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "StudyMatrix RAG Agent API"}


@app.get("/agent/status")
def get_agent_status() -> Dict[str, Any]:
    """Return Hugging Face and Gmail connection status."""
    hf_token = os.getenv("HF_TOKEN", "")
    gmail_status = gmail_client.get_connection_status()
    return {
        "huggingface": {
            "token_configured": bool(hf_token),
            "chat_model": os.getenv("HF_CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            "embed_model": os.getenv("HF_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        },
        "gmail": gmail_status,
    }


@app.get("/agent/tools")
def list_agent_tools() -> List[Dict[str, Any]]:
    """List all exposed tools available to the autonomous agent."""
    return agent.get_tool_definitions()


@app.post("/agent/run")
def run_agent(request: AgentRunRequest) -> Dict[str, Any]:
    """Execute the agent loop synchronously and return the complete step trace and answer."""
    steps: List[Dict[str, Any]] = []
    final_answer: str = ""

    for step in agent.run_stream(request.goal):
        steps.append(step.model_dump())
        if step.step_type == "final_answer":
            final_answer = step.content or ""

    return {
        "goal": request.goal,
        "steps": steps,
        "final_answer": final_answer,
    }


@app.post("/agent/run/stream")
def run_agent_stream(request: AgentRunRequest) -> StreamingResponse:
    """Stream agent execution step-by-step using Server-Sent Events (SSE)."""

    def event_generator():
        for step in agent.run_stream(request.goal):
            yield f"data: {json.dumps(step.model_dump())}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/agent/auth/gmail")
def trigger_gmail_auth() -> Dict[str, Any]:
    """Trigger the interactive OAuth consent flow for Gmail."""
    try:
        gmail_client.authenticate(interactive=True)
        return {"success": True, "message": "Gmail authenticated successfully."}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/lectures")
def list_lectures() -> Dict[str, Any]:
    """List all indexed lecture resources available for RAG."""
    lectures_file = Path("data/lectures.json")
    if not lectures_file.exists():
        return {"lectures": {}}
    try:
        data = json.loads(lectures_file.read_text(encoding="utf-8"))
        return {"lectures": data}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read lectures: {exc}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
