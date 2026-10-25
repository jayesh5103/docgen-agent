"""FastAPI application for the Autonomous Document Agent service."""

from functools import lru_cache
import logging
from pathlib import Path
import re
from typing import Any
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from app.agent import DocumentAgent
from app.config import Settings
from app.errors import AgentError, LLMOutputError, LLMUnavailableError
from app.llm import build_llm
from app.schemas import AgentRequest, AgentResponse
from app.tools import default_registry

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("docgen-agent")

app = FastAPI(
    title="Autonomous Document Agent",
    description="FastAPI service planning and generating executive business documents in Word (.docx) format.",
    version="1.0.0",
)

DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


# Autonomous Document Agent - FastAPI Service (Configured for Groq)

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


def get_agent(settings: Settings = Depends(get_settings)) -> DocumentAgent:
    """Build or retrieve cached DocumentAgent instance."""
    llm = build_llm(settings)
    tools = default_registry()
    return DocumentAgent(llm=llm, tools=tools, settings=settings)


# -----------------------------------------------------------------------------
# Exception Handlers
# -----------------------------------------------------------------------------

@app.exception_handler(LLMUnavailableError)
def handle_llm_unavailable(request: Request, exc: LLMUnavailableError) -> JSONResponse:
    logger.error("LLM unavailable: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc), "request_id": getattr(request.state, "request_id", None)},
    )


@app.exception_handler(LLMOutputError)
def handle_llm_output_error(request: Request, exc: LLMOutputError) -> JSONResponse:
    logger.error("LLM output error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(exc), "request_id": getattr(request.state, "request_id", None)},
    )


@app.exception_handler(AgentError)
def handle_agent_error(request: Request, exc: AgentError) -> JSONResponse:
    logger.error("Agent execution error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(exc), "request_id": getattr(request.state, "request_id", None)},
    )


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Health check reporting provider and active model."""
    return {
        "status": "ok",
        "provider": settings.provider,
        "model": settings.model,
    }


@app.get("/tools")
def list_tools(agent: DocumentAgent = Depends(get_agent)) -> dict[str, Any]:
    """List all available tools and descriptions."""
    tool_list = []
    for name in agent.tools.names():
        t = agent.tools.get(name)
        if t:
            tool_list.append({
                "name": t.name,
                "description": t.description,
                "example": t.example,
            })
    return {"tools": tool_list}


@app.post("/agent", response_model=AgentResponse)
def generate_document(
    request: AgentRequest,
    agent: DocumentAgent = Depends(get_agent),
) -> Any:
    """Generate a document from a natural-language business request."""
    logger.info("Received document generation request (%d chars)", len(request.request))
    response = agent.run(request)

    if response.status == "rejected":
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=response.model_dump(),
        )

    return response


@app.get("/documents/{request_id}")
def download_document(
    request_id: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Download the generated .docx document by request ID."""
    if not ID_PATTERN.match(request_id):
        raise HTTPException(status_code=404, detail="Invalid document ID format.")

    output_dir = Path(settings.output_dir)
    matches = list(output_dir.glob(f"*-{request_id}.docx"))
    if not matches or not matches[0].exists():
        raise HTTPException(status_code=404, detail=f"Document with ID '{request_id}' not found.")

    target_file = matches[0]
    return FileResponse(
        path=str(target_file),
        media_type=DOCX_MEDIA_TYPE,
        filename=target_file.name,
    )
