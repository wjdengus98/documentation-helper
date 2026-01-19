import json
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.core import run_llm

app = FastAPI(title="LangChain Documentation Helper")

# Templates 설정
templates = Jinja2Templates(directory="templates")


def _format_sources(context_docs: list[Any]) -> list[str]:
    """Extract source URLs from context documents."""
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main chat page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(request: Request):
    """Handle chat messages and return AI response."""
    form_data = await request.form()
    user_message = form_data.get("message", "").strip()

    if not user_message:
        return HTMLResponse(
            content='<div class="text-red-400">Please enter a message.</div>',
            status_code=400
        )

    try:
        # Run the RAG pipeline
        result: Dict[str, Any] = run_llm(user_message)
        answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
        sources = _format_sources(result.get("context", []))

        # Return the response as HTML fragment for HTMX
        return templates.TemplateResponse(
            "partials/message.html",
            {
                "request": request,
                "role": "assistant",
                "content": answer,
                "sources": sources
            }
        )
    except Exception as e:
        return HTMLResponse(
            content=f'<div class="text-red-400">Error: {str(e)}</div>',
            status_code=500
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
