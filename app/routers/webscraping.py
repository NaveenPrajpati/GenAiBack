import os
import json
import requests
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(
    prefix="/webscraping",
    tags=["webscraping"],
    responses={404: {"description": "Not found"}},
)


# ── Tool ──

@tool
def scrape_webpage(url: str) -> str:
    """Scrape a webpage and extract its main text content from the given URL."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Truncate to avoid token limits
        if len(text) > 8000:
            text = text[:8000] + "\n\n[Content truncated...]"

        return text if text.strip() else "Could not extract meaningful text from this page."

    except requests.RequestException as e:
        return f"Error fetching URL: {str(e)}"


# ── Agent ──

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_agent(
    model=model,
    tools=[scrape_webpage],
    system_prompt=(
        "You are a web content analyst. When the user provides a URL:\n"
        "1. Use the scrape_webpage tool to fetch the page content.\n"
        "2. Analyze the extracted content.\n"
        "3. Provide a clear, well-structured summary covering the key points.\n\n"
        "If the user asks a specific question about the page, answer it based on "
        "the scraped content. Always cite specific details from the page."
    ),
)


# ── Request schema ──

class QueryRequest(BaseModel):
    text: str


# ── Non-streaming endpoint ──

@router.post("/scrap")
async def scrap(request: QueryRequest):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": request.text}]}
    )
    # Extract the final AI message content
    ai_message = result["messages"][-1].content
    return {"summary": ai_message}


# ── Streaming endpoint (token-by-token for typing effect) ──

@router.post("/scrap/stream")
async def scrap_stream(request: QueryRequest):
    async def event_generator():
        try:
            async for event in agent.astream_events(
                {"messages": [{"role": "user", "content": request.text}]},
                version="v2",
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    # Individual token from the LLM
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        data = json.dumps({"token": chunk.content})
                        yield f"data: {data}\n\n"

                elif kind == "on_tool_start":
                    # Notify frontend that a tool is being used
                    tool_name = event.get("name", "tool")
                    data = json.dumps({"status": f"Using {tool_name}..."})
                    yield f"data: {data}\n\n"

                elif kind == "on_tool_end":
                    data = json.dumps({"status": "Analyzing content..."})
                    yield f"data: {data}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )