import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(
    prefix="/email-assistant",
    tags=["email-assistant"],
    responses={404: {"description": "Not found"}},
)


# ── Tools ──


@tool
def fix_grammar(text: str) -> str:
    """Fix grammar, spelling, and punctuation errors in the given email text.
    Use this when the user wants to correct mistakes in their email."""
    # The agent itself will handle this via the LLM — this tool acts as a
    # routing signal. In a production app you could call a dedicated grammar
    # API here, but for our case the agent's LLM does the heavy lifting.
    return f"GRAMMAR_CHECK_REQUEST: {text}"


@tool
def summarize_thread(thread: str) -> str:
    """Summarize a long email thread or conversation into key points.
    Use this when the user pastes an email thread and wants a summary."""
    return f"SUMMARIZE_REQUEST: {thread}"


@tool
def draft_reply(context: str) -> str:
    """Draft a professional reply to an email based on the given context.
    The context should include the original email and any instructions
    about tone or key points to address."""
    return f"DRAFT_REPLY_REQUEST: {context}"


@tool
def improve_email(email_text: str) -> str:
    """Suggest improvements to make an email more professional, clear,
    and effective. Analyze tone, structure, clarity, and word choice."""
    return f"IMPROVE_REQUEST: {email_text}"


# ── Agent ──

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

agent = create_agent(
    model=model,
    tools=[fix_grammar, summarize_thread, draft_reply, improve_email],
    system_prompt=(
        "You are an expert email assistant. You help users with email-related tasks.\n\n"
        "Your capabilities:\n"
        "1. **Draft replies**: Write professional email responses based on context the user provides.\n"
        "2. **Summarize threads**: Condense long email conversations into key points and action items.\n"
        "3. **Fix grammar**: Correct spelling, grammar, and punctuation errors.\n"
        "4. **Improve emails**: Suggest improvements for tone, clarity, structure, and professionalism.\n\n"
        "Guidelines:\n"
        "- Always use the appropriate tool for the task.\n"
        "- After using a tool, provide the polished result directly to the user.\n"
        "- When drafting replies, ask about tone (formal/casual) if not specified.\n"
        "- When summarizing, highlight action items and key decisions.\n"
        "- When improving, explain what you changed and why.\n"
        "- Keep your language professional but approachable.\n"
        "- Format emails properly with greeting, body, and sign-off."
    ),
)


# ── Request / Response schemas ──

class EmailRequest(BaseModel):
    text: str
    action: str = "auto"  # "auto", "draft", "summarize", "grammar", "improve"


# ── Non-streaming endpoint ──

@router.post("/process")
async def process_email(request: EmailRequest):
    # Build user message based on action
    user_message = _build_user_message(request.text, request.action)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]}
    )
    ai_message = result["messages"][-1].content
    return {"result": ai_message}


# ── Streaming endpoint ──

@router.post("/process/stream")
async def process_email_stream(request: EmailRequest):
    user_message = _build_user_message(request.text, request.action)

    async def event_generator():
        try:
            async for event in agent.astream_events(
                {"messages": [{"role": "user", "content": user_message}]},
                version="v2",
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        data = json.dumps({"token": chunk.content})
                        yield f"data: {data}\n\n"

                elif kind == "on_tool_start":
                    tool_name = event.get("name", "tool")
                    status_map = {
                        "fix_grammar": "Checking grammar...",
                        "summarize_thread": "Summarizing thread...",
                        "draft_reply": "Drafting reply...",
                        "improve_email": "Analyzing email...",
                    }
                    status = status_map.get(tool_name, f"Using {tool_name}...")
                    data = json.dumps({"status": status})
                    yield f"data: {data}\n\n"

                elif kind == "on_tool_end":
                    data = json.dumps({"status": "Generating response..."})
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


# ── Helper ──

def _build_user_message(text: str, action: str) -> str:
    """Build a clear user message based on the selected action."""
    if action == "draft":
        return (
            f"Draft a professional reply to this email:\n\n{text}"
        )
    elif action == "summarize":
        return (
            f"Summarize this email thread into key points and action items:\n\n{text}"
        )
    elif action == "grammar":
        return (
            f"Fix all grammar, spelling, and punctuation errors in this email. "
            f"Show the corrected version:\n\n{text}"
        )
    elif action == "improve":
        return (
            f"Improve this email for clarity, tone, and professionalism. "
            f"Show the improved version and briefly explain the changes:\n\n{text}"
        )
    else:
        # "auto" — let the agent figure out the intent
        return text