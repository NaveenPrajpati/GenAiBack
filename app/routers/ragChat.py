import os
import uuid
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
supabase: Client = create_client(supabase_url, supabase_key)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)


# ── Supabase helpers (imported by rag.py) ─────────────────────────────────────


def create_chat(title: str) -> str:
    """Insert a new chat and return its id."""
    now = datetime.now(timezone.utc).isoformat()
    row = (
        supabase.table("rag_chats")
        .insert({"title": title, "created_at": now, "updated_at": now})
        .execute()
    )
    return row.data[0]["id"]


def save_messages(
    chat_id: str,
    question: str,
    answer: str,
    sources: list,
    ingestions: list[str],
) -> None:
    """Insert user + assistant messages and bump the chat's updated_at."""
    now = datetime.now(timezone.utc).isoformat()
    supabase.table("rag_messages").insert(
        [
            {
                "chat_id": chat_id,
                "role": "user",
                "content": question,
                "ingestions": ingestions,
                "created_at": now,
            },
            {
                "chat_id": chat_id,
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "created_at": now,
            },
        ]
    ).execute()
    supabase.table("rag_chats").update({"updated_at": now}).eq("id", chat_id).execute()


# ── Request / response models ──────────────────────────────────────────────────


class CreateChatRequest(BaseModel):
    title: Optional[str] = None


# ── Routes ─────────────────────────────────────────────────────────────────────


@router.post("", status_code=201)
async def create_chat_route(body: CreateChatRequest):
    """Create a new chat session. Title defaults to 'New Chat'."""
    try:
        now = datetime.now(timezone.utc).isoformat()
        title = (body.title or "New Chat")[:200]
        row = (
            supabase.table("rag_chats")
            .insert({"title": title, "created_at": now, "updated_at": now})
            .execute()
        )
        return {"message": "chat created", "data": row.data[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", status_code=200)
async def get_all_chats():
    """Return all chats ordered by most recently updated."""
    try:
        result = (
            supabase.table("rag_chats")
            .select("*")
            .order("updated_at", desc=True)
            .execute()
        )
        return {"message": "chats fetched", "data": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{chat_id}/messages", status_code=200)
async def get_messages(chat_id: str):
    """Return all messages for a chat ordered chronologically."""
    try:
        # Verify chat exists
        chat = (
            supabase.table("rag_chats")
            .select("id, title, created_at, updated_at")
            .eq("id", chat_id)
            .single()
            .execute()
        )
        messages = (
            supabase.table("rag_messages")
            .select("*")
            .eq("chat_id", chat_id)
            .order("created_at", desc=False)
            .execute()
        )
        return {
            "message": "messages fetched",
            "data": {"chat": chat.data, "messages": messages.data},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{chat_id}", status_code=200)
async def delete_chat(chat_id: str):
    """Delete a chat and all its messages (cascade)."""
    try:
        supabase.table("rag_chats").delete().eq("id", chat_id).execute()
        return {"message": "chat deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
