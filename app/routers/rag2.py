import asyncio
import os
import shutil
import tempfile
import uuid
import requests
import numpy as np
import redis.asyncio as aioredis
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Form, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.retrievers import PineconeHybridSearchRetriever
from dotenv import load_dotenv
import json
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.document_transformers import LongContextReorder
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from supabase import create_client, Client
from app.utils.spliters import getSpliter
from app.routers.ragChat import create_chat, save_messages

import logging

load_dotenv()
logger = logging.getLogger(__name__)

# ── Pinecone setup ─────────────────────────────────────────────────────────────

api_key = os.getenv("PINECONE_KEY")
if not api_key:
    raise ValueError("PINECONE_KEY environment variable not set")

pc = Pinecone(api_key=api_key)
index_name = "rag-hybrid"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="dotproduct",  # dotproduct required for native Pinecone sparse+dense hybrid
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# ── Supabase setup ─────────────────────────────────────────────────────────────

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
supabase: Client = create_client(supabase_url, supabase_key)

# ── Redis semantic cache setup ─────────────────────────────────────────────────

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
_redis: aioredis.Redis = aioredis.from_url(_REDIS_URL, decode_responses=True)

_CACHE_PREFIX = "rag:cache:"
_CACHE_INDEX_PREFIX = "rag:cache_idx:"  # one Redis Set per ingestions scope
_CACHE_TTL = 60 * 60 * 24              # 24 h
_CACHE_THRESHOLD = 0.95                # cosine similarity required for a hit

# ── Router & shared models ─────────────────────────────────────────────────────

router = APIRouter(
    prefix="/rag",
    tags=["rag"],
    responses={404: {"description": "Not found"}},
)

embeddings = OpenAIEmbeddings()
text_splitter = getSpliter(strategy="recursive", embeddings=None)

# Pre-trained BM25 sparse encoder (MS MARCO — good general-purpose baseline)
bm25_encoder = BM25Encoder().default()

hybrid_retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings,
    sparse_encoder=bm25_encoder,
    index=index,
    top_k=10,
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
reranker = CrossEncoderReranker(model=_cross_encoder, top_n=5)
reorder = LongContextReorder()

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid_retriever,
)

query_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the user's question based only on the provided context. "
            "If the context doesn't contain enough information to answer, say so.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

# In-memory job tracker — fast status checks without hitting Supabase
_ingestion_jobs: dict[str, dict] = {}


# ── Semantic cache helpers ─────────────────────────────────────────────────────


def _scope_key(ingestions: list[str]) -> str:
    """Stable string that identifies the set of sources being searched."""
    return "|".join(sorted(ingestions)) if ingestions else "__all__"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


async def _cache_lookup(query_embedding: list[float], scope: str) -> dict | None:
    """Return cached payload if a semantically similar query exists, else None."""
    try:
        index_key = f"{_CACHE_INDEX_PREFIX}{scope}"
        cache_keys = await _redis.smembers(index_key)
        if not cache_keys:
            return None

        best_sim, best_payload = 0.0, None
        for key in cache_keys:
            raw = await _redis.get(key)
            if not raw:
                continue
            entry = json.loads(raw)
            sim = _cosine_similarity(query_embedding, entry["embedding"])
            if sim > best_sim:
                best_sim, best_payload = sim, entry

        return best_payload if best_sim >= _CACHE_THRESHOLD else None
    except Exception:
        logger.warning("Redis cache lookup failed — proceeding without cache")
        return None


async def _cache_save(
    query_embedding: list[float],
    scope: str,
    sources: list,
    answer: str,
) -> None:
    """Persist a new Q&A pair to the semantic cache."""
    try:
        entry_id = str(uuid.uuid4())
        cache_key = f"{_CACHE_PREFIX}{entry_id}"
        index_key = f"{_CACHE_INDEX_PREFIX}{scope}"

        payload = json.dumps({"embedding": query_embedding, "sources": sources, "answer": answer})
        await _redis.setex(cache_key, _CACHE_TTL, payload)
        await _redis.sadd(index_key, cache_key)
        await _redis.expire(index_key, _CACHE_TTL)
    except Exception:
        logger.warning("Redis cache save failed — result not cached")


# ── URL loader (BeautifulSoup) ─────────────────────────────────────────────────


def _load_url(url: str) -> list[Document]:
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return [Document(page_content=text, metadata={"source": url})]


# ── Supabase helpers ───────────────────────────────────────────────────────────


def _create_log(doc_id: str, source: str, file_type: str, ingested_at: str) -> None:
    supabase.table("rag_ingestion_logs").insert(
        {
            "doc_id": doc_id,
            "source": source,
            "file_type": file_type,
            "status": "queued",
            "ingested_at": ingested_at,
        }
    ).execute()


def _update_log(doc_id: str, updates: dict) -> None:
    supabase.table("rag_ingestion_logs").update(updates).eq("doc_id", doc_id).execute()


# ── Evaluation helpers ─────────────────────────────────────────────────────────


async def _score_doc_relevance(question: str, content: str) -> float:
    """Binary relevance judgment: 1.0 = relevant, 0.0 = not relevant."""
    msg = await llm.ainvoke(
        [
            SystemMessage(
                content=(
                    "You are a relevance judge. Given a query and a document chunk, "
                    "reply only YES if the document helps answer the query, or NO if it does not."
                )
            ),
            HumanMessage(content=f"Query: {question}\n\nDocument: {content[:600]}"),
        ]
    )
    return 1.0 if "YES" in msg.content.upper() else 0.0


async def _score_recall(question: str, context: str) -> float:
    """Approximate recall: how completely does the context cover the question (0–1)."""
    msg = await llm.ainvoke(
        [
            SystemMessage(
                content=(
                    "You are an evaluation assistant. Rate how completely the context "
                    "contains the information needed to fully answer the question. "
                    "0.0 = context is useless, 1.0 = context fully covers the answer. "
                    "Return only a decimal number."
                )
            ),
            HumanMessage(content=f"Question: {question}\n\nContext: {context[:2000]}"),
        ]
    )
    try:
        return round(max(0.0, min(1.0, float(msg.content.strip()))), 3)
    except ValueError:
        return 0.0


async def _score_hallucination(context: str, answer: str) -> float:
    """Hallucination rate: fraction of the answer not supported by context (0=grounded, 1=hallucinated)."""
    msg = await llm.ainvoke(
        [
            SystemMessage(
                content=(
                    "You are a hallucination detector. Rate what fraction of the answer "
                    "contains claims NOT supported by the provided context. "
                    "0.0 = fully grounded, 1.0 = fully hallucinated. Return only a decimal number."
                )
            ),
            HumanMessage(content=f"Context: {context[:2000]}\n\nAnswer: {answer}"),
        ]
    )
    try:
        return round(max(0.0, min(1.0, float(msg.content.strip()))), 3)
    except ValueError:
        return 0.0


async def _run_evaluation(question: str, docs: list, context: str, answer: str) -> dict:
    """Run precision, recall, and hallucination checks in parallel."""
    relevance_tasks = [_score_doc_relevance(question, doc.page_content) for doc in docs]
    results = await asyncio.gather(
        *relevance_tasks,
        _score_recall(question, context),
        _score_hallucination(context, answer),
    )
    relevance_scores = list(results[: len(docs)])
    recall_score = results[-2]
    hallucination_rate = results[-1]
    precision = (
        sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    )
    return {
        "retrieval_precision": round(precision, 3),
        "recall_score": recall_score,
        "hallucination_rate": hallucination_rate,
    }


# ── Source citation builder ────────────────────────────────────────────────────


def _build_sources(docs: list) -> list[dict]:
    """Return deduplicated source citations with chunk text, page, and confidence score."""
    seen: set = set()
    sources = []
    for doc in docs:
        key = doc.page_content[:120]
        if key in seen:
            continue
        seen.add(key)
        # CrossEncoderReranker writes relevance_score into metadata after reranking
        raw_score = doc.metadata.get("relevance_score")
        sources.append(
            {
                "chunk_text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page_number": doc.metadata.get(
                    "page"
                ),  # PyPDFLoader sets "page" (0-indexed)
                "confidence_score": (
                    round(float(raw_score), 4) if raw_score is not None else None
                ),
                "doc_id": doc.metadata.get("doc_id"),
            }
        )
    return sources


# ── Async ingestion background task ───────────────────────────────────────────


async def _run_ingestion(
    job_id: str,
    loader_fn,
    display_source: str,
    file_type: str,
    tmp_path: Optional[str],
) -> None:
    try:
        _ingestion_jobs[job_id]["status"] = "processing"
        await asyncio.to_thread(_update_log, job_id, {"status": "processing"})

        documents = await asyncio.to_thread(loader_fn)
        chunks = await asyncio.to_thread(text_splitter.split_documents, documents)

        ingested_at = datetime.now(timezone.utc).isoformat()
        texts = [chunk.page_content for chunk in chunks]
        total = len(chunks)
        metadatas = [
            {
                "doc_id": job_id,
                "source": display_source,
                "file_type": file_type,
                "ingested_at": ingested_at,
                "chunk_index": i,
                "total_chunks": total,
            }
            for i in range(total)
        ]

        # Upsert dense (OpenAI) + sparse (BM25) vectors into Pinecone
        await asyncio.to_thread(hybrid_retriever.add_texts, texts, metadatas=metadatas)

        completed_at = datetime.now(timezone.utc).isoformat()
        _ingestion_jobs[job_id].update(
            {
                "status": "completed",
                "chunks": len(chunks),
                "source": display_source,
                "completed_at": completed_at,
            }
        )
        await asyncio.to_thread(
            _update_log,
            job_id,
            {
                "status": "completed",
                "chunks": len(chunks),
                "completed_at": completed_at,
            },
        )

    except Exception as exc:
        _ingestion_jobs[job_id].update({"status": "failed", "error": str(exc)})
        await asyncio.to_thread(
            _update_log, job_id, {"status": "failed", "error": str(exc)}
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Routes ─────────────────────────────────────────────────────────────────────


@router.get("/get-files", status_code=200)
async def get_all_files():
    try:
        result = (
            supabase.table("rag_ingestion_logs")
            .select("*")
            .order("ingested_at", desc=True)
            .execute()
        )
        return {"message": "list fetched", "data": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch files: {str(e)}")


@router.post("/ingest/{action}", status_code=202)
async def ingest_document(
    background_tasks: BackgroundTasks,
    action: str,
    data: str = Form(...),
    file: Optional[UploadFile] = File(None),
):
    parsed = json.loads(data)
    tmp_path = None
    job_id = str(uuid.uuid4())

    if action == "url":
        url = parsed["url"]
        loader_fn = lambda: _load_url(url)
        display_source = url
        file_type = "url"

    else:
        if not file:
            raise HTTPException(status_code=400, detail="File is required")
        if file.content_type not in ["application/pdf", "text/plain"]:
            raise HTTPException(
                status_code=400, detail="Only PDF and text files are supported"
            )

        suffix = ".pdf" if file.content_type == "application/pdf" else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        loader_fn = (
            (lambda p=tmp_path: PyPDFLoader(p).load())
            if suffix == ".pdf"
            else (lambda p=tmp_path: TextLoader(p).load())
        )
        display_source = file.filename
        file_type = "pdf" if suffix == ".pdf" else "text"

    queued_at = datetime.now(timezone.utc).isoformat()
    _ingestion_jobs[job_id] = {
        "status": "queued",
        "source": display_source,
        "queued_at": queued_at,
    }

    await asyncio.to_thread(_create_log, job_id, display_source, file_type, queued_at)

    background_tasks.add_task(
        _run_ingestion, job_id, loader_fn, display_source, file_type, tmp_path
    )

    return {"job_id": job_id, "status": "queued", "source": display_source}


@router.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str):
    job = _ingestion_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


class QueryRequest(BaseModel):
    question: str
    evaluate: bool = False
    ingestions: list[str] = []    # doc_ids to scope the search; empty = all
    chat_id: Optional[str] = None  # omit to auto-create a new chat


@router.post("/query")
async def query_documents(request: QueryRequest):
    # Multi-query expansion over Pinecone hybrid retriever (dense + sparse)
    # multi_query = MultiQueryRetriever.from_llm(retriever=hybrid_retriever, llm=llm)

    pinecone_filter = (
        {"doc_id": {"$in": request.ingestions}} if request.ingestions else None
    )
    base = (
        PineconeHybridSearchRetriever(
            embeddings=embeddings,
            sparse_encoder=bm25_encoder,
            index=index,
            top_k=10,
            filter=pinecone_filter,
        )
        if pinecone_filter
        else hybrid_retriever
    )

    # Cross-encoder reranking (writes relevance_score into doc.metadata)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base,
    )

    docs = compression_retriever.invoke(request.question)

    if not docs:
        return {
            "answer": "No relevant documents found. Please upload documents first.",
            "sources": [],
        }

    docs = reorder.transform_documents(docs)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer the user's question based only on the provided context. "
                "If the context doesn't contain enough information to answer, say so.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    response = await (prompt | llm).ainvoke(
        {"context": context, "question": request.question}
    )

    # sources = _build_sources(docs)
    result: dict = {"answer": response.content, "sources": {}}

    if request.evaluate:
        result["evaluation"] = await _run_evaluation(
            request.question, docs, context, response.content
        )

    return result


@router.post("/query/stream")
async def query_documents_stream(request: QueryRequest):
    invoke_kwargs = (
        {"filter": {"doc_id": {"$in": request.ingestions}}} if request.ingestions else {}
    )
    scope = _scope_key(request.ingestions)

    # Resolve or create the chat session before streaming starts
    chat_id = request.chat_id or await asyncio.to_thread(
        create_chat, request.question[:200]
    )

    async def generate():
        try:
            # ── Semantic cache check ───────────────────────────────────────────
            query_embedding = await asyncio.to_thread(
                embeddings.embed_query, request.question
            )
            cached = await _cache_lookup(query_embedding, scope)

            if cached:
                logger.info("Semantic cache hit: %s", request.question[:60])
                yield f"data: {json.dumps({'type': 'sources', 'sources': cached['sources'], 'cached': True})}\n\n"
                for word in cached["answer"].split(" "):
                    yield f"data: {json.dumps({'type': 'token', 'token': word + ' '})}\n\n"
                    await asyncio.sleep(0)
                asyncio.ensure_future(
                    asyncio.to_thread(
                        save_messages,
                        chat_id, request.question, cached["answer"],
                        cached["sources"], request.ingestions,
                    )
                )
                yield f"data: {json.dumps({'type': 'done', 'cached': True, 'chat_id': chat_id})}\n\n"
                return

            # ── Cache miss: full retrieval + generation ────────────────────────
            docs = await asyncio.to_thread(
                compression_retriever.invoke, request.question, **invoke_kwargs
            )

            if not docs:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No relevant documents found. Please upload documents first.'})}\n\n"
                return

            docs = reorder.transform_documents(docs)
            context = "\n\n".join(doc.page_content for doc in docs)

            sources = _build_sources(docs)
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            full_answer = ""
            async for chunk in (query_prompt | llm).astream(
                {"context": context, "question": request.question}
            ):
                token = chunk.content
                if token:
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

            # Fire-and-forget: semantic cache + message persistence
            asyncio.ensure_future(_cache_save(query_embedding, scope, sources, full_answer))
            asyncio.ensure_future(
                asyncio.to_thread(
                    save_messages,
                    chat_id, request.question, full_answer, sources, request.ingestions,
                )
            )

            if request.evaluate:
                evaluation = await _run_evaluation(
                    request.question, docs, context, full_answer
                )
                yield f"data: {json.dumps({'type': 'evaluation', 'evaluation': evaluation})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'chat_id': chat_id})}\n\n"

        except Exception as exc:
            logger.exception("Stream generation failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
