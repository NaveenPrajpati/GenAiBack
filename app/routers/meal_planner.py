from typing import TypedDict, Optional, List, Literal, Annotated
from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from app.dependencies import get_current_user
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, END
from app.core.config import supabase
from datetime import date, timedelta, datetime
from langchain_core.output_parsers import StrOutputParser
from langgraph.types import interrupt, Command
import uuid

load_dotenv()


mealRouter = APIRouter(
    prefix="/meal-planner",
    tags=["summarizer"],
    responses={404: {"description": "Not found"}},
)


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# llm = ChatOllama(model="llama3:latest", temperature=0)
class QueryRequest(BaseModel):
    text: str
    plan_id: Optional[str] = None


class ProfileState(TypedDict):
    id: uuid
    user: uuid
    display_name: str
    diet: Literal["vegetarian", "non-vegetarian"]
    protein_target: int


class PlannerState(TypedDict, total=False):
    query: str
    intent: str
    profile: ProfileState
    user_id: str
    thread_id: str
    memory: dict
    plan_status: Optional[str]
    plan_id: Optional[str]
    suggestions: Optional[list]
    meal_slots: Optional[list]


graph = StateGraph(PlannerState)


class ProfileOutput(BaseModel):
    display_name: str
    diet: str
    protein_target: int


class LogOutput(BaseModel):
    receipe: str
    day_of_week: int
    meal_type: str
    conflict: bool
    suggestion: Optional[str]


class GroceryItem(BaseModel):
    plan_id: Optional[str] = None
    name: str
    qty: Optional[float] = None
    unit: Optional[str] = None
    checked: bool = False


class ReceipeOutput(BaseModel):
    name: str
    ingredients: list[GroceryItem] = []
    protein_g: Optional[int] = None
    prep_minutes: Optional[int] = None
    source_url: Optional[str] = None
    summary: Optional[str] = None


class IntentOutput(BaseModel):
    intent: str


today = date.today()


def get_monday(today: date = date.today()) -> str:
    # weekday(): Monday=0, Sunday=6
    monday = today - timedelta(days=today.weekday())
    return monday.isoformat()


week_start = get_monday()


def create_meal_plans():
    try:

        supabase.table("meal_plans").insert(
            {
                "week_start": week_start,
                "status": "draft",
            }
        ).execute()
    except Exception as e:
        print(e)


def classify_intent(state: PlannerState):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at extracting what is intent of text is it log, research , plan , query",
            ),
            ("human", "{text}"),
        ]
    )
    chain = prompt | llm.with_structured_output(IntentOutput)
    result: IntentOutput = chain.invoke({"text": state.get("query", "")})
    print(result)
    return {
        "intent": result.intent,
    }


class QueryOutput(BaseModel):
    meal_type: List[str]
    time: str


async def findRecepieInDb(
    recepie: Optional[str] = None, filters: Optional[QueryOutput] = None
):
    try:
        res = (
            supabase.table("recipes")
            .select("id, name, protein_g")
            .eq("name", recepie)
            .maybe_single()
            .execute()
        )
        print("findRecepieInDb result:", res)
        return res.data if res else None
    except Exception as e:
        print("findRecepieInDb error:", e)
        return None


async def InsertRecepieInDb(recepie: ReceipeOutput):
    try:
        res = (
            supabase.table("recipes")
            .insert(recepie.model_dump(mode="json", exclude_none=True))
            .execute()
        )
        print("InsertRecepieInDb result:", res.data)
        return res.data
    except Exception as e:
        print("InsertRecepieInDb error:", e)
        return None


async def insertRecepieInMealSlot(data: dict):
    try:
        res = (
            supabase.table("meal_slots")
            .insert(
                {
                    "plan_id": data["plan_id"],
                    "day_of_week": data["day_of_week"],
                    "meal_type": data["meal_type"],
                    "recipe_id": data["recipe_id"],
                    "recipe_name": data["recipe_name"],
                    "protein_g": data["protein_g"],
                }
            )
            .execute()
        )
        print("insertRecepieInMealSlot result:", res.data)
        return res.data
    except Exception as e:
        print("insertRecepieInMealSlot error:", e)
        return None


async def log_agent(state: PlannerState):
    findPrompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract the recipe, day_of_week (Monday=0), and meal_type from the text.\n"
                "User diet: {diet}. Disliked: {disliked}.\n"
                "If the requested dish conflicts with their diet, set conflict=true "
                "and suggest an alternative instead of extracting it.",
            ),
            ("human", "{text}"),
        ]
    )
    searchReceipePrompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at finding nutrients and cocking time , ingredients of given receipe",
            ),
            ("human", "{text}"),
        ]
    )

    profile = state.get("profile") or {}
    memory = state.get("memory") or {}

    chain = findPrompt | llm.with_structured_output(LogOutput)
    result: LogOutput = await chain.ainvoke(
        {
            "text": state["query"],
            "diet": profile.get("diet", "vegetarian"),
            "disliked": memory.get("disliked_dishes", []),
        }
    )
    print("log data", result)
    if result.conflict:
        return {
            "intent": "log",
            "log_status": "conflict",
            "suggestion": result.suggestion,
        }
    recepie = result.receipe
    recepiePresent = await findRecepieInDb(recepie)
    if not recepiePresent:
        print("receipe nnot presennt")
        chain2 = searchReceipePrompt | llm.with_structured_output(ReceipeOutput)
        result2: ReceipeOutput = await chain2.ainvoke({"text": recepie})
        print("receipe data", result2)
        inseted = await InsertRecepieInDb(result2)
        await insertRecepieInMealSlot(
            {
                "plan_id": state.get("plan_id"),
                "day_of_week": result.day_of_week,
                "meal_type": result.meal_type.lower(),
                "recipe_id": inseted[0]["id"] if inseted else None,
                "recipe_name": inseted[0]["name"],
                "protein_g": inseted[0]["protein_g"] if inseted else None,
            }
        )
    else:
        await insertRecepieInMealSlot(
            {
                "plan_id": state.get("plan_id"),
                "day_of_week": result.day_of_week,
                "meal_type": result.meal_type.lower(),
                "recipe_id": recepiePresent["id"],
                "recipe_name": recepiePresent["name"],
                "protein_g": recepiePresent["protein_g"],
            }
        )
    return {
        "intent": "log",
    }


async def findMealSlotsInDb(
    recepie: Optional[str] = None, filters: Optional[QueryOutput] = None
):
    try:
        res = (
            supabase.table("meal_slots")
            .select("day_of_week, meal_type, recipe_name, protein_g")
            .in_("meal_type", filters.meal_type)
            .execute()
        )
        print("findRecepieInDb result:", res)
        return res.data if res else None
    except Exception as e:
        print("findRecepieInDb error:", e)
        return None


async def query_agent(state: PlannerState):
    findPrompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at extracting information about having food , so tell me what types of meal (dinner,lunch,breakfast ) user want to know and for when (today,week)",
            ),
            ("human", "{text}"),
        ]
    )

    chain = findPrompt | llm.with_structured_output(QueryOutput)
    result: QueryOutput = await chain.ainvoke({"text": state["query"]})
    print("query data", result)
    slots = await findMealSlotsInDb(filters=result)
    return {
        "intent": "query",
        "meal_slots": slots or [],
    }


class Meals(BaseModel):
    meal_type: str
    recipe_name: str
    protein_g: int
    prep_minutes: int


class ResearchOutput(BaseModel):
    suggestions: List[Meals]


async def research_agent(state: PlannerState):
    suggestionPrompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at food nutrients and meals from which we can get (protiens, carbs , good fats etc) and what is best time (lunch, dinner, breakfast) to eat these meals "
                "User diet: {diet}. Disliked: {disliked}.\n",
            ),
            ("human", "{text}"),
        ]
    )
    profile = state.get("profile") or {}
    memory = state.get("memory") or {}
    chain = suggestionPrompt | llm.with_structured_output(ResearchOutput)
    result: ResearchOutput = await chain.ainvoke(
        {
            "text": state["query"],
            "diet": profile.get("diet", "vegetarian"),
            "disliked": memory.get("disliked_dishes", []),
        }
    )
    print("research data", result)
    return {
        "intent": "research",
        "suggestions": result.suggestions,
    }


class MealSlots(BaseModel):
    plan_id: Optional[str] = None
    day_of_week: int = 0
    meal_type: Literal["dinner", "lunch", "breakfast"]
    recipe_id: Optional[str] = None
    recipe_name: Optional[str] = None
    protein_g: Optional[int] = None


class PlanOutput(BaseModel):
    plan: list[MealSlots] = []


async def remember(user_id: str, key: str, value):
    try:
        supabase.table("memory").upsert(
            {"user_id": user_id, "key": key, "value": value},
            on_conflict="user_id,key",
        ).execute()
    except Exception as e:
        print("remember error:", e)


async def plan_agent(state: PlannerState):
    # LangGraph re-runs this node from the top on resume. Check Supabase first
    # so we reuse the original proposal instead of creating a duplicate.
    approval_id = None
    proposed = None
    try:
        existing_row = (
            supabase.table("approvals")
            .select("id, payload")
            .eq("thread_id", state["thread_id"])
            .eq("status", "pending")
            .maybe_single()
            .execute()
        )
        if existing_row and existing_row.data:
            approval_id = existing_row.data["id"]
            proposed = existing_row.data["payload"]["plan"]
    except Exception as e:
        print("approval lookup error:", e)

    if not approval_id:
        # First run: generate plan via LLM and insert approval.
        suggestionPrompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert at planning diet plan , so plan diet for week my week start from monday means monday is day of week 0 ,recipes for dinner, lunch, breakfast for all days of week along with protien in grams in each meal "
                    "User diet: {diet}. Disliked: {disliked}.\n",
                ),
                ("human", "{text}"),
            ]
        )
        profile = state.get("profile") or {}
        memory = state.get("memory") or {}
        chain = suggestionPrompt | llm.with_structured_output(PlanOutput)
        result: PlanOutput = await chain.ainvoke(
            {
                "text": state["query"],
                "diet": profile.get("diet", "vegetarian"),
                "disliked": memory.get("disliked_dishes", []),
            }
        )
        print("plan data", result)
        proposed = [slot.model_dump(mode="json") for slot in result.plan]

        try:
            res = (
                supabase.table("approvals")
                .insert(
                    {
                        "user_id": state["user_id"],
                        "thread_id": state["thread_id"],
                        "action_type": "save_plan",
                        "payload": {"week_start": week_start, "plan": proposed},
                        "status": "pending",
                    }
                )
                .execute()
            )
            approval_id = res.data[0]["id"] if res.data else None
        except Exception as e:
            print("approval insert error:", e)

    decision = interrupt(
        {
            "type": "save_plan",
            "approval_id": approval_id,
            "week_start": week_start,
            "plan": proposed,
        }
    )

    if decision != "approved":
        if approval_id:
            supabase.table("approvals").update(
                {"status": "rejected", "resolved_at": datetime.now().isoformat()}
            ).eq("id", approval_id).execute()
        return {"intent": "plan", "plan_status": "rejected"}

    # Approved: create the plan row, then insert all slots.
    plan_id = None
    try:
        plan_row = (
            supabase.table("meal_plans")
            .insert(
                {
                    "user": state["user_id"],
                    "week_start": week_start,
                    "status": "approved",
                }
            )
            .execute()
        )
        plan_id = plan_row.data[0]["id"] if plan_row.data else None
    except Exception as e:
        print("meal_plan insert error:", e)

    existing = (state.get("memory") or {}).get("liked_dishes", [])
    merged = list(
        dict.fromkeys(existing + [s["recipe_name"] for s in (proposed or [])])
    )
    await remember(state["user_id"], "liked_dishes", merged)

    for slot in proposed or []:
        try:
            supabase.table("meal_slots").insert(
                {
                    "plan_id": plan_id,
                    "day_of_week": slot["day_of_week"],
                    "meal_type": slot["meal_type"].lower(),
                    "recipe_name": slot["recipe_name"],
                    "protein_g": slot["protein_g"],
                }
            ).execute()
        except Exception as e:
            print("slot insert error:", e)

    if approval_id:
        supabase.table("approvals").update(
            {"status": "approved", "resolved_at": datetime.now().isoformat()}
        ).eq("id", approval_id).execute()

    return {"intent": "plan", "plan_status": "approved", "plan_id": plan_id}


async def load_memory(state: PlannerState):
    user_id = state["user_id"]
    profile = {}
    memory = {}

    try:
        prof = (
            supabase.table("profiles")
            .select("display_name, diet, protein_target")
            .eq("id", user_id)
            .maybe_single()
            .execute()
        )
        if prof and prof.data:
            profile = prof.data
    except Exception as e:
        print("load profile error:", e)

    try:
        rows = (
            supabase.table("memory")
            .select("key, value")
            .eq("user_id", user_id)
            .execute()
        )
        if rows and rows.data:
            memory = {r["key"]: r["value"] for r in rows.data}
    except Exception as e:
        print("load memory error:", e)

    return {"profile": profile, "memory": memory}


def decide_agent(state: PlannerState):

    if state.get("intent") == "log":
        return "log_agent"
    elif state.get("intent") == "query":
        return "query_agent"
    elif state.get("intent") == "research":
        return "research_agent"
    elif state.get("intent") == "plan":
        return "plan_agent"

    return END


graph.add_node("load_memory", load_memory)
graph.add_node("classify_intent", classify_intent)
graph.add_node("log_agent", log_agent)
graph.add_node("query_agent", query_agent)
graph.add_node("research_agent", research_agent)
graph.add_node("plan_agent", plan_agent)
graph.add_edge(START, "load_memory")
graph.add_edge("load_memory", "classify_intent")
graph.add_conditional_edges(
    "classify_intent",
    decide_agent,
    ["log_agent", "query_agent", "research_agent", "plan_agent", END],
)


@mealRouter.post("/query")
async def ask(
    body: QueryRequest,
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    agent = request.app.state.agent
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = await agent.ainvoke(
        {
            "query": body.text,
            "user_id": current_user["uid"],
            "thread_id": thread_id,
            "plan_id": body.plan_id,
        },
        config=config,
    )
    print("fineal--", result)

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return {
            "status": "needs_approval",
            "thread_id": thread_id,  # app sends this back to /approve
            "proposal": payload,
        }

    return {"status": "done", "result": result}


class ApproveRequest(BaseModel):
    thread_id: str
    decision: Literal["approved", "rejected"]


@mealRouter.post("/approve")
async def approve(
    body: ApproveRequest,
    request: Request,
    _auth: Annotated[dict, Depends(get_current_user)],
):
    agent = request.app.state.agent
    config = {"configurable": {"thread_id": body.thread_id}}

    snapshot = await agent.aget_state(config)
    if not snapshot or not snapshot.next:
        raise HTTPException(
            status_code=404,
            detail="No pending approval for this thread. The server may have restarted — please re-submit your plan request.",
        )

    result = await agent.ainvoke(Command(resume=body.decision), config=config)
    return {"status": "done", "result": result}


@mealRouter.get("/approve")
async def list_approvals(current_user: Annotated[dict, Depends(get_current_user)]):
    user_id = current_user["uid"]
    print("---", user_id)
    try:
        result = (
            supabase.table("approvals").select("*").eq("user_id", user_id).execute()
        )
        print(result)

        if not result.data:
            return {"status": "done", "message": "now approval found", "result": []}

        return {"status": "done", "result": result.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@mealRouter.get("/plans")
async def getPlans(current_user: Annotated[dict, Depends(get_current_user)]):
    user_id = current_user["uid"]
    print("---", user_id)
    try:
        result = supabase.table("meal_plans").select("*").eq("user", user_id).execute()
        print(result)

        if not result.data:
            return {"status": "done", "message": "plans not found", "result": []}

        return {"status": "done", "message": "plans fetched", "result": result.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@mealRouter.get("/profile")
async def getprofiles(current_user: Annotated[dict, Depends(get_current_user)]):
    user_id = current_user["uid"]
    print("---", user_id)
    try:
        result = supabase.table("profiles").select("*").eq("user", user_id).execute()
        print(result)

        if not result.data:
            return {"status": "done", "message": "profile not found", "result": []}

        return {"status": "done", "message": "profile fetched", "result": result.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ProfileUpdateBody(BaseModel):
    display_name: str
    diet: str
    protein_target: int


@mealRouter.patch("/profile/{profile}")
async def updateProfile(
    profile: uuid.UUID,
    body: ProfileUpdateBody,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    user_id = current_user["uid"]
    print("---", user_id)
    try:
        result = (
            supabase.table("profiles")
            .update(
                {
                    "display_name": body.display_name,
                    "diet": body.diet,
                    "protein_target": body.protein_target,
                }
            )
            .eq("id", profile)
            .execute()
        )
        print(result)

        if not result.data:
            return {"status": "done", "message": "profile not found", "result": []}

        return {"status": "done", "message": "profile fetched", "result": result.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class Trigger(BaseModel):
    id: str
    name: str
    schedule: str
    action_type: str
    enabled: bool = True
    last_run_at: Optional[datetime] = None


@mealRouter.post("/toggle-trigger")
async def toggle_trigger(current_user: Annotated[dict, Depends(get_current_user)]):
    user_id = current_user["uid"]

    try:
        result = supabase.table("triggers").select("*").eq("user_id", user_id).execute()
        if result and result.data:
            for t in result.data or []:
                supabase.table("triggers").update({"enabled": not t["enabled"]}).eq(
                    "id", t["id"]
                ).execute()
        else:
            supabase.table("triggers").insert(
                {
                    "user_id": user_id,
                    "name": "plan my schedule",
                    "schedule": "30 18 * * 0",
                    "action_type": "schedule",
                    "enabled": True,
                    "last_run_at": None,
                }
            ).execute()

        return {"status": "done"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def run_triggers(agent):
    print("This job runs every sunday on 6:30 pm")
    now = datetime.now()
    try:
        result = supabase.table("triggers").select("*").eq("enabled", True).execute()
        for t in result.data or []:
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            await agent.ainvoke(
                {
                    "query": "Plan my meals for next week",
                    "user_id": t["user_id"],
                    "thread_id": thread_id,
                },
                config=config,
            )
            supabase.table("triggers").update({"last_run_at": now.isoformat()}).eq(
                "id", t["id"]
            ).execute()
    except Exception as e:
        print(e)
