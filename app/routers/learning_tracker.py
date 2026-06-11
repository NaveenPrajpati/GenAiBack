from typing import TypedDict, Optional, List, Literal, Annotated
from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from app.dependencies import get_current_user
from app.core.llm import llm
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, END
from app.core.config import supabase
from datetime import date, timedelta, datetime, timezone
from langgraph.types import interrupt, Command
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
import logging
import uuid
import os
import httpx
from app.database import get_db

load_dotenv()

logger = logging.getLogger(__name__)


mealRouter = APIRouter(
    prefix="/learning",
    tags=["learning"],
    responses={404: {"description": "Not found"}},
)


class QueryRequest(BaseModel):
    text: str
    roadmapId: Optional[str] = None
    thread_id: Optional[str] = None


class TopicNode(BaseModel):
    id: str
    order: int
    title: str
    description: str
    prerequisites: List[str]
    estimated_hours: Optional[int] = None
    resources: Optional[List[str]] = None
    covered: Optional[bool] = False


class RoadmapOutput(BaseModel):
    id: str
    title: str
    summary: str
    status: Literal["active", "archived", "completed"] = "archived"
    userid: str
    total_estimated_hours: Optional[int] = None
    stages: List[str]
    topics: List[TopicNode]


class LearningState(TypedDict, total=False):
    query: str
    intent: str
    current_user: dict
    userId: str
    thread_id: str
    memory: dict
    plan_status: Optional[str]
    log_status: Optional[str]
    conflict: Optional[dict]
    roadmapId: Optional[str]
    suggestions: Optional[list]
    meal_slots: Optional[list]
    roadmap: Optional[RoadmapOutput]
    roadmap_status: Optional[str]
    next_topic: str


graph = StateGraph(LearningState)


class GroceryItem(BaseModel):
    roadmapId: Optional[str] = None
    name: str
    qty: Optional[float] = None
    unit: Optional[str] = None
    checked: bool = False


class RecipeOutput(BaseModel):
    name: str
    ingredients: list[GroceryItem] = []
    protein_g: Optional[int] = None
    prep_minutes: Optional[int] = None
    source_url: Optional[str] = None
    summary: Optional[str] = None


class IntentOutput(BaseModel):
    intent: str


def get_monday(today: Optional[date] = None) -> str:
    # weekday(): Monday=0, Sunday=6
    if today is None:
        today = date.today()
    monday = today - timedelta(days=today.weekday())
    return monday.isoformat()


# Shared prompt: given a recipe name, ask the LLM to fill in nutrients,
# cooking time and ingredients. Used by both roadmap_agent and conflict resolution.
searchRecipePrompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at finding nutrients and cooking time , ingredients of given recipe",
        ),
        ("human", "{text}"),
    ]
)


async def classify_intent(state: LearningState):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Classify the user message into one intent:\n"
                "- create_roadmap: user asks what to study \n"
                "- explain: user asks for an explanation of a concept, topic, or step\n"
                "- quiz: user requests a quiz or test on a topic\n"
                "- submit_quiz: user is submitting answers to a quiz for evaluation\n"
                "- find_resources: user asks for resources, links, books, or materials on a topic\n"
                "- update_progress: user marks progress, completes a step, or logs learning activity\n"
                "- query_roadmap: user asks what to do next in their learning roadmap or user wants to view or check the current state of their roadmap\n"
                "- modify_roadmap: user wants to change, restructure, or regenerate their roadmap\n"
                "Reply with one word only: create_roadmap, explain, quiz, submit_quiz, find_resources, update_progress, query_roadmap, or modify_roadmap.",
            ),
            ("human", "{text}"),
        ]
    )
    chain = prompt | llm.with_structured_output(IntentOutput)
    result: IntentOutput = await chain.ainvoke({"text": state.get("query", "")})
    logger.info("%s", result)
    return {
        "intent": result.intent,
    }


class ProgressOutput(BaseModel):
    nextTopic: str
    covedTopic: list[str]
    remainingTopics: list[list]
    topicsToUpdate: list[str]


class TutorOutput(BaseModel):
    topic: str


class UpdateProgressOutput(BaseModel):
    topic: str
    covered: Optional[bool] = False


async def findRecipeInDb(
    recipe: Optional[str] = None, filters: Optional[TutorOutput] = None
):
    try:
        res = (
            supabase.table("recipes")
            .select("id, name, protein_g")
            .ilike("name", recipe)
            .maybe_single()
            .execute()
        )
        logger.info("findRecipeInDb result: %s", res)
        return res.data if res else None
    except Exception as e:
        logger.error("findRecipeInDb error: %s", e)
        return None


async def insertRecipeInDb(recipe: RecipeOutput):
    try:
        res = (
            supabase.table("recipes")
            .insert(recipe.model_dump(mode="json", exclude_none=True))
            .execute()
        )
        logger.info("insertRecipeInDb result: %s", res.data)
        return res.data
    except Exception as e:
        logger.error("insertRecipeInDb error: %s", e)
        return None


async def insertRecipeInMealSlot(data: dict):
    try:
        # Upsert (not insert) so re-logging the same plan/day/meal_type replaces
        # the slot instead of creating a duplicate row. Requires a unique
        # constraint on (roadmapId, day_of_week, meal_type) — see migrations/.
        res = (
            supabase.table("meal_slots")
            .upsert(
                {
                    "roadmapId": data["roadmapId"],
                    "day_of_week": data["day_of_week"],
                    "meal_type": data["meal_type"],
                    "recipe_id": data["recipe_id"],
                    "recipe_name": data["recipe_name"],
                    "protein_g": data["protein_g"],
                },
                on_conflict="roadmapId,day_of_week,meal_type",
            )
            .execute()
        )
        logger.info("insertRecipeInMealSlot result: %s", res.data)
        return res.data
    except Exception as e:
        logger.error("insertRecipeInMealSlot error: %s", e)
        return None


async def log_recipe_to_slot(
    roadmapId: str, recipe_name: str, day_of_week: int, meal_type: str
):
    """Find-or-create a recipe by name, then attach it to a meal slot.

    Shared by roadmap_agent and the conflict-resolution endpoint so the
    find/enrich/insert logic lives in exactly one place.
    """
    recipe_present = await findRecipeInDb(recipe_name)
    if recipe_present:
        recipe_id = recipe_present["id"]
        name = recipe_present["name"]
        protein = recipe_present["protein_g"]
    else:
        logger.info("recipe not present: %s", recipe_name)
        chain = searchRecipePrompt | llm.with_structured_output(RecipeOutput)
        details: RecipeOutput = await chain.ainvoke({"text": recipe_name})
        logger.info("recipe data %s", details)
        inserted = await insertRecipeInDb(details)
        recipe_id = inserted[0]["id"] if inserted else None
        name = inserted[0]["name"] if inserted else recipe_name
        protein = inserted[0]["protein_g"] if inserted else None

    return await insertRecipeInMealSlot(
        {
            "roadmapId": roadmapId,
            "day_of_week": day_of_week,
            "meal_type": meal_type.lower(),
            "recipe_id": recipe_id,
            "recipe_name": name,
            "protein_g": protein,
        }
    )


async def insertRoadmapToDb(
    roadmap: RoadmapOutput, userId: Optional[str] = None
) -> Optional[str]:
    try:
        doc = roadmap.model_dump()
        doc["userId"] = userId
        doc["createdAt"] = datetime.now(timezone.utc).isoformat()
        res = await get_db()["roadmaps"].insert_one(doc)
        logger.info("insertRoadmapToDb inserted: %s", res.inserted_id)
        return str(res.inserted_id)
    except Exception as e:
        logger.error("insertRoadmapToDb error: %s", e)
        return None


async def roadmap_agent(state: LearningState):
    from bson import ObjectId

    is_modify = state.get("intent") == "modify_roadmap"
    existingApproval = await get_db()["approvals"].find_one(
        {"threadId": state.get("thread_id"), "status": "pending"}
    )
    if existingApproval:
        approval_id = str(existingApproval["_id"])
        action_type = existingApproval.get(
            "action", "update_roadmap" if is_modify else "save_roadmap"
        )
        result = RoadmapOutput(**existingApproval["payload"])
        logger.info("roadmap approval already exists: %s", approval_id)
    else:
        if is_modify:
            # Fetch the existing roadmap so the LLM can operate on it
            existing_roadmap = None
            roadmapId = state.get("roadmapId")
            if roadmapId:
                try:
                    from bson import ObjectId

                    doc = await get_db()["roadmaps"].find_one(
                        {"_id": ObjectId(roadmapId)}
                    )
                    if doc:
                        doc.pop("_id", None)
                        existing_roadmap = doc
                except Exception as e:
                    logger.error("roadmap fetch error: %s", e)

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert curriculum designer. The user wants to modify an existing learning roadmap.\n"
                        "Apply the requested change (add topic, remove topic, reorder, adjust hours, update resources, etc.).\n"
                        "Return the full updated roadmap — keep all unchanged topics intact.\n"
                        "Maintain correct order values and prerequisite links after any structural change.\n"
                        "Existing roadmap:\n{existing_roadmap}",
                    ),
                    ("human", "{text}"),
                ]
            )
            chain = prompt | llm.with_structured_output(RoadmapOutput)
            result: RoadmapOutput = await chain.ainvoke(
                {"text": state["query"], "existing_roadmap": existing_roadmap or "none"}
            )
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert curriculum designer and learning path architect.\n"
                        "Given a topic the user wants to learn, produce a complete, sequenced roadmap:\n"
                        "1. Break the subject into ordered topics (order field starts at 1).\n"
                        "2. For each topic list its prerequisites by title — only topics that appear earlier in the list.\n"
                        "3. Group topics into broad stages (e.g. Foundations, Intermediate, Advanced).\n"
                        "4. Estimate realistic study hours per topic and a total.\n"
                        "5. Suggest 1-2 free learning resources (course names, docs, book titles) per topic.\n"
                        "Personalize based on the exact subject in the user query. Be specific and practical.",
                    ),
                    ("human", "{text}"),
                ]
            )
            chain = prompt | llm.with_structured_output(RoadmapOutput)
            result: RoadmapOutput = await chain.ainvoke({"text": state["query"]})
        logger.info("roadmap_agent result: %s", result)

        action_type = "update_roadmap" if is_modify else "save_roadmap"

        approval_id = None
        try:

            res = await get_db()["approvals"].insert_one(
                {
                    "userId": state.get("userId"),
                    "threadId": state.get("thread_id"),
                    "action": action_type,
                    "payload": result.model_dump(),
                    "status": "pending",
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                }
            )
            approval_id = str(res.inserted_id)
            logger.info("roadmap pending approval created: %s", approval_id)
        except Exception as e:
            logger.error("roadmap approval insert error: %s", e)

    # Pause — send roadmap + approval_id to client for review
    decision = interrupt(
        {"type": action_type, "approvalId": approval_id, "roadmap": result.model_dump()}
    )

    if decision != "approved":
        if approval_id:
            try:
                await get_db()["approvals"].update_one(
                    {"_id": ObjectId(approval_id)},
                    {
                        "$set": {
                            "status": "rejected",
                            "resolvedAt": datetime.now(timezone.utc).isoformat(),
                        }
                    },
                )
            except Exception as e:
                logger.error("approval reject update error: %s", e)
        return {"intent": state.get("intent"), "roadmap_status": "rejected"}

    # Approved — update approval status then persist roadmap
    if approval_id:
        try:
            await get_db()["approvals"].update_one(
                {"_id": ObjectId(approval_id)},
                {
                    "$set": {
                        "status": "approved",
                        "resolvedAt": datetime.now(timezone.utc).isoformat(),
                    }
                },
            )
        except Exception as e:
            logger.error("approval approve update error: %s", e)

    if is_modify and state.get("roadmapId"):
        try:
            await get_db()["roadmaps"].replace_one(
                {"_id": ObjectId(state["roadmapId"])},
                {
                    **result.model_dump(),
                    "userId": state.get("userId"),
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                },
            )
            saved_roadmapId = state["roadmapId"]
        except Exception as e:
            logger.error("roadmap update error: %s", e)
            saved_roadmapId = None
    else:
        saved_roadmapId = await insertRoadmapToDb(result, state.get("userId"))

    return {
        "intent": state.get("intent"),
        "roadmap_status": "approved",
        "roadmapId": saved_roadmapId,
        "roadmap": result.model_dump(),
    }


async def tutor_agent(state: LearningState):

    if state.get("intent") == "whats_next":
        existing_roadmap = None
        roadmapId = state.get("roadmapId")
        if roadmapId:
            try:
                from bson import ObjectId

                doc = await get_db()["roadmaps"].find_one({"_id": ObjectId(roadmapId)})
                if doc:
                    doc.pop("_id", None)
                    existing_roadmap = doc
            except Exception as e:
                logger.error("roadmap fetch error: %s", e)
            findPrompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert curriculum designer. The user wants to know that what should he or she learn next so check existing roadmap what user coverd till now suggest him what should he learn next according to roadmap.\n",
                        "Existing roadmap:\n{existing_roadmap}",
                    ),
                    ("human", "{text}"),
                ]
            )

        chain = findPrompt | llm.with_structured_output(TutorOutput)
        result: TutorOutput = await chain.ainvoke({"text": state["query"]})
        logger.info("query data %s", result)

        return {
            "next_topic": result.topic,
        }


async def udateTopic(topic: TopicNode, userId: Optional[str] = None) -> Optional[str]:
    try:
        doc = topic.model_dump()
        doc["id"] = topic.id
        res = await get_db()["roadmaps"].update_one(doc)
        logger.info("insertRoadmapToDb inserted: %s", res.inserted_id)
        return str(res.inserted_id)
    except Exception as e:
        logger.error("insertRoadmapToDb error: %s", e)
        return None


async def progress_agent(state: LearningState):
    if state.get("intent") == "query_roadmap":
        existing_roadmap = None
        roadmapId = state.get("roadmapId")
        if roadmapId:
            try:
                from bson import ObjectId

                doc = await get_db()["roadmaps"].find_one({"_id": ObjectId(roadmapId)})
                if doc:
                    doc.pop("_id", None)
                    existing_roadmap = doc
            except Exception as e:
                logger.error("roadmap fetch error: %s", e)
            findPrompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert curriculum designer. The user wants to know that what should he or she learn next so check existing roadmap what user coverd till now suggest him what should he learn next according to roadmap.\n",
                        "Existing roadmap:\n{existing_roadmap}",
                    ),
                    ("human", "{text}"),
                ]
            )

        chain = findPrompt | llm.with_structured_output(ProgressOutput)
        result: ProgressOutput = await chain.ainvoke({"text": state["query"]})
        logger.info("query data %s", result)

        return {
            "next_topic": result.topic,
        }
    elif state.get("intent") == "update_progress":
        findPrompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert curriculum designer. you have my this roadmap find or extract of topic for user is talking about and what does user want to update does is coverd it means covered:True of  \n",
                    "Existing roadmap:\n{existing_roadmap}",
                ),
                ("human", "{text}"),
            ]
        )

        chain = findPrompt | llm.with_structured_output(ProgressOutput)
        result: ProgressOutput = await chain.ainvoke({"text": state["query"]})

        udateTopic()
        logger.info("query data %s", result)
        return {
            "roadmap": result.topic,
        }


class NutritionData(BaseModel):
    calories: float = 0
    protein_g: float = 0
    carbs_g: float = 0
    fat_g: float = 0


class ResearchMeal(BaseModel):
    meal_type: str
    recipe_name: str
    ingredients: list[str]
    prep_minutes: int
    nutrition: Optional[NutritionData] = None


class ResearchOutput(BaseModel):
    suggestions: List[ResearchMeal]


@tool
async def get_nutrition(ingredients: list[str]) -> dict:
    """Fetch accurate nutrition data for a recipe from the Edamam API.
    Call this for every meal you suggest.
    Pass ingredients with quantities e.g. ['200g chicken breast', '1 cup rice'].
    Returns calories, protein_g, carbs_g, fat_g for the full recipe."""
    app_id = os.getenv("EDAMAM_APP_ID", "")
    app_key = os.getenv("EDAMAM_APP_KEY", "")
    if not app_id or not app_key:
        return {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://api.edamam.com/api/nutrition-details",
                params={"app_id": app_id, "app_key": app_key},
                json={"ingr": ingredients},
            )
            if resp.status_code != 200:
                logger.error(f"Edamam error {resp.status_code}: {resp.text}")
                return {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}
            data = resp.json()
            n = data.get("totalNutrients", {})
            return {
                "calories": round(data.get("calories", 0), 1),
                "protein_g": round(n.get("PROCNT", {}).get("quantity", 0), 1),
                "carbs_g": round(n.get("CHOCDF", {}).get("quantity", 0), 1),
                "fat_g": round(n.get("FAT", {}).get("quantity", 0), 1),
            }
    except Exception as e:
        logger.error(f"nutrition API error: {e}")
        return {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}


_research_tools = [get_nutrition]
_research_tool_node = ToolNode(_research_tools)
_research_llm = llm.bind_tools(_research_tools)


async def research_agent(state: LearningState):
    current_user = state.get("current_user") or {}
    memory = state.get("memory") or {}

    messages = [
        SystemMessage(
            content=(
                "You are a nutrition expert. Suggest meals matching the user's request.\n"
                "For EVERY meal you suggest, call get_nutrition with its ingredient list "
                "(with quantities e.g. '200g chicken breast') to get accurate nutrition data.\n"
                f"User diet: {current_user.get('diet', 'vegetarian')}. "
                f"Disliked: {memory.get('disliked_dishes', [])}."
            )
        ),
        HumanMessage(content=state["query"]),
    ]

    # Tool-calling loop: LLM calls get_nutrition for each meal it suggests
    while True:
        response = await _research_llm.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        tool_results = await _research_tool_node.ainvoke({"messages": messages})
        messages.extend(tool_results["messages"])

    # Final pass: extract structured output from the enriched conversation
    structured: ResearchOutput = await llm.with_structured_output(
        ResearchOutput
    ).ainvoke(
        messages
        + [
            HumanMessage(
                content="Return all meal suggestions with their nutrition data in structured format."
            )
        ]
    )
    logger.info("research data %s", structured)
    return {
        "intent": "research",
        "suggestions": [m.model_dump() for m in structured.suggestions],
    }


class MealSlots(BaseModel):
    roadmapId: Optional[str] = None
    day_of_week: int = 0
    meal_type: Literal["dinner", "lunch", "breakfast"]
    recipe_id: Optional[str] = None
    recipe_name: Optional[str] = None
    protein_g: Optional[int] = None


class PlanOutput(BaseModel):
    plan: list[MealSlots] = []


async def remember(userId: str, key: str, value):
    try:
        supabase.table("memory").upsert(
            {"userId": userId, "key": key, "value": value},
            on_conflict="userId,key",
        ).execute()
    except Exception as e:
        logger.error("remember error: %s", e)


async def get_disliked_dishes(userId: str) -> list:
    """Return the user's current disliked_dishes list (empty on miss/error)."""
    try:
        row = (
            supabase.table("memory")
            .select("value")
            .eq("userId", userId)
            .eq("key", "disliked_dishes")
            .maybe_single()
            .execute()
        )
        return list(row.data["value"]) if row and row.data else []
    except Exception as e:
        logger.error("get_disliked_dishes error: %s", e)
        return []


async def add_disliked_dish(userId: str, dish: str) -> list:
    """Append a dish to the user's disliked_dishes memory (de-duplicated)."""
    existing = await get_disliked_dishes(userId)
    merged = list(dict.fromkeys(existing + [dish]))
    await remember(userId, "disliked_dishes", merged)
    return merged


async def remove_disliked_dish(userId: str, dish: str) -> list:
    """Remove a dish from the user's disliked_dishes memory."""
    existing = await get_disliked_dishes(userId)
    merged = [d for d in existing if d != dish]
    await remember(userId, "disliked_dishes", merged)
    return merged


async def build_grocery_list(roadmapId: str) -> list:
    """Aggregate ingredients across every meal slot in a plan into a shopping
    list. Quantities accumulate per (ingredient name, unit), counting each slot
    separately so a dish eaten N times contributes N times."""
    try:
        slots_res = (
            supabase.table("meal_slots")
            .select("recipe_id, recipe_name")
            .eq("roadmapId", roadmapId)
            .execute()
        )
        slots = slots_res.data or []
    except Exception as e:
        logger.error("build_grocery_list slots error: %s", e)
        return []

    ids = list({s["recipe_id"] for s in slots if s.get("recipe_id")})
    names = list({s["recipe_name"] for s in slots if s.get("recipe_name")})

    # Slots from the log path carry a recipe_id; plan-generated slots only have a
    # recipe_name. Look up ingredients by both so either kind resolves.
    by_id: dict = {}
    by_name: dict = {}
    try:
        if ids:
            r = (
                supabase.table("recipes")
                .select("id, name, ingredients")
                .in_("id", ids)
                .execute()
            )
            for rec in r.data or []:
                by_id[rec["id"]] = rec.get("ingredients") or []
        if names:
            r = (
                supabase.table("recipes")
                .select("id, name, ingredients")
                .in_("name", names)
                .execute()
            )
            for rec in r.data or []:
                by_name[rec["name"]] = rec.get("ingredients") or []
    except Exception as e:
        logger.error("build_grocery_list recipes error: %s", e)

    agg: dict = {}
    for s in slots:
        ingredients = by_id.get(s.get("recipe_id")) if s.get("recipe_id") else None
        if ingredients is None:
            ingredients = by_name.get(s.get("recipe_name"), [])
        for ing in ingredients:
            name = (ing.get("name") or "").strip()
            if not name:
                continue
            unit = ing.get("unit")
            qty = ing.get("qty")
            key = (name.lower(), unit)
            entry = agg.setdefault(
                key, {"name": name, "qty": None, "unit": unit, "checked": False}
            )
            if qty is not None:
                entry["qty"] = (entry["qty"] or 0) + qty

    return sorted(agg.values(), key=lambda x: x["name"].lower())


async def plan_agent(state: LearningState):
    # LangGraph re-runs this node from the top on resume. Check Supabase first
    # so we reuse the original proposal instead of creating a duplicate.
    week_start = get_monday()
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
        logger.error("approval lookup error: %s", e)

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
        current_user = state.get("current_user") or {}
        memory = state.get("memory") or {}
        chain = suggestionPrompt | llm.with_structured_output(PlanOutput)
        result: PlanOutput = await chain.ainvoke(
            {
                "text": state["query"],
                "diet": current_user.get("diet", "vegetarian"),
                "disliked": memory.get("disliked_dishes", []),
            }
        )
        logger.info("plan data %s", result)
        proposed = [slot.model_dump(mode="json") for slot in result.plan]

        try:
            res = (
                supabase.table("approvals")
                .insert(
                    {
                        "userId": state["userId"],
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
            logger.error("approval insert error: %s", e)

    is_update = state.get("intent") == "update" or bool(state.get("roadmapId"))
    action_type = "update_plan" if is_update else "save_plan"

    decision = interrupt(
        {
            "type": action_type,
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
        return {"intent": state.get("intent", "plan"), "plan_status": "rejected"}

    # Approved: for update reuse the existing plan row; for new plan create one.
    roadmapId = state.get("roadmapId") if is_update else None
    if not roadmapId:
        try:
            plan_row = (
                supabase.table("meal_plans")
                .insert(
                    {
                        "user": state["userId"],
                        "week_start": week_start,
                        "status": "approved",
                    }
                )
                .execute()
            )
            roadmapId = plan_row.data[0]["id"] if plan_row.data else None
        except Exception as e:
            logger.error("meal_plan insert error: %s", e)
    else:
        # Clear existing slots so we start fresh with the new proposal.
        try:
            supabase.table("meal_slots").delete().eq("roadmapId", roadmapId).execute()
        except Exception as e:
            logger.error("meal_slots clear error: %s", e)

    existing = (state.get("memory") or {}).get("liked_dishes", [])
    merged = list(
        dict.fromkeys(existing + [s["recipe_name"] for s in (proposed or [])])
    )
    await remember(state["userId"], "liked_dishes", merged)

    for slot in proposed or []:
        try:
            supabase.table("meal_slots").upsert(
                {
                    "roadmapId": roadmapId,
                    "day_of_week": slot["day_of_week"],
                    "meal_type": slot["meal_type"].lower(),
                    "recipe_name": slot["recipe_name"],
                    "protein_g": slot["protein_g"],
                },
                on_conflict="roadmapId,day_of_week,meal_type",
            ).execute()
        except Exception as e:
            logger.error("slot insert error: %s", e)

    if approval_id:
        supabase.table("approvals").update(
            {"status": "approved", "resolved_at": datetime.now().isoformat()}
        ).eq("id", approval_id).execute()

    return {
        "intent": state.get("intent", "plan"),
        "plan_status": "approved",
        "roadmapId": roadmapId,
    }


async def load_memory(state: LearningState):
    userId = state["userId"]
    memory = {}

    try:
        memories = get_db()["memories"]
        result = memories.find_one({userId: state["userId"]})
        if result:
            print(result)
            # memory = {r["key"]: r["value"] for r in rows.data}
    except Exception as e:
        logger.error("load memory error: %s", e)

    return {"memory": memory}


def decide_agent(state: LearningState):
    intent = state.get("intent")
    if intent == "create_roadmap":
        return "roadmap_agent"
    elif intent == "whats_next":
        return "progress_agent"
    elif intent == "explain":
        return "tutor_agent"
    # elif intent == "quiz":
    #     return "research_agent"
    # elif intent == "submit_quiz":
    #     return "research_agent"
    # elif intent == "find_resources":
    #     return "research_agent"
    elif intent == "update_progress":
        return "roadmap_agent"
    elif intent == "query_roadmap":
        return "tutor_agent"
    elif intent == "modify_roadmap":
        return "plan_agent"
    return END


graph.add_node("load_memory", load_memory)
graph.add_node("classify_intent", classify_intent)
graph.add_node("roadmap_agent", roadmap_agent)
graph.add_node("progress_agent", progress_agent)
graph.add_node("tutor_agent", tutor_agent)
graph.add_node("plan_agent", plan_agent)
graph.add_edge(START, "classify_intent")
# graph.add_edge("load_memory", "classify_intent")
graph.add_conditional_edges(
    "classify_intent",
    decide_agent,
    ["roadmap_agent", "tutor_agent", "plan_agent", END],
)


async def verify_plan_ownership(roadmapId: str, userId: str) -> bool:
    """Return True if the plan exists and belongs to the user."""
    try:
        res = (
            supabase.table("meal_plans")
            .select("id")
            .eq("id", roadmapId)
            .eq("user", userId)
            .maybe_single()
            .execute()
        )
        return bool(res and res.data)
    except Exception as e:
        logger.error("verify_plan_ownership error: %s", e)
        return False


@mealRouter.post("/query")
async def ask(
    body: QueryRequest,
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    agent = request.app.state.agent

    thread_id = body.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    _excluded = {"_id", "expires_at", "password_hash"}
    user_data = {k: v for k, v in current_user.items() if k not in _excluded}
    result = await agent.ainvoke(
        {
            "query": body.text,
            "userId": current_user["uid"],
            "thread_id": thread_id,
            "roadmapId": body.roadmapId,
            "current_user": user_data,
        },
        config=config,
    )
    logger.info("final -- %s", result)

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


@mealRouter.post("/approvals")
async def approve(
    body: ApproveRequest,
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    agent = request.app.state.agent
    config = {"configurable": {"thread_id": body.thread_id}}

    # The thread/approval must belong to the caller (prevents IDOR where a user
    # approves or rejects someone else's pending plan by guessing the thread_id).
    approval = None
    try:
        approval = await get_db()["approvals"].find_one(
            {"threadId": body.thread_id, "status": "pending"}
        )
        logger.info("approval found: %s", approval)
    except Exception as e:
        logger.error("approval ownership lookup error: %s", e)

    if not approval:
        raise HTTPException(
            status_code=404, detail="No pending approval for this thread."
        )
    if approval["userId"] != current_user["uid"]:
        raise HTTPException(
            status_code=403, detail="You do not have access to this approval."
        )

    snapshot = await agent.aget_state(config)
    if not snapshot or not snapshot.next:
        raise HTTPException(
            status_code=404,
            detail="No pending approval for this thread. The server may have restarted — please re-submit your plan request.",
        )

    result = await agent.ainvoke(Command(resume=body.decision), config=config)
    return {"status": "done", "result": result}


class ResolveConflictRequest(BaseModel):
    roadmapId: str
    recipe: str  # the suggested (or chosen) dish to act on
    day_of_week: int
    meal_type: Literal["dinner", "lunch", "breakfast"]
    decision: Literal["accept", "reject"]


@mealRouter.post("/resolve-conflict")
async def resolve_conflict(
    body: ResolveConflictRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Finish a 'conflict' from roadmap_agent.

    accept → log the suggested recipe into the slot.
    reject → record the suggestion as a disliked dish so it isn't offered again.
    """
    if not await verify_plan_ownership(body.roadmapId, current_user["uid"]):
        raise HTTPException(
            status_code=403, detail="You do not have access to this plan."
        )

    if body.decision == "reject":
        await add_disliked_dish(current_user["uid"], body.recipe)
        return {"status": "done", "log_status": "rejected"}

    slot = await log_recipe_to_slot(
        body.roadmapId, body.recipe, body.day_of_week, body.meal_type
    )
    return {"status": "done", "log_status": "logged", "slot": slot}


class DislikeRequest(BaseModel):
    dish: str


@mealRouter.get("/disliked")
async def list_disliked(current_user: Annotated[dict, Depends(get_current_user)]):
    return {"status": "done", "result": await get_disliked_dishes(current_user["uid"])}


@mealRouter.post("/disliked")
async def add_disliked(
    body: DislikeRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    merged = await add_disliked_dish(current_user["uid"], body.dish)
    return {"status": "done", "result": merged}


@mealRouter.delete("/disliked")
async def delete_disliked(
    body: DislikeRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    merged = await remove_disliked_dish(current_user["uid"], body.dish)
    return {"status": "done", "result": merged}


@mealRouter.get("/meal-slots/{roadmapId}")
async def get_meal_slots(
    roadmapId: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    if not await verify_plan_ownership(roadmapId, current_user["uid"]):
        raise HTTPException(
            status_code=403, detail="You do not have access to this plan."
        )
    try:
        res = (
            supabase.table("meal_slots")
            .select("id, day_of_week, meal_type, recipe_id, recipe_name, protein_g")
            .eq("roadmapId", roadmapId)
            .order("day_of_week")
            .order("meal_type")
            .execute()
        )
        return {"status": "done", "roadmapId": roadmapId, "slots": res.data or []}
    except Exception as e:
        logger.error("get_meal_slots error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to fetch meal slots.")


@mealRouter.get("/grocery-list/{roadmapId}")
async def grocery_list(
    roadmapId: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    if not await verify_plan_ownership(roadmapId, current_user["uid"]):
        raise HTTPException(
            status_code=403, detail="You do not have access to this plan."
        )
    items = await build_grocery_list(roadmapId)
    return {"status": "done", "roadmapId": roadmapId, "result": items}


@mealRouter.get("/approvals")
async def list_approvals(current_user: Annotated[dict, Depends(get_current_user)]):
    userId = current_user["uid"]
    logger.info("--- %s", userId)
    try:
        cursor = get_db()["approvals"].find({"userId": userId, "status": "pending"})
        docs = await cursor.to_list(None)
        for doc in docs:
            doc["_id"] = str(doc["_id"])
        logger.info("approvals found: %s", len(docs))

        if not docs:
            return {"status": "done", "message": "no approval found", "result": []}

        return {"status": "done", "result": docs}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@mealRouter.get("/roadmaps")
async def getPlans(current_user: Annotated[dict, Depends(get_current_user)]):
    userId = current_user["uid"]
    logger.info("--- %s", userId)
    try:
        cursor = get_db()["roadmaps"].find({"userId": userId})
        docs = await cursor.to_list(None)
        for doc in docs:
            doc["_id"] = str(doc["_id"])
        logger.info("approvals found: %s", len(docs))

        if not docs:
            return {"status": "done", "message": "roadmaps not found", "result": []}

        return {"status": "done", "message": "roadmaps fetched", "result": docs}

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
    userId = current_user["uid"]

    try:
        result = supabase.table("triggers").select("*").eq("userId", userId).execute()
        if result and result.data:
            for t in result.data or []:
                supabase.table("triggers").update({"enabled": not t["enabled"]}).eq(
                    "id", t["id"]
                ).execute()
        else:
            supabase.table("triggers").insert(
                {
                    "userId": userId,
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
    logger.info("This job runs every sunday on 6:30 pm")
    now = datetime.now()
    week_start = get_monday()
    try:
        # Only meal-plan schedules — other features (e.g. personal_assistant)
        # share this table with their own action_type.
        triggers = (
            supabase.table("triggers")
            .select("*")
            .eq("enabled", True)
            .eq("action_type", "schedule")
            .execute()
        )
    except Exception as e:
        logger.error("run_triggers fetch error: %s", e)
        return

    for t in triggers.data or []:
        # Per-user isolation: one user's failure must not abort the whole sweep.
        try:
            thread_id = str(uuid.uuid4())

            # Check if user already has an approved plan to re-use
            latest = (
                supabase.table("meal_plans")
                .select("id, meal_slots(*)")
                .eq("user", t["userId"])
                .eq("status", "approved")
                .order("createdAt", desc=True)
                .limit(1)
                .maybe_single()
                .execute()
            )

            if latest and latest.data:
                # Re-use existing plan: create approval directly without LLM
                slots = latest.data.get("meal_slots", [])
                proposed = [
                    {
                        "roadmapId": latest.data["id"],
                        "day_of_week": s["day_of_week"],
                        "meal_type": s["meal_type"],
                        "recipe_name": s["recipe_name"],
                        "protein_g": s["protein_g"],
                    }
                    for s in slots
                ]
                supabase.table("approvals").insert(
                    {
                        "userId": t["userId"],
                        "thread_id": thread_id,
                        "action_type": "save_plan",
                        "payload": {"week_start": week_start, "plan": proposed},
                        "status": "pending",
                    }
                ).execute()
                logger.info(
                    f"[trigger] Approval created for existing plan, user={t['userId']}"
                )
            else:
                # No existing plan: invoke agent to generate one
                config = {"configurable": {"thread_id": thread_id}}
                agent_result = await agent.ainvoke(
                    {
                        "query": "Plan my meals for next week",
                        "userId": t["userId"],
                        "thread_id": thread_id,
                    },
                    config=config,
                )
                if "__interrupt__" in agent_result:
                    logger.info(
                        f"[trigger] New plan approval created, user={t['userId']}"
                    )

            supabase.table("triggers").update({"last_run_at": now.isoformat()}).eq(
                "id", t["id"]
            ).execute()
        except Exception as e:
            logger.error(f"[trigger] error for user={t.get('userId')}: {e}")
