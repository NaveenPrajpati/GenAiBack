from typing import TypedDict, Optional
from fastapi import APIRouter
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, END
from app.core.config import supabase
from datetime import date, timedelta, datetime
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

mealRouter = APIRouter(
    prefix="/meal-planner",
    tags=["summarizer"],
    responses={404: {"description": "Not found"}},
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# llm = ChatOllama(model="llama3:latest", temperature=0)
plandId = "4676c66a-fce3-4704-a37d-3501eb5e952a"


class QueryRequest(BaseModel):
    text: str


class ProfileState(TypedDict):
    display_name: str
    diet: str
    protein_target: int


class PlannerState(TypedDict):
    query: str
    intent: str
    profile: ProfileState


graph = StateGraph(PlannerState)


class ProfileOutput(BaseModel):
    display_name: str
    diet: str
    protein_target: int


class LogOutput(BaseModel):
    receipe: str
    day_of_week: int
    meal_type: str


class GroceryItem(BaseModel):
    plan_id: str = plandId
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


def insertDataindb(data: ProfileOutput):
    try:

        supabase.table("profiles").insert(
            {
                "display_name": data.display_name,
                "diet": data.diet,
                "protein_target": data.protein_target,
            }
        ).execute()
    except Exception as e:
        print(e)


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


def classify_intent1(state: PlannerState):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at extracting user profile info. Extract name, diet preference, and daily protein target (in grams) from the text.",
            ),
            ("human", "{text}"),
        ]
    )
    chain = prompt | llm.with_structured_output(ProfileOutput)
    result: ProfileOutput = chain.invoke({"text": state["query"]})
    print(result)
    insertDataindb(result)
    return {
        "profile": {
            "display_name": result.display_name,
            "diet": result.diet,
            "protein_target": result.protein_target,
        }
    }


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
    result: IntentOutput = chain.invoke({"text": state["query"]})
    print(result)
    return {
        "intent": result.intent,
    }


async def findRecepieInDb(recepie: str):
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
                    "plan_id": plandId,
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
                "You are an expert at extracting recepie related thing from text, tell me which recepie on what day_of_week like if my plan starts with monday so for thrusday its 4th day so id just need value(like 4) and what is meal type like (breakfast,lunch, dinner,snack)  user want to log ",
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

    chain = findPrompt | llm.with_structured_output(LogOutput)
    result: LogOutput = await chain.ainvoke({"text": state["query"]})
    print("log data", result)
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


def decide_agent(state: PlannerState):

    if state.get("intent") == "log":
        return "log_agent"

    return END


graph.add_node("classify_intent", classify_intent)
graph.add_node("log_agent", log_agent)
graph.add_edge(START, "classify_intent")
graph.add_conditional_edges("classify_intent", decide_agent, ["log_agent", END])

agent = graph.compile()


@mealRouter.post("/query")
async def summarize(body: QueryRequest):
    result = await agent.ainvoke({"query": body.text})
    print("fineal--", result)
    return {"profile": "nothing"}
