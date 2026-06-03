import operator
import os
from typing import Annotated, TypedDict
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AnyMessage, PlainTextContentBlock
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import json
from langgraph.graph import START , StateGraph , END
from langgraph.checkpoint.postgres import PostgresSaver  
from app.core.config import supabase
from langchain_openai import ChatOpenAI

load_dotenv()

mealRouter = APIRouter(
    prefix="/meal-planner",
    tags=["summarizer"],
    responses={404: {"description": "Not found"}},
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


parser = StrOutputParser()


config = {
        "configurable": {
            "thread_id": "1"
        }
    }
def insertData() -> None:
    supabase.table("meal_planner").insert({
 
    }).execute()

llm=ChatOpenAI()

class QueryRequest(TypedDict):
    text: Annotated[list[AnyMessage], operator.add]
    
class PlannerState(TypedDict):
    messages: str
    intent:str
    memory:str
    draft_plan:str

graph=StateGraph(PlannerState)




def classify_intent(state: PlannerState):

  prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in finding intent , name , diet ,protein target, "
      
        ),
        ("human", "{text}"),
    ])
    agent1 = prompt | llm
    agent1.invoke(state.get('messages'))
    return {"messages": ''}

graph.add_node("classify_intent",classify_intent)

graph.add_edge(START,'classify_intent')

agent=graph.compile(PlannerState)

@mealRouter.post("/query")
async def summarize(request: QueryRequest):
    print({"text": request.text})
    return {"summary": ''}



    chain = prompt | model | parser

    async def event_generator():
        try:
            async for chunk in chain.astream({"text": request.text}):
                # Send each token as an SSE event
                data = json.dumps({"token": chunk})
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
            "X-Accel-Buffering": "no",  # Disable nginx buffering if behind nginx
        },
    )
