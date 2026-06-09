from fastapi import FastAPI, Cookie, Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from vector_db import VectorStore
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, MessagesState
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.sqlite import SqliteSaver
from uuid import uuid4
import sqlite3

import os
load_dotenv()

langfuse_handler = CallbackHandler()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
builder = StateGraph(state_schema=MessagesState)


store = VectorStore("Petergray_psychology")

llm = ChatOllama(
    model="qwen2.5:1.5b"
)
# llm=ChatDeepSeek(
#     model="	deepseek-v4-flash"
# )
def chat_node(state: MessagesState):
    
    history=state["messages"]
    
        
    latest_user_msg=history[-1].content 
    
    # hyde=store.generate_hyde(latest_user_msg)
    # print("\n=== HYDE ===")
    # print(hyde)
    # docs=store.retrieve(latest_user_msg)
    # if docs:
    #     docs = store.rerank(
    #     latest_user_msg,
    #     docs,
    #     top_k=5
    #     )
    # context = "\n".join(
    #     doc.page_content
    #     for doc in docs
    # )

    # sys_message = SystemMessage(
    #     content=f"""
    # You are a psychology therapist.
    # Use the provided context when necessary.

    # Context:
    # {context}

   
    # """)


    # prompt = [sys_message] + history
    print("history:")
    
    for msg in history:
        print(msg.type, msg.content)
    prompt=history
    response = llm.invoke(prompt)
    return {"messages": [response]}

builder.add_node("chatllm",chat_node)
builder.add_edge(START,"chatllm")

conn = sqlite3.connect(
    "checkpoints.db",
    check_same_thread=False
)

memory = SqliteSaver(conn)
chat_app=builder.compile(checkpointer=memory)

class ChatRequest(BaseModel):
    query: str


@app.get("/", response_class=HTMLResponse)
def home():

    with open(
        "templates/index.html",
        encoding="utf-8"
    ) as f:
        return f.read()


# @app.post("/chat")
# async def chat(request: ChatRequest):
    
#     query = request.query  
#     thread_id="1"
#     state_update={"messages":[HumanMessage(content=query)]}
#     response = chat_app.invoke(
#         state_update,
#         config={
#         "callbacks": [langfuse_handler],
#         "configurable":{
#             "thread_id":thread_id
#         }
        
#     })

#     return {
#         "answer": response["messages"][-1].content,

#     }

@app.post("/chat")
async def chat(
    request: ChatRequest,
    response: Response,
    anonymous_user_id: str = Cookie(None)
):
    
    if not anonymous_user_id:
        anonymous_user_id = str(uuid4())

        response.set_cookie(
            key="anonymous_user_id",
            value=anonymous_user_id,
            httponly=True,
            max_age=60 * 60 * 24 * 365  # 1 year
        )

    result = chat_app.invoke(
        {"messages": [HumanMessage(content=request.query)]},
        config={
            "callbacks": [langfuse_handler],
            "configurable": {
                "thread_id": anonymous_user_id,
                "user_id": anonymous_user_id
            }
        }
    )

    return {
        "answer": result["messages"][-1].content
    }