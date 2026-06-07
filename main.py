from fastapi import FastAPI
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


store = VectorStore("Psychology")

llm = ChatOllama(
    model="llama3.1:8b"
)
def chat_node(state: MessagesState):
    
    history=state["messages"]
    
        
    latest_user_msg=history[-1].content 
    docs=store.retrieve(latest_user_msg)
    docs = store.rerank(
    latest_user_msg,
    docs,
    top_k=5
)
    context = "\n".join(
        doc.page_content
        for doc,score in docs
    )

    sys_message = SystemMessage(
        content=f"""
    You are a psychology therapist.
    Use the provided context when necessary.

    Context:
    {context}

   
    """)


    prompt = [sys_message] + history
    response = llm.invoke(prompt)
    return {"messages": [response]}

builder.add_node("chatllm",chat_node)
builder.add_edge(START,"chatllm")

memory=MemorySaver()
chat_app=builder.compile(checkpointer=memory)
thread_id="2"
class ChatRequest(BaseModel):
    query: str


@app.get("/", response_class=HTMLResponse)
def home():

    with open(
        "templates/index.html",
        encoding="utf-8"
    ) as f:
        return f.read()


@app.post("/chat")
async def chat(request: ChatRequest):
    
    query = request.query
    
    state_update={"messages":[HumanMessage(content=query)]}
    response = chat_app.invoke(
        state_update,
        config={
        "callbacks": [langfuse_handler],
        "configurable":{
            "thread_id":thread_id
        }
        
    })

    return {
        "answer": response["messages"][-1].content,

    }