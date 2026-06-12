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
from typing import Literal
import sqlite3
import torch
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
class Route(BaseModel):
    category: Literal["chatllm","rag"]
    
    
# llm=ChatDeepSeek(
#     model="	deepseek-v4-flash"
# )
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

model_path = r"models\finetuned ModernBERT"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path
)
id2label = {
    0: "All-or-nothing thinking",
    1: "Emotional Reasoning",
    2: "Fortune-Telling",
    3: "Labeling",
    4: "Magnification",
    5: "Mental filter",
    6: "Mind Reading",
    7: "No Distortion",
    8: "Overgeneralization",
    9: "Personalization",
    10: "Should statements"
}
def resolve_no_distortion(top_results):

    no_dist = next(
        (x for x in top_results
         if x["distortion"] == "No Distortion"),
        None
    )

    if not no_dist:
        return top_results

    no_score = no_dist["confidence"]

    other_sum = sum(
        x["confidence"]
        for x in top_results
        if x["distortion"] != "No Distortion"
    )

    MARGIN = 0.10

    if no_score > other_sum + MARGIN:
        return [no_dist]

    return [
        x for x in top_results
        if x["distortion"] != "No Distortion"
    ]
def detect_distortions(text, threshold=0.30, top_k=4):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits)[0]

    results = [
        {
            "distortion": id2label[i],
            "confidence": round(float(probs[i]), 4)
        }
        for i in range(len(probs))
    ]

    results.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    active_distortions = [
        item
        for item in results
    ][:top_k]

    return resolve_no_distortion(active_distortions)
def chat_node(state: MessagesState):
    
    SYSTEM_PROMPT = """
    You are an empathetic psychological support assistant.

    IMPORTANT RULES:

    1. Do NOT give advice.
    2. Do NOT suggest activities.
    3. Do NOT provide coping strategies.
    4. Do NOT try to solve the user's problem.
    5. Focus on understanding.

    When a user shares a feeling, first:
    - reflect what they said
    - explore their experience
    - ask clarifying questions one at a time

    Example:

        User: "Lately I've been feeling very anxious and don't know how to relieve it."

        Assistant:
        "What do you think is contributing most to that anxiety right now?"

        User:
        "But I always feel like whatever I do isn't good enough, and I'm very afraid of failure."

        Assistant:
        "You mentioned feeling not good enough and fearing failure. How does that affect your ability to focus on tasks?"

        User:
        "But I can never concentrate, and I procrastinate a lot."
    """
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT)
    ] + state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }
    

def rag_node(state: MessagesState):
    history=state["messages"]
    
        
    latest_user_msg=history[-1].content 
    
    hyde=store.generate_hyde(latest_user_msg)
    print("\n=== HYDE ===")
    print(hyde)
    docs=store.retrieve(latest_user_msg)
    if docs:
        docs = store.rerank(
        latest_user_msg,
        docs,
        top_k=5
        )
    context = "\n".join(
        doc.page_content
        for doc in docs
    )

    sys_message = SystemMessage(
        content=f"""
    You are a psychology therapist.
    Use the provided context when necessary.

    Context:
    {context}

   
    """)


    prompt = [sys_message,HumanMessage(content=latest_user_msg)]
    response = llm.invoke(prompt)
    return {"messages": [response]}
def router_node(state: MessagesState):
    prompt=state["messages"][-1].content

    EMOTIONAL_WORDS = {
    "anxious",
    "sad",
    "lonely",
    "worried",
    "depressed",
    "stress",
    "angry",
    "tired",
    "fed up"
    }
    PSYCHOLOGY_WORDS = {
    "cbt",
    "conditioning",
    "attachment",
    "memory",
    "cognitive distortion",
    "psychology",
    "theory"
    }
    matched_words = [word for word in EMOTIONAL_WORDS if word in prompt.lower()]
    if matched_words:
        return "chatllm"
    router_llm=llm.with_structured_output(Route)
    result = router_llm.invoke(f"""
    Classify the following user message.

    User Message:
    {prompt}

    Return:
    - rag → if the user is asking about a psychology concept, theory, definition, explanation, or factual knowledge.
    - chatllm → if the user is discussing personal experiences, emotions, worries, feelings,  seeking support or just having a conversion.

    Return only one word:
    rag
    or
    chatllm
    """)
    return result.category


builder.add_node("chatllm",chat_node)
builder.add_node("rag",rag_node)
builder.add_conditional_edges(START,router_node)

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
            max_age=60 * 60 * 24 * 365
        )

    
    distortions = detect_distortions(request.query)

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
        "answer": result["messages"][-1].content,
        "distortions": distortions  
    }
