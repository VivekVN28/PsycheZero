from vector_db import VectorStore
from langchain_ollama import ChatOllama
from main import router_node
from langchain_core.messages import HumanMessage
llm=ChatOllama(
    model="qwen2.5:1.5b"
)

test_cases = [
    {
        "query": "What is CBT?",
        "expected": "rag"
    },
    {
        "query": "Explain attachment theory",
        "expected": "rag"
    },
    {
        "query": "I feel lonely",
        "expected": "chatllm"
    },
    {
        "query": "I am anxious about exams",
        "expected": "chatllm"
    },
]
for case in test_cases:

    state = {
        "messages": [
            HumanMessage(content=case["query"])
        ]
    }

    predicted = router_node(state)

    print(
        case["query"],
        predicted,
        case["expected"]
    )
    if predicted == case["expected"]:
        correct += 1

accuracy = correct / len(test_cases)

print(f"Accuracy: {accuracy:.2%}")

