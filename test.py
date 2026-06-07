from vector_db import VectorStore
from langchain_ollama import ChatOllama


llm=ChatOllama(
    model="llama3.2:1b"
)
new=VectorStore(collection_name="Psychology")

# new.add_documents(r"D:\AI Engineering\docs\Termination of decline welbeing.pdf","APA")
query="what are types of resilience ?"

documents=new.retrieve(query)
context="\n".join([doc.page_content for doc in documents])
print(context)
prompt = f"""
Use only the provided context.


Context:
{context}

Question:
{query}
"""
response=llm.invoke(query)
print(response.content)
print(response)

