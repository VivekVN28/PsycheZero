from langchain_core.documents import Document
from vector_db import VectorStore
store=VectorStore("Petergray_psychology")

# docs=Document(
#     page_content=""
#     )
# store.add_documents([docs],source_name="")
# results = store.vectorstore.get()
store.add_documents(r"D:\AI Engineering\Psychologydoc.pdf","Peter Gray Text")
results=store.vectorstore.get()
print(results.keys())
print(results['metadatas'][10])