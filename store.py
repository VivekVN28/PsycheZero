from vector_db import VectorStore

psy=VectorStore("Petergray_psychology")
results=psy.vectorstore.get()
print(results.keys())
print(results['documents'][0:15])
print(results['data'][0:15])