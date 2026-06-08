from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,WebBaseLoader
from langchain_ollama import OllamaEmbeddings,ChatOllama
from sentence_transformers import SentenceTransformer,CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)
llm = ChatOllama(
    model="qwen2.5:1.5b"
)

# embedding_model=OllamaEmbeddings(model="nomic-embed-text")

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

class VectorStore:
    def __init__(self,collection_name,embedding_model=embedding_model,persist_directory="./chroma_db"):
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_directory
        )
        self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    def add_documents(self,docs,source_name):
        loader=PyPDFLoader(docs)
        docs=loader.load()
        chunks=self.splitter.split_documents(docs)
        print(f"Number of chunks: {len(chunks)}")
        for chunk in chunks:
            chunk.metadata["source"]=source_name
        return self.vectorstore.add_documents(chunks)
    def generate_hyde(self,query):
        prompt = f"""
        Write a short psychology article answering:

        {query}

        Do not mention that this is hypothetical.
        """

        response = llm.invoke(prompt)

        return response.content
    def rerank(self,query, docs, top_k=5):

        pairs = [
            (query, doc.page_content)
            for doc in docs
        ]

        scores = reranker.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )
        print(f"Reranked chunks: {ranked}")
        ranked_doc=[doc for doc, score in ranked[:top_k]]
        return 
    def retrieve(self,query:str,k:int=10):

        results=self.vectorstore.similarity_search_with_score(query,k=5)
        print("\n=== RETRIEVAL SCORES ===")
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n[{i}] Score={score:.4f}")
            print(f"Source={doc.metadata.get('source')}")
            print(doc.page_content[:300])
            print("=" * 80)

        docs = [doc for doc, score in results]
        
        print(f"Retrieved {len(results)} docs")

        print(f"Retrieved chunks: {docs}")
        return docs
    


