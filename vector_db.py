from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import SentenceTransformer,CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
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
    def rerank(query, docs, top_k=5):

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
        return [doc for doc, score in ranked[:top_k]]
    def retrieve(self,query:str,k:int=10):

        results=self.vectorstore.similarity_search_with_score(query,k=5)
        print("\n=== RETRIEVAL SCORES ===")

        threshold=0.6
        docs=[]
        for doc, score in results:
            if score<=threshold:
                docs.append(doc)
        print(f"Retrieved {len(docs)} docs")
        return docs
    


