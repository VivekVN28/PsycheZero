from sentence_transformers import SentenceTransformer, CrossEncoder

# Embedding model
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
embedding_model.save("./models/bge-base-en-v1.5")

# Reranker
reranker = CrossEncoder("BAAI/bge-reranker-base")
reranker.model.save_pretrained("./models/bge-reranker-base")
reranker.tokenizer.save_pretrained("./models/bge-reranker-base")