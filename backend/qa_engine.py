from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_vector_store(text_chunks):
    embeddings = model.encode(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

def get_top_chunks(user_query, text_chunks, index, embeddings, top_k=3):
    query_embedding = model.encode([user_query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    top_chunks = [text_chunks[i] for i in indices[0]]
    return top_chunks
