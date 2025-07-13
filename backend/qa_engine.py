from sklearn.neighbors import NearestNeighbors
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_vector_store(text_chunks):
    embeddings = model.encode(text_chunks)
    index = NearestNeighbors(n_neighbors=5, metric='cosine')
    index.fit(embeddings)
    return index, embeddings

def get_top_chunks(question, text_chunks, index, embeddings):
    question_embedding = model.encode([question])
    distances, indices = index.kneighbors(question_embedding)
    return [text_chunks[i] for i in indices[0]]
