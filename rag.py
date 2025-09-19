import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load Sentence Transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 1: Parse the uploaded file
def parse_documents(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split into documents
    raw_docs = re.split(r"=== Document ID: \d+ ===", text)
    parsed_docs = []

    for doc in raw_docs:
        if doc.strip() == "":
            continue

        user_match = re.search(r"User:\s*(.*?)\s*Response:", doc, re.DOTALL)
        response_match = re.search(r"Response:\s*(.*?)\s*\[Emotion:", doc, re.DOTALL)
        emotion_match = re.search(r"\[Emotion:\s*(.*?)\]", doc)
        tone_match = re.search(r"\[Tone:\s*(.*?)\]", doc)

        if user_match and response_match and emotion_match and tone_match:
            parsed_docs.append({
                "user": user_match.group(1).strip(),
                "response": response_match.group(1).strip(),
                "emotion": emotion_match.group(1).strip(),
                "tone": tone_match.group(1).strip()
            })
    
    return parsed_docs

# Step 2: Build FAISS index from user utterances
def build_faiss_index(user_texts):
    embeddings = model.encode(user_texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

# Step 3: Search the index
def search_query(query, docs, index, top_k=2):
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)

    print(f"\nTop {top_k} results for query: \"{query}\"\n")
    for i in range(top_k):
        idx = indices[0][i]
        doc = docs[idx]
        print(f"Response: {doc['response']}")
        print(f"[Emotion: {doc['emotion']}] [Tone: {doc['tone']}]\n")

# ===== MAIN USAGE =====
# Replace this with your uploaded file path
FILE_PATH = "rag_formatted_data.txt"  # <-- Change this to match your actual uploaded filename

# Parse documents
documents = parse_documents(FILE_PATH)

# Embed only user utterances
user_utterances = [doc["user"] for doc in documents]

# Build FAISS index
index, embeddings = build_faiss_index(user_utterances)

# Sample query
query = "I had pain while eating something cold"
search_query(query, documents, index, top_k=3)