import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ========================== LOAD MODELS ==========================

# SentenceTransformer model for embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Gemma model for generation
gemma_model_id = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(gemma_model_id)
gemma_model = AutoModelForCausalLM.from_pretrained(gemma_model_id, device_map={"": "cpu"})
generator = pipeline("text-generation", model=gemma_model, tokenizer=tokenizer, device=-1)

# ========================== STEP 1: PARSE DOCUMENT ==========================

def parse_documents(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

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

# ========================== STEP 2: BUILD FAISS ==========================

def build_faiss_index(user_texts):
    embeddings = embed_model.encode(user_texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

# ========================== STEP 3: RETRIEVE AND FORMAT PROMPT ==========================

def retrieve_and_format_prompt(query, docs, index, top_k=3):
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)

    retrieved_chunks = ""
    for i in range(top_k):
        idx = indices[0][i]
        doc = docs[idx]
        retrieved_chunks += f"[Emotion: {doc['emotion']}] [Tone: {doc['tone']}]\nResponse: {doc['response']}\n\n"

    # Format final prompt
    full_prompt = f"""
You are an empathetic assistant who understands the user's concerns deeply. The user says: "{query}"

Below are previous relevant responses, along with their emotional tones and speaking styles. Carefully analyze these to capture not only the emotions and tone but also the frequency of words, style of speaking, and the user's communication patterns.

Use this context to generate your reply. Your response should:

- Reflect the exact emotional tone and style as shown in the examples.
- Mimic the user's way of speaking, including common phrases, word choice, and sentence structure.
- Maintain empathy and relevance to the user's question.
- Leverage your knowledge base to provide clear, helpful, and contextually appropriate information.

Here are the past responses to guide you:

{retrieved_chunks.strip()}

Now, craft a response that closely matches the above style and tone while addressing the user's query naturally and helpfully.
"""

    return full_prompt

# ========================== STEP 4: GENERATE RESPONSE ==========================

def generate_response(prompt):
    output = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    return output[0]['generated_text']

def remove_bracketed_phrases(text):
    """
    Removes phrases enclosed in parentheses from the text, e.g., (Confirming), (Questioning).
    
    Args:
        text (str): Input string with bracketed annotations.
    
    Returns:
        str: Cleaned string without bracketed annotations.
    """
    # Remove all parenthetical phrases
    cleaned = re.sub(r'\([^)]*\)', '', text)
    
    # Replace multiple spaces with a single space
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    
    return cleaned.strip()


# ========================== MAIN EXECUTION ==========================

def main(user_query):
    # Step A: Load and parse
    FILE_PATH = "rag_formatted_data.txt"  # Replace with actual path
    documents = parse_documents(FILE_PATH)

    # Step B: Build index
    user_utterances = [doc["user"] for doc in documents]
    index, embeddings = build_faiss_index(user_utterances)

    # Step C: Take user query
    # user_query = ""

    # Step D: Get prompt with retrieved context
    final_prompt = retrieve_and_format_prompt(user_query, documents, index, top_k=5)

    # Step E: Generate model output
    print("==== Final Prompt to Model ====\n")
    print(final_prompt)
    print("\n==== Gemma Model Response ====\n")
    generated_text=generate_response(final_prompt)
    # Extract only after last "Response:"
    response_only_with_emotions = generated_text.split("Response:")[-1].strip()
    final_response = remove_bracketed_phrases(response_only_with_emotions)
    print(final_response)
    return final_response


if __name__ == '__main__':
    main("I had pain while eating something cold")