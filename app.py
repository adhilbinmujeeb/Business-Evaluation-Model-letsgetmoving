import streamlit as st
import requests
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# MongoDB connection
mongo_uri = "mongodb+srv://adhil:root@cluster0.uxsar.mongodb.net/"
client = MongoClient(mongo_uri)
db = client["business_rag"]
collection = db["questions"]

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Groq API Key (Replace with your actual key)
GROQ_API_KEY = "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
"
GROQ_API_URL = "https://api.groq.com/v1/chat/completions"

# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Function to retrieve similar questions
def search_questions(query, top_n=5):
    query_embedding = embedding_model.encode(query).tolist()
    all_questions = list(collection.find({}, {"question": 1, "category": 1, "subcategory": 1, "embedding": 1}))

    # Compute similarity scores
    for q in all_questions:
        q["similarity"] = cosine_similarity(query_embedding, q["embedding"])

    # Sort by similarity and return top_n results
    sorted_questions = sorted(all_questions, key=lambda x: x["similarity"], reverse=True)[:top_n]

    return [q["question"] for q in sorted_questions]

# Function to refine questions with Groq LLM
def refine_questions_with_groq(questions, user_query):
    prompt = f"""
    You are an expert in business analysis. Based on the user's input: "{user_query}",
    refine the following questions to make them more precise and context-aware:

    {questions}

    Provide a more structured and relevant set of questions.
    """

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mixtral-8x7b-32768",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    response = requests.post(GROQ_API_URL, json=data, headers=headers)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Error retrieving response from Groq API"

# Streamlit UI
st.title("RAG-Powered Business Question Generator")

# User input
user_query = st.text_input("Enter business type or key details:", "")

if st.button("Generate Questions"):
    if user_query:
        retrieved_questions = search_questions(user_query)

        if retrieved_questions:
            st.subheader("Refining Questions with Groq AI...")
            refined_questions = refine_questions_with_groq(retrieved_questions, user_query)

            st.subheader("AI-Generated Questions:")
            st.write(refined_questions)
        else:
            st.warning("No relevant questions found!")
    else:
        st.warning("Please enter a query!")
