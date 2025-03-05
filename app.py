import streamlit as st
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# Connect to MongoDB
mongo_uri = "mongodb+srv://adhil:root@cluster0.uxsar.mongodb.net/"
client = MongoClient(mongo_uri)
db = client["business_rag"]
collection = db["questions"]

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Function to search similar questions
def search_questions(query, top_n=5):
    query_embedding = embedding_model.encode(query).tolist()
    all_questions = list(collection.find({}, {"question": 1, "category": 1, "subcategory": 1, "embedding": 1}))

    # Compute similarity scores
    for q in all_questions:
        q["similarity"] = cosine_similarity(query_embedding, q["embedding"])

    # Sort by similarity and get top_n results
    sorted_questions = sorted(all_questions, key=lambda x: x["similarity"], reverse=True)[:top_n]

    return sorted_questions

# Streamlit UI
st.title("RAG-Enabled Business Question Generator")

# User input for business details
user_query = st.text_input("Enter business type or key details:", "")

if st.button("Generate Questions"):
    if user_query:
        results = search_questions(user_query)
        
        if results:
            st.subheader("Suggested Questions:")
            for i, res in enumerate(results):
                st.write(f"**{i+1}. {res['question']}**")
                st.caption(f"Category: {res['category']} | Subcategory: {res['subcategory']}")
        else:
            st.warning("No relevant questions found!")
    else:
        st.warning("Please enter a query!")
