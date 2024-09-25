from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS

import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import os
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS to handle cross-origin requests if necessary

# Paths for saving embeddings and loading documents
embedding_file_path = os.path.join(os.getcwd(), "document_embeddings.pkl")
documents_file_path = os.path.join(os.getcwd(), "documents.txt")

# Load pre-trained model and tokenizer for embeddings
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
model = AutoModel.from_pretrained(embedding_model_name)

# Function to load documents from an external file
def load_documents(document_file=documents_file_path):
    with open(document_file, "r", encoding="utf-8") as f:
        documents = f.read().splitlines()  # Assuming each document is on a new line
    return documents

# Function to convert text to embeddings
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Get the embeddings
    return embeddings.cpu().numpy()

# Function to prepare FAISS index and save/load embeddings
def prepare_faiss_index(documents, embedding_file=embedding_file_path):
    if os.path.exists(embedding_file):
        # Load existing embeddings if available
        print(f"Loading document embeddings from {embedding_file}...")
        with open(embedding_file, "rb") as f:
            document_vectors = pickle.load(f)
    else:
        # Embed and save documents if embeddings file doesn't exist
        print("Embedding documents and saving them to file...")
        document_vectors = [embed_text(doc) for doc in documents]
        document_vectors = np.vstack(document_vectors)
        with open(embedding_file, "wb") as f:
            pickle.dump(document_vectors, f)

    # Initialize FAISS index
    embedding_dim = document_vectors.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(document_vectors)
    return index

# Load documents and prepare FAISS index
documents = load_documents()
index = prepare_faiss_index(documents)

# Load pre-trained QA model
qa_model_name = "deepset/xlm-roberta-large-squad2"  # Supports Arabic QA
qa_pipeline = pipeline("question-answering", model=qa_model_name)

# Retrieve documents from FAISS index based on the query
def retrieve(query):
    query_vector = embed_text(query)  # Embed the query
    distances, indices = index.search(query_vector, k=3)  # Top 3 results
    results = [documents[i].strip() for i in indices[0]]  # Return the top 3 matching documents
    return " ".join(results)  # Combine the results for better context

# Chat endpoint that handles queries
@app.route('/', methods=['POST'])
def chat():
    user_input = request.json.get("query")
    retrieved_docs = retrieve(user_input)

    # Prepare input for QA model
    qa_input = {
        'question': user_input,
        'context': retrieved_docs
    }

    # Get the answer from the QA model
    response = qa_pipeline(qa_input)
    answer = response['answer']

    # Log the conversation to a file
    with open(os.path.join(os.getcwd(), "chat_logs.txt"), "a", encoding="utf-8") as f:
        f.write(f"Question: {user_input}\n")
        f.write(f"Answer: {answer}\n")
        f.write("=" * 50 + "\n")  # Separator for readability

    return jsonify({"response": answer})

# Run the app using Gunicorn
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
