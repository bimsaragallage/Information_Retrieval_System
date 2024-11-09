import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the crawled documents
def load_documents(crawled_dir):
    documents = []
    doc_names = []
    
    for file_name in os.listdir(crawled_dir):
        file_path = os.path.join(crawled_dir, file_name)
        
        if file_path.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(content)
                doc_names.append(file_name)
    
    return documents, doc_names

# TF-IDF Vectorization
def vectorize_documents(documents):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix

# Process query into a vector using the same TF-IDF scheme
def process_query(query, vectorizer):
    query_vector = vectorizer.transform([query])
    return query_vector

# Rank documents based on cosine similarity
def rank_documents(query_vector, tfidf_matrix):
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    return cosine_similarities

# Main function
def search_documents(query, crawled_dir="crawled_pages"):
    # Load documents from directory
    documents, doc_names = load_documents(crawled_dir)
    
    # Vectorize documents using TF-IDF
    vectorizer, tfidf_matrix = vectorize_documents(documents)
    
    # Process query
    query_vector = process_query(query, vectorizer)
    
    # Compute cosine similarities
    cosine_similarities = rank_documents(query_vector, tfidf_matrix)
    
    # Rank documents by similarity
    ranked_indices = np.argsort(-cosine_similarities)  # Sort in descending order
    
    # Output the results
    print(f"Query: {query}")
    print("Top 5 matching documents:")
    for idx in ranked_indices[:5]:
        print(f"Document: {doc_names[idx]}, Similarity: {cosine_similarities[idx]}")

if __name__ == "__main__":
    query = "carrot"  # Example query
    search_documents(query)
