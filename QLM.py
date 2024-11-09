import os
import nltk
from collections import defaultdict, Counter
import math

# Tokenize document into words
def tokenize(text):
    return nltk.word_tokenize(text.lower())

# Build unigram language model for each document
def build_language_model(documents):
    doc_models = []
    term_frequencies = defaultdict(int)
    total_terms = 0

    for doc in documents:
        tokenized_doc = tokenize(doc)
        doc_length = len(tokenized_doc)
        term_counts = Counter(tokenized_doc)
        doc_models.append((term_counts, doc_length))
        for term, count in term_counts.items():
            term_frequencies[term] += count
            total_terms += count

    collection_model = {term: freq / total_terms for term, freq in term_frequencies.items()}

    return doc_models, collection_model

# Calculate Dirichlet-smoothed probability of a term in a document
def dirichlet_smoothing(term, term_count, doc_length, collection_model, mu=2000):
    collection_prob = collection_model.get(term, 0)
    return (term_count + mu * collection_prob) / (doc_length + mu)

# Calculate Jelinek-Mercer smoothed probability of a term in a document
def jelinek_mercer_smoothing(term, term_count, doc_length, collection_model, lambda_=0.1):
    collection_prob = collection_model.get(term, 0)
    doc_prob = term_count / doc_length if doc_length > 0 else 0
    return lambda_ * doc_prob + (1 - lambda_) * collection_prob

# Compute the query likelihood score for a document
def query_likelihood(query, doc_model, doc_length, collection_model, smoothing='dirichlet', mu=2000, lambda_=0.1):
    query_tokens = tokenize(query)
    score = 0
    
    for term in query_tokens:
        term_count = doc_model.get(term, 0)
        
        if smoothing == 'dirichlet':
            prob = dirichlet_smoothing(term, term_count, doc_length, collection_model, mu)
        elif smoothing == 'jelinek_mercer':
            prob = jelinek_mercer_smoothing(term, term_count, doc_length, collection_model, lambda_)
        
        score += prob
    
    return score

# Rank documents based on query likelihood
def rank_documents(query, documents, smoothing='dirichlet', mu=2000, lambda_=0.1):
    # Build language models for documents
    doc_models, collection_model = build_language_model(documents)

    # Compute query likelihood for each document
    scores = []
    for idx, (doc_model, doc_length) in enumerate(doc_models):
        score = query_likelihood(query, doc_model, doc_length, collection_model, smoothing, mu, lambda_)
        scores.append((idx, score))

    # Sort documents by score (descending)
    ranked_docs = sorted(scores, key=lambda x: x[1], reverse=True)
    
    return ranked_docs

# Load crawled documents (from previous section)
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

# Main function
def search_documents(query, crawled_dir="crawled_pages", smoothing='dirichlet', top_k=5):
    # Load documents from directory
    documents, doc_names = load_documents(crawled_dir)
    
    # Rank documents using Query Likelihood Model
    ranked_docs = rank_documents(query, documents, smoothing=smoothing)

    # Output the top-ranked documents
    print(f"Query: {query}")
    print(f"Top {top_k} matching documents:")
    for idx, score in ranked_docs[:top_k]:
        print(f"Document: {doc_names[idx]}, Score: {score}")

if __name__ == "__main__":
    query = "carrot"  # Example query
    search_documents(query, smoothing='dirichlet', top_k=5)  # You can change top_k and smoothing method
