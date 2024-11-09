import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import nltk
from collections import defaultdict, Counter
import math


def tokenize(text):
    return nltk.word_tokenize(text.lower())

# ------------------- VSM Model --------------------

def compute_tfidf(documents):
    doc_term_frequencies = []
    term_document_frequencies = defaultdict(int)

    # Calculate term frequencies and document frequencies
    for doc in documents:
        term_counts = Counter(tokenize(doc))
        doc_term_frequencies.append(term_counts)
        for term in term_counts:
            term_document_frequencies[term] += 1

    # Compute TF-IDF for each document
    num_docs = len(documents)
    tfidf_documents = []
    for doc_index, term_counts in enumerate(doc_term_frequencies):
        tfidf = {}
        for term, count in term_counts.items():
            if count > 0:
                tf = 1 + math.log(count)
            else:
                tf = 0
            idf = math.log(num_docs / (1 + term_document_frequencies[term]))
            tfidf[term] = tf * idf
        tfidf_documents.append(tfidf)

    return tfidf_documents

# Calculate cosine similarity for query and document vectors
def cosine_similarity(query_vector, document_vector):
    dot_product = sum(query_vector.get(term, 0) * document_vector.get(term, 0) for term in query_vector)
    query_magnitude = math.sqrt(sum(val ** 2 for val in query_vector.values()))
    doc_magnitude = math.sqrt(sum(val ** 2 for val in document_vector.values()))
    return dot_product / (query_magnitude * doc_magnitude) if query_magnitude * doc_magnitude != 0 else 0

def vsm_search(query, documents):
    tfidf_documents = compute_tfidf(documents)
    query_tokens = Counter(tokenize(query))
    
    query_vector = {}
    for term in query_tokens:
        query_vector[term] = query_tokens[term]  # In real use, use TF-IDF

    # Compute similarities
    similarities = []
    for idx, doc_tfidf in enumerate(tfidf_documents):
        sim = cosine_similarity(query_vector, doc_tfidf)
        similarities.append((idx, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

# ------------------- QLM Model --------------------

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

def query_likelihood(query, doc_model, doc_length, collection_model, mu=2000):
    query_tokens = tokenize(query)
    score = 0
    
    for term in query_tokens:
        term_count = doc_model.get(term, 0)
        collection_prob = collection_model.get(term, 0)
        prob = (term_count + mu * collection_prob) / (doc_length + mu)
        score += prob
    
    return score

def qlm_search(query, documents):
    doc_models, collection_model = build_language_model(documents)
    
    scores = []
    for idx, (doc_model, doc_length) in enumerate(doc_models):
        score = query_likelihood(query, doc_model, doc_length, collection_model)
        scores.append((idx, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

# ------------------- Load Documents --------------------

def load_documents(crawled_dir="crawled_pages"):
    documents = []
    doc_names = []
    doc_urls = []

    for file_name in os.listdir(crawled_dir):
        file_path = os.path.join(crawled_dir, file_name)
        
        if file_path.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(content)
                doc_names.append(file_name)

                # Generate the URL from the file name (assuming the domain is part of the file name)
                base_url = "http://" + file_name.replace("_", "/").replace(".txt", "")
                doc_urls.append(base_url)

    return documents, doc_names, doc_urls

# ------------------- Dash Application --------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Cookingclassy Retrieval System", style={'textAlign': 'center'}),
    html.Div([
        dcc.Input(
            id='query-input', 
            type='text', 
            placeholder='Enter your query', 
            style={
                'width': '50%', 
                'padding': '10px', 
                'fontSize': '20px',
                'borderRadius': '25px',  # Rounded search bar
                'border': '1px solid #ccc',
                'boxShadow': '0px 4px 8px rgba(0, 0, 0, 0.2)'
            }
        ),
        html.Button(
            'Search', 
            id='search-button', 
            n_clicks=0, 
            style={
                'fontSize': '20px', 
                'padding': '10px 20px', 
                'marginLeft': '10px',
                'borderRadius': '25px',  # Rounded button
                'backgroundColor': '#007bff', 
                'color': 'white', 
                'border': 'none',
                'boxShadow': '0px 4px 8px rgba(0, 0, 0, 0.2)'
            }
        )
    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'marginTop': '20px'}),
    
    dcc.Tabs([
        dcc.Tab(
            label='VSM Results', 
            children=[html.Div(id='vsm-results')],
            style={
                'borderRadius': '20px 20px 0 0',  # Rounded top edges
                'padding': '10px', 
                'fontWeight': 'bold'
            },
            selected_style={
                'borderRadius': '20px 20px 0 0', 
                'padding': '10px', 
                'fontWeight': 'bold', 
                'backgroundColor': '#007bff',
                'color': 'white'
            }
        ),
        dcc.Tab(
            label='QLM Results', 
            children=[html.Div(id='qlm-results')],
            style={
                'borderRadius': '20px 20px 0 0',  # Rounded top edges
                'padding': '10px', 
                'fontWeight': 'bold'
            },
            selected_style={
                'borderRadius': '20px 20px 0 0', 
                'padding': '10px', 
                'fontWeight': 'bold', 
                'backgroundColor': '#007bff',
                'color': 'white'
            }
        ),
    ], style={
        'marginTop': '20px', 
        'borderRadius': '20px', 
        'overflow': 'hidden',  # To prevent overflow from rounded edges
        'boxShadow': '0px 4px 8px rgba(0, 0, 0, 0.2)'
    }),
])

@app.callback(
    [dash.dependencies.Output('vsm-results', 'children'),
     dash.dependencies.Output('qlm-results', 'children')],
    [dash.dependencies.Input('search-button', 'n_clicks')],
    [dash.dependencies.State('query-input', 'value')]
)
def update_results(n_clicks, query):
    if not query:
        return "Enter a query to search.", "Enter a query to search."
    
    documents, doc_names, doc_urls = load_documents()  # Load documents and URLs

    # Generate summaries from each document, starting after the second "Submit"
    summaries = []
    for doc in documents:
        # Find the position of the second occurrence of "Submit"
        first_idx = doc.lower().find('submit')
        second_idx = doc.lower().find('submit', first_idx + 1)
        
        if second_idx != -1:
            # Start the summary from the text after the second "Submit"
            summary_text = doc[second_idx + len('submit'):].strip()[:220] + '...'
        else:
            summary_text = doc[:150] + '...'  # Fallback if there isn't a second "Submit"
        summaries.append(summary_text)

    # VSM search
    vsm_ranking = vsm_search(query, documents)
    vsm_result = html.Ul([
        html.Li([
            html.A(f"{doc_names[idx]}, Score: {score:.4f}", href=doc_urls[idx], target="_blank", style={'fontSize': '22px'}),
            html.P(summaries[idx], style={'fontSize': '15px','marginTop': '5px', 'fontStyle': 'italic'})
        ], style={'marginBottom': '35px'})
        for idx, score in vsm_ranking[:10]
    ])

    # QLM search
    qlm_ranking = qlm_search(query, documents)
    qlm_result = html.Ul([
        html.Li([
            html.A(f"{doc_names[idx]}, Score: {score:.4f}", href=doc_urls[idx], target="_blank", style={'fontSize': '22px'}),
            html.P(summaries[idx], style={'fontSize': '15px','marginTop': '5px', 'fontStyle': 'italic'})
        ], style={'marginBottom': '35px'})
        for idx, score in qlm_ranking[:10]
    ])

    return vsm_result, qlm_result

if __name__ == '__main__':
    app.run_server(debug=True)