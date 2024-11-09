import os
import string
import json
import nltk
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')

# Initialize stemmer
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to normalize text
def normalize(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Convert to lowercase, remove punctuation, and apply stemming
    normalized_tokens = [
        stemmer.stem(token.lower()) 
        for token in tokens 
        if token.isalpha() and token.lower() not in stop_words  # Remove stopwords and punctuation
    ]
    
    return normalized_tokens

# Create inverted index with term frequencies
def create_inverted_index(folder_path):
    inverted_index = defaultdict(lambda: defaultdict(int))
    
    # Loop through all .txt files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
                # Normalize and tokenize text
                tokens = normalize(text)
                
                # Count token frequencies
                for token in tokens:
                    inverted_index[token][file_name] += 1
    
    return inverted_index

# Store inverted index in a file
def save_index(inverted_index, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(inverted_index, f, indent=4)

# Folder containing crawled web pages (saved as .txt files)
folder_path = 'crawled_pages'

# Create the inverted index with term frequencies
inverted_index = create_inverted_index(folder_path)

# Save the inverted index to a file (JSON format)
output_file = 'inverted_index.json'
save_index(inverted_index, output_file)

print(f"Inverted index with term frequencies saved to {output_file}")