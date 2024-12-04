# Information Retrieval System for Document Search

## Project Overview

This project focuses on building an information retrieval system designed for efficient document search and ranking using the QLM (Query Likelihood Model) and VSM (Vector Space Model). The system is built to crawl data from web pages, process and index them, and then use these indexed documents to retrieve and rank search results for user queries.

The main web page used for crawling data is [Cooking Classy](https://www.cookingclassy.com/), a recipe-focused website that provides varied content for testing the retrieval system.

## Project Structure

Below is a list of key files and their roles in the system:

### 1. `crawler.py`

- **Description**: This file handles the web crawling process, extracting data from the target web page (`https://www.cookingclassy.com/`). It collects raw content and saves it for further processing.
- **Functionality**:
  - Connects to the specified web page.
  - Scrapes text and HTML content from web articles and recipes.
  - Saves the crawled data in a structured format.

### 2. `indexing.py`

- **Description**: Responsible for processing the crawled data and creating an inverted index for fast lookups.
- **Functionality**:
  - Parses and tokenizes the text from the crawled pages.
  - Creates an inverted index mapping terms to their occurrences in the documents.
  - Stores the inverted index as a JSON file (`inverted_index.json`) for efficient access.

### 3. `inverted_index.json`

- **Description**: A JSON file that holds the data structure of the inverted index.
- **Purpose**: Used to facilitate quick searches and retrievals by mapping keywords to their document locations.

### 4. `VSM.py`

- **Description**: Implements the Vector Space Model for document ranking and retrieval.
- **Functionality**:
  - Converts documents and queries into vectors.
  - Calculates similarity scores between the query vector and document vectors.
  - Returns a ranked list of documents based on similarity scores.

### 5. `QLM.py`

- **Description**: Implements the Query Likelihood Model for document retrieval.
- **Functionality**:
  - Estimates the likelihood of a document matching a given query.
  - Ranks documents by their probabilities of relevance to the query.
  - Returns the top-ranked documents.

### 6. `app.py`

- **Description**: Contains the Dash app that runs the information retrieval system on a local server.
- **Functionality**:
  - Provides a user interface with two tabs:
    - **QLM Tab**: Displays the top 10 results for a given query using the Query Likelihood Model.
    - **VSM Tab**: Shows the search results using the Vector Space Model.
  - Hosts the Dash app on `localhost` to allow users to interact with the system.

### 7. `README.md`

- **Description**: This markdown file provides documentation for the project, explaining the purpose, structure, and instructions for usage.

## Setup and Running the Project

### Prerequisites

- **Python**: Version 3.11 or higher.
- **Libraries**: Ensure the following libraries are installed:
  - `dash`
  - `pandas`
  - `numpy`
  - `requests`
  - `beautifulsoup4`
  - `scikit-learn` (for VSM)

- **Python Libraries**: Install required libraries with the following command:
  ```bash
  pip install dash pandas numpy requests beautifulsoup4 scikit-learn

![Dash — Mozilla Firefox 12_4_2024 12_37_46 PM](https://github.com/user-attachments/assets/ba748f20-bb3a-4011-8b44-9a60352e6a25)

