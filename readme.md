# BM25Okapi PyO3 Module

A high-performance Rust implementation of the BM25Okapi algorithm optimized for multicore processing, exposed to Python via PyO3. This module leverages Rust's concurrency and safety features to provide efficient text search capabilities within Python applications.

## Features

- **High Performance:** Utilizes Rust's `rayon` crate for parallel processing, ensuring efficient handling of large corpora.
- **Python Integration:** Seamlessly integrates with Python using PyO3.
- **Custom Tokenization:** Supports custom tokenizers through Python callback functions.
- **Persistent Storage:** Provides methods to save and load BM25 indexes.
- **Thread-Safe:** Ensures thread-safe operations.
- **Easy to Use:** Simple API for indexing, scoring, and retrieving top documents.

## Installation

### Prerequisites

- **Rust:** Install from [rust-lang.org](https://www.rust-lang.org/tools/install).
- **Python:** Compatible with Python 3.6 and above.
- **maturin:** Install via pip: `pip install maturin`.

### Building the Module

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/yourusername/bm25_pyo3.git
cd bm25_pyo3
maturin develop --release
```

### Usage
```python
from bm25_pyo3 import BM25Okapi

# Sample corpus
corpus = [
    "this is the first document",
    "this document is the second document",
    "and this is the third one",
    "is this the first document",
]

# Optional: Define a custom tokenizer
def my_tokenizer(text):
    return text.lower().split()

# Initialize BM25Okapi instance
bm25 = BM25Okapi(corpus=corpus, tokenizer=my_tokenizer)

# Define a query
query = "first document"
tokenized_query = my_tokenizer(query)

# Get scores for all documents
scores = bm25.get_scores(tokenized_query)
print("Scores:", scores)

# Retrieve top N documents
top_docs = bm25.get_top_n(query=tokenized_query, documents=corpus, n=2)
print("Top Documents:", top_docs)

scores_loaded = bm25_loaded.get_scores(tokenized_query)
print("Scores from loaded index:", scores_loaded)
```
