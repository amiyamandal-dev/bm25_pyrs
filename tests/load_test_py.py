import time
import random
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from rank_bm25 import BM25Okapi
from collections import defaultdict

random.seed(42)

def generate_random_document(word_count, vocabulary):
    """Generates a random document with a specified number of words."""
    return ' '.join(random.choices(vocabulary, k=word_count))

def generate_corpus(num_documents, avg_word_count=100):
    """Generates a corpus of random documents."""
    vocabulary_size = 100000  # Define vocabulary size
    vocabulary = [''.join(random.choices(string.ascii_lowercase, k=5)) for _ in range(vocabulary_size)]
    corpus = [generate_random_document(avg_word_count, vocabulary) for _ in range(num_documents)]
    return corpus, vocabulary

def generate_queries(vocabulary, num_queries, query_length=5):
    """Generates a list of random queries."""
    return [' '.join(random.choices(vocabulary, k=query_length)) for _ in range(num_queries)]

def perform_query(bm25_instance, query_terms):
    """Performs a single BM25 query and returns the score or top N results."""
    try:
        # Split query into terms
        query = query_terms.split()
        # Get scores
        scores = bm25_instance.get_scores(query)
        # Optionally, get top N documents
        # top_docs = bm25_instance.get_top_n(query, docs, n=5)
        return scores  # or return top_docs
    except Exception as e:
        return str(e)

def load_test_bm25(
    num_documents=100000,     # Total number of documents in the corpus
    avg_word_count=100,      # Average number of words per document
    num_queries=1000,        # Total number of queries to perform
    concurrency_level=50,    # Number of concurrent threads
    query_length=5           # Number of terms in each query
):
    """
    Performs a load test on the BM25Okapi model.

    Parameters:
    - num_documents: Total documents in the corpus
    - avg_word_count: Average words per document
    - num_queries: Total number of queries to execute
    - concurrency_level: Number of threads to use concurrently
    - query_length: Number of terms in each query
    """
    print("Generating corpus...")
    start_time = time.time()
    corpus, vocabulary = generate_corpus(num_documents, avg_word_count)
    generation_time = time.time() - start_time
    print(f"Corpus generated in {generation_time:.2f} seconds.")

    print("Initializing BM25Okapi model...")
    start_time = time.time()
    bm25 = BM25Okapi(corpus, tokenizer=None)  # Assuming default tokenizer (whitespace)
    initialization_time = time.time() - start_time
    print(f"BM25Okapi initialized in {initialization_time:.2f} seconds.")

    print("Generating queries...")
    queries = generate_queries(vocabulary, num_queries, query_length)
    print(f"{num_queries} queries generated.")

    # Prepare for concurrent execution
    results = []
    errors = 0
    start_time = time.time()

    print("Starting load test...")
    with ThreadPoolExecutor(max_workers=concurrency_level) as executor:
        # Submit all queries to the executor
        future_to_query = {executor.submit(perform_query, bm25, query): query for query in queries}

        # As each future completes, process the result
        for future in as_completed(future_to_query):
            result = future.result()
            if isinstance(result, str):
                # An error occurred
                errors += 1
            else:
                results.append(result)  # You can process the scores as needed

    total_time = time.time() - start_time
    print("Load test completed.")
    print(f"Total queries executed: {num_queries}")
    print(f"Successful queries: {num_queries - errors}")
    print(f"Failed queries: {errors}")
    print(f"Total time taken: {total_time:.2f} seconds.")
    print(f"Throughput: {num_queries / total_time:.2f} queries per second.")

    # Optionally, analyze results
    # For example, compute average score, top N documents frequency, etc.
    # Here, we'll just print the number of results
    print(f"Total results collected: {len(results)}")

if __name__ == "__main__":
    # Customize load test parameters as needed
    load_test_bm25(
        num_documents=100000,
        avg_word_count=1000,
        num_queries=100,
        concurrency_level=50,
        query_length=5
    )
