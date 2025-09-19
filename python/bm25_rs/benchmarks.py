"""
Benchmarking utilities for BM25-RS.
"""

import time
import random
import string
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from . import BM25Okapi, BM25Plus, BM25L
except ImportError:
    # Fallback for development
    from bm25_rs import BM25Okapi, BM25Plus, BM25L


def generate_random_corpus(
    num_docs: int,
    avg_doc_length: int = 100,
    vocab_size: int = 10000,
    seed: Optional[int] = None
) -> List[str]:
    """
    Generate a random corpus for benchmarking.
    
    Args:
        num_docs: Number of documents to generate
        avg_doc_length: Average document length in words
        vocab_size: Size of vocabulary
        seed: Random seed for reproducibility
        
    Returns:
        List of document strings
    """
    if seed is not None:
        random.seed(seed)
    
    # Generate vocabulary
    vocabulary = [
        ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
        for _ in range(vocab_size)
    ]
    
    # Generate documents with realistic length distribution
    corpus = []
    for _ in range(num_docs):
        doc_length = max(10, int(random.normalvariate(avg_doc_length, avg_doc_length * 0.3)))
        words = random.choices(vocabulary, k=doc_length)
        corpus.append(' '.join(words))
    
    return corpus


def generate_queries(
    vocabulary: List[str],
    num_queries: int,
    query_length_range: tuple = (2, 5),
    seed: Optional[int] = None
) -> List[str]:
    """
    Generate random queries for benchmarking.
    
    Args:
        vocabulary: Vocabulary to sample from
        num_queries: Number of queries to generate
        query_length_range: (min, max) query length
        seed: Random seed
        
    Returns:
        List of query strings
    """
    if seed is not None:
        random.seed(seed)
    
    queries = []
    for _ in range(num_queries):
        query_length = random.randint(*query_length_range)
        words = random.choices(vocabulary[:1000], k=query_length)  # Use common words
        queries.append(' '.join(words))
    
    return queries


def benchmark_initialization(
    corpus_sizes: List[int],
    bm25_class=BM25Okapi,
    **bm25_kwargs
) -> Dict[str, Any]:
    """
    Benchmark BM25 initialization performance.
    
    Args:
        corpus_sizes: List of corpus sizes to test
        bm25_class: BM25 class to benchmark
        **bm25_kwargs: Arguments for BM25 constructor
        
    Returns:
        Benchmark results dictionary
    """
    results = {
        'corpus_sizes': corpus_sizes,
        'init_times': [],
        'docs_per_second': [],
        'class_name': bm25_class.__name__
    }
    
    for size in corpus_sizes:
        print(f"Benchmarking initialization with {size:,} documents...")
        
        corpus = generate_random_corpus(size, seed=42)
        
        start_time = time.time()
        bm25 = bm25_class(corpus, **bm25_kwargs)
        init_time = time.time() - start_time
        
        results['init_times'].append(init_time)
        results['docs_per_second'].append(size / init_time)
        
        print(f"  {init_time:.3f}s ({size/init_time:.0f} docs/sec)")
        
        del bm25, corpus  # Clean up memory
    
    return results


def benchmark_query_performance(
    corpus_size: int = 10000,
    num_queries: int = 1000,
    bm25_class=BM25Okapi,
    **bm25_kwargs
) -> Dict[str, Any]:
    """
    Benchmark query performance.
    
    Args:
        corpus_size: Size of corpus
        num_queries: Number of queries to test
        bm25_class: BM25 class to benchmark
        **bm25_kwargs: Arguments for BM25 constructor
        
    Returns:
        Benchmark results dictionary
    """
    print(f"Benchmarking query performance with {corpus_size:,} docs, {num_queries} queries...")
    
    corpus = generate_random_corpus(corpus_size, seed=42)
    vocabulary = [word for doc in corpus[:100] for word in doc.split()]  # Sample vocabulary
    queries = generate_queries(vocabulary, num_queries, seed=42)
    
    # Initialize BM25
    start_time = time.time()
    bm25 = bm25_class(corpus, **bm25_kwargs)
    init_time = time.time() - start_time
    
    # Benchmark queries
    query_times = []
    for query in queries:
        query_tokens = query.lower().split()
        
        start_time = time.time()
        scores = bm25.get_scores(query_tokens)
        query_time = time.time() - start_time
        
        query_times.append(query_time)
    
    total_query_time = sum(query_times)
    avg_query_time = total_query_time / len(query_times)
    qps = len(query_times) / total_query_time
    
    results = {
        'corpus_size': corpus_size,
        'num_queries': num_queries,
        'init_time': init_time,
        'total_query_time': total_query_time,
        'avg_query_time': avg_query_time,
        'queries_per_second': qps,
        'class_name': bm25_class.__name__
    }
    
    print(f"  Initialization: {init_time:.3f}s")
    print(f"  Total query time: {total_query_time:.3f}s")
    print(f"  Average query time: {avg_query_time*1000:.2f}ms")
    print(f"  Queries per second: {qps:.1f}")
    
    return results


def benchmark_concurrent_queries(
    corpus_size: int = 10000,
    num_threads: int = 4,
    queries_per_thread: int = 250,
    bm25_class=BM25Okapi,
    **bm25_kwargs
) -> Dict[str, Any]:
    """
    Benchmark concurrent query performance.
    
    Args:
        corpus_size: Size of corpus
        num_threads: Number of concurrent threads
        queries_per_thread: Queries per thread
        bm25_class: BM25 class to benchmark
        **bm25_kwargs: Arguments for BM25 constructor
        
    Returns:
        Benchmark results dictionary
    """
    print(f"Benchmarking concurrent queries: {num_threads} threads, {queries_per_thread} queries each...")
    
    corpus = generate_random_corpus(corpus_size, seed=42)
    vocabulary = [word for doc in corpus[:100] for word in doc.split()]
    
    # Initialize BM25
    bm25 = bm25_class(corpus, **bm25_kwargs)
    
    # Generate queries for all threads
    all_queries = generate_queries(vocabulary, num_threads * queries_per_thread, seed=42)
    
    def worker_function(queries: List[str]) -> List[float]:
        """Worker function for concurrent testing."""
        times = []
        for query in queries:
            query_tokens = query.lower().split()
            start_time = time.time()
            scores = bm25.get_scores(query_tokens)
            times.append(time.time() - start_time)
        return times
    
    # Split queries among threads
    queries_per_worker = len(all_queries) // num_threads
    query_chunks = [
        all_queries[i:i + queries_per_worker] 
        for i in range(0, len(all_queries), queries_per_worker)
    ]
    
    # Run concurrent benchmark
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker_function, chunk) for chunk in query_chunks]
        all_times = []
        for future in as_completed(futures):
            all_times.extend(future.result())
    
    total_time = time.time() - start_time
    total_queries = len(all_times)
    concurrent_qps = total_queries / total_time
    
    results = {
        'corpus_size': corpus_size,
        'num_threads': num_threads,
        'total_queries': total_queries,
        'total_time': total_time,
        'concurrent_qps': concurrent_qps,
        'avg_query_time': sum(all_times) / len(all_times),
        'class_name': bm25_class.__name__
    }
    
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Concurrent QPS: {concurrent_qps:.1f}")
    print(f"  Average query time: {results['avg_query_time']*1000:.2f}ms")
    
    return results


def benchmark_bm25(
    corpus_sizes: Optional[List[int]] = None,
    query_corpus_size: int = 10000,
    num_queries: int = 1000,
    bm25_variants: Optional[List[type]] = None,
    include_concurrent: bool = True,
    **bm25_kwargs
) -> Dict[str, Any]:
    """
    Comprehensive BM25 benchmark suite.
    
    Args:
        corpus_sizes: Corpus sizes for initialization benchmark
        query_corpus_size: Corpus size for query benchmarks
        num_queries: Number of queries to test
        bm25_variants: BM25 classes to benchmark
        include_concurrent: Whether to include concurrent benchmarks
        **bm25_kwargs: Arguments for BM25 constructors
        
    Returns:
        Complete benchmark results
    """
    if corpus_sizes is None:
        corpus_sizes = [1000, 5000, 10000, 25000]
    
    if bm25_variants is None:
        bm25_variants = [BM25Okapi, BM25Plus, BM25L]
    
    print("=" * 60)
    print("BM25-RS Comprehensive Benchmark Suite")
    print("=" * 60)
    
    all_results = {
        'initialization': {},
        'query_performance': {},
        'concurrent_performance': {}
    }
    
    # Benchmark each variant
    for bm25_class in bm25_variants:
        class_name = bm25_class.__name__
        print(f"\n--- Benchmarking {class_name} ---")
        
        # Initialization benchmark
        print(f"\n{class_name} Initialization:")
        init_results = benchmark_initialization(corpus_sizes, bm25_class, **bm25_kwargs)
        all_results['initialization'][class_name] = init_results
        
        # Query performance benchmark
        print(f"\n{class_name} Query Performance:")
        query_results = benchmark_query_performance(
            query_corpus_size, num_queries, bm25_class, **bm25_kwargs
        )
        all_results['query_performance'][class_name] = query_results
        
        # Concurrent benchmark
        if include_concurrent:
            print(f"\n{class_name} Concurrent Performance:")
            concurrent_results = benchmark_concurrent_queries(
                query_corpus_size, 4, 250, bm25_class, **bm25_kwargs
            )
            all_results['concurrent_performance'][class_name] = concurrent_results
    
    print("\n" + "=" * 60)
    print("Benchmark completed!")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    # Run benchmarks when called directly
    results = benchmark_bm25()
    
    # Print summary
    print("\n=== BENCHMARK SUMMARY ===")
    for variant_name, init_data in results['initialization'].items():
        query_data = results['query_performance'][variant_name]
        concurrent_data = results['concurrent_performance'][variant_name]
        
        print(f"\n{variant_name}:")
        print(f"  Init (10K docs): {init_data['docs_per_second'][2]:.0f} docs/sec")
        print(f"  Query performance: {query_data['queries_per_second']:.0f} QPS")
        print(f"  Concurrent performance: {concurrent_data['concurrent_qps']:.0f} QPS")