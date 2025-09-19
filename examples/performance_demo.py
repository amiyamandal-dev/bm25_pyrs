#!/usr/bin/env python3
"""
Performance demonstration for BM25-RS.
"""

import time
import random
import string
from concurrent.futures import ThreadPoolExecutor
from bm25_rs import BM25Okapi
from bm25_rs.benchmarks import generate_random_corpus, generate_queries


def demo_initialization_speed():
    """Demonstrate initialization performance."""
    print("Initialization Speed Demo")
    print("-" * 30)
    
    corpus_sizes = [1000, 5000, 10000, 25000]
    
    for size in corpus_sizes:
        corpus = generate_random_corpus(size, seed=42)
        
        start_time = time.time()
        bm25 = BM25Okapi(corpus)
        init_time = time.time() - start_time
        
        docs_per_sec = size / init_time
        print(f"{size:5d} docs: {init_time:.3f}s ({docs_per_sec:,.0f} docs/sec)")


def demo_query_speed():
    """Demonstrate query performance."""
    print("\n\nQuery Speed Demo")
    print("-" * 20)
    
    # Generate test data
    corpus = generate_random_corpus(10000, seed=42)
    vocabulary = [word for doc in corpus[:50] for word in doc.split()]
    queries = generate_queries(vocabulary, 1000, seed=42)
    
    # Initialize BM25
    print("Initializing BM25 with 10,000 documents...")
    start_time = time.time()
    bm25 = BM25Okapi(corpus)
    init_time = time.time() - start_time
    print(f"Initialization: {init_time:.3f}s")
    
    # Benchmark queries
    print("\nRunning 1,000 queries...")
    query_times = []
    
    start_time = time.time()
    for query in queries:
        query_tokens = query.lower().split()
        
        query_start = time.time()
        scores = bm25.get_scores(query_tokens)
        query_time = time.time() - query_start
        
        query_times.append(query_time)
    
    total_time = time.time() - start_time
    avg_time = sum(query_times) / len(query_times)
    qps = len(queries) / total_time
    
    print(f"Total time: {total_time:.3f}s")
    print(f"Average query time: {avg_time*1000:.2f}ms")
    print(f"Queries per second: {qps:.1f}")
    print(f"Min query time: {min(query_times)*1000:.2f}ms")
    print(f"Max query time: {max(query_times)*1000:.2f}ms")


def demo_concurrent_performance():
    """Demonstrate concurrent query performance."""
    print("\n\nConcurrent Performance Demo")
    print("-" * 30)
    
    # Generate test data
    corpus = generate_random_corpus(5000, seed=42)
    vocabulary = [word for doc in corpus[:50] for word in doc.split()]
    
    bm25 = BM25Okapi(corpus)
    
    def worker_function(num_queries):
        """Worker function for concurrent testing."""
        queries = generate_queries(vocabulary, num_queries, seed=random.randint(1, 1000))
        times = []
        
        for query in queries:
            query_tokens = query.lower().split()
            start_time = time.time()
            scores = bm25.get_scores(query_tokens)
            times.append(time.time() - start_time)
        
        return times
    
    # Test different thread counts
    thread_counts = [1, 2, 4, 8]
    queries_per_thread = 100
    
    for num_threads in thread_counts:
        print(f"\nTesting with {num_threads} thread(s), {queries_per_thread} queries each:")
        
        start_time = time.time()
        
        if num_threads == 1:
            # Sequential execution
            all_times = worker_function(queries_per_thread)
        else:
            # Concurrent execution
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker_function, queries_per_thread) 
                          for _ in range(num_threads)]
                all_times = []
                for future in futures:
                    all_times.extend(future.result())
        
        total_time = time.time() - start_time
        total_queries = len(all_times)
        qps = total_queries / total_time
        avg_query_time = sum(all_times) / len(all_times)
        
        print(f"  Total queries: {total_queries}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  QPS: {qps:.1f}")
        print(f"  Avg query time: {avg_query_time*1000:.2f}ms")


def demo_batch_operations():
    """Demonstrate batch operation performance."""
    print("\n\nBatch Operations Demo")
    print("-" * 25)
    
    corpus = generate_random_corpus(10000, seed=42)
    bm25 = BM25Okapi(corpus)
    
    query = ["machine", "learning", "algorithm"]
    
    # Test different batch sizes
    batch_sizes = [10, 100, 1000, 5000]
    
    print(f"Query: {query}")
    print("Batch size performance:")
    
    for batch_size in batch_sizes:
        doc_ids = random.sample(range(len(corpus)), min(batch_size, len(corpus)))
        
        # Warm up
        for _ in range(5):
            bm25.get_batch_scores(query, doc_ids)
        
        # Benchmark
        start_time = time.time()
        num_iterations = 50
        for _ in range(num_iterations):
            scores = bm25.get_batch_scores(query, doc_ids)
        
        total_time = time.time() - start_time
        ops_per_sec = num_iterations / total_time
        avg_time = total_time / num_iterations
        
        print(f"  {batch_size:4d} docs: {ops_per_sec:6.1f} ops/sec, {avg_time*1000:.2f}ms avg")


def demo_memory_efficiency():
    """Demonstrate memory efficiency."""
    print("\n\nMemory Efficiency Demo")
    print("-" * 25)
    
    try:
        import psutil
        import os
        import gc
        
        process = psutil.Process(os.getpid())
        
        corpus_sizes = [1000, 5000, 10000, 20000]
        
        print("Memory usage by corpus size:")
        
        for size in corpus_sizes:
            # Clean up before measurement
            gc.collect()
            mem_before = process.memory_info().rss / (1024 * 1024)
            
            # Create corpus and BM25 instance
            corpus = generate_random_corpus(size, avg_doc_length=50, seed=42)
            bm25 = BM25Okapi(corpus)
            
            # Measure memory after
            mem_after = process.memory_info().rss / (1024 * 1024)
            mem_used = mem_after - mem_before
            mem_per_doc = mem_used / size * 1024  # KB per document
            
            print(f"  {size:5d} docs: {mem_used:6.1f} MB total, {mem_per_doc:.2f} KB/doc")
            
            # Clean up
            del bm25, corpus
            gc.collect()
            
    except ImportError:
        print("psutil not available - install with: pip install psutil")


def demo_scaling_behavior():
    """Demonstrate scaling behavior with corpus size."""
    print("\n\nScaling Behavior Demo")
    print("-" * 25)
    
    corpus_sizes = [1000, 2000, 5000, 10000, 20000]
    query = ["test", "query", "performance"]
    
    print("Query performance vs corpus size:")
    print("Size     | Init Time | Query Time | QPS")
    print("-" * 40)
    
    for size in corpus_sizes:
        corpus = generate_random_corpus(size, seed=42)
        
        # Measure initialization time
        start_time = time.time()
        bm25 = BM25Okapi(corpus)
        init_time = time.time() - start_time
        
        # Measure query time (average of 100 queries)
        query_times = []
        for _ in range(100):
            start_time = time.time()
            scores = bm25.get_scores(query)
            query_times.append(time.time() - start_time)
        
        avg_query_time = sum(query_times) / len(query_times)
        qps = 1.0 / avg_query_time
        
        print(f"{size:5d}    | {init_time:8.3f}s | {avg_query_time*1000:9.2f}ms | {qps:6.1f}")


def main():
    """Run all performance demonstrations."""
    print("BM25-RS Performance Demonstration")
    print("=" * 50)
    
    demo_initialization_speed()
    demo_query_speed()
    demo_concurrent_performance()
    demo_batch_operations()
    demo_memory_efficiency()
    demo_scaling_behavior()
    
    print("\n" + "=" * 50)
    print("Performance demonstration completed!")


if __name__ == "__main__":
    main()