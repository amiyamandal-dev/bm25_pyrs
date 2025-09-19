#!/usr/bin/env python3
"""
Basic usage example for BM25-RS.
"""

from bm25_rs import BM25Okapi, BM25Plus, BM25L
from bm25_rs.utils import tokenize_text, SimpleTokenizer


def main():
    """Demonstrate basic BM25-RS usage."""
    
    # Sample document corpus
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "never gonna give you up never gonna let you down",
        "the answer to life the universe and everything is 42", 
        "to be or not to be that is the question",
        "may the force be with you",
        "hello world this is a test document",
        "python is a great programming language",
        "rust is fast and memory safe",
        "machine learning and artificial intelligence",
        "information retrieval and search engines"
    ]
    
    print("BM25-RS Basic Usage Example")
    print("=" * 40)
    
    # Example 1: Basic BM25Okapi usage
    print("\n1. Basic BM25Okapi Usage:")
    bm25 = BM25Okapi(corpus)
    
    query = "quick brown fox"
    query_tokens = query.lower().split()
    
    print(f"Query: '{query}'")
    print(f"Corpus size: {bm25.corpus_size}")
    print(f"Average document length: {bm25.avgdl:.2f}")
    
    # Get relevance scores
    scores = bm25.get_scores(query_tokens)
    print(f"\nRelevance scores: {[f'{s:.4f}' for s in scores]}")
    
    # Get top documents
    top_docs = bm25.get_top_n(query_tokens, corpus, n=3)
    print(f"\nTop 3 documents:")
    for i, (doc, score) in enumerate(top_docs, 1):
        print(f"  {i}. Score: {score:.4f} - '{doc}'")
    
    # Example 2: Custom tokenizer
    print("\n\n2. Custom Tokenizer Example:")
    
    def custom_tokenizer(text):
        """Custom tokenizer that removes short words."""
        tokens = text.lower().split()
        return [token for token in tokens if len(token) > 2]
    
    bm25_custom = BM25Okapi(corpus, tokenizer=custom_tokenizer)
    scores_custom = bm25_custom.get_scores(custom_tokenizer(query))
    
    print(f"Query with custom tokenizer: {custom_tokenizer(query)}")
    print(f"Scores: {[f'{s:.4f}' for s in scores_custom]}")
    
    # Example 3: Using SimpleTokenizer utility
    print("\n\n3. SimpleTokenizer Utility:")
    
    tokenizer = SimpleTokenizer(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        min_token_length=2
    )
    
    sample_text = "The quick, brown fox jumps over the lazy dog!"
    tokens = tokenizer(sample_text)
    print(f"Original: '{sample_text}'")
    print(f"Tokenized: {tokens}")
    
    # Example 4: Comparing BM25 variants
    print("\n\n4. Comparing BM25 Variants:")
    
    query_tokens = ["python", "programming"]
    
    bm25_okapi = BM25Okapi(corpus)
    bm25_plus = BM25Plus(corpus)
    bm25_l = BM25L(corpus)
    
    scores_okapi = bm25_okapi.get_scores(query_tokens)
    scores_plus = bm25_plus.get_scores(query_tokens)
    scores_l = bm25_l.get_scores(query_tokens)
    
    print(f"Query: {query_tokens}")
    print(f"BM25Okapi scores: {[f'{s:.4f}' for s in scores_okapi]}")
    print(f"BM25Plus scores:  {[f'{s:.4f}' for s in scores_plus]}")
    print(f"BM25L scores:     {[f'{s:.4f}' for s in scores_l]}")
    
    # Find document with highest score for each variant
    for name, scores in [("BM25Okapi", scores_okapi), ("BM25Plus", scores_plus), ("BM25L", scores_l)]:
        best_idx = scores.index(max(scores))
        print(f"{name} top result: '{corpus[best_idx]}' (score: {scores[best_idx]:.4f})")
    
    # Example 5: Batch operations
    print("\n\n5. Batch Operations:")
    
    # Score only specific documents
    doc_ids = [0, 2, 4, 6]  # Score only these documents
    batch_scores = bm25.get_batch_scores(query_tokens, doc_ids)
    
    print(f"Scoring documents {doc_ids} for query {query_tokens}:")
    for i, (doc_id, score) in enumerate(zip(doc_ids, batch_scores)):
        print(f"  Doc {doc_id}: {score:.4f} - '{corpus[doc_id][:50]}...'")
    
    # Example 6: Parameter tuning
    print("\n\n6. Parameter Tuning:")
    
    # Test different k1 values
    k1_values = [0.5, 1.2, 1.5, 2.0]
    query_tokens = ["machine", "learning"]
    
    print(f"Query: {query_tokens}")
    print("k1 parameter effects:")
    
    for k1 in k1_values:
        bm25_tuned = BM25Okapi(corpus, k1=k1)
        scores = bm25_tuned.get_scores(query_tokens)
        max_score = max(scores)
        best_doc_idx = scores.index(max_score)
        print(f"  k1={k1}: max_score={max_score:.4f}, best_doc='{corpus[best_doc_idx][:40]}...'")


if __name__ == "__main__":
    main()