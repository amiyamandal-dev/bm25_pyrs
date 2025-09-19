"""
Basic tests for BM25-RS implementations.
"""

import pytest
from typing import List

try:
    from bm25_rs import BM25Okapi, BM25Plus, BM25L
except ImportError:
    pytest.skip("BM25-RS not available", allow_module_level=True)


@pytest.fixture
def sample_corpus():
    """Sample corpus for testing."""
    return [
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


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        "quick brown fox",
        "never gonna give",
        "answer to life",
        "to be or not",
        "force be with",
        "hello world",
        "python programming",
        "rust memory safe",
        "machine learning",
        "search engines"
    ]


class TestBM25Okapi:
    """Test BM25Okapi implementation."""
    
    def test_initialization(self, sample_corpus):
        """Test BM25Okapi initialization."""
        bm25 = BM25Okapi(sample_corpus)
        
        assert bm25.corpus_size == len(sample_corpus)
        assert bm25.k1 == 1.5  # default value
        assert bm25.b == 0.75  # default value
        assert bm25.epsilon == 0.25  # default value
        assert bm25.avgdl > 0
    
    def test_custom_parameters(self, sample_corpus):
        """Test BM25Okapi with custom parameters."""
        bm25 = BM25Okapi(sample_corpus, k1=2.0, b=0.5, epsilon=0.1)
        
        assert bm25.k1 == 2.0
        assert bm25.b == 0.5
        assert bm25.epsilon == 0.1
    
    def test_get_scores(self, sample_corpus):
        """Test get_scores method."""
        bm25 = BM25Okapi(sample_corpus)
        query = ["quick", "brown", "fox"]
        
        scores = bm25.get_scores(query)
        
        assert len(scores) == len(sample_corpus)
        assert all(isinstance(score, float) for score in scores)
        assert all(score >= 0 for score in scores)
        
        # First document should have highest score for this query
        assert scores[0] == max(scores)
    
    def test_get_batch_scores(self, sample_corpus):
        """Test get_batch_scores method."""
        bm25 = BM25Okapi(sample_corpus)
        query = ["quick", "brown", "fox"]
        doc_ids = [0, 2, 4, 6, 8]
        
        batch_scores = bm25.get_batch_scores(query, doc_ids)
        all_scores = bm25.get_scores(query)
        
        assert len(batch_scores) == len(doc_ids)
        
        # Batch scores should match individual scores
        for i, doc_id in enumerate(doc_ids):
            assert abs(batch_scores[i] - all_scores[doc_id]) < 1e-10
    
    def test_get_top_n(self, sample_corpus):
        """Test get_top_n method."""
        bm25 = BM25Okapi(sample_corpus)
        query = ["quick", "brown", "fox"]
        
        top_docs = bm25.get_top_n(query, sample_corpus, n=3)
        
        assert len(top_docs) == 3
        assert all(isinstance(doc, str) and isinstance(score, float) 
                  for doc, score in top_docs)
        
        # Results should be sorted by score (descending)
        scores = [score for _, score in top_docs]
        assert scores == sorted(scores, reverse=True)
        
        # First result should be the document with "quick brown fox"
        assert "quick brown fox" in top_docs[0][0]
    
    def test_empty_query(self, sample_corpus):
        """Test behavior with empty query."""
        bm25 = BM25Okapi(sample_corpus)
        
        scores = bm25.get_scores([])
        assert len(scores) == len(sample_corpus)
        assert all(score == 0.0 for score in scores)
    
    def test_nonexistent_terms(self, sample_corpus):
        """Test query with terms not in corpus."""
        bm25 = BM25Okapi(sample_corpus)
        query = ["nonexistent", "terms", "xyz123"]
        
        scores = bm25.get_scores(query)
        assert len(scores) == len(sample_corpus)
        assert all(score == 0.0 for score in scores)
    
    def test_custom_tokenizer(self, sample_corpus):
        """Test with custom tokenizer."""
        def custom_tokenizer(text):
            # Simple tokenizer that also removes short words
            tokens = text.lower().split()
            return [token for token in tokens if len(token) > 2]
        
        bm25 = BM25Okapi(sample_corpus, tokenizer=custom_tokenizer)
        query = ["quick", "brown", "fox"]
        
        scores = bm25.get_scores(query)
        assert len(scores) == len(sample_corpus)
        assert isinstance(scores[0], float)


class TestBM25Plus:
    """Test BM25Plus implementation."""
    
    def test_initialization(self, sample_corpus):
        """Test BM25Plus initialization."""
        bm25 = BM25Plus(sample_corpus)
        
        assert bm25.corpus_size == len(sample_corpus)
        assert bm25.k1 == 1.5
        assert bm25.b == 0.75
        assert bm25.delta == 1.0  # default for BM25Plus
    
    def test_custom_delta(self, sample_corpus):
        """Test BM25Plus with custom delta."""
        bm25 = BM25Plus(sample_corpus, delta=0.5)
        assert bm25.delta == 0.5
    
    def test_get_scores(self, sample_corpus):
        """Test BM25Plus scoring."""
        bm25 = BM25Plus(sample_corpus)
        query = ["quick", "brown", "fox"]
        
        scores = bm25.get_scores(query)
        
        assert len(scores) == len(sample_corpus)
        assert all(isinstance(score, float) for score in scores)
        assert all(score >= 0 for score in scores)


class TestBM25L:
    """Test BM25L implementation."""
    
    def test_initialization(self, sample_corpus):
        """Test BM25L initialization."""
        bm25 = BM25L(sample_corpus)
        
        assert bm25.corpus_size == len(sample_corpus)
        assert bm25.k1 == 1.5
        assert bm25.b == 0.75
        assert bm25.delta == 0.5  # default for BM25L
    
    def test_get_scores(self, sample_corpus):
        """Test BM25L scoring."""
        bm25 = BM25L(sample_corpus)
        query = ["quick", "brown", "fox"]
        
        scores = bm25.get_scores(query)
        
        assert len(scores) == len(sample_corpus)
        assert all(isinstance(score, float) for score in scores)
        assert all(score >= 0 for score in scores)


class TestComparisons:
    """Test comparisons between BM25 variants."""
    
    def test_variant_differences(self, sample_corpus):
        """Test that different BM25 variants produce different scores."""
        query = ["quick", "brown", "fox"]
        
        bm25_okapi = BM25Okapi(sample_corpus)
        bm25_plus = BM25Plus(sample_corpus)
        bm25_l = BM25L(sample_corpus)
        
        scores_okapi = bm25_okapi.get_scores(query)
        scores_plus = bm25_plus.get_scores(query)
        scores_l = bm25_l.get_scores(query)
        
        # Scores should be different between variants
        assert scores_okapi != scores_plus
        assert scores_okapi != scores_l
        assert scores_plus != scores_l
        
        # But all should identify the same top document
        top_okapi = scores_okapi.index(max(scores_okapi))
        top_plus = scores_plus.index(max(scores_plus))
        top_l = scores_l.index(max(scores_l))
        
        assert top_okapi == top_plus == top_l == 0  # "quick brown fox" document


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_corpus(self):
        """Test initialization with empty corpus."""
        with pytest.raises(ValueError, match="Corpus size must be greater than zero"):
            BM25Okapi([])
    
    def test_invalid_doc_ids(self, sample_corpus):
        """Test batch scoring with invalid document IDs."""
        bm25 = BM25Okapi(sample_corpus)
        query = ["test"]
        
        # Test out of range document IDs
        with pytest.raises(ValueError, match="One or more document IDs are out of range"):
            bm25.get_batch_scores(query, [0, 1, 999])
    
    def test_mismatched_documents(self, sample_corpus):
        """Test get_top_n with mismatched document list."""
        bm25 = BM25Okapi(sample_corpus)
        query = ["test"]
        wrong_docs = sample_corpus[:-1]  # Remove one document
        
        with pytest.raises(ValueError, match="documents given don't match"):
            bm25.get_top_n(query, wrong_docs, n=3)


if __name__ == "__main__":
    pytest.main([__file__])