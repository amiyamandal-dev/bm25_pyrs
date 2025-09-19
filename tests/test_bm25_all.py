import unittest

from bm25_rs import BM25Okapi, BM25L, BM25Plus


# Sample tokenizer: splits on whitespace and lowercases the tokens
def simple_tokenizer(doc):
    return doc.lower().split()


class TestBM25Okapi(unittest.TestCase):
    def setUp(self):
        # Example corpus with non-empty documents
        self.corpus = [
            "The quick brown fox jumps over the lazy dog",
            "Never jump over the lazy dog quickly",
            "Bright foxes leap over lazy dogs in summer"
        ]
        # Initialize BM25Okapi without a custom tokenizer
        self.bm25 = BM25Okapi(self.corpus, tokenizer=None)

    def test_get_scores_basic(self):
        query = ["quick", "lazy"]
        scores = self.bm25.get_scores(query)
        self.assertEqual(len(scores), len(self.corpus))
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

    def test_get_batch_scores(self):
        query = ["over", "lazy"]
        doc_ids = [0, 2]
        scores = self.bm25.get_batch_scores(query, doc_ids)
        self.assertEqual(len(scores), len(doc_ids))
        # Both documents contain the query terms, so both should have positive scores
        self.assertGreater(scores[0], 0)
        self.assertGreater(scores[1], 0)

    def test_get_top_n(self):
        query = ["over", "lazy"]
        top_n = self.bm25.get_top_n(query, self.corpus, n=2)
        expected_top_docs = [self.corpus[1], self.corpus[2]]
        top_n = [x[0] for x in top_n]
        self.assertEqual(top_n, expected_top_docs)


class TestBM25L(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "Data science is an interdisciplinary field",
            "Machine learning is a subset of data science",
            "Artificial intelligence and data science are closely related",
        ]
        self.tokenizer = simple_tokenizer
        self.bm25l = BM25L(self.corpus, tokenizer=self.tokenizer, delta=0.5)

    def test_get_scores(self):
        query = ["data", "science"]
        scores = self.bm25l.get_scores(query)
        self.assertEqual(len(scores), 3)
        # Verify that all documents have non-zero scores
        self.assertTrue(all(score > 0 for score in scores))

    def test_batch_scores(self):
        query = ["machine", "learning"]
        doc_ids = [1]
        scores = self.bm25l.get_batch_scores(query, doc_ids)
        self.assertEqual(len(scores), 1)
        self.assertGreater(scores[0], 0)

    def test_get_top_n(self):
        query = ["artificial", "intelligence"]
        documents = self.corpus
        top_n = self.bm25l.get_top_n(query, documents, n=1)

        # Expected top document
        expected_top_doc = self.corpus[2]

        # Extract the document and score from the top_n result
        top_doc, top_score = top_n[0]

        # Assert that the top document matches the expected document
        self.assertEqual(top_doc, expected_top_doc)

        # Optionally, verify that the score is a float and greater than zero
        self.assertIsInstance(top_score, float)
        self.assertGreater(top_score, 0)


class TestBM25Plus(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "Natural language processing enables computers to understand human language",
            "Deep learning empowers natural language processing",
            "Understanding language is key to AI and machine learning",
        ]
        self.tokenizer = simple_tokenizer
        self.bm25plus = BM25Plus(self.corpus, tokenizer=self.tokenizer, delta=1)

    def test_get_scores_basic(self):
        query = ["language", "learning"]
        scores = self.bm25plus.get_scores(query)
        self.assertEqual(len(scores), 3)
        # Check that we get meaningful scores - documents with query terms should have positive scores
        # Document 0: contains "language" 
        # Document 1: contains "learning"
        # Document 2: contains both "learning" and "language" (as "AI and machine learning")
        self.assertGreater(scores[0], 0)  # Contains "language"
        self.assertGreater(scores[1], 0)  # Contains "learning" 
        self.assertGreater(scores[2], 0)  # Contains "learning"

    def test_get_scores_empty_query(self):
        query = []
        scores = self.bm25plus.get_scores(query)
        self.assertTrue(all(score == 0 for score in scores))

    def test_get_scores_nonexistent_term(self):
        query = ["quantum"]
        scores = self.bm25plus.get_scores(query)
        self.assertTrue(all(score == 0 for score in scores))

    def test_get_batch_scores(self):
        query = ["ai", "machine"]
        doc_ids = [2]
        scores = self.bm25plus.get_batch_scores(query, doc_ids)
        self.assertEqual(len(scores), 1)
        self.assertGreater(scores[0], 0)

    def test_get_top_n(self):
        query = ["deep", "learning"]
        documents = self.corpus
        top_n = self.bm25plus.get_top_n(query, documents, n=2)
        expected_top_docs = [self.corpus[1], self.corpus[2]]
        top_n = [x[0] for x in top_n]
        self.assertEqual(top_n, expected_top_docs)


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.empty_corpus = []
        self.empty_query = []
        self.tokenizer = simple_tokenizer
        self.bm25_okapi = BM25Okapi(["Test document"], tokenizer=self.tokenizer)
        self.bm25_l = BM25L(["Another test document"], tokenizer=self.tokenizer)
        self.bm25_plus = BM25Plus(["Yet another test document"], tokenizer=self.tokenizer)

    def test_empty_corpus_initialization(self):
        with self.assertRaises(ValueError):
            BM25Okapi(self.empty_corpus, tokenizer=None)

    def test_empty_query_scores(self):
        bm25 = BM25Okapi(["Test document"], tokenizer=self.tokenizer)
        scores = bm25.get_scores(self.empty_query)
        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0], 0)

    def test_query_with_all_terms_nonexistent(self):
        bm25 = BM25Okapi(["Sample document"], tokenizer=self.tokenizer)
        query = ["nonexistent", "unknown"]
        scores = bm25.get_scores(query)
        self.assertEqual(scores, [0])

    def test_documents_with_zero_length(self):
        with self.assertRaises(ValueError):
            _ = BM25Okapi([], tokenizer=None)  # One document with zero length

    def test_get_top_n_with_n_exceeding_corpus(self):
        corpus = ["doc1", "doc2"]
        bm25 = BM25Okapi(corpus, tokenizer=None)
        query = ["doc1"]
        top_n = bm25.get_top_n(query, corpus, n=5)
        self.assertEqual(len(top_n), 2)

    def test_get_batch_scores_with_invalid_doc_ids(self):
        bm25 = BM25Okapi(["doc1"], tokenizer=None)
        with self.assertRaises(ValueError):
            bm25.get_batch_scores(["doc1"], [1])  # Invalid doc_id


if __name__ == '__main__':
    unittest.main()
