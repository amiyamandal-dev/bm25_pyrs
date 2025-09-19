# BM25-RS: High-Performance BM25 for Python

[![PyPI version](https://badge.fury.io/py/bm25-rs.svg)](https://badge.fury.io/py/bm25-rs)
[![Python versions](https://img.shields.io/pypi/pyversions/bm25-rs.svg)](https://pypi.org/project/bm25-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A blazingly fast BM25 implementation in Rust with Python bindings. This library provides high-performance text search capabilities with multiple BM25 variants, optimized for both speed and memory efficiency.

## 🚀 Features

- **🔥 High Performance**: 4000+ queries per second with sub-millisecond latency
- **🧵 Thread-Safe**: Perfect linear scaling with concurrent queries  
- **💾 Memory Efficient**: Optimized data structures with 30% less memory usage
- **🎯 Multiple Variants**: BM25Okapi, BM25Plus, and BM25L implementations
- **🐍 Python Integration**: Seamless integration with Python via PyO3
- **⚡ Batch Operations**: Efficient batch scoring for multiple documents
- **🔧 Custom Tokenization**: Support for custom tokenizers via Python callbacks

## 📦 Installation

Install from PyPI:

```bash
pip install bm25-rs
```

Or install with development dependencies:

```bash
pip install bm25-rs[dev,benchmark]
```

## 🏃‍♂️ Quick Start

```python
from bm25_rs import BM25Okapi

# Sample corpus
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "never gonna give you up never gonna let you down",
    "the answer to life the universe and everything is 42",
    "to be or not to be that is the question",
    "may the force be with you",
]

# Initialize BM25
bm25 = BM25Okapi(corpus)

# Search query
query = "the quick brown"
query_tokens = query.lower().split()

# Get relevance scores for all documents
scores = bm25.get_scores(query_tokens)
print(f"Scores: {scores}")

# Get top-k most relevant documents
top_docs = bm25.get_top_n(query_tokens, corpus, n=3)
print(f"Top documents: {top_docs}")
```

## 🎯 Advanced Usage

### Custom Tokenization

```python
def custom_tokenizer(text):
    # Your custom tokenization logic
    return text.lower().split()

bm25 = BM25Okapi(corpus, tokenizer=custom_tokenizer)
```

### Batch Operations

```python
# Score specific documents efficiently
doc_ids = [0, 2, 4]  # Document indices to score
scores = bm25.get_batch_scores(query_tokens, doc_ids)
```

### Multiple BM25 Variants

```python
from bm25_rs import BM25Okapi, BM25Plus, BM25L

# Standard BM25Okapi
bm25_okapi = BM25Okapi(corpus, k1=1.5, b=0.75, epsilon=0.25)

# BM25Plus (handles term frequency saturation)
bm25_plus = BM25Plus(corpus, k1=1.5, b=0.75, delta=1.0)

# BM25L (length normalization variant)
bm25_l = BM25L(corpus, k1=1.5, b=0.75, delta=0.5)
```

### Performance Optimization

```python
# For large corpora, use chunked processing
scores = bm25.get_scores_chunked(query_tokens, chunk_size=1000)

# Get only top-k indices (faster when you don't need full documents)
top_indices = bm25.get_top_n_indices(query_tokens, n=10)
```

## 📊 Performance Benchmarks

Performance comparison on a corpus of 10,000 documents:

| Operation | Throughput | Latency |
|-----------|------------|---------|
| Initialization | 190K docs/sec | - |
| Single Query | 4,400 QPS | 0.23ms |
| Batch Queries | 73K ops/sec | 0.01ms |
| Concurrent (4 threads) | 17,600 QPS | 0.06ms |

Memory usage: ~30% less than pure Python implementations.

## 🔧 API Reference

### BM25Okapi

```python
class BM25Okapi:
    def __init__(
        self, 
        corpus: List[str], 
        tokenizer: Optional[Callable] = None,
        k1: float = 1.5, 
        b: float = 0.75, 
        epsilon: float = 0.25
    )
    
    def get_scores(self, query: List[str]) -> List[float]
    def get_batch_scores(self, query: List[str], doc_ids: List[int]) -> List[float]
    def get_top_n(self, query: List[str], documents: List[str], n: int = 5) -> List[Tuple[str, float]]
    def get_top_n_indices(self, query: List[str], n: int = 5) -> List[Tuple[int, float]]
    def get_scores_chunked(self, query: List[str], chunk_size: int = 1000) -> List[float]
```

### Parameters

- **k1** (float): Controls term frequency saturation (default: 1.5)
- **b** (float): Controls length normalization (default: 0.75)  
- **epsilon** (float): IDF normalization parameter for BM25Okapi (default: 0.25)
- **delta** (float): Term frequency normalization for BM25Plus/BM25L (default: 1.0/0.5)

## 🛠️ Development

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/bm25-rs.git
cd bm25-rs

# Install development dependencies
pip install -e .[dev]

# Build the Rust extension
maturin develop --release
```

### Running Tests

```bash
pytest tests/
```

### Benchmarking

```bash
python benchmarks/benchmark.py
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [PyO3](https://pyo3.rs/) for Python-Rust interoperability
- Uses [Rayon](https://github.com/rayon-rs/rayon) for parallel processing
- Inspired by the [rank-bm25](https://github.com/dorianbrown/rank_bm25) Python library

## 📈 Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.

---

**Made with ❤️ and ⚡ by the BM25-RS team**
