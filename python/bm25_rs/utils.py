"""
Utility functions for BM25-RS package.
"""

import re
import string
from typing import List, Optional, Callable, Union


def tokenize_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_stopwords: bool = False,
    stopwords: Optional[List[str]] = None,
) -> List[str]:
    """
    Simple text tokenization utility.

    Args:
        text: Input text to tokenize
        lowercase: Convert to lowercase
        remove_punctuation: Remove punctuation marks
        remove_stopwords: Remove common stopwords
        stopwords: Custom stopwords list

    Returns:
        List of tokens
    """
    if lowercase:
        text = text.lower()

    if remove_punctuation:
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

    # Split on whitespace
    tokens = text.split()

    if remove_stopwords:
        if stopwords is None:
            # Basic English stopwords
            stopwords = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
                'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if'
            }
        else:
            stopwords = set(stopwords)

        tokens = [token for token in tokens if token not in stopwords]

    return tokens


def preprocess_corpus(
    corpus: List[str],
    tokenizer: Optional[Callable[[str], List[str]]] = None,
    **tokenizer_kwargs
) -> List[str]:
    """
    Preprocess a corpus of documents.

    Args:
        corpus: List of document strings
        tokenizer: Custom tokenizer function
        **tokenizer_kwargs: Arguments for default tokenizer

    Returns:
        List of preprocessed document strings
    """
    if tokenizer is None:
        tokenizer = lambda text: tokenize_text(text, **tokenizer_kwargs)

    processed_corpus = []
    for doc in corpus:
        tokens = tokenizer(doc)
        processed_corpus.append(' '.join(tokens))

    return processed_corpus


def validate_corpus(corpus: List[str]) -> None:
    """
    Validate corpus format and content.

    Args:
        corpus: List of document strings

    Raises:
        ValueError: If corpus is invalid
    """
    if not isinstance(corpus, list):
        raise ValueError("Corpus must be a list of strings")

    if len(corpus) == 0:
        raise ValueError("Corpus cannot be empty")

    for i, doc in enumerate(corpus):
        if not isinstance(doc, str):
            raise ValueError(f"Document at index {i} must be a string, got {type(doc)}")

        if len(doc.strip()) == 0:
            raise ValueError(f"Document at index {i} is empty or contains only whitespace")


def validate_query(query: List[str]) -> None:
    """
    Validate query format.

    Args:
        query: List of query terms

    Raises:
        ValueError: If query is invalid
    """
    if not isinstance(query, list):
        raise ValueError("Query must be a list of strings")

    if len(query) == 0:
        raise ValueError("Query cannot be empty")

    for i, term in enumerate(query):
        if not isinstance(term, str):
            raise ValueError(f"Query term at index {i} must be a string, got {type(term)}")


class SimpleTokenizer:
    """
    A simple, configurable tokenizer class.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = False,
        stopwords: Optional[List[str]] = None,
        min_token_length: int = 1,
        max_token_length: Optional[int] = None,
    ):
        """
        Initialize the tokenizer.

        Args:
            lowercase: Convert tokens to lowercase
            remove_punctuation: Remove punctuation from tokens
            remove_stopwords: Filter out stopwords
            stopwords: Custom stopwords list
            min_token_length: Minimum token length
            max_token_length: Maximum token length
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length

        if stopwords is None and remove_stopwords:
            self.stopwords = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
                'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if'
            }
        else:
            self.stopwords = set(stopwords) if stopwords else set()

    def __call__(self, text: str) -> List[str]:
        """
        Tokenize the input text.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        return tokenize_text(
            text,
            lowercase=self.lowercase,
            remove_punctuation=self.remove_punctuation,
            remove_stopwords=self.remove_stopwords,
            stopwords=list(self.stopwords) if self.stopwords else None,
        )


def create_tokenizer(
    tokenizer_type: str = "simple",
    **kwargs
) -> Callable[[str], List[str]]:
    """
    Factory function to create tokenizers.

    Args:
        tokenizer_type: Type of tokenizer ("simple", "whitespace", "regex")
        **kwargs: Tokenizer-specific arguments

    Returns:
        Tokenizer function
    """
    if tokenizer_type == "simple":
        return SimpleTokenizer(**kwargs)
    elif tokenizer_type == "whitespace":
        return lambda text: text.lower().split() if kwargs.get("lowercase", True) else text.split()
    elif tokenizer_type == "regex":
        pattern = kwargs.get("pattern", r'\b\w+\b')
        lowercase = kwargs.get("lowercase", True)

        def regex_tokenizer(text: str) -> List[str]:
            tokens = re.findall(pattern, text)
            return [token.lower() for token in tokens] if lowercase else tokens

        return regex_tokenizer
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
