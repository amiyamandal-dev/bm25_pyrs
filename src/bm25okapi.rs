use ahash::{AHashMap, AHashSet};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::sync::Arc;
use string_interner::{DefaultBackend, DefaultSymbol, StringInterner};

/// BM25Okapi structure with necessary fields - optimized version
#[pyclass]
pub struct BM25Okapi {
    #[pyo3(get)]
    k1: f64,
    #[pyo3(get)]
    b: f64,
    #[pyo3(get)]
    epsilon: f64,
    #[pyo3(get)]
    corpus_size: usize,
    #[pyo3(get)]
    avgdl: f64,
    // Optimized data structures
    doc_freqs: Arc<Vec<AHashMap<DefaultSymbol, u32>>>, // Use symbols and u32 for better cache performance
    idf: Arc<AHashMap<DefaultSymbol, f64>>,            // Use AHashMap for better performance
    doc_len: Arc<Vec<u32>>, // Use u32 instead of usize for better cache performance
    interner: Arc<StringInterner<DefaultBackend>>, // String interner for vocabulary
    tokenizer: Option<Py<PyAny>>, // Optional Python tokenizer
    // Precomputed values for faster scoring
    k1_plus1: f64,
    one_minus_b: f64,
    b_over_avgdl: f64,
}

#[pymethods]
impl BM25Okapi {
    #[new]
    #[pyo3(signature = (corpus, tokenizer=None, k1=None, b=None, epsilon=None))]
    pub fn new(
        py: Python,
        corpus: Vec<String>,
        tokenizer: Option<Bound<PyAny>>,
        k1: Option<f64>,
        b: Option<f64>,
        epsilon: Option<f64>,
    ) -> PyResult<Self> {
        let k1 = k1.unwrap_or(1.5);
        let b = b.unwrap_or(0.75);
        let epsilon = epsilon.unwrap_or(0.25);

        let tokenizer = tokenizer.map(|tk| tk.into());
        let mut interner = StringInterner::default();

        // Tokenize the corpus
        let tokenized_corpus = if let Some(ref tokenizer_py) = tokenizer {
            // Sequential tokenization due to GIL
            Self::tokenize_corpus_with_py(py, &corpus, tokenizer_py)?
        } else {
            // Optimized parallel tokenization with string interning
            corpus
                .par_iter()
                .map(|doc| {
                    doc.split_whitespace()
                        .map(|s| s.to_lowercase())
                        .collect::<SmallVec<[String; 16]>>() // Use SmallVec for better performance on small docs
                })
                .collect()
        };

        let corpus_size = tokenized_corpus.len();
        if corpus_size == 0 {
            return Err(PyErr::new::<PyValueError, _>(
                "Corpus size must be greater than zero.",
            ));
        }

        // Initialize structures with optimized data types
        let (nd, doc_freqs, doc_len, avgdl) =
            Self::initialize_optimized(tokenized_corpus, &mut interner);

        // Calculate IDF with optimized algorithm
        let idf_map = Self::calc_idf_optimized(nd, corpus_size, epsilon);

        // Precompute frequently used values
        let k1_plus1 = k1 + 1.0;
        let one_minus_b = 1.0 - b;
        let b_over_avgdl = b / avgdl;

        Ok(BM25Okapi {
            k1,
            b,
            epsilon,
            corpus_size,
            avgdl,
            doc_freqs: Arc::new(doc_freqs),
            idf: Arc::new(idf_map),
            doc_len: Arc::new(doc_len),
            interner: Arc::new(interner),
            tokenizer,
            k1_plus1,
            one_minus_b,
            b_over_avgdl,
        })
    }

    /// Calculates BM25 scores for all documents given a query - optimized version
    pub fn get_scores(&self, query: Vec<String>) -> PyResult<Vec<f64>> {
        if self.corpus_size == 0 {
            return Ok(vec![]);
        }

        // Convert query terms to symbols for faster lookup
        let query_symbols: SmallVec<[DefaultSymbol; 8]> = query
            .iter()
            .filter_map(|term| self.interner.get(term))
            .collect();

        if query_symbols.is_empty() {
            return Ok(vec![0.0; self.corpus_size]);
        }

        // Precompute query term IDFs - use symbols for faster lookup
        let query_terms: SmallVec<[(DefaultSymbol, f64); 8]> = query_symbols
            .iter()
            .filter_map(|&symbol| self.idf.get(&symbol).map(|&idf_val| (symbol, idf_val)))
            .collect();

        if query_terms.is_empty() {
            return Ok(vec![0.0; self.corpus_size]);
        }

        // Use precomputed values for better performance
        let k1_plus1 = self.k1_plus1;
        let one_minus_b = self.one_minus_b;
        let b_over_avgdl = self.b_over_avgdl;
        let k1 = self.k1;

        // Compute scores in parallel with optimized algorithm
        let scores: Vec<f64> = (0..self.corpus_size)
            .into_par_iter()
            .map(|i| {
                let doc_freq = &self.doc_freqs[i];
                let dl = self.doc_len[i] as f64;
                let norm_factor = k1 * (one_minus_b + b_over_avgdl * dl);

                // Use fold with early termination for better performance
                query_terms.iter().fold(0.0, |score, &(symbol, idf_val)| {
                    if let Some(&freq) = doc_freq.get(&symbol) {
                        let freq_f64 = freq as f64;
                        let numerator = freq_f64 * k1_plus1;
                        let denominator = freq_f64 + norm_factor;
                        score + idf_val * (numerator / denominator)
                    } else {
                        score
                    }
                })
            })
            .collect();

        Ok(scores)
    }

    /// Retrieves the top N documents for a given query, along with their scores
    #[pyo3(signature = (query, documents, n=None))]
    pub fn get_top_n(
        &self,
        query: Vec<String>,
        documents: Vec<String>,
        n: Option<usize>,
    ) -> PyResult<Vec<(String, f64)>> {
        let n = n.unwrap_or(5);
        if self.corpus_size != documents.len() {
            return Err(PyErr::new::<PyValueError, _>(
                "The documents given don't match the index corpus!",
            ));
        }

        let scores = self.get_scores(query)?;

        // Use optimized top-k selection for better performance
        let top_indices = crate::optimizations::select_top_k_indices(&scores, n);

        let top_n: Vec<(String, f64)> = top_indices
            .into_iter()
            .map(|i| (documents[i].clone(), scores[i]))
            .collect();

        Ok(top_n)
    }

    /// Optimized method to get only top N document indices (faster when you don't need the full documents)
    #[pyo3(signature = (query, n=None))]
    pub fn get_top_n_indices(
        &self,
        query: Vec<String>,
        n: Option<usize>,
    ) -> PyResult<Vec<(usize, f64)>> {
        let n = n.unwrap_or(5);
        let scores = self.get_scores(query)?;

        let top_indices = crate::optimizations::select_top_k_indices(&scores, n);

        let result: Vec<(usize, f64)> = top_indices.into_iter().map(|i| (i, scores[i])).collect();

        Ok(result)
    }

    /// Batch scoring with chunked processing for better cache performance
    #[pyo3(signature = (query, chunk_size=None))]
    pub fn get_scores_chunked(
        &self,
        query: Vec<String>,
        chunk_size: Option<usize>,
    ) -> PyResult<Vec<f64>> {
        let chunk_size = chunk_size.unwrap_or(1000);

        if self.corpus_size == 0 {
            return Ok(vec![]);
        }

        // Convert query terms to symbols for faster lookup
        let query_symbols: SmallVec<[DefaultSymbol; 8]> = query
            .iter()
            .filter_map(|term| self.interner.get(term))
            .collect();

        if query_symbols.is_empty() {
            return Ok(vec![0.0; self.corpus_size]);
        }

        // Precompute query term IDFs
        let query_terms: SmallVec<[(DefaultSymbol, f64); 8]> = query_symbols
            .iter()
            .filter_map(|&symbol| self.idf.get(&symbol).map(|&idf_val| (symbol, idf_val)))
            .collect();

        if query_terms.is_empty() {
            return Ok(vec![0.0; self.corpus_size]);
        }

        // Use precomputed values
        let k1_plus1 = self.k1_plus1;
        let one_minus_b = self.one_minus_b;
        let b_over_avgdl = self.b_over_avgdl;
        let k1 = self.k1;

        // Process in chunks for better cache performance
        let scores = crate::optimizations::process_documents_in_chunks(
            self.corpus_size,
            chunk_size,
            |start, end| {
                (start..end)
                    .into_par_iter()
                    .map(|i| {
                        let doc_freq = &self.doc_freqs[i];
                        let dl = self.doc_len[i] as f64;
                        let norm_factor = k1 * (one_minus_b + b_over_avgdl * dl);

                        crate::optimizations::compute_bm25_score_vectorized(
                            &query_terms,
                            doc_freq,
                            dl,
                            norm_factor,
                            k1_plus1,
                        )
                    })
                    .collect()
            },
        );

        Ok(scores)
    }

    /// Calculates BM25 scores for a batch of documents given a query - optimized version
    pub fn get_batch_scores(&self, query: Vec<String>, doc_ids: Vec<usize>) -> PyResult<Vec<f64>> {
        if doc_ids.is_empty() {
            return Ok(vec![]);
        }

        if doc_ids.iter().any(|&di| di >= self.corpus_size) {
            return Err(PyErr::new::<PyValueError, _>(
                "One or more document IDs are out of range.",
            ));
        }

        // Convert query terms to symbols for faster lookup
        let query_symbols: SmallVec<[DefaultSymbol; 8]> = query
            .iter()
            .filter_map(|term| self.interner.get(term))
            .collect();

        if query_symbols.is_empty() {
            return Ok(vec![0.0; doc_ids.len()]);
        }

        // Precompute query term IDFs
        let query_terms: SmallVec<[(DefaultSymbol, f64); 8]> = query_symbols
            .iter()
            .filter_map(|&symbol| self.idf.get(&symbol).map(|&idf_val| (symbol, idf_val)))
            .collect();

        if query_terms.is_empty() {
            return Ok(vec![0.0; doc_ids.len()]);
        }

        // Use precomputed values
        let k1_plus1 = self.k1_plus1;
        let one_minus_b = self.one_minus_b;
        let b_over_avgdl = self.b_over_avgdl;
        let k1 = self.k1;

        // Compute scores in parallel with optimized algorithm
        let scores: Vec<f64> = doc_ids
            .into_par_iter()
            .map(|i| {
                let doc_freq = &self.doc_freqs[i];
                let dl = self.doc_len[i] as f64;
                let norm_factor = k1 * (one_minus_b + b_over_avgdl * dl);

                query_terms.iter().fold(0.0, |score, &(symbol, idf_val)| {
                    if let Some(&freq) = doc_freq.get(&symbol) {
                        let freq_f64 = freq as f64;
                        let numerator = freq_f64 * k1_plus1;
                        let denominator = freq_f64 + norm_factor;
                        score + idf_val * (numerator / denominator)
                    } else {
                        score
                    }
                })
            })
            .collect();

        Ok(scores)
    }
}

impl BM25Okapi {
    /// Tokenizes the corpus using the provided tokenizer
    fn tokenize_corpus_with_py(
        py: Python,
        corpus: &[String],
        tokenizer_py: &Py<PyAny>,
    ) -> PyResult<Vec<SmallVec<[String; 16]>>> {
        // Sequential tokenization due to GIL requirements
        let mut tokenized_corpus = Vec::with_capacity(corpus.len());
        for doc in corpus {
            let tokens: Vec<String> = tokenizer_py
                .call1(py, (doc,))
                .map_err(|e| PyValueError::new_err(format!("Tokenizer failed: {}", e)))?
                .extract(py)
                .map_err(|e| PyValueError::new_err(format!("Failed to extract tokens: {}", e)))?;
            tokenized_corpus.push(SmallVec::from_vec(tokens));
        }

        Ok(tokenized_corpus)
    }

    /// Optimized initialization with string interning and better data structures
    fn initialize_optimized(
        corpus: Vec<SmallVec<[String; 16]>>,
        interner: &mut StringInterner<DefaultBackend>,
    ) -> (
        AHashMap<DefaultSymbol, u32>,      // nd
        Vec<AHashMap<DefaultSymbol, u32>>, // doc_freqs
        Vec<u32>,                          // doc_len
        f64,                               // avgdl
    ) {
        let corpus_size = corpus.len();

        // Pre-intern all unique terms for better memory efficiency
        let mut all_terms = AHashSet::new();
        for doc in &corpus {
            for term in doc {
                all_terms.insert(term.as_str());
            }
        }

        // Intern all terms at once
        let _: Vec<_> = all_terms
            .iter()
            .map(|term| interner.get_or_intern(term))
            .collect();

        // Calculate document lengths and frequencies in parallel with optimized data structures
        let doc_data: Vec<(AHashMap<DefaultSymbol, u32>, u32, AHashSet<DefaultSymbol>)> = corpus
            .into_par_iter()
            .map(|doc| {
                let mut freq_map = AHashMap::with_capacity(doc.len().min(64)); // Reasonable initial capacity
                let mut unique_terms = AHashSet::with_capacity(doc.len().min(64));

                for term in &doc {
                    // This is safe because we pre-interned all terms
                    if let Some(symbol) = interner.get(term) {
                        *freq_map.entry(symbol).or_insert(0) += 1;
                        unique_terms.insert(symbol);
                    }
                }

                (freq_map, doc.len() as u32, unique_terms)
            })
            .collect();

        // Collect doc_freqs and doc_len with better memory layout
        let mut doc_freqs = Vec::with_capacity(corpus_size);
        let mut doc_len = Vec::with_capacity(corpus_size);
        let mut total_len = 0u64;

        for (freq_map, len, _) in &doc_data {
            doc_freqs.push(freq_map.clone());
            doc_len.push(*len);
            total_len += *len as u64;
        }

        let avgdl = total_len as f64 / corpus_size as f64;

        // Compute nd (document frequencies) more efficiently
        let mut nd = AHashMap::new();
        for (_, _, unique_terms) in &doc_data {
            for &symbol in unique_terms {
                *nd.entry(symbol).or_insert(0) += 1;
            }
        }

        (nd, doc_freqs, doc_len, avgdl)
    }

    /// Optimized IDF calculation with better numerical stability
    fn calc_idf_optimized(
        nd: AHashMap<DefaultSymbol, u32>,
        corpus_size: usize,
        epsilon: f64,
    ) -> AHashMap<DefaultSymbol, f64> {
        let corpus_size_f64 = corpus_size as f64;

        // Compute initial IDF values in parallel
        let idf_values: Vec<(DefaultSymbol, f64)> = nd
            .par_iter()
            .map(|(&symbol, &doc_freq)| {
                let doc_freq_f64 = doc_freq as f64;
                // Use more numerically stable IDF calculation
                let idf = ((corpus_size_f64 - doc_freq_f64 + 0.5) / (doc_freq_f64 + 0.5)).ln();
                (symbol, idf)
            })
            .collect();

        // Compute average IDF for negative adjustment
        let idf_sum: f64 = idf_values.par_iter().map(|(_, idf)| *idf).sum();
        let average_idf = idf_sum / idf_values.len() as f64;
        let eps = epsilon * average_idf;

        // Adjust negative IDFs and collect into AHashMap
        idf_values
            .into_iter()
            .map(|(symbol, idf)| {
                let adjusted_idf = if idf < 0.0 { eps } else { idf };
                (symbol, adjusted_idf)
            })
            .collect()
    }
}
