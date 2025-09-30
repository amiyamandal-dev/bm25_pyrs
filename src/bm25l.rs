use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use ahash::AHashMap;
use smallvec::SmallVec;
use string_interner::{StringInterner, DefaultSymbol, DefaultBackend};
use std::sync::Arc;

/// BM25L structure with necessary fields - optimized version
#[pyclass]
pub struct BM25L {
    #[pyo3(get)]
    k1: f64,
    #[pyo3(get)]
    b: f64,
    #[pyo3(get)]
    delta: f64,
    #[pyo3(get)]
    corpus_size: usize,
    #[pyo3(get)]
    avgdl: f64,
    // Optimized data structures
    doc_freqs: Arc<Vec<AHashMap<DefaultSymbol, u32>>>,
    idf: Arc<AHashMap<DefaultSymbol, f64>>,
    doc_len: Arc<Vec<u32>>,
    interner: Arc<StringInterner<DefaultBackend>>,
    tokenizer: Option<Py<PyAny>>,
    // Precomputed values
    k1_plus1: f64,
    one_minus_b: f64,
    b_over_avgdl: f64,
}

#[pymethods]
impl BM25L {
    #[new]
    #[pyo3(signature = (corpus, tokenizer=None, k1=None, b=None, delta=None))]
    pub fn new(
        py: Python,
        corpus: Vec<String>,
        tokenizer: Option<Bound<PyAny>>,
        k1: Option<f64>,
        b: Option<f64>,
        delta: Option<f64>,
    ) -> PyResult<Self> {
        let k1 = k1.unwrap_or(1.5);
        let b = b.unwrap_or(0.75);
        let delta = delta.unwrap_or(0.5);

        let tokenizer = tokenizer.map(|tokenizer| tokenizer.into());
        let mut interner = StringInterner::default();

        // Tokenize corpus with optimization
        let tokenized_corpus = if let Some(ref tokenizer_py) = tokenizer {
            Self::tokenize_corpus_with_py(py, &corpus, tokenizer_py)?
        } else {
            corpus
                .par_iter()
                .map(|doc| {
                    doc.split_whitespace()
                        .map(|s| s.to_lowercase())
                        .collect::<SmallVec<[String; 16]>>()
                })
                .collect()
        };

        let corpus_size = tokenized_corpus.len();
        if corpus_size == 0 {
            return Err(PyErr::new::<PyValueError, _>(
                "Corpus size must be greater than zero.",
            ));
        }

        // Initialize with optimized structures
        let (nd, doc_freqs, doc_len, avgdl) = Self::initialize_optimized(tokenized_corpus, &mut interner);
        let idf_map = Self::calc_idf_optimized(nd, corpus_size);

        // Precompute values
        let k1_plus1 = k1 + 1.0;
        let one_minus_b = 1.0 - b;
        let b_over_avgdl = b / avgdl;

        Ok(BM25L {
            k1,
            b,
            delta,
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

    /// Calculates BM25L scores for a given query - optimized version
    pub fn get_scores(&self, query: Vec<String>) -> PyResult<Vec<f64>> {
        if self.corpus_size == 0 {
            return Ok(vec![]);
        }

        // Convert query terms to symbols
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
        let delta = self.delta;

        let scores: Vec<f64> = (0..self.corpus_size)
            .into_par_iter()
            .map(|i| {
                let doc_freq = &self.doc_freqs[i];
                let dl = self.doc_len[i] as f64;
                let denominator = one_minus_b + b_over_avgdl * dl;

                query_terms.iter().fold(0.0, |score, &(symbol, idf_val)| {
                    if let Some(&freq) = doc_freq.get(&symbol) {
                        let freq_f64 = freq as f64;
                        let ctd = if denominator > 0.0 {
                            freq_f64 / denominator
                        } else {
                            0.0
                        };
                        let numerator = k1_plus1 * (ctd + delta);
                        let denom = k1 + ctd + delta;
                        if denom > 0.0 {
                            score + idf_val * numerator / denom
                        } else {
                            score
                        }
                    } else {
                        score
                    }
                })
            })
            .collect();

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

        // Convert query terms to symbols
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
        let delta = self.delta;

        let scores: Vec<f64> = doc_ids
            .into_par_iter()
            .map(|i| {
                let doc_freq = &self.doc_freqs[i];
                let dl = self.doc_len[i] as f64;
                let denominator = one_minus_b + b_over_avgdl * dl;

                query_terms.iter().fold(0.0, |score, &(symbol, idf_val)| {
                    if let Some(&freq) = doc_freq.get(&symbol) {
                        let freq_f64 = freq as f64;
                        let ctd = if denominator > 0.0 {
                            freq_f64 / denominator
                        } else {
                            0.0
                        };
                        let numerator = k1_plus1 * (ctd + delta);
                        let denom = k1 + ctd + delta;
                        if denom > 0.0 {
                            score + idf_val * numerator / denom
                        } else {
                            score
                        }
                    } else {
                        score
                    }
                })
            })
            .collect();

        Ok(scores)
    }

    /// Retrieves the top N documents with their scores
    pub fn get_top_n(
        &self,
        query: Vec<String>,
        documents: Vec<String>,
        n: usize,
    ) -> PyResult<Vec<(String, f64)>> {
        if self.corpus_size != documents.len() {
            return Err(PyErr::new::<PyValueError, _>(
                "The documents given don't match the index corpus!",
            ));
        }

        let scores = self.get_scores(query)?;
        let mut doc_scores: Vec<(String, f64)> =
            documents.into_iter().zip(scores.into_iter()).collect();

        doc_scores.par_sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_n: Vec<(String, f64)> = doc_scores.into_iter().take(n).collect();

        Ok(top_n)
    }
}

impl BM25L {
    /// Tokenizes the corpus using the provided tokenizer
    fn tokenize_corpus_with_py(
        py: Python,
        corpus: &[String],
        tokenizer_py: &Py<PyAny>,
    ) -> PyResult<Vec<SmallVec<[String; 16]>>> {
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

    /// Optimized initialization with string interning
    fn initialize_optimized(
        corpus: Vec<SmallVec<[String; 16]>>,
        interner: &mut StringInterner<DefaultBackend>,
    ) -> (
        AHashMap<DefaultSymbol, u32>,
        Vec<AHashMap<DefaultSymbol, u32>>,
        Vec<u32>,
        f64,
    ) {
        let corpus_size = corpus.len();

        // Pre-intern all unique terms
        let mut all_terms = std::collections::HashSet::new();
        for doc in &corpus {
            for term in doc {
                all_terms.insert(term.as_str());
            }
        }

        let _: Vec<_> = all_terms.iter().map(|term| interner.get_or_intern(term)).collect();

        // Process documents in parallel
        let doc_data: Vec<(AHashMap<DefaultSymbol, u32>, u32, std::collections::HashSet<DefaultSymbol>)> = corpus
            .into_par_iter()
            .map(|doc| {
                let mut freq_map = AHashMap::with_capacity(doc.len().min(64));
                let mut unique_terms = std::collections::HashSet::with_capacity(doc.len().min(64));
                
                for term in &doc {
                    if let Some(symbol) = interner.get(term) {
                        *freq_map.entry(symbol).or_insert(0) += 1;
                        unique_terms.insert(symbol);
                    }
                }
                
                (freq_map, doc.len() as u32, unique_terms)
            })
            .collect();

        // Collect results
        let mut doc_freqs = Vec::with_capacity(corpus_size);
        let mut doc_len = Vec::with_capacity(corpus_size);
        let mut total_len = 0u64;

        for (freq_map, len, _) in &doc_data {
            doc_freqs.push(freq_map.clone());
            doc_len.push(*len);
            total_len += *len as u64;
        }

        let avgdl = total_len as f64 / corpus_size as f64;

        // Compute document frequencies
        let mut nd = AHashMap::new();
        for (_, _, unique_terms) in &doc_data {
            for &symbol in unique_terms {
                *nd.entry(symbol).or_insert(0) += 1;
            }
        }

        (nd, doc_freqs, doc_len, avgdl)
    }

    /// Optimized IDF calculation for BM25L
    fn calc_idf_optimized(
        nd: AHashMap<DefaultSymbol, u32>,
        corpus_size: usize,
    ) -> AHashMap<DefaultSymbol, f64> {
        let corpus_size_f64 = corpus_size as f64;

        nd.into_iter()
            .map(|(symbol, freq)| {
                let freq_f64 = freq as f64;
                let idf_val = (corpus_size_f64 + 1.0).ln() - (freq_f64 + 0.5).ln();
                (symbol, idf_val)
            })
            .collect()
    }
}
