use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// BM25Okapi structure with necessary fields
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
    doc_freqs: Arc<Vec<HashMap<String, usize>>>, // Wrapped in Arc for shared read-only access
    idf: Arc<HashMap<String, f64>>,              // IDF values
    doc_len: Arc<Vec<usize>>,                    // Document lengths
    tokenizer: Option<Py<PyAny>>,                // Optional Python tokenizer
}

#[pymethods]
impl BM25Okapi {
    #[new]
    pub fn new(
        py: Python,
        corpus: Vec<String>,
        tokenizer: Option<&PyAny>,
        k1: Option<f64>,
        b: Option<f64>,
        epsilon: Option<f64>,
    ) -> PyResult<Self> {
        let k1 = k1.unwrap_or(1.5);
        let b = b.unwrap_or(0.75);
        let epsilon = epsilon.unwrap_or(0.25);

        let tokenizer = tokenizer.map(|tk| tk.into());

        let mut bm25 = BM25Okapi {
            k1,
            b,
            epsilon,
            corpus_size: 0,
            avgdl: 0.0,
            doc_freqs: Arc::new(Vec::new()),
            idf: Arc::new(HashMap::new()),
            doc_len: Arc::new(Vec::new()),
            tokenizer,
        };

        // Tokenize the corpus
        let tokenized_corpus = if let Some(ref tokenizer_py) = bm25.tokenizer {
            // Sequential tokenization due to GIL
            bm25.tokenize_corpus(py, &corpus)?
        } else {
            // Parallel tokenization
            corpus
                .par_iter()
                .map(|doc| doc.split_whitespace().map(|s| s.to_lowercase()).collect())
                .collect()
        };

        bm25.corpus_size = tokenized_corpus.len();
        if bm25.corpus_size == 0 {
            return Err(PyErr::new::<PyValueError, _>(
                "Corpus size must be greater than zero.",
            ));
        }

        // Initialize structures
        let (nd, doc_freqs, doc_len, avgdl) = bm25.initialize(tokenized_corpus);

        // Calculate IDF
        let idf_map = bm25.calc_idf(nd);

        // Update the BM25Okapi instance
        bm25.idf = Arc::new(idf_map);
        bm25.doc_freqs = Arc::new(doc_freqs);
        bm25.doc_len = Arc::new(doc_len);
        bm25.avgdl = avgdl;

        Ok(bm25)
    }

    /// Calculates BM25 scores for all documents given a query
    pub fn get_scores(&self, query: Vec<String>) -> PyResult<Vec<f64>> {
        if self.corpus_size == 0 {
            return Ok(vec![]);
        }

        // Shared data
        let idf = Arc::clone(&self.idf);
        let doc_freqs = Arc::clone(&self.doc_freqs);
        let doc_len = Arc::clone(&self.doc_len);
        let avgdl = self.avgdl;
        let k1 = self.k1;
        let b = self.b;

        // Precompute query term IDFs and filter out terms not in the corpus
        let query_terms: Vec<(&String, f64)> = query
            .iter()
            .filter_map(|q| idf.get(q).map(|&idf_val| (q, idf_val)))
            .collect();

        if query_terms.is_empty() {
            // None of the query terms are in the corpus
            return Ok(vec![0.0; self.corpus_size]);
        }

        // Precompute (k1 + 1)
        let k1_plus1 = k1 + 1.0;

        // Compute scores in parallel
        let scores: Vec<f64> = (0..self.corpus_size)
            .into_par_iter()
            .map(|i| {
                let doc_freq = &doc_freqs[i];
                let dl = doc_len[i] as f64;
                let denom = k1 * (1.0 - b + b * dl / avgdl);

                query_terms.iter().fold(0.0, |mut score, &(q, idf_val)| {
                    if let Some(&freq) = doc_freq.get(q) {
                        let freq = freq as f64;
                        let numerator = freq * k1_plus1;
                        let denominator = freq + denom;
                        if denominator > 0.0 {
                            score += idf_val * (numerator / denominator);
                        }
                    }
                    score
                })
            })
            .collect();

        Ok(scores)
    }

    /// Retrieves the top N documents for a given query, along with their scores
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

        // Create a vector of (document, score) pairs
        let mut doc_scores: Vec<(String, f64)> =
            documents.into_iter().zip(scores.into_iter()).collect();

        // Sort the documents by score in descending order
        doc_scores.par_sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top n documents with their scores
        let top_n: Vec<(String, f64)> = doc_scores.into_iter().take(n).collect();

        Ok(top_n)
    }

    /// Calculates BM25 scores for a batch of documents given a query
    pub fn get_batch_scores(&self, query: Vec<String>, doc_ids: Vec<usize>) -> PyResult<Vec<f64>> {
        if doc_ids.is_empty() {
            return Ok(vec![]);
        }

        if doc_ids.iter().any(|&di| di >= self.corpus_size) {
            return Err(PyErr::new::<PyValueError, _>(
                "One or more document IDs are out of range.",
            ));
        }

        // Shared data
        let idf = Arc::clone(&self.idf);
        let doc_freqs = Arc::clone(&self.doc_freqs);
        let doc_len = Arc::clone(&self.doc_len);
        let avgdl = self.avgdl;
        let k1 = self.k1;
        let b = self.b;

        // Precompute query term IDFs and filter out terms not in the corpus
        let query_terms: Vec<(&String, f64)> = query
            .iter()
            .filter_map(|q| idf.get(q).map(|&idf_val| (q, idf_val)))
            .collect();

        if query_terms.is_empty() {
            // None of the query terms are in the corpus
            return Ok(vec![0.0; doc_ids.len()]);
        }

        // Precompute (k1 + 1)
        let k1_plus1 = k1 + 1.0;

        // Compute scores in parallel
        let scores: Vec<f64> = doc_ids
            .into_par_iter()
            .map(|i| {
                let doc_freq = &doc_freqs[i];
                let dl = doc_len[i] as f64;
                let denom = k1 * (1.0 - b + b * dl / avgdl);

                query_terms.iter().fold(0.0, |mut score, &(q, idf_val)| {
                    if let Some(&freq) = doc_freq.get(q) {
                        let freq = freq as f64;
                        let numerator = freq * k1_plus1;
                        let denominator = freq + denom;
                        if denominator > 0.0 {
                            score += idf_val * (numerator / denominator);
                        }
                    }
                    score
                })
            })
            .collect();

        Ok(scores)
    }
}

impl BM25Okapi {
    /// Tokenizes the corpus using the provided tokenizer
    fn tokenize_corpus(&self, py: Python, corpus: &[String]) -> PyResult<Vec<Vec<String>>> {
        let tokenizer_py = self.tokenizer.as_ref().unwrap();

        // Sequential tokenization due to GIL requirements
        let mut tokenized_corpus = Vec::with_capacity(corpus.len());
        for doc in corpus {
            let tokens: Vec<String> = tokenizer_py
                .call1(py, (doc,))
                .map_err(|e| PyValueError::new_err(format!("Tokenizer failed: {}", e)))?
                .extract(py)
                .map_err(|e| PyValueError::new_err(format!("Failed to extract tokens: {}", e)))?;
            tokenized_corpus.push(tokens);
        }

        Ok(tokenized_corpus)
    }

    /// Initializes document frequencies and other metrics
    fn initialize(
        &self,
        corpus: Vec<Vec<String>>,
    ) -> (
        HashMap<String, usize>,      // nd
        Vec<HashMap<String, usize>>, // doc_freqs
        Vec<usize>,                  // doc_len
        f64,                         // avgdl
    ) {
        let corpus_size = corpus.len();

        // Calculate document lengths and frequencies in parallel
        let doc_data: Vec<(HashMap<String, usize>, usize, HashSet<String>)> = corpus
            .into_par_iter()
            .map(|doc| {
                let mut freq_map = HashMap::new();
                for term in &doc {
                    *freq_map.entry(term.clone()).or_insert(0) += 1;
                }
                let unique_terms: HashSet<String> = freq_map.keys().cloned().collect();
                (freq_map, doc.len(), unique_terms)
            })
            .collect();

        // Collect doc_freqs and doc_len
        let doc_freqs: Vec<HashMap<String, usize>> = doc_data
            .iter()
            .map(|(freq_map, _, _)| freq_map.clone())
            .collect();

        let doc_len: Vec<usize> = doc_data.iter().map(|(_, len, _)| *len).collect();

        let total_len: usize = doc_len.iter().sum();
        let avgdl = total_len as f64 / corpus_size as f64;

        // Compute nd (document frequencies)
        let mut nd = HashMap::new();
        for (_, _, unique_terms) in &doc_data {
            for term in unique_terms {
                *nd.entry(term.clone()).or_insert(0) += 1;
            }
        }

        (nd, doc_freqs, doc_len, avgdl)
    }

    /// Calculates the inverse document frequency (IDF) with negative IDF adjustment
    fn calc_idf(&self, nd: HashMap<String, usize>) -> HashMap<String, f64> {
        let corpus_size = self.corpus_size as f64;
        let epsilon = self.epsilon;

        // Compute initial IDF values
        let idf_values: Vec<(String, f64)> = nd
            .par_iter()
            .map(|(term, &doc_freq)| {
                let idf = ((corpus_size - doc_freq as f64 + 0.5) / (doc_freq as f64 + 0.5)).ln();
                (term.clone(), idf)
            })
            .collect();

        // Compute average IDF and adjust negative values
        let idf_sum: f64 = idf_values.par_iter().map(|(_, idf)| *idf).sum();
        let average_idf = idf_sum / idf_values.len() as f64;
        let eps = epsilon * average_idf;

        // Adjust negative IDFs
        idf_values
            .into_par_iter()
            .map(
                |(term, idf)| {
                    if idf < 0.0 {
                        (term, eps)
                    } else {
                        (term, idf)
                    }
                },
            )
            .collect()
    }
}
