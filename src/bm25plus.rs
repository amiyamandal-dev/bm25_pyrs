use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// BM25Plus structure with necessary fields
#[pyclass]
pub struct BM25Plus {
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
    doc_freqs: Arc<Vec<HashMap<String, usize>>>,
    idf: Arc<HashMap<String, f64>>,
    doc_len: Arc<Vec<usize>>,
    tokenizer: Option<Py<PyAny>>,
}

#[pymethods]
impl BM25Plus {
    #[new]
    pub fn new(
        py: Python,
        corpus: Vec<String>,
        tokenizer: Option<&PyAny>,
        k1: Option<f64>,
        b: Option<f64>,
        delta: Option<f64>,
    ) -> PyResult<Self> {
        let k1 = k1.unwrap_or(1.5);
        let b = b.unwrap_or(0.75);
        let delta = delta.unwrap_or(1.0); // Default delta for BM25Plus is 1.0

        let tokenizer = tokenizer.map(|tokenizer| tokenizer.into());

        let mut bm25 = BM25Plus {
            k1,
            b,
            delta,
            corpus_size: 0,
            avgdl: 0.0,
            doc_freqs: Arc::new(Vec::new()),
            idf: Arc::new(HashMap::new()),
            doc_len: Arc::new(Vec::new()),
            tokenizer,
        };

        // Tokenize corpus
        let tokenized_corpus = if let Some(ref tokenizer_py) = bm25.tokenizer {
            bm25.tokenize_corpus(py, &corpus)?
        } else {
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
        let idf_map = bm25.calc_idf(nd);
        bm25.idf = Arc::new(idf_map);
        bm25.doc_freqs = Arc::new(doc_freqs);
        bm25.doc_len = Arc::new(doc_len);
        bm25.avgdl = avgdl;

        Ok(bm25)
    }

    /// Calculates BM25Plus scores for a given query
    pub fn get_scores(&self, query: Vec<String>) -> PyResult<Vec<f64>> {
        if self.corpus_size == 0 {
            return Ok(vec![]);
        }

        let idf = Arc::clone(&self.idf);
        let doc_freqs = Arc::clone(&self.doc_freqs);
        let doc_len = Arc::clone(&self.doc_len);
        let avgdl = self.avgdl;
        let k1 = self.k1;
        let b = self.b;
        let delta = self.delta;

        let query_terms: Vec<(&String, f64)> = query
            .iter()
            .filter_map(|q| idf.get(q).map(|&idf_val| (q, idf_val)))
            .collect();

        if query_terms.is_empty() {
            return Ok(vec![0.0; self.corpus_size]);
        }

        let k1_plus1 = k1 + 1.0;

        let scores: Vec<f64> = (0..self.corpus_size)
            .into_par_iter()
            .map(|i| {
                let doc_freq = &doc_freqs[i];
                let dl = doc_len[i] as f64;
                let denom = k1 * (1.0 - b + b * dl / avgdl);

                query_terms.iter().fold(0.0, |mut score, &(q, idf_val)| {
                    if let Some(&freq) = doc_freq.get(q) {
                        let freq = freq as f64;
                        let numerator = delta + freq * k1_plus1;
                        let denominator = denom + freq;
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

        let idf = Arc::clone(&self.idf);
        let doc_freqs = Arc::clone(&self.doc_freqs);
        let doc_len = Arc::clone(&self.doc_len);
        let avgdl = self.avgdl;
        let k1 = self.k1;
        let b = self.b;
        let delta = self.delta;

        let query_terms: Vec<(&String, f64)> = query
            .iter()
            .filter_map(|q| idf.get(q).map(|&idf_val| (q, idf_val)))
            .collect();

        if query_terms.is_empty() {
            return Ok(vec![0.0; doc_ids.len()]);
        }

        let k1_plus1 = k1 + 1.0;

        let scores: Vec<f64> = doc_ids
            .into_par_iter()
            .map(|i| {
                let doc_freq = &doc_freqs[i];
                let dl = doc_len[i] as f64;
                let denom = k1 * (1.0 - b + b * dl / avgdl);

                query_terms.iter().fold(0.0, |mut score, &(q, idf_val)| {
                    if let Some(&freq) = doc_freq.get(q) {
                        let freq = freq as f64;
                        let numerator = delta + freq * k1_plus1;
                        let denominator = denom + freq;
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

impl BM25Plus {
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

    /// Calculates the inverse document frequency (IDF)
    fn calc_idf(&self, nd: HashMap<String, usize>) -> HashMap<String, f64> {
        let corpus_size = self.corpus_size as f64;

        nd.into_par_iter()
            .map(|(word, freq)| {
                let idf_val = (corpus_size + 1.0).ln() - (freq as f64).ln();
                (word, idf_val)
            })
            .collect()
    }
}
