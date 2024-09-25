use pyo3::prelude::*;
use pyo3::types::PyAny;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// A Rust implementation of BM25Okapi optimized for multicore processing.
#[pyclass]
pub struct BM25Okapi {
    #[pyo3(get, set)]
    k1: f64,
    #[pyo3(get, set)]
    b: f64,
    #[pyo3(get, set)]
    epsilon: f64,
    #[pyo3(get, set)]
    corpus_size: usize,
    #[pyo3(get, set)]
    avgdl: f64,
    #[pyo3(get, set)]
    doc_freqs: Vec<HashMap<String, usize>>,
    idf: Arc<HashMap<String, f64>>, // Arc for thread-safe shared access
    #[pyo3(get, set)]
    doc_len: Vec<usize>,
    tokenizer: Option<Py<PyAny>>,
}

#[pymethods]
impl BM25Okapi {
    /// Constructor for BM25Okapi
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

        let tokenizer = tokenizer.map(|tokenizer| tokenizer.into());

        // Initialize BM25 structure without tokenization
        let mut bm25 = BM25Okapi {
            k1,
            b,
            epsilon,
            corpus_size: 0,
            avgdl: 0.0,
            doc_freqs: Vec::new(),
            idf: Arc::new(HashMap::new()),
            doc_len: Vec::new(),
            tokenizer,
        };

        // Tokenize corpus if tokenizer is provided
        let tokenized_corpus = if bm25.tokenizer.is_some() {
            // Tokenization with Python tokenizer must be sequential due to GIL limitations
            bm25.tokenize_corpus(py, &corpus)?
        } else {
            // Parallel tokenization without Python involvement
            corpus
                .par_iter()
                .map(|doc| doc.split_whitespace().map(|s| s.to_string()).collect())
                .collect()
        };

        // Initialize BM25 structures
        let nd = bm25.initialize(tokenized_corpus);
        let idf_map = bm25.calc_idf(nd);
        bm25.idf = Arc::new(idf_map);

        Ok(bm25)
    }

    /// Calculates BM25 scores for all documents given a query
    pub fn get_scores(&self, _py: Python, query: Vec<String>) -> PyResult<Vec<f64>> {
        if self.corpus_size == 0 {
            return Ok(vec![]);
        }

        // Prepare shared data
        let idf = Arc::clone(&self.idf);
        let doc_freqs = &self.doc_freqs;
        let doc_len = &self.doc_len;
        let avgdl = self.avgdl;
        let k1 = self.k1;
        let b = self.b;

        // Create a mapping from query terms to their idf values
        let query_terms: Vec<(&String, &f64)> =
            query.iter().filter_map(|q| idf.get_key_value(q)).collect();
        if query_terms.is_empty() {
            // None of the query terms are in the corpus
            return Ok(vec![0.0; self.corpus_size]);
        }

        // Compute scores in parallel over documents
        let scores: Vec<f64> = (0..self.corpus_size)
            .into_par_iter()
            .map(|i| {
                let mut score = 0.0;
                let doc_freq = &doc_freqs[i];
                let dl = doc_len[i] as f64;
                let normalization = k1 * (1.0 - b + b * dl / avgdl);

                for &(q, &idf_q) in &query_terms {
                    let freq = *doc_freq.get(q).unwrap_or(&0) as f64;
                    if freq > 0.0 {
                        let numerator = freq * (k1 + 1.0);
                        let denominator = freq + normalization;
                        score += idf_q * (numerator / denominator);
                    }
                }
                score
            })
            .collect();

        Ok(scores)
    }

    /// Calculates BM25 scores for a batch of documents given a query
    pub fn get_batch_scores(
        &self,
        _py: Python,
        query: Vec<String>,
        doc_ids: Vec<usize>,
    ) -> PyResult<Vec<f64>> {
        if doc_ids.is_empty() {
            return Ok(vec![]);
        }

        // Prepare shared data
        let idf = Arc::clone(&self.idf);
        let doc_freqs = &self.doc_freqs;
        let doc_len = &self.doc_len;
        let avgdl = self.avgdl;
        let k1 = self.k1;
        let b = self.b;

        // Create a mapping from query terms to their idf values
        let query_terms: Vec<(&String, &f64)> =
            query.iter().filter_map(|q| idf.get_key_value(q)).collect();
        if query_terms.is_empty() {
            // None of the query terms are in the corpus
            return Ok(vec![0.0; doc_ids.len()]);
        }

        // Compute scores in parallel over specified document IDs
        let scores: Vec<f64> = doc_ids
            .par_iter()
            .map(|&di| {
                if di >= doc_freqs.len() {
                    // Invalid document ID
                    return 0.0;
                }
                let mut score = 0.0;
                let doc_freq = &doc_freqs[di];
                let dl = doc_len[di] as f64;
                let normalization = k1 * (1.0 - b + b * dl / avgdl);

                for &(q, &idf_q) in &query_terms {
                    let freq = *doc_freq.get(q).unwrap_or(&0) as f64;
                    if freq > 0.0 {
                        let numerator = freq * (k1 + 1.0);
                        let denominator = freq + normalization;
                        score += idf_q * (numerator / denominator);
                    }
                }
                score
            })
            .collect();

        Ok(scores)
    }

    /// Retrieves the top N documents for a given query
    pub fn get_top_n(
        &self,
        py: Python,
        query: Vec<String>,
        documents: Vec<String>,
        n: Option<usize>,
    ) -> PyResult<Vec<String>> {
        let n = n.unwrap_or(5);
        if self.corpus_size != documents.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "The documents given don't match the index corpus!",
            ));
        }

        let scores = self.get_scores(py, query)?;

        // Create a vector of (index, score)
        let mut idx_scores: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();

        // Sort by score in descending order
        idx_scores.par_sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top n indices
        let top_n_indices: Vec<usize> = idx_scores.iter().take(n).map(|(i, _)| *i).collect();

        // Collect the top N documents
        let top_n = top_n_indices
            .into_iter()
            .map(|i| documents[i].clone())
            .collect();

        Ok(top_n)
    }
}

// Implement internal methods outside of #[pymethods]
impl BM25Okapi {
    /// Tokenizes the corpus using the provided tokenizer
    pub fn tokenize_corpus(&self, py: Python, corpus: &Vec<String>) -> PyResult<Vec<Vec<String>>> {
        let tokenizer_py = match &self.tokenizer {
            Some(tk) => tk,
            None => return Ok(vec![]),
        };

        // Sequential tokenization due to GIL requirements
        let mut tokenized_corpus = Vec::with_capacity(corpus.len());
        for doc in corpus {
            let tokens: Vec<String> = tokenizer_py
                .call1(py, (doc,))
                .expect("Tokenizer failed to execute")
                .extract(py)
                .expect("Failed to extract tokens");
            tokenized_corpus.push(tokens);
        }

        Ok(tokenized_corpus)
    }

    /// Initializes document frequencies and other metrics
    pub fn initialize(&mut self, corpus: Vec<Vec<String>>) -> HashMap<String, usize> {
        use rayon::prelude::*;
        use std::sync::Mutex;

        let corpus_size = corpus.len();
        let avgdl =
            corpus.par_iter().map(|doc| doc.len()).sum::<usize>() as f64 / corpus_size as f64;

        // Shared data structures protected by Mutex for thread-safe mutation
        let nd = Mutex::new(HashMap::new());
        let doc_freqs = Mutex::new(Vec::with_capacity(corpus_size));
        let doc_len = Mutex::new(Vec::with_capacity(corpus_size));

        // Parallel processing of documents
        corpus.into_par_iter().for_each(|document| {
            let doc_length = document.len();

            // Calculate frequencies using a local HashMap
            let frequencies = document.iter().fold(HashMap::new(), |mut freq, word| {
                *freq.entry(word.clone()).or_insert(0) += 1;
                freq
            });

            // Update global data structures
            {
                let mut nd_lock = nd.lock().unwrap();
                for word in frequencies.keys() {
                    *nd_lock.entry(word.clone()).or_insert(0) += 1;
                }
            }
            {
                let mut doc_freqs_lock = doc_freqs.lock().unwrap();
                doc_freqs_lock.push(frequencies);
            }
            {
                let mut doc_len_lock = doc_len.lock().unwrap();
                doc_len_lock.push(doc_length);
            }
        });

        self.corpus_size = corpus_size;
        self.avgdl = avgdl;
        self.doc_freqs = doc_freqs.into_inner().unwrap();
        self.doc_len = doc_len.into_inner().unwrap();

        nd.into_inner().unwrap()
    }

    /// Calculates the inverse document frequency (IDF)
    pub fn calc_idf(&self, nd: HashMap<String, usize>) -> HashMap<String, f64> {
        // Preallocate HashMap with expected size
        let mut idf: HashMap<String, f64> = HashMap::with_capacity(nd.len());
        let mut idf_sum = 0.0;
        let mut negative_idfs = Vec::new();

        // Compute IDF for each term
        for (word, freq) in nd.iter() {
            // Adding 0.5 to numerator and denominator for smoothing
            let idf_val =
                ((self.corpus_size as f64 - *freq as f64 + 0.5) / (*freq as f64 + 0.5)).ln();
            idf.insert(word.clone(), idf_val);
            idf_sum += idf_val;
            if idf_val < 0.0 {
                negative_idfs.push(word.clone());
            }
        }

        // Calculate average IDF
        let average_idf = if !idf.is_empty() {
            idf_sum / idf.len() as f64
        } else {
            0.0
        };

        let eps = self.epsilon * average_idf;

        // Apply flooring to negative IDF values
        for word in negative_idfs {
            idf.insert(word, eps);
        }

        idf
    }
}

/// A Python module implemented in Rust.
#[pymodule]
pub fn bm25_pyrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BM25Okapi>()?;
    Ok(())
}
