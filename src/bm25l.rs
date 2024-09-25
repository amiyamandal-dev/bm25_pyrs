use pyo3::prelude::*;
use pyo3::types::PyAny;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[pyclass]
pub struct BM25L {
    #[pyo3(get, set)]
    k1: f64,
    #[pyo3(get, set)]
    b: f64,
    #[pyo3(get, set)]
    delta: f64,
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
impl BM25L {
    /// Constructor for BM25Okapi
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
        let delta = delta.unwrap_or(0.25);

        let tokenizer = tokenizer.map(|tokenizer| tokenizer.into());

        // Initialize BM25 structure without tokenization
        let mut bm25 = BM25L {
            k1,
            b,
            delta,
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

    pub fn get_scores(&self, _py: Python, query: Vec<String>) -> PyResult<Vec<f64>> {}
}

impl BM25L {
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
        let idf: Mutex<HashMap<String, f64>> = Mutex::new(HashMap::with_capacity(nd.len()));

        nd.par_iter().for_each(|(word, freq)| {
            let idf_val = ((self.corpus_size as f64 + 1.0) / (*freq as f64 + 0.5)).ln();
            let mut idf_guard = idf.lock().unwrap();
            idf_guard.insert(word.clone(), idf_val);
        });

        let idf = idf.into_inner().unwrap();
        idf
    }
}
