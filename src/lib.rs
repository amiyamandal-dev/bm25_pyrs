use std::collections::HashMap;

use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
struct BM25Okapi {
    #[pyo3(get)]
    corpus_size: f64,
    #[pyo3(get)]
    avgdl: f64,
    #[pyo3(get)]
    doc_freqs: Vec<HashMap<String, usize>>,
    #[pyo3(get)]
    doc_len: Vec<usize>,
    #[pyo3(get)]
    idf: HashMap<String, f64>,
    #[pyo3(get)]
    k1: f64,
    #[pyo3(get)]
    b: f64,
    #[pyo3(get)]
    delta: f64,
    #[pyo3(get)]
    average_idf: f64,
    #[pyo3(get)]
    epsilon: f64,
}

fn calc_idf(b: &mut BM25Okapi, nd: &HashMap<String, f64>) {
    let mut idf_sum = 0.0;
    let mut negative_idfs = Vec::new();

    // Reserve capacity to avoid reallocations
    let mut idf_updates = HashMap::with_capacity(nd.len());

    for (word, freq) in nd.into_iter() {
        let idf = f64::ln(b.corpus_size - freq + 0.5) - f64::ln(freq + 0.5);
        idf_updates.insert(word.clone(), idf);
        idf_sum += idf;
        if idf < 0.0 {
            negative_idfs.push(word.clone());
        }
    }
    // Update idf in b with idf_updates
    b.idf.extend(idf_updates);

    b.average_idf = idf_sum / b.idf.len() as f64;
    let eps = b.epsilon * b.average_idf;

    for i in negative_idfs {
        // Update negative idfs in b.idf directly
        *b.idf.entry(i).or_insert(eps) = eps;
    }
}

fn initialize(b: &mut BM25Okapi, corpus: &[Vec<String>]) -> HashMap<String, f64> {
    let mut nd: HashMap<String, f64> = HashMap::new();
    let mut num_doc = 0;
    // Reserve capacity to avoid reallocations
    b.doc_len.reserve(corpus.len());
    b.doc_freqs.reserve(corpus.len());

    for doc in corpus {
        b.doc_len.push(doc.len());
        num_doc += doc.len();

        let mut frequencies = HashMap::new();
        for word in doc {
            *frequencies.entry(word.to_string()).or_insert(0) += 1;
        }

        for (word, _) in &frequencies {
            *nd.entry(word.to_string()).or_insert(0.0) += 1.0;
        }
        b.doc_freqs.push(frequencies);
        b.corpus_size += 1.0;
    }

    b.avgdl = num_doc as f64 / b.corpus_size;
    nd
}

#[pymethods]
impl BM25Okapi {
    #[new]
    fn new(corpus: Vec<Vec<String>>, epsilon: f64) -> Self {
        let mut bm25 = BM25Okapi {
            corpus_size: 0.0,
            avgdl: 0.0,
            doc_freqs: Vec::new(),
            doc_len: Vec::new(),
            idf: HashMap::new(),
            k1: 1.5,
            b: 0.75,
            delta: 0.5,
            average_idf: 0.0,
            epsilon,
        };
        let nd = initialize(&mut bm25, &corpus);
        calc_idf(&mut bm25, &nd);
        bm25
    }

    fn get_scores(&mut self, query: Vec<String>) -> Vec<f64> {
        let mut scores: Vec<f64> = vec![0.0; self.corpus_size as usize];
        let doc_len = &self.doc_len;
        for q in query.iter() {
            let q_freq: Vec<f64> = self
                .doc_freqs
                .par_iter() // Parallel iteration over documents
                .map(|doc| doc.get(q).map_or(0.0, |&t| t as f64)) // Get frequency for term q in each document
                .collect(); // Collect into a vector

            let idf_score = match self.idf.get(q) {
                Some(t) => *t,
                None => 0.0,
            };
            scores
                .par_iter_mut()
                .enumerate() // Parallel iteration over scores
                .for_each(|(i, score)| {
                    let freq = q_freq[i];
                    let doc_len_i = doc_len[i] as f64;
                    let bm25_score = idf_score
                        * (freq * (self.k1 + 1.0)
                            / (freq + self.k1 * (1.0 - self.b + self.b * doc_len_i / self.avgdl)));
                    *score += bm25_score; // Update the score
                });
        }
        scores
    }
}

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pymodule]
fn bm25_pyrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<BM25Okapi>()?;
    Ok(())
}
