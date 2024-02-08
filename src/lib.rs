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
        let nd = bm25.initialize(corpus);
        bm25._calc_idf(nd);
        bm25
    }

    fn initialize(&mut self, corpus: Vec<Vec<String>>) -> HashMap<String, f64> {
        let mut nd: HashMap<String, f64> = HashMap::new();
        let mut num_doc = 0;

        for doc in corpus.iter() {
            self.doc_len.push(doc.len());
            num_doc += doc.len();

            let mut frequencies = HashMap::new();
            for word in doc.iter() {
                *frequencies.entry(word.to_string()).or_insert(0) += 1;
            }
            self.doc_freqs.push(frequencies.clone());
            for (word, _) in frequencies.into_iter() {
                *nd.entry(word).or_insert(0.0) += 1.0;
            }
            self.corpus_size += 1.0;
        }

        self.avgdl = num_doc as f64 / self.corpus_size;
        nd
    }

    fn _calc_idf(&mut self, nd: HashMap<String, f64>) {
        let mut idf_sum = 0.0;
        let mut negative_idfs: Vec<_> = Vec::new();

        for (word, freq) in nd.into_iter() {
            let idf = f64::ln(self.corpus_size - freq + 0.5) - f64::ln(freq + 0.5);
            self.idf.insert(word.clone(), idf);
            idf_sum += idf;
            if idf < 0.0 {
                negative_idfs.push(word.clone());
            }
        }
        self.average_idf = idf_sum / self.idf.len() as f64;
        let eps = self.epsilon * self.average_idf;
        for i in negative_idfs.iter() {
            self.idf.insert(i.clone(), eps);
        }
    }

    fn get_scores(&mut self, query: Vec<String>)->Vec<f64>{
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
            scores.par_iter_mut().enumerate()  // Parallel iteration over scores
                .for_each(|(i, score)| {
                    let freq = q_freq[i];
                    let doc_len_i = doc_len[i] as f64;
                    let bm25_score = idf_score * (freq * (self.k1 + 1.0) / (freq + self.k1 * (1.0 - self.b + self.b * doc_len_i / self.avgdl)));
                    *score += bm25_score;  // Update the score
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
