use pyo3::prelude::*;

mod bm25l;
mod bm25okapi;
mod bm25plus;

/// A Python module implemented in Rust.
#[pymodule]
pub fn bm25_pyrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<bm25okapi::BM25Okapi>()?;
    m.add_class::<bm25plus::BM25Plus>()?;
    m.add_class::<bm25l::BM25L>()?;
    Ok(())
}
