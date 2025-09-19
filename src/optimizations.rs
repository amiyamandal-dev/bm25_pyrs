// Additional optimization utilities for BM25 implementations

use rayon::prelude::*;
use smallvec::SmallVec;
use string_interner::DefaultSymbol;

/// Optimized query preprocessing that can be reused across multiple documents
#[derive(Clone)]
pub struct PreprocessedQuery {
    pub symbols: SmallVec<[DefaultSymbol; 8]>,
    pub idf_values: SmallVec<[(DefaultSymbol, f64); 8]>,
    pub is_empty: bool,
}

impl PreprocessedQuery {
    pub fn new() -> Self {
        Self {
            symbols: SmallVec::new(),
            idf_values: SmallVec::new(),
            is_empty: true,
        }
    }
    
    pub fn len(&self) -> usize {
        self.idf_values.len()
    }
}

/// SIMD-friendly scoring function for better performance on modern CPUs
#[inline(always)]
pub fn compute_bm25_score_vectorized(
    query_terms: &[(DefaultSymbol, f64)],
    doc_freq: &ahash::AHashMap<DefaultSymbol, u32>,
    _dl: f64,
    norm_factor: f64,
    k1_plus1: f64,
) -> f64 {
    // Unroll the loop for better performance with small query sizes
    match query_terms.len() {
        0 => 0.0,
        1 => {
            let (symbol, idf_val) = query_terms[0];
            if let Some(&freq) = doc_freq.get(&symbol) {
                let freq_f64 = freq as f64;
                let numerator = freq_f64 * k1_plus1;
                let denominator = freq_f64 + norm_factor;
                idf_val * (numerator / denominator)
            } else {
                0.0
            }
        }
        2 => {
            let mut score = 0.0;
            for &(symbol, idf_val) in query_terms {
                if let Some(&freq) = doc_freq.get(&symbol) {
                    let freq_f64 = freq as f64;
                    let numerator = freq_f64 * k1_plus1;
                    let denominator = freq_f64 + norm_factor;
                    score += idf_val * (numerator / denominator);
                }
            }
            score
        }
        _ => {
            // For longer queries, use the standard loop
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
        }
    }
}

/// Cache-friendly batch processing for better memory access patterns
pub fn process_documents_in_chunks<F>(
    total_docs: usize,
    chunk_size: usize,
    processor: F,
) -> Vec<f64>
where
    F: Fn(usize, usize) -> Vec<f64> + Send + Sync,
{
    let chunks: Vec<(usize, usize)> = (0..total_docs)
        .step_by(chunk_size)
        .map(|start| (start, (start + chunk_size).min(total_docs)))
        .collect();

    chunks
        .into_par_iter()
        .flat_map(|(start, end)| processor(start, end))
        .collect()
}

/// Optimized top-k selection using partial sorting
pub fn select_top_k_indices(scores: &[f64], k: usize) -> Vec<usize> {
    if k >= scores.len() {
        let mut indices: Vec<usize> = (0..scores.len()).collect();
        indices.par_sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        return indices;
    }

    // Use partial sort for better performance when k << n
    let mut indexed_scores: Vec<(usize, f64)> = scores
        .iter()
        .enumerate()
        .map(|(i, &score)| (i, score))
        .collect();

    // Partial sort - only sort the top k elements
    indexed_scores.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed_scores[..k].sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    indexed_scores[..k].iter().map(|(i, _)| *i).collect()
}