#[derive(Debug, Clone)]
pub struct Batch {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u32>>,
}

/// Pad/truncate token id sequences into a fixed length batch.
pub fn make_batch(seqs: &[Vec<u32>], max_len: usize, pad_id: u32) -> Batch {
    let mut input_ids = Vec::with_capacity(seqs.len());
    let mut attention_mask = Vec::with_capacity(seqs.len());

    for s in seqs {
        let mut ids = s.clone();
        ids.truncate(max_len);

        let mut mask = vec![1u32; ids.len()];

        while ids.len() < max_len {
            ids.push(pad_id);
            mask.push(0u32);
        }

        input_ids.push(ids);
        attention_mask.push(mask);
    }

    Batch { input_ids, attention_mask }
}