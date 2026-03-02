use burn::data::dataset::Dataset;

/// One synthetic Q&A item:
/// - input_ids: token ids
/// - start/end: span labels
#[derive(Clone, Debug)]
pub struct SpanItem {
    pub input_ids: Vec<i32>,
    pub start: i32,
    pub end: i32,
}

pub struct SyntheticSpanDataset {
    items: Vec<SpanItem>,
}

impl SyntheticSpanDataset {
    pub fn new(num_items: usize, seq_len: usize, vocab_size: usize) -> Self {
        let mut items = Vec::with_capacity(num_items);

        for i in 0..num_items {
            // simple deterministic tokens
            let mut input_ids = Vec::with_capacity(seq_len);
            for j in 0..seq_len {
                let tok = ((i + j) % vocab_size) as i32;
                input_ids.push(tok);
            }

            // deterministic "answer span"
            let start = (i % (seq_len / 2).max(1)) as i32;
            let end = (start + 3).min((seq_len - 1) as i32);

            items.push(SpanItem { input_ids, start, end });
        }

        Self { items }
    }
}

impl Dataset<SpanItem> for SyntheticSpanDataset {
    fn get(&self, index: usize) -> Option<SpanItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}