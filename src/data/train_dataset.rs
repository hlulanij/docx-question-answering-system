use burn::data::dataset::Dataset;

#[derive(Debug, Clone)]
pub struct TextItem {
    pub text: String,
}

pub struct TextDataset {
    items: Vec<TextItem>,
}

impl TextDataset {
    pub fn new(items: Vec<TextItem>) -> Self {
        Self { items }
    }

    /// Train/validation split
    pub fn split(self, train_ratio: f32) -> (Self, Self) {
        let len = self.items.len();
        let split_idx = ((len as f32) * train_ratio).round() as usize;
        let split_idx = split_idx.clamp(1, len.saturating_sub(1).max(1));

        let train = self.items[..split_idx].to_vec();
        let val = self.items[split_idx..].to_vec();

        (Self { items: train }, Self { items: val })
    }
}

impl Dataset<TextItem> for TextDataset {
    fn get(&self, index: usize) -> Option<TextItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}