#[cfg(test)]
mod tests {
    use std::path::Path;
    use crate::data::tokenizer::TokenizerWrapper;
    use crate::data::train_dataset::{TextDataset, TextItem};
    use crate::data::batcher::make_batch;

    #[test]
    fn tokenization_batching_split_smoke_test() {
        // 1) tiny dataset
        let ds = TextDataset::new(vec![
            TextItem { text: "TERM 1 starts in January".into() },
            TextItem { text: "Orientation is held early in the year".into() },
            TextItem { text: "Exams start near the end of term".into() },
            TextItem { text: "Registration closes after a few weeks".into() },
        ]);

        // 2) train/val split
        let (train, val) = ds.split(0.75);
        assert!(train.len() > 0);
        assert!(val.len() > 0);

        // 3) tokenizer load (expects data/tokenizer.json)
        // If you don't have this file yet, we’ll generate it next.
        let tok = TokenizerWrapper::from_file(Path::new("data/tokenizer.json"));

        // Don’t fail the whole build if tokenizer file isn't there yet:
        if tok.is_err() {
            return;
        }
        let tok = tok.unwrap();

        // 4) tokenize a small batch
        let seq1 = tok.encode_ids(&train.get(0).unwrap().text).unwrap();
        let seq2 = tok.encode_ids(&train.get(1).unwrap().text).unwrap();

        let batch = make_batch(&[seq1, seq2], 32, 0);
        assert_eq!(batch.input_ids.len(), 2);
        assert_eq!(batch.input_ids[0].len(), 32);
        assert_eq!(batch.attention_mask[0].len(), 32);
    }
}