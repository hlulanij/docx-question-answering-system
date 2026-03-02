use burn::data::dataset::Dataset;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// One processed training example
#[derive(Debug, Clone)]
pub struct QaSample {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
}

/// Must match what prepare.rs writes (we only need text)
#[derive(Debug, Deserialize)]
struct ExampleJsonl {
    pub text: String,
}

pub struct QaDataset {
    samples: Vec<QaSample>,
}

impl QaDataset {
    pub fn from_jsonl(path: &Path, max_len: usize) -> Self {
        // ✅ NO PANIC if file missing
        if !path.exists() {
            println!(
                "⚠️ Dataset file {:?} not found — returning empty dataset (no crash).",
                path
            );
            return Self {
                samples: Vec::new(),
            };
        }

        let file = File::open(path).expect("Failed to open data/examples.jsonl");
        let reader = BufReader::new(file);

        let mut vocab: HashMap<String, i64> = HashMap::new();
        vocab.insert("[PAD]".into(), 0);
        vocab.insert("[UNK]".into(), 1);

        let mut next_id: i64 = 2;
        let mut samples = Vec::new();

        for line in reader.lines() {
            let line = line.expect("Invalid line read");
            if line.trim().is_empty() {
                continue;
            }

            let raw: ExampleJsonl = serde_json::from_str(&line).expect("Invalid JSON line");

            let mut input_ids = Vec::new();
            for token in raw.text.split_whitespace() {
                let id = *vocab.entry(token.to_lowercase()).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                });
                input_ids.push(id);
            }

            // truncate
            input_ids.truncate(max_len);

            // attention mask
            let mut attention_mask = vec![1; input_ids.len()];

            // padding
            while input_ids.len() < max_len {
                input_ids.push(0);
                attention_mask.push(0);
            }

            samples.push(QaSample {
                input_ids,
                attention_mask,
            });
        }

        println!(
            "✅ QaDataset loaded {} samples from {:?}",
            samples.len(),
            path
        );

        Self { samples }
    }

    pub fn split(self, train_ratio: f32) -> (Self, Self) {
        let len = self.samples.len();

        if len < 2 {
            return (
                Self {
                    samples: self.samples.clone(),
                },
                Self {
                    samples: Vec::new(),
                },
            );
        }

        let split_idx = ((len as f32) * train_ratio).max(1.0) as usize;

        let train = self.samples[..split_idx].to_vec();
        let val = self.samples[split_idx..].to_vec();

        (Self { samples: train }, Self { samples: val })
    }
}

impl Dataset<QaSample> for QaDataset {
    fn get(&self, index: usize) -> Option<QaSample> {
        self.samples.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}
