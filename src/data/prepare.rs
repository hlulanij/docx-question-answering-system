use crate::data::chunking::{chunk_by_month, MonthChunk};
use crate::data::docx_loader::load_docx_text;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    pub source_file: String,
    pub chunk_id: String, // e.g. "JANUARY_2024" or "CHUNK_1"
    pub text: String,
}

/// Load all .docx files in data/docs and create chunk examples.
pub fn build_examples_from_docs(docs_dir: &Path) -> Result<Vec<Example>, Box<dyn Error>> {
    let mut examples = Vec::new();

    for entry in fs::read_dir(docs_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("docx") {
            continue;
        }

        let file_name = path.file_name().unwrap().to_string_lossy().to_string();
        let full_text = load_docx_text(&path)?;

        let chunks: Vec<MonthChunk> = chunk_by_month(&full_text);

        for c in chunks {
            let id = if c.year != 0 {
                format!("{}_{}", c.month, c.year)
            } else {
                c.month.clone()
            };

            examples.push(Example {
                source_file: file_name.clone(),
                chunk_id: id,
                text: c.text,
            });
        }
    }

    Ok(examples)
}

/// Save examples as JSON Lines file (one JSON object per line)
pub fn save_examples_jsonl(examples: &[Example], out_path: &Path) -> Result<(), Box<dyn Error>> {
    let file = File::create(out_path)?;
    let mut writer = BufWriter::new(file);

    for ex in examples {
        let line = serde_json::to_string(ex)?;
        writeln!(writer, "{line}")?;
    }

    Ok(())
}
