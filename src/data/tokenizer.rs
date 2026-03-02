use std::path::Path;
use tokenizers::Tokenizer;

/// Load a tokenizer from data/tokenizer.json.
pub fn load_tokenizer(path: &Path) -> Tokenizer {
    let p = path
        .to_str()
        .expect("Tokenizer path is not valid UTF-8");

    Tokenizer::from_file(p).expect("Failed to load tokenizer.json")
}

/// Encode text into token IDs using the loaded tokenizer.
pub fn encode_ids(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    // NOTE: No named args in Rust: encode(text, add_special_tokens)
    let enc = tokenizer.encode(text, true).expect("Tokenization failed");
    enc.get_ids().to_vec()
}