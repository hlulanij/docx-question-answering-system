use std::path::Path;

use crate::data::docx_loader::load_docx_text;

pub fn answer_question(doc_path: &Path, question: &str) -> String {
    let text = match load_docx_text(doc_path) {
        Ok(t) => t,
        Err(e) => return format!("❌ Failed to load document: {e}"),
    };

    let q_words: Vec<String> = question
        .to_lowercase()
        .split_whitespace()
        .map(|w| w.chars().filter(|c| c.is_alphanumeric()).collect::<String>())
        .filter(|w| !w.is_empty())
        .collect();

    // split doc into lines (best for calendars/tables)
    let lines: Vec<String> = text
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    let mut scored: Vec<(usize, String)> = Vec::new();

    for line in lines {
        let l = line.to_lowercase();
        let mut score = 0usize;

        for w in &q_words {
            if l.contains(w) {
                score += 1;
            }
        }

        if score > 0 {
            scored.push((score, line));
        }
    }

    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.len().cmp(&b.1.len())));

    if scored.is_empty() {
        return "Answer:\n1. No relevant match found.\n2. No relevant match found.".to_string();
    }

    let top: Vec<String> = scored
        .into_iter()
        .take(2)
        .enumerate()
        .map(|(i, (_, s))| format!("{}. {}", i + 1, cap_len(&s, 180)))
        .collect();

    format!("Answer:\n{}", top.join("\n"))
}

fn cap_len(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let shortened: String = s.chars().take(max_chars).collect();
        format!("{shortened}…")
    }
}