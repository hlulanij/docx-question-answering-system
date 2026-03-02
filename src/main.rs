#![recursion_limit = "256"]

mod data;
mod inference;

use std::path::Path;

fn main() {
    println!("✅ DOCX Question Answering System (Inference Demo)");

    let docs = [
        "data/docs/calendar_2024.docx",
        "data/docs/calendar_2025.docx",
        "data/docs/calendar_2026.docx",
    ];

    let questions = [
        "When does TERM 1 start?",
        "When does TERM 1 end?",
        "When is orientation?",
        "When does registration start?",
        "When does registration close?",
        "When do exams start?",
        "When do exams end?",
        "When does the university close?",
        "When does the university reopen?",
        "When is the mid-year break?",
    ];

    for doc in docs {
        let doc_path = Path::new(doc);

        println!("\n==============================");
        println!("📄 Document: {}", doc);

        for q in questions {
            println!("\nQ: {}", q);

            // ✅ FIX: NO named arguments in Rust
            let answer = inference::answer_question(doc_path, q);

            println!("{}", answer);
        }
    }
}