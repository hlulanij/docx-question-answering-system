mod docx;

use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  cargo run -- extract <path-to-docx>");
        eprintln!("Example:");
        eprintln!("  cargo run -- extract data/graduation_info.docx");
        return Ok(());
    }

    match args[1].as_str() {
        "extract" => {
            if args.len() < 3 {
                eprintln!("Missing docx path. Example:");
                eprintln!("  cargo run -- extract data/graduation_info.docx");
                return Ok(());
            }
            let path = &args[2];
            let text = docx::extract::extract_text_from_docx(path)?;
            println!("--- Extracted Text Start ---");
            println!("{text}");
            println!("--- Extracted Text End ---");
        }
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            eprintln!("Try: cargo run -- extract <path-to-docx>");
        }
    }

    Ok(())
}
