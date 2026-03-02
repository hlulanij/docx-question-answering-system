use docx_rs::read_docx;
use std::fs::File;
use std::io::Read;

/// Extract plain text from a .docx file.
/// This focuses on paragraph text which is sufficient for the assignment data pipeline.
pub fn extract_text_from_docx(path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let doc = read_docx(&buffer)?;
    let mut text = String::new();

    for child in doc.document.children {
        if let docx_rs::DocumentChild::Paragraph(p) = child {
            for pc in p.children {
                if let docx_rs::ParagraphChild::Run(r) = pc {
                    for rc in r.children {
                        if let docx_rs::RunChild::Text(t) = rc {
                            text.push_str(&t.text);
                        }
                    }
                }
            }
            text.push('\n');
        }
    }

    Ok(text)
}
