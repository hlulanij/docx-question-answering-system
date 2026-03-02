use docx_rs::{read_docx, DocumentChild};
use std::error::Error;
use std::fs;
use std::path::Path;

pub fn load_docx_text<P: AsRef<Path>>(path: P) -> Result<String, Box<dyn Error>> {
    // docx-rs expects bytes
    let bytes = fs::read(path)?;
    let docx = read_docx(&bytes)?; // read_docx(buf: &[u8]) [4](https://docs.rs/docx-rs/latest/docx_rs/fn.read_docx.html)

    let mut out = String::new();

    for child in docx.document.children {
        extract_child_text(child, &mut out);
        out.push('\n');
    }

    Ok(clean_whitespace(out))
}

fn extract_child_text(child: DocumentChild, out: &mut String) {
    match child {
        // Paragraphs
        docx_rs::DocumentChild::Paragraph(para) => {
            for p_child in para.children {
                if let docx_rs::ParagraphChild::Run(run) = p_child {
                    for r_child in run.children {
                        if let docx_rs::RunChild::Text(t) = r_child {
                            out.push_str(&t.text);
                            out.push(' ');
                        }
                    }
                }
            }
        }

        // Tables
        docx_rs::DocumentChild::Table(table) => {
            for row in table.rows {
                if let docx_rs::TableChild::TableRow(table_row) = row {
                    for cell in table_row.cells {
                        if let docx_rs::TableRowChild::TableCell(cell) = cell {
                            for content in cell.children {
                                match content {
                                    docx_rs::TableCellContent::Paragraph(p) => {
                                        // same logic as paragraph
                                        for p_child in p.children {
                                            if let docx_rs::ParagraphChild::Run(run) = p_child {
                                                for r_child in run.children {
                                                    if let docx_rs::RunChild::Text(t) = r_child {
                                                        out.push_str(&t.text);
                                                        out.push(' ');
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    docx_rs::TableCellContent::Table(nested_table) => {
                                        // handle nested tables (Word calendars often nest)
                                        extract_child_text(
                                            docx_rs::DocumentChild::Table(Box::new(nested_table)),
                                            out,
                                        );
                                    }
                                    _ => {}
                                }
                            }
                            out.push(' ');
                        }
                    }
                    out.push('\n');
                }
            }
        }

        _ => {}
    }
}

fn clean_whitespace(s: String) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}