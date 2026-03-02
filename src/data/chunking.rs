#[derive(Debug, Clone)]
pub struct MonthChunk {
    pub year: i32,
    pub month: String,
    pub text: String,
}

/// Split a whole document's extracted text into chunks, one per month.
/// Expected headings like: "JANUARY 2024", "FEBRUARY 2025", etc.
pub fn chunk_by_month(full_text: &str) -> Vec<MonthChunk> {
    let mut chunks: Vec<MonthChunk> = Vec::new();

    let mut current_month: Option<String> = None;
    let mut current_year: Option<i32> = None;
    let mut buffer = String::new();

    for line in full_text.lines() {
        let trimmed = line.trim();

        // Detect "MONTH YEAR" heading
        if let Some((month, year)) = parse_month_year_heading(trimmed) {
            // If we were already collecting a previous month, finalize it
            if let (Some(m), Some(y)) = (current_month.take(), current_year.take()) {
                if !buffer.trim().is_empty() {
                    chunks.push(MonthChunk {
                        year: y,
                        month: m,
                        text: buffer.trim().to_string(),
                    });
                }
                buffer.clear();
            }

            // Start new chunk
            current_month = Some(month);
            current_year = Some(year);

            // Keep the heading inside the chunk text (useful for QA)
            buffer.push_str(trimmed);
            buffer.push('\n');
        } else {
            // Normal content line
            if current_month.is_some() && current_year.is_some() {
                buffer.push_str(trimmed);
                buffer.push('\n');
            }
        }
    }

    // Finalize last chunk
    if let (Some(m), Some(y)) = (current_month, current_year) {
        if !buffer.trim().is_empty() {
            chunks.push(MonthChunk {
                year: y,
                month: m,
                text: buffer.trim().to_string(),
            });
        }
    }

    chunks
}

/// Recognize headings like: "JANUARY 2024"
fn parse_month_year_heading(s: &str) -> Option<(String, i32)> {
    // quick reject
    if s.len() < 8 {
        return None;
    }

    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() != 2 {
        return None;
    }

    let month = parts[0];
    let year_str = parts[1];

    // year must be 4 digits
    let year: i32 = year_str.parse().ok()?;
    if year < 1900 || year > 2100 {
        return None;
    }

    // month must be one of these
    if !is_month_name(month) {
        return None;
    }

    Some((month.to_string(), year))
}

fn is_month_name(s: &str) -> bool {
    matches!(
        s,
        "JANUARY"
            | "FEBRUARY"
            | "MARCH"
            | "APRIL"
            | "MAY"
            | "JUNE"
            | "JULY"
            | "AUGUST"
            | "SEPTEMBER"
            | "OCTOBER"
            | "NOVEMBER"
            | "DECEMBER"
    )
}
