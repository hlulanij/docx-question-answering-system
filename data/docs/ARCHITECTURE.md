# System Architecture

## High-Level Flow
DOCX File → Text Extraction → Tokenization → Transformer Encoder → Answer Prediction

## Components
- Document Loader: Extracts text from .docx files
- Tokenizer: Converts text to token IDs
- Model: Transformer encoder (6 layers planned)
- Trainer: Handles loss, backpropagation, and checkpoints
- Inference Engine: Accepts questions and outputs answers
``
