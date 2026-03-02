# System Architecture

## Overview
The Question Answering system follows a modular, pipeline-based architecture designed to separate concerns between data processing, model computation, and user interaction.

## High-Level Data Flow
1. DOCX document is loaded from disk
2. Text is extracted and cleaned
3. Text is tokenized and batched
4. Tokens are passed through a transformer encoder
5. Output logits are projected to answer predictions
6. Results are returned via the command-line interface

## Core Components

### Document Loader
Responsible for extracting plain text from Microsoft Word (.docx) documents using the `docx-rs` library.

### Data Pipeline
Handles tokenization, batching, and dataset splitting. Implements the Burn `Dataset` trait to integrate seamlessly with the training framework.

### Model
A transformer-based neural network consisting of:
- Token embeddings
- Positional embeddings
- Six transformer encoder layers
- Output projection layer

### Training Pipeline
Controls the training loop, loss calculation, backpropagation, metric logging, and checkpoint saving.

### Inference Engine
Loads trained model checkpoints and processes user questions to generate answers.
