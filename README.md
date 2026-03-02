# DOCX Question Answering System (Rust + Burn)

## Overview
This project implements a Question Answering (Q&A) system that reads Microsoft Word (.docx) documents and answers natural language questions about their content. The system is built in Rust using the Burn deep learning framework.

This project was developed for **SEG 580S – Software Engineering Deep Learning Systems**.

## Features
- Loads and processes `.docx` documents
- Tokenizes and batches text data
- Transformer-based Question Answering model
- Training pipeline with configurable hyperparameters
- Command-line interface for training and inference

## Tech Stack
- Rust (Edition 2021)
- Burn 0.20.1
- docx-rs
- tokenizers
- serde / serde_json

## Project Structure
- `src/` – Core system implementation
- `data/` – Sample documents and datasets
- `docs/` – Project report and documentation

## Running the Project
```bash
cargo build
cargo run -- train
cargo run -- ask --question "Your question here"
