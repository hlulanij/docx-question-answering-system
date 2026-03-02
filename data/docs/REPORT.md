# SEG 580S Assignment 1
## Question Answering System using Rust and Burn

**Student Name:** Hlulani Mathebula  
**Course:** SEG 580S – Software Engineering Deep Learning Systems  
**Framework:** Burn (Rust)

---

## 1. Introduction

This project implements a Question Answering (Q&A) system that reads Microsoft Word (.docx) documents and answers natural language questions about their content. The motivation for this project is to demonstrate the design and implementation of a complete deep learning system using Rust and the Burn framework.

The system is designed to process real-world document data, train a transformer-based neural network, and provide answers via a command-line interface. This assignment focuses on the full machine learning lifecycle, including data preparation, model architecture, training, and inference.

Key design decisions include the use of a transformer encoder architecture, modular system design, and strict adherence to the Burn framework and required dependency versions.

---

## 2. Implementation

### 2.1 Architecture Details

The system architecture consists of the following major components:

- **Document Loader**: Extracts raw text from `.docx` files
- **Data Pipeline**: Tokenizes text, batches samples, and creates training/validation splits
- **Model**: Transformer-based Question Answering network
- **Training Pipeline**: Handles loss computation, backpropagation, and checkpointing
- **Inference System**: Loads trained models and answers user questions via CLI

The neural network architecture is based on a transformer encoder with token embeddings, positional embeddings, and a stack of six transformer layers, followed by an output projection layer for answer prediction.

---

### 2.2 Data Pipeline

Microsoft Word documents are loaded using the `docx-rs` crate and converted into plain text. The extracted text is cleaned and segmented into training examples.

Tokenization is performed using the `tokenizers` library, converting text into token IDs suitable for model input. The dataset is batched and split into training and validation subsets to allow performance evaluation during training.

---
2.4 Code Walkthrough

The system is organized into clearly separated modules:

1. **Document Processing (`src/docx`)**  
   Responsible for loading and extracting text from `.docx` files.

2. **Data Pipeline (`src/data_pipeline` and `src/nlp`)**  
   Handles tokenization, batching, and dataset preparation using the Burn Dataset trait.

3. **Model (`src/model`)**  
   Implements a transformer-based Question Answering model with token embeddings, positional embeddings, and multiple transformer encoder layers.

4. **Training (`src/train`)**  
   Contains the training loop, loss calculation, backpropagation, checkpoint saving, and configuration management.

5. **Inference (`src/inference`)**  
   Loads trained model checkpoints and answers user questions through a command-line interface.

This modular structure improves maintainability, testability, and clarity of the system.


