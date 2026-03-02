# SEG 580S Assignment 1
## Question Answering System using Rust and Burn

**Student:** Hlulani Mathebula

---

## 1. Introduction
This project implements a Question Answering (Q&A) system that reads Microsoft Word (.docx) documents and answers natural language questions about their content. The system is developed in Rust using the Burn deep learning framework.

The objective is to demonstrate an end-to-end deep learning pipeline including data processing, model training, and inference.

---

## 2. Implementation

### 2.1 Architecture Details
The system consists of the following components:
- DOCX document loader
- Tokenization and batching pipeline
- Transformer-based Question Answering model
- Training pipeline
- Command-line inference interface

The neural network architecture is based on a transformer encoder with token and positional embeddings.

---

### 2.2 Data Pipeline
Word documents are loaded using the `docx-rs` crate and converted into plain text. The extracted text is tokenized and prepared for training using batching and dataset splitting.

---

### 2.3 Training Strategy
The model is trained using gradient-based optimization with configurable hyperparameters such as learning rate, batch size, and number of epochs. Training metrics such as loss are tracked during training.

---

## 3. Experiments and Results

### 3.1 Training Results
Training and validation loss values are monitored to evaluate convergence.

---

### 3.2 Model Performance
Example questions and answers include:
- What is the month and date of the 2026 End of Year Graduation Ceremony?
- How many times did the HDC hold meetings in 2024?

System strengths, weaknesses, and failure cases are discussed.

---

## 4. Conclusion
This project demonstrates the construction of a transformer-based Question Answering system in Rust. Challenges included managing model configuration and training complexity.

Future work includes improving model accuracy, scaling to larger datasets, and optimizing inference performance.
``
