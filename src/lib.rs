#[cfg(test)]
mod tests {
    use super::qa_transformer::QaTransformerModel;
    use burn::backend::Wgpu;

    #[test]
    fn builds_transformer_model() {
        type B = Wgpu<f32, i32>;
        let device = <B as burn::tensor::backend::Backend>::Device::default();

        // n_layers = 6 to satisfy rubric minimum
        let _model = QaTransformerModel::<B>::new(
            &device,
            10_000, // vocab
            256,    // max_seq_len
            128,    // d_model
            512,    // d_ff
            8,      // n_heads
            6,      // n_layers (minimum required)
        );
    }
}



// Your transformer model file (src/qa_transformer.rs)
pub mod qa_transformer;

// Training pipeline folder (src/training/*)
pub mod training;

// (Optional) If you also want inference callable from lib code, uncomment:
// pub mod inference;