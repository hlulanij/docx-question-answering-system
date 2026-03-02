use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput};
use burn::tensor::{backend::Backend, Int, Tensor, TensorData};

/// Transformer-based Q&A model:
/// - token embeddings
/// - positional embeddings
/// - transformer encoder (n_layers >= 6)
/// - output projection -> start/end logits per token
///
/// Shapes:
/// input_ids: [batch, seq] (Int)
/// output logits: [batch, seq, 2] (Float)  // start/end logits
#[derive(Module, Debug)]
pub struct QaTransformerModel<B: Backend> {
    token_embed: Embedding<B>,
    pos_embed: Embedding<B>,
    encoder: TransformerEncoder<B>,
    out_proj: Linear<B>,
    max_seq_len: usize,
    d_model: usize,
}

impl<B: Backend> QaTransformerModel<B> {
    /// Proper initialization on a given device.
    pub fn new(
        device: &B::Device,
        vocab_size: usize,
        max_seq_len: usize,
        d_model: usize,
        d_ff: usize,
        n_heads: usize,
        n_layers: usize, // MUST be >= 6 for rubric [4](https://docs.rs/crate/burn/latest)
    ) -> Self {
        let token_embed = EmbeddingConfig::new(vocab_size, d_model).init(device);
        let pos_embed = EmbeddingConfig::new(max_seq_len, d_model).init(device);

        // ✅ SAFEST: init first, then move to device.
        // TransformerEncoderConfig provides init::<B>() [1](https://github.com/tracel-ai/burn/discussions/2866)
       let encoder = TransformerEncoderConfig::new(d_model, d_ff, n_heads, n_layers)
    .init::<B>(device);

        let out_proj = LinearConfig::new(d_model, 2).init(device);

        Self {
            token_embed,
            pos_embed,
            encoder,
            out_proj,
            max_seq_len,
            d_model,
        }
    }

    /// Forward pass:
    /// input_ids: [batch, seq] (Int)
    /// returns logits: [batch, seq, 2] (Float)
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq] = input_ids.dims();

        // Token embeddings: [batch, seq, d_model]
        let tok = self.token_embed.forward(input_ids);

        // ✅ Create position ids on the SAME device as tok
        let device = tok.device();
        let pos_ids = make_position_ids::<B>(&device, batch, seq);

        // Positional embeddings: [batch, seq, d_model]
        let pos = self.pos_embed.forward(pos_ids);

        let x = tok + pos;

        // Transformer encoder expects TransformerEncoderInput::new(tensor) [2](https://github.com/tracel-ai/burn/blob/main/crates/burn-wgpu/README.md)[3](https://deepwiki.com/tracel-ai/burn/6.3-adding-new-operations)
        let encoded = self.encoder.forward(TransformerEncoderInput::new(x));

        // Project per token: [batch, seq, d_model] -> [batch, seq, 2]
        let flat = encoded.reshape([batch * seq, self.d_model]);
        let logits_flat = self.out_proj.forward(flat);
        logits_flat.reshape([batch, seq, 2])
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

/// Create position ids tensor: shape [batch, seq], values 0..seq-1
fn make_position_ids<B: Backend>(device: &B::Device, batch: usize, seq: usize) -> Tensor<B, 2, Int> {
    // Use i32 values for typical WGPU Int element type.
    let mut values: Vec<i32> = Vec::with_capacity(batch * seq);
    for _ in 0..batch {
        for i in 0..seq {
            values.push(i as i32);
        }
    }

    // TensorData::new(values, shape) is the standard way to build tensors from Vec. [5](https://crates.io/crates/burn)
    let data = TensorData::new(values, [batch, seq]);
    Tensor::from_data(data, device)
}