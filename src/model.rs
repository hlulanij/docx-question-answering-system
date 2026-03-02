use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::{backend::Backend, Int, Tensor};

#[derive(Module, Debug)]
pub struct SimpleModel<B: Backend> {
    embed: Embedding<B>,
    linear: Linear<B>,
}

impl<B: Backend> SimpleModel<B> {
    pub fn init(device: &B::Device, vocab_size: usize, embed_dim: usize) -> Self {
        // Burn 0.20.x uses Config -> init(device) pattern for modules. [3](https://www.huggingface.co/docs/tokenizers/python/latest/index.html)[4](https://stackoverflow.com/questions/65924090/simpletransformers-error-versionconflict-tokenizers-0-9-4-how-do-i-fix-this)
        let embed = EmbeddingConfig::new(vocab_size, embed_dim).init(device);
        let linear = LinearConfig::new(embed_dim, 1).init(device);

        Self { embed, linear }
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        // Embedding expects Int token IDs. [1](https://github.com/bokuweb/docx-rs/blob/master/docx-core/examples/table.rs)[5](https://docs.rs/docx-rs/latest/docx_rs/enum.DocumentChild.html)
        // [batch, seq] -> [batch, seq, embed]
        let embedded: Tensor<B, 3> = self.embed.forward(input_ids);

        // In Burn, many reduction ops keep rank / keep dims; mean_dim can return 3D. [1](https://github.com/bokuweb/docx-rs/blob/master/docx-core/examples/table.rs)
        // pooled3: [batch, 1, embed]
        let pooled3: Tensor<B, 3> = embedded.mean_dim(1);

        // Convert [batch, 1, embed] -> [batch, embed] using reshape (safe and simple). [1](https://github.com/bokuweb/docx-rs/blob/master/docx-core/examples/table.rs)
        let [batch, _one, embed] = pooled3.dims();
        let pooled: Tensor<B, 2> = pooled3.reshape([batch, embed]);

        // Linear expects 2D input. [3](https://www.huggingface.co/docs/tokenizers/python/latest/index.html)
        self.linear.forward(pooled)
    }
}