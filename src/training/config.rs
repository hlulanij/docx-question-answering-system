use burn::config::Config;

#[derive(Config, Debug)]
pub struct TrainingConfig {
    // --- training hyperparameters ---
    #[config(default = 2)]
    pub num_epochs: usize,

    #[config(default = 8)]
    pub batch_size: usize,

    #[config(default = 1e-3)]
    pub lr: f64,

    #[config(default = 42)]
    pub seed: u64,

    // --- model hyperparameters ---
    #[config(default = 10_000)]
    pub vocab_size: usize,

    #[config(default = 128)]
    pub max_seq_len: usize,

    #[config(default = 128)]
    pub d_model: usize,

    #[config(default = 512)]
    pub d_ff: usize,

    #[config(default = 8)]
    pub n_heads: usize,

    // MUST be >= 6 for rubric
    #[config(default = 6)]
    pub n_layers: usize,

    // --- artifacts ---
    #[config(default = "artifacts")]
    pub artifact_dir: String,
}
