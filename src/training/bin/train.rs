use burn::backend::{Autodiff, Wgpu};
use burn::backend::wgpu::WgpuDevice;

use word_doc_qa::training::config::TrainingConfig;
use word_doc_qa::training::train_loop::run_training;

fn main() {
    type Backend = Autodiff<Wgpu<f32, i32>>;
    let device = WgpuDevice::default();

    // Default config (you can later load from a file)
    let cfg = TrainingConfig::new();

    run_training::<Backend>(&device, &cfg);
}