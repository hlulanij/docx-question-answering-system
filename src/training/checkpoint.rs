use std::fs;
use std::path::Path;

use burn::record::CompactRecorder;
use burn::tensor::backend::Backend;

pub fn reset_dir<P: AsRef<Path>>(path: P) {
    let p = path.as_ref();
    fs::remove_dir_all(p).ok();
    fs::create_dir_all(p).ok();
}

pub fn ensure_dir<P: AsRef<Path>>(path: P) {
    fs::create_dir_all(path).ok();
}

/// Save model record to a file (Burn recorder).
pub fn save_model<B: Backend, M: burn::module::Module<B>>(
    model: &M,
    dir: &str,
    name: &str,
) {
    ensure_dir(dir);
    let path = format!("{}/{}", dir, name);

    // CompactRecorder is a standard Burn recorder optimized for compactness. [5](https://burn.dev/docs/burn/record/type.CompactRecorder.html)
    model.clone()
        .save_file(path, &CompactRecorder::new())
        .expect("Failed to save model checkpoint");
}