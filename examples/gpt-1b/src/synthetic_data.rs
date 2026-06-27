use std::marker::PhantomData;

use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor};

/// Synthetic data loader for language modeling.
///
/// Generates random token sequences.  The target equals the input
/// (for random data the distinction between input and target is negligible).
/// Implements `Iterator<Item = (Tensor<B, 2, Int>, Tensor<B, 2, Int>)>`.
pub struct SyntheticDataLoader<B: Backend> {
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
    num_batches: usize,
    current: usize,
    device: B::Device,
    _backend: PhantomData<B>,
}

impl<B: Backend> SyntheticDataLoader<B> {
    /// Creates a new synthetic data loader.
    ///
    /// * `batch_size` - Samples per batch.
    /// * `seq_len` - Sequence length per sample.
    /// * `vocab_size` - Token range `[0, vocab_size)`.
    /// * `num_batches` - How many batches to yield.
    /// * `device` - Burn device for tensor allocation.
    pub fn new(
        batch_size: usize,
        seq_len: usize,
        vocab_size: usize,
        num_batches: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            batch_size,
            seq_len,
            vocab_size,
            num_batches,
            current: 0,
            device: device.clone(),
            _backend: PhantomData,
        }
    }

    /// Total number of batches configured.
    #[allow(dead_code)]
    pub fn num_batches(&self) -> usize {
        self.num_batches
    }

    /// Current batch index (0-based).
    #[allow(dead_code)]
    pub fn current_batch(&self) -> usize {
        self.current
    }

    /// Resets the iterator to the beginning.
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.current = 0;
    }
}

impl<B: Backend> Iterator for SyntheticDataLoader<B> {
    type Item = (Tensor<B, 2, Int>, Tensor<B, 2, Int>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.num_batches {
            return None;
        }
        self.current += 1;

        let input: Tensor<B, 2> = Tensor::random(
            [self.batch_size, self.seq_len],
            Distribution::Uniform(0.0, self.vocab_size as f64),
            &self.device,
        );
        let input = input.int();

        // For causal LM the target would normally be input shifted by 1.
        // With random data the target is set to the input directly.
        let target = input.clone();

        Some((input, target))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray;

    fn default_device() -> NdArrayDevice {
        NdArrayDevice::default()
    }

    #[test]
    fn test_dataloader_output_shapes() {
        let device = default_device();
        let mut loader = SyntheticDataLoader::<TestBackend>::new(4, 32, 1000, 5, &device);
        let (input, target) = loader.next().expect("should produce a batch");
        assert_eq!(input.dims(), [4, 32]);
        assert_eq!(target.dims(), [4, 32]);
    }

    #[test]
    fn test_dataloader_respects_num_batches() {
        let device = default_device();
        let mut loader = SyntheticDataLoader::<TestBackend>::new(2, 16, 500, 3, &device);
        let mut count = 0;
        for _ in loader.by_ref() {
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_dataloader_exhaustion() {
        let device = default_device();
        let mut loader = SyntheticDataLoader::<TestBackend>::new(2, 8, 100, 1, &device);
        assert!(loader.next().is_some());
        assert!(loader.next().is_none());
    }

    #[test]
    fn test_dataloader_reset() {
        let device = default_device();
        let mut loader = SyntheticDataLoader::<TestBackend>::new(1, 4, 10, 2, &device);
        loader.next();
        assert_eq!(loader.current_batch(), 1);
        loader.reset();
        assert_eq!(loader.current_batch(), 0);
        assert!(loader.next().is_some());
    }

    #[test]
    fn test_dataloader_values_in_range() {
        let device = default_device();
        let vocab_size = 100;
        let mut loader = SyntheticDataLoader::<TestBackend>::new(4, 16, vocab_size, 1, &device);
        let (input, _) = loader.next().expect("should produce a batch");
        let data = input.into_data();
        let values: Vec<i64> = data.to_vec().expect("i64");
        for (i, &v) in values.iter().enumerate() {
            assert!(v >= 0 && v < vocab_size as i64, "index {i}: {v} not in [0, {vocab_size})");
        }
    }

    #[test]
    fn test_dataloader_zero_batches() {
        let device = default_device();
        let mut loader = SyntheticDataLoader::<TestBackend>::new(1, 1, 10, 0, &device);
        assert!(loader.next().is_none());
    }

    #[test]
    fn test_dataloader_num_batches() {
        let loader =
            SyntheticDataLoader::<TestBackend>::new(1, 1, 10, 7, &default_device());
        assert_eq!(loader.num_batches(), 7);
    }
}
