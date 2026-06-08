use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, EmbeddingConfig, LayerNormConfig, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use crate::transformer::{TransformerBlock, create_causal_mask};

/// GPT model hyper-parameters.
///
/// Use [`GptConfig::gpt_1b`] for the ~1B parameter variant or
/// [`GptConfig::test_config`] for a small config suitable for CPU testing.
#[derive(Debug, Clone)]
pub struct GptConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub dropout: f64,
}

impl GptConfig {
    /// ~1B parameter configuration (GPT-2 scale).
    pub fn gpt_1b() -> Self {
        Self {
            vocab_size: 50257,
            d_model: 2048,
            num_layers: 24,
            num_heads: 32,
            d_ff: 8192,
            max_seq_len: 2048,
            dropout: 0.1,
        }
    }

    /// Small test config for fast CPU execution.
    pub fn test_config() -> Self {
        Self {
            vocab_size: 1000,
            d_model: 128,
            num_layers: 2,
            num_heads: 4,
            d_ff: 512,
            max_seq_len: 64,
            dropout: 0.1,
        }
    }
}

/// GPT decoder-only language model.
///
/// Architecture: token embedding + position embedding → N transformer blocks
/// → final layer norm → LM head.
#[derive(Module, Debug)]
pub struct GptModel<B: Backend> {
    pub token_embedding: burn::nn::Embedding<B>,
    pub position_embedding: burn::nn::Embedding<B>,
    pub blocks: Vec<TransformerBlock<B>>,
    pub ln_f: burn::nn::LayerNorm<B>,
    pub lm_head: burn::nn::Linear<B>,
    pub dropout: Dropout,
    pub d_model: usize,
    pub max_seq_len: usize,
    pub vocab_size: usize,
}

impl<B: Backend> GptModel<B> {
    /// Creates a new GPT model from the given configuration.
    pub fn new(config: &GptConfig, device: &B::Device) -> Self {
        let token_embedding = EmbeddingConfig::new(config.vocab_size, config.d_model).init(device);
        let position_embedding =
            EmbeddingConfig::new(config.max_seq_len, config.d_model).init(device);

        let blocks = (0..config.num_layers)
            .map(|_| {
                TransformerBlock::new(
                    config.d_model,
                    config.num_heads,
                    config.d_ff,
                    config.dropout,
                    device,
                )
            })
            .collect();

        let ln_f = LayerNormConfig::new(config.d_model).init(device);
        let lm_head = LinearConfig::new(config.d_model, config.vocab_size).init(device);

        Self {
            token_embedding,
            position_embedding,
            blocks,
            ln_f,
            lm_head,
            dropout: DropoutConfig::new(config.dropout).init(),
            d_model: config.d_model,
            max_seq_len: config.max_seq_len,
            vocab_size: config.vocab_size,
        }
    }

    /// Forward pass producing logits over the vocabulary.
    ///
    /// `input`: `[batch_size, seq_len]` integer token IDs.
    ///
    /// Returns `[batch_size, seq_len, vocab_size]` logits.
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input.dims();
        let device = input.device();

        let token_emb = self.token_embedding.forward(input);
        let positions = create_position_ids::<B>(batch_size, seq_len, &device);
        let pos_emb = self.position_embedding.forward(positions);

        let mut hidden = token_emb + pos_emb;
        hidden = self.dropout.forward(hidden);

        let mask_2d = create_causal_mask::<B>(seq_len, &device);
        let mask_4d = mask_2d.reshape([1, 1, seq_len, seq_len]);

        for block in &self.blocks {
            hidden = block.forward(hidden, Some(mask_4d.clone()));
        }

        hidden = self.ln_f.forward(hidden);
        self.lm_head.forward(hidden)
    }

    /// Returns the number of transformer blocks.
    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }
}

/// Creates a `[batch_size, seq_len]` tensor of position IDs
/// where each row is `[0, 1, ..., seq_len-1]`.
fn create_position_ids<B: Backend>(
    batch_size: usize,
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let mut data: Vec<i64> = Vec::with_capacity(batch_size * seq_len);
    for _ in 0..batch_size {
        for j in 0..seq_len {
            data.push(j as i64);
        }
    }
    Tensor::<B, 2, Int>::from_data(TensorData::new(data, vec![batch_size, seq_len]), device)
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
    fn test_gpt_config_1b() {
        let config = GptConfig::gpt_1b();
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.d_model, 2048);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.d_ff, 8192);
        assert_eq!(config.max_seq_len, 2048);
    }

    #[test]
    fn test_gpt_model_forward_shape() {
        let device = default_device();
        let config = GptConfig::test_config();
        let model = GptModel::<TestBackend>::new(&config, &device);

        let batch_size = 2;
        let seq_len = 16;
        let input = Tensor::<TestBackend, 2>::random(
            [batch_size, seq_len],
            burn::tensor::Distribution::Uniform(0.0, config.vocab_size as f64),
            &device,
        )
        .int();

        let logits = model.forward(input);
        assert_eq!(logits.dims(), [batch_size, seq_len, config.vocab_size]);
    }

    #[test]
    fn test_gpt_model_num_layers() {
        let device = default_device();
        let config = GptConfig::test_config();
        let model = GptModel::<TestBackend>::new(&config, &device);
        assert_eq!(model.num_layers(), 2);
    }

    #[test]
    fn test_gpt_model_single_sample() {
        let device = default_device();
        let config = GptConfig::test_config();
        let model = GptModel::<TestBackend>::new(&config, &device);

        let input = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1i64, 2, 3, 4], vec![1, 4]),
            &device,
        );
        let logits = model.forward(input);
        assert_eq!(logits.dims(), [1, 4, config.vocab_size]);
    }

    #[test]
    fn test_position_ids() {
        let device = default_device();
        let pos = create_position_ids::<TestBackend>(2, 5, &device);
        let data = pos.into_data();
        let values: Vec<i64> = data.to_vec().expect("i64");
        assert_eq!(values.len(), 10);
        for i in 0..2 {
            for j in 0..5 {
                assert_eq!(values[i * 5 + j], j as i64);
            }
        }
    }

    #[test]
    fn test_gpt_model_logits_finite() {
        let device = default_device();
        let config = GptConfig::test_config();
        let model = GptModel::<TestBackend>::new(&config, &device);

        let input = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![0i64, 1, 2, 3, 4], vec![1, 5]),
            &device,
        );
        let logits = model.forward(input);
        let data = logits.into_data();
        let values: Vec<f32> = data.to_vec().expect("f32");
        for (i, &v) in values.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] is not finite: {v}");
        }
    }
}
