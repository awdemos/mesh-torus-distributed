use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

/// Multi-head scaled dot-product attention.
///
/// Projects to Q/K/V, splits into `num_heads`, computes scaled dot-product
/// attention with an optional 4D mask `[1, 1, seq, seq]`, concatenates
/// heads, and applies output projection.
#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub dropout: Dropout,
    pub num_heads: usize,
    pub d_head: usize,
    pub d_model: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new(d_model: usize, num_heads: usize, dropout_prob: f64, device: &B::Device) -> Self {
        assert!(
            d_model % num_heads == 0,
            "d_model ({}) must be divisible by num_heads ({})",
            d_model,
            num_heads
        );
        let d_head = d_model / num_heads;
        Self {
            q_proj: LinearConfig::new(d_model, d_model).init(device),
            k_proj: LinearConfig::new(d_model, d_model).init(device),
            v_proj: LinearConfig::new(d_model, d_model).init(device),
            out_proj: LinearConfig::new(d_model, d_model).init(device),
            dropout: DropoutConfig::new(dropout_prob).init(),
            num_heads,
            d_head,
            d_model,
        }
    }

    /// Forward pass.
    ///
    /// `x`: `[batch, seq, d_model]`
    /// `mask`: optional `[1, 1, seq, seq]` (0 for allowed, -inf for masked)
    ///
    /// Returns `[batch, seq, d_model]`.
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4>>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = q
            .reshape([batch_size, seq_len, self.num_heads, self.d_head])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch_size, seq_len, self.num_heads, self.d_head])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch_size, seq_len, self.num_heads, self.d_head])
            .swap_dims(1, 2);

        let k_t = k.swap_dims(2, 3);
        let scale = (self.d_head as f64).sqrt().recip();
        let mut scores = q.matmul(k_t) * scale;

        if let Some(m) = mask {
            scores = scores + m;
        }

        let attn = softmax(scores, 3);
        let attn = self.dropout.forward(attn);

        let out = attn.matmul(v);
        let out = out
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, self.d_model]);

        self.out_proj.forward(out)
    }
}

/// Two-layer feed-forward network with GELU activation.
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    pub dropout: Dropout,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(d_model: usize, d_ff: usize, dropout_prob: f64, device: &B::Device) -> Self {
        Self {
            linear1: LinearConfig::new(d_model, d_ff).init(device),
            linear2: LinearConfig::new(d_ff, d_model).init(device),
            dropout: DropoutConfig::new(dropout_prob).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(x);
        let x = gelu(x);
        let x = self.dropout.forward(x);
        self.linear2.forward(x)
    }
}

/// Pre-LN transformer block with residual connections.
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    pub attention: MultiHeadAttention<B>,
    pub ff: FeedForward<B>,
    pub norm1: LayerNorm<B>,
    pub norm2: LayerNorm<B>,
    pub dropout: Dropout,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout_prob: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            attention: MultiHeadAttention::new(d_model, num_heads, dropout_prob, device),
            ff: FeedForward::new(d_model, d_ff, dropout_prob, device),
            norm1: LayerNormConfig::new(d_model).init(device),
            norm2: LayerNormConfig::new(d_model).init(device),
            dropout: DropoutConfig::new(dropout_prob).init(),
        }
    }

    /// Forward pass.
    ///
    /// `x`: `[batch, seq, d_model]`
    /// `mask`: optional `[1, 1, seq, seq]` causal mask
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4>>) -> Tensor<B, 3> {
        let residual = x.clone();
        let x = self.norm1.forward(x);
        let x = self.attention.forward(x, mask);
        let x = self.dropout.forward(x);
        let x = x + residual;

        let residual = x.clone();
        let x = self.norm2.forward(x);
        let x = self.ff.forward(x);
        let x = self.dropout.forward(x);
        x + residual
    }
}

/// Creates a `[seq_len, seq_len]` causal attention mask.
///
/// Returns 0 for allowed positions (lower triangle + diagonal)
/// and `f32::NEG_INFINITY` for masked positions (upper triangle).
pub fn create_causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut data = Vec::with_capacity(seq_len * seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            data.push(if j > i { f32::NEG_INFINITY } else { 0.0 });
        }
    }
    Tensor::from_data(TensorData::new(data, vec![seq_len, seq_len]), device)
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
    fn test_mha_output_shape() {
        let device = default_device();
        let d_model = 64;
        let num_heads = 4;
        let batch_size = 2;
        let seq_len = 8;

        let mha = MultiHeadAttention::new(d_model, num_heads, 0.1, &device);
        let x = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_model],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let output = mha.forward(x, None);
        assert_eq!(output.dims(), [batch_size, seq_len, d_model]);
    }

    #[test]
    fn test_mha_with_causal_mask() {
        let device = default_device();
        let d_model = 32;
        let num_heads = 4;
        let batch_size = 2;
        let seq_len = 6;

        let mha = MultiHeadAttention::new(d_model, num_heads, 0.0, &device);
        let x = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_model],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let mask_2d = create_causal_mask::<TestBackend>(seq_len, &device);
        let mask_4d = mask_2d.reshape([1, 1, seq_len, seq_len]);
        let output = mha.forward(x, Some(mask_4d));
        assert_eq!(output.dims(), [batch_size, seq_len, d_model]);
    }

    #[test]
    fn test_ff_output_shape() {
        let device = default_device();
        let d_model = 64;
        let d_ff = 256;
        let batch_size = 2;
        let seq_len = 8;

        let ff = FeedForward::new(d_model, d_ff, 0.1, &device);
        let x = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_model],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let output = ff.forward(x);
        assert_eq!(output.dims(), [batch_size, seq_len, d_model]);
    }

    #[test]
    fn test_transformer_block_output_shape() {
        let device = default_device();
        let d_model = 64;
        let num_heads = 4;
        let d_ff = 256;
        let batch_size = 2;
        let seq_len = 8;

        let block = TransformerBlock::new(d_model, num_heads, d_ff, 0.1, &device);
        let x = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_model],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let output = block.forward(x, None);
        assert_eq!(output.dims(), [batch_size, seq_len, d_model]);
    }

    #[test]
    fn test_transformer_block_with_mask() {
        let device = default_device();
        let d_model = 32;
        let num_heads = 2;
        let d_ff = 128;
        let batch_size = 1;
        let seq_len = 4;

        let block = TransformerBlock::new(d_model, num_heads, d_ff, 0.0, &device);
        let x = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_model],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let mask_2d = create_causal_mask::<TestBackend>(seq_len, &device);
        let mask_4d = mask_2d.reshape([1, 1, seq_len, seq_len]);
        let output = block.forward(x, Some(mask_4d));
        assert_eq!(output.dims(), [batch_size, seq_len, d_model]);
    }

    #[test]
    fn test_causal_mask_shape() {
        let device = default_device();
        let mask = create_causal_mask::<TestBackend>(10, &device);
        assert_eq!(mask.dims(), [10, 10]);
    }

    #[test]
    fn test_causal_mask_values() {
        let device = default_device();
        let mask = create_causal_mask::<TestBackend>(4, &device);
        let data = mask.into_data();
        let values: Vec<f32> = data.to_vec().expect("f32");
        for i in 0..4 {
            for j in 0..4 {
                let v = values[i * 4 + j];
                if j > i {
                    assert!(v.is_infinite() && v.is_sign_negative(),
                        "[{i},{j}]: expected -inf, got {v}");
                } else {
                    assert!((v - 0.0).abs() < 1e-6,
                        "[{i},{j}]: expected 0, got {v}");
                }
            }
        }
    }

    #[test]
    fn test_mha_d_head() {
        let mha = MultiHeadAttention::<TestBackend>::new(128, 8, 0.1, &default_device());
        assert_eq!(mha.d_head, 16);
    }

    #[test]
    #[should_panic(expected = "divisible by num_heads")]
    fn test_mha_invalid_d_model() {
        MultiHeadAttention::<TestBackend>::new(127, 4, 0.1, &default_device());
    }
}
