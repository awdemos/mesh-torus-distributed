use float8::{F8E4M3, F8E5M2};

use super::config::Fp8Format;
use super::fp8_tensor::Fp8Tensor;
use super::scaling::fp8_format_max;

pub trait Quantize {
    fn to_fp8(values: &[f32], format: Fp8Format) -> Fp8Tensor;
    fn from_fp8(tensor: &Fp8Tensor) -> Vec<f32>;
}

pub struct Fp8Quantizer;

impl Quantize for Fp8Quantizer {
    fn to_fp8(values: &[f32], format: Fp8Format) -> Fp8Tensor {
        let amax = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if amax == 0.0 {
            1.0
        } else {
            fp8_format_max(format) / amax
        };
        let data = quantize_bytes(values, format, scale);
        Fp8Tensor::new(data, format, scale).with_amax_history(vec![amax])
    }

    fn from_fp8(tensor: &Fp8Tensor) -> Vec<f32> {
        match tensor.block_scales {
            Some(ref block_scales) => {
                let block_size = if !block_scales.is_empty() {
                    tensor.data.len().div_ceil(block_scales.len())
                } else {
                    tensor.data.len()
                };
                tensor
                    .data
                    .iter()
                    .enumerate()
                    .map(|(i, &bits)| {
                        let block_idx = i / block_size;
                        let scale = block_scales.get(block_idx).copied().unwrap_or(1.0);
                        dequantize_single(bits, tensor.format) / scale
                    })
                    .collect()
            }
            None => tensor
                .data
                .iter()
                .map(|&bits| dequantize_single(bits, tensor.format) / tensor.scale)
                .collect(),
        }
    }
}

pub fn to_fp8(values: &[f32], format: Fp8Format) -> Fp8Tensor {
    Fp8Quantizer::to_fp8(values, format)
}

pub fn from_fp8(tensor: &Fp8Tensor) -> Vec<f32> {
    Fp8Quantizer::from_fp8(tensor)
}

pub fn to_fp8_with_scale(values: &[f32], format: Fp8Format, scale: f32) -> Fp8Tensor {
    let amax = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let data = quantize_bytes(values, format, scale);
    Fp8Tensor::new(data, format, scale).with_amax_history(vec![amax])
}

pub fn to_fp8_per_block(values: &[f32], format: Fp8Format, block_size: usize) -> Fp8Tensor {
    assert!(block_size > 0, "block_size must be > 0");
    let num_blocks = values.len().div_ceil(block_size);
    let mut data = Vec::with_capacity(values.len());
    let mut block_scales = Vec::with_capacity(num_blocks);

    for block_idx in 0..num_blocks {
        let start = block_idx * block_size;
        let end = (start + block_size).min(values.len());
        let block = &values[start..end];

        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if amax == 0.0 {
            1.0
        } else {
            fp8_format_max(format) / amax
        };
        block_scales.push(scale);

        for &v in block {
            data.push(quantize_single(v, format, scale));
        }
    }

    let overall_amax = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    Fp8Tensor::new(data, format, block_scales.first().copied().unwrap_or(1.0))
        .with_block_scales(block_scales)
        .with_amax_history(vec![overall_amax])
}

fn quantize_single(value: f32, format: Fp8Format, scale: f32) -> u8 {
    let scaled = value * scale;
    match format {
        Fp8Format::E4M3 => F8E4M3::from_f32(scaled).to_bits(),
        Fp8Format::E5M2 => F8E5M2::from_f32(scaled).to_bits(),
    }
}

fn quantize_bytes(values: &[f32], format: Fp8Format, scale: f32) -> Vec<u8> {
    values.iter().map(|&v| quantize_single(v, format, scale)).collect()
}

fn dequantize_single(bits: u8, format: Fp8Format) -> f32 {
    match format {
        Fp8Format::E4M3 => F8E4M3::from_bits(bits).to_f32(),
        Fp8Format::E5M2 => F8E5M2::from_bits(bits).to_f32(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn relative_error(original: f32, roundtripped: f32) -> f32 {
        if original.abs() < 1e-8 {
            return (original - roundtripped).abs();
        }
        ((original - roundtripped) / original).abs()
    }

    #[rstest]
    #[case::e4m3(Fp8Format::E4M3)]
    #[case::e5m2(Fp8Format::E5M2)]
    fn roundtrip_zeros(#[case] format: Fp8Format) {
        let values = vec![0.0, 0.0, 0.0];
        let tensor = to_fp8(&values, format);
        let recovered = from_fp8(&tensor);
        for (orig, rec) in values.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 1e-6);
        }
    }

    #[rstest]
    #[case::e4m3(Fp8Format::E4M3, 0.07)]
    #[case::e5m2(Fp8Format::E5M2, 0.15)]
    fn roundtrip_fp8_exact_values(#[case] format: Fp8Format, #[case] tolerance: f32) {
        let values = vec![0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0];
        let tensor = to_fp8(&values, format);
        let recovered = from_fp8(&tensor);
        for (i, (orig, rec)) in values.iter().zip(recovered.iter()).enumerate() {
            let err = relative_error(*orig, *rec);
            assert!(
                err < tolerance,
                "index {i}: orig={orig}, rec={rec}, rel_err={err:.4}, tol={tolerance}"
            );
        }
    }

    #[rstest]
    #[case::e4m3(Fp8Format::E4M3, 0.07)]
    #[case::e5m2(Fp8Format::E5M2, 0.15)]
    fn roundtrip_general_values(#[case] format: Fp8Format, #[case] tolerance: f32) {
        let values = vec![0.1, 0.3, 0.7, 1.3, 2.7, 5.0];
        let tensor = to_fp8(&values, format);
        let recovered = from_fp8(&tensor);
        for (i, (orig, rec)) in values.iter().zip(recovered.iter()).enumerate() {
            let err = relative_error(*orig, *rec);
            assert!(
                err < tolerance,
                "index {i}: orig={orig}, rec={rec}, rel_err={err:.4}, tol={tolerance}"
            );
        }
    }

    #[test]
    fn roundtrip_negative_values() {
        let values = vec![-1.0, -2.0, -0.5, 0.5, 1.0, 2.0];
        let tensor = to_fp8(&values, Fp8Format::E4M3);
        let recovered = from_fp8(&tensor);
        for (i, (orig, rec)) in values.iter().zip(recovered.iter()).enumerate() {
            let err = relative_error(*orig, *rec);
            assert!(
                err < 0.07,
                "index {i}: orig={orig}, rec={rec}, rel_err={err:.4}"
            );
        }
    }

    #[test]
    fn roundtrip_preserves_shape() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = to_fp8(&values, Fp8Format::E4M3);
        assert_eq!(tensor.len(), 5);
        let recovered = from_fp8(&tensor);
        assert_eq!(recovered.len(), 5);
    }

    #[test]
    fn to_fp8_sets_amax_history() {
        let values = vec![1.0, -3.0, 2.0];
        let tensor = to_fp8(&values, Fp8Format::E4M3);
        assert_eq!(tensor.amax_history.len(), 1);
        assert!((tensor.amax_history[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn to_fp8_computes_scale() {
        let values = vec![1.0, 2.0, 4.0];
        let tensor = to_fp8(&values, Fp8Format::E4M3);
        let expected_scale = 416.0 / 4.0;
        assert!((tensor.scale - expected_scale).abs() < 1e-3);
    }

    #[test]
    fn per_block_quantization() {
        let values = vec![1.0, 2.0, 100.0, 200.0];
        let tensor = to_fp8_per_block(&values, Fp8Format::E4M3, 2);
        assert!(tensor.block_scales.is_some());
        let scales = tensor.block_scales.as_ref().unwrap();
        assert_eq!(scales.len(), 2);
        let recovered = from_fp8(&tensor);
        for (i, (orig, rec)) in values.iter().zip(recovered.iter()).enumerate() {
            let err = relative_error(*orig, *rec);
            assert!(
                err < 0.07,
                "index {i}: orig={orig}, rec={rec}, rel_err={err:.4}"
            );
        }
    }

    #[test]
    fn per_block_better_than_per_tensor_for_heterogeneous() {
        let values = vec![1.0, 2.0, 100.0, 200.0];
        let per_tensor = to_fp8(&values, Fp8Format::E4M3);
        let per_tensor_rec = from_fp8(&per_tensor);
        let per_block = to_fp8_per_block(&values, Fp8Format::E4M3, 2);
        let per_block_rec = from_fp8(&per_block);

        let tensor_err: f32 = values
            .iter()
            .zip(per_tensor_rec.iter())
            .map(|(o, r)| relative_error(*o, *r))
            .fold(0.0f32, f32::max);
        let block_err: f32 = values
            .iter()
            .zip(per_block_rec.iter())
            .map(|(o, r)| relative_error(*o, *r))
            .fold(0.0f32, f32::max);

        assert!(
            block_err <= tensor_err + 0.01,
            "per-block max error ({block_err:.4}) should not be worse than per-tensor ({tensor_err:.4})"
        );
    }

    #[test]
    fn to_fp8_with_custom_scale() {
        let values = vec![1.0, 2.0, 3.0];
        let tensor = to_fp8_with_scale(&values, Fp8Format::E4M3, 100.0);
        assert!((tensor.scale - 100.0).abs() < 1e-6);
        assert_eq!(tensor.len(), 3);
    }

    #[test]
    fn quantizer_trait_impl() {
        let values = vec![1.0, 2.0, 3.0];
        let tensor = Fp8Quantizer::to_fp8(&values, Fp8Format::E4M3);
        let recovered = Fp8Quantizer::from_fp8(&tensor);
        assert_eq!(recovered.len(), 3);
    }
}
