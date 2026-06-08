use float8::{F8E4M3, F8E5M2};

use super::config::Fp8Format;

#[derive(Debug, Clone, PartialEq)]
pub struct Fp8Tensor {
    pub data: Vec<u8>,
    pub format: Fp8Format,
    pub scale: f32,
    pub block_scales: Option<Vec<f32>>,
    pub amax_history: Vec<f32>,
}

impl Fp8Tensor {
    pub fn new(data: Vec<u8>, format: Fp8Format, scale: f32) -> Self {
        Self {
            data,
            format,
            scale,
            block_scales: None,
            amax_history: Vec::new(),
        }
    }

    pub fn with_block_scales(mut self, scales: Vec<f32>) -> Self {
        self.block_scales = Some(scales);
        self
    }

    pub fn with_amax_history(mut self, history: Vec<f32>) -> Self {
        self.amax_history = history;
        self
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn amax(&self) -> f32 {
        if let Some(&last) = self.amax_history.last() {
            return last;
        }
        if self.scale > 0.0 {
            let fp8_max = match self.format {
                Fp8Format::E4M3 => F8E4M3::MAX.to_f32(),
                Fp8Format::E5M2 => F8E5M2::MAX.to_f32(),
            };
            return fp8_max / self.scale;
        }
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn new_tensor_basic() {
        let t = Fp8Tensor::new(vec![0u8; 4], Fp8Format::E4M3, 1.0);
        assert_eq!(t.len(), 4);
        assert!(!t.is_empty());
        assert_eq!(t.format, Fp8Format::E4M3);
        assert!((t.scale - 1.0).abs() < 1e-6);
        assert!(t.block_scales.is_none());
        assert!(t.amax_history.is_empty());
    }

    #[test]
    fn empty_tensor() {
        let t = Fp8Tensor::new(vec![], Fp8Format::E5M2, 1.0);
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
    }

    #[test]
    fn with_block_scales_builder() {
        let t = Fp8Tensor::new(vec![0u8; 4], Fp8Format::E4M3, 1.0)
            .with_block_scales(vec![2.0, 4.0]);
        assert_eq!(t.block_scales.as_deref(), Some([2.0, 4.0].as_slice()));
    }

    #[test]
    fn with_amax_history_builder() {
        let t = Fp8Tensor::new(vec![0u8; 4], Fp8Format::E4M3, 1.0)
            .with_amax_history(vec![1.0, 2.0, 3.0]);
        assert_eq!(t.amax_history, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn amax_from_history() {
        let t = Fp8Tensor::new(vec![0u8; 4], Fp8Format::E4M3, 1.0)
            .with_amax_history(vec![1.0, 5.0, 3.0]);
        assert!((t.amax() - 3.0).abs() < 1e-6);
    }

    #[rstest]
    #[case::e4m3(Fp8Format::E4M3)]
    #[case::e5m2(Fp8Format::E5M2)]
    fn amax_from_scale(#[case] format: Fp8Format) {
        let fp8_max = match format {
            Fp8Format::E4M3 => F8E4M3::MAX.to_f32(),
            Fp8Format::E5M2 => F8E5M2::MAX.to_f32(),
        };
        let t = Fp8Tensor::new(vec![0u8; 4], format, 208.0);
        let expected = fp8_max / 208.0;
        assert!((t.amax() - expected).abs() < 1e-3);
    }

    #[test]
    fn clone_equality() {
        let t = Fp8Tensor::new(vec![1, 2, 3], Fp8Format::E4M3, 2.0)
            .with_amax_history(vec![4.0]);
        assert_eq!(t, t.clone());
    }
}
