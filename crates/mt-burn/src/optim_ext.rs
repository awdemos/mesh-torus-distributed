use std::vec::Vec;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn::module::{Module, ParamId};
use mt_core::precision::{Fp8Format, Fp8Tensor, MixedPrecisionConfig, from_fp8, to_fp8};

pub trait Fp8Optimizer<B: Backend> {
    fn step_fp8<M: Module<B>>(
        &mut self,
        lr: f64,
        module: M,
        grads: Fp8Gradients,
        config: &MixedPrecisionConfig,
    ) -> M;
}

#[derive(Debug, Clone, Default)]
pub struct Fp8Gradients {
    grads: Vec<(ParamId, Vec<f32>)>,
}

impl Fp8Gradients {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, id: ParamId, values: Vec<f32>) {
        self.grads.push((id, values));
    }

    pub fn remove(&mut self, id: ParamId) -> Option<Vec<f32>> {
        let pos = self.grads.iter().position(|(pid, _)| *pid == id)?;
        Some(self.grads.remove(pos).1)
    }

    pub fn len(&self) -> usize {
        self.grads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.grads.is_empty()
    }
}

pub fn update_weight_fp8<B: Backend, const D: usize>(
    weight: Tensor<B, D>,
    grad_fp8: &Fp8Tensor,
    lr: f64,
) -> Tensor<B, D> {
    let weight_data = weight.into_data();
    let dims: Vec<usize> = weight_data.shape.as_slice().to_vec();
    let weight_f32: Vec<f32> = weight_data.to_vec().expect("f32 conversion");
    let grad_f32 = from_fp8(grad_fp8);
    let device = B::Device::default();

    let updated: Vec<f32> = weight_f32
        .iter()
        .zip(grad_f32.iter())
        .map(|(&w, &g)| w - lr as f32 * g)
        .collect();

    let data = TensorData::new(updated, dims);
    Tensor::from_data(data, &device)
}

pub fn cast_to_fp8(values: &[f32], format: Fp8Format) -> Fp8Tensor {
    to_fp8(values, format)
}

pub fn cast_from_fp8(tensor: &Fp8Tensor) -> Vec<f32> {
    from_fp8(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};
    use burn::module::ParamId;
    use burn_ndarray::NdArray;
    use burn_ndarray::NdArrayDevice;
    use rstest::rstest;

    type TestBackend = NdArray;

    fn default_device() -> NdArrayDevice {
        Default::default()
    }

    #[test]
    fn fp8_gradients_register_remove() {
        let mut grads = Fp8Gradients::new();
        let id = ParamId::from(1u64);
        assert!(grads.is_empty());

        grads.register(id, vec![1.0, 2.0, 3.0]);
        assert_eq!(grads.len(), 1);

        let values = grads.remove(id);
        assert_eq!(values, Some(vec![1.0, 2.0, 3.0]));
        assert!(grads.is_empty());
    }

    #[test]
    fn fp8_gradients_remove_missing() {
        let mut grads = Fp8Gradients::new();
        let id = ParamId::from(2u64);
        assert!(grads.remove(id).is_none());
    }

    #[rstest]
    #[case::e4m3_forward(Fp8Format::E4M3)]
    #[case::e5m2_backward(Fp8Format::E5M2)]
    fn cast_roundtrip(#[case] format: Fp8Format) {
        let values = vec![1.0f32, 2.0, 3.0, 0.5, -1.0];
        let fp8 = cast_to_fp8(&values, format);
        let recovered = cast_from_fp8(&fp8);
        assert_eq!(recovered.len(), values.len());
    }

    #[test]
    fn update_weight_basic() {
        let device = default_device();
        let weight = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0], vec![3]),
            &device,
        );
        let grad_values = vec![0.1f32, 0.2, 0.3];
        let grad_fp8 = to_fp8(&grad_values, Fp8Format::E5M2);

        let updated = update_weight_fp8::<TestBackend, 1>(
            weight, &grad_fp8, 1.0,
        );

        let updated_data = updated.into_data();
        let updated_f32: Vec<f32> = updated_data.to_vec().expect("f32");
        let grad_f32 = from_fp8(&grad_fp8);

        for (i, (w, g)) in [1.0f32, 2.0, 3.0].iter().zip(grad_f32.iter()).enumerate() {
            let expected = w - g;
            let diff = (updated_f32[i] - expected).abs();
            assert!(diff < 0.1, "index {i}: expected ~{expected}, got {}, diff={diff}", updated_f32[i]);
        }
    }

    #[test]
    fn update_weight_2d() {
        let device = default_device();
        let weight = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]),
            &device,
        );
        let grad_values = vec![0.1f32, 0.1, 0.1, 0.1];
        let grad_fp8 = to_fp8(&grad_values, Fp8Format::E5M2);

        let updated = update_weight_fp8::<TestBackend, 2>(
            weight, &grad_fp8, 0.1,
        );

        let updated_data = updated.into_data();
        let updated_f32: Vec<f32> = updated_data.to_vec().expect("f32");

        for (i, val) in updated_f32.iter().enumerate() {
            let original = [1.0f32, 2.0, 3.0, 4.0][i];
            assert!(
                *val < original,
                "index {i}: weight should decrease with positive grad, got {val} >= {original}"
            );
        }
    }

    #[test]
    fn mixed_precision_config_defaults() {
        let config = MixedPrecisionConfig::default();
        assert_eq!(config.forward_format, Fp8Format::E4M3);
        assert_eq!(config.backward_format, Fp8Format::E5M2);
    }
}
