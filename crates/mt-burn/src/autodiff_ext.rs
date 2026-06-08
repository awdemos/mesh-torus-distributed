    use burn::tensor::{Tensor, TensorData, backend::Backend};
use mt_core::precision::{Fp8Format, to_fp8, from_fp8};

pub struct GradientHook;

impl GradientHook {
    pub fn cast_gradients_to_e5m2<B: Backend, const D: usize>(
        tensor: &Tensor<B, D>,
    ) -> Tensor<B, D> {
        let device = tensor.device();
        let data = tensor.to_data();
        let f32_values: Vec<f32> = data.to_vec().expect("f32 conversion");
        let fp8_tensor = to_fp8(&f32_values, Fp8Format::E5M2);
        let dequantized = from_fp8(&fp8_tensor);
        let shape: Vec<usize> = (0..D).map(|_| 1).collect();
        let new_data = TensorData::new(dequantized, shape);
        Tensor::from_data(new_data, &device)
    }

    pub fn cast_gradients_to_e5m2_preserving_shape<B: Backend, const D: usize>(
        tensor: &Tensor<B, D>,
    ) -> Tensor<B, D> {
        let device = tensor.device();
        let shape = tensor.shape();
        let dims: Vec<usize> = shape.dims::<D>().to_vec();
        let data = tensor.to_data();
        let f32_values: Vec<f32> = data.to_vec().expect("f32 conversion");
        let fp8_tensor = to_fp8(&f32_values, Fp8Format::E5M2);
        let dequantized = from_fp8(&fp8_tensor);
        let new_data = TensorData::new(dequantized, dims);
        Tensor::from_data(new_data, &device)
    }
}

pub struct AutodiffExt;

impl AutodiffExt {
    pub fn gradient_cast_e5m2<B: Backend, const D: usize>(
        grad: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let device = grad.device();
        let shape = grad.shape();
        let dims: Vec<usize> = shape.dims::<D>().to_vec();
        let data = grad.into_data();
        let f32_values: Vec<f32> = data.to_vec().expect("f32 conversion");
        let fp8 = to_fp8(&f32_values, Fp8Format::E5M2);
        let dequantized = from_fp8(&fp8);
        let new_data = TensorData::new(dequantized, dims);
        Tensor::from_data(new_data, &device)
    }

    pub fn weight_cast_e4m3<B: Backend, const D: usize>(
        weight: &Tensor<B, D>,
    ) -> Tensor<B, D> {
        let device = weight.device();
        let shape = weight.shape();
        let dims: Vec<usize> = shape.dims::<D>().to_vec();
        let data = weight.to_data();
        let f32_values: Vec<f32> = data.to_vec().expect("f32 conversion");
        let fp8 = to_fp8(&f32_values, Fp8Format::E4M3);
        let dequantized = from_fp8(&fp8);
        let new_data = TensorData::new(dequantized, dims);
        Tensor::from_data(new_data, &device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};
    use burn_ndarray::NdArray;
    use burn_ndarray::NdArrayDevice;

    type TestBackend = NdArray;

    fn default_device() -> NdArrayDevice {
        Default::default()
    }

    fn relative_error(original: f32, roundtripped: f32) -> f32 {
        if original.abs() < 1e-6 {
            return (original - roundtripped).abs();
        }
        ((original - roundtripped) / original).abs()
    }

    #[test]
    fn gradient_cast_e5m2_1d() {
        let device = default_device();
        let values = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let tensor = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(values.clone(), vec![5]),
            &device,
        );
        let cast = AutodiffExt::gradient_cast_e5m2(tensor);
        let cast_data = cast.into_data();
        let cast_f32: Vec<f32> = cast_data.to_vec().expect("f32");

        for (i, (orig, rec)) in values.iter().zip(cast_f32.iter()).enumerate() {
            let err = relative_error(*orig, *rec);
            assert!(
                err < 0.15,
                "index {i}: orig={orig}, rec={rec}, rel_err={err:.4}"
            );
        }
    }

    #[test]
    fn gradient_cast_e5m2_2d() {
        let device = default_device();
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(values.clone(), vec![2, 3]),
            &device,
        );
        let cast = AutodiffExt::gradient_cast_e5m2(tensor);
        let cast_data = cast.into_data();
        let cast_f32: Vec<f32> = cast_data.to_vec().expect("f32");

        assert_eq!(cast_f32.len(), 6);
        for (i, (orig, rec)) in values.iter().zip(cast_f32.iter()).enumerate() {
            let err = relative_error(*orig, *rec);
            assert!(
                err < 0.15,
                "index {i}: orig={orig}, rec={rec}, rel_err={err:.4}"
            );
        }
    }

    #[test]
    fn weight_cast_e4m3_1d() {
        let device = default_device();
        let values = vec![1.0f32, 2.0, 3.0];
        let tensor = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(values.clone(), vec![3]),
            &device,
        );
        let cast = AutodiffExt::weight_cast_e4m3(&tensor);
        let cast_data = cast.into_data();
        let cast_f32: Vec<f32> = cast_data.to_vec().expect("f32");

        for (i, (orig, rec)) in values.iter().zip(cast_f32.iter()).enumerate() {
            let err = relative_error(*orig, *rec);
            assert!(
                err < 0.07,
                "index {i}: orig={orig}, rec={rec}, rel_err={err:.4}"
            );
        }
    }

    #[test]
    fn weight_cast_e4m3_preserves_shape() {
        let device = default_device();
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(values, vec![2, 3]),
            &device,
        );
        let cast = AutodiffExt::weight_cast_e4m3(&tensor);
        assert_eq!(cast.shape().dims::<2>(), [2, 3]);
    }

    #[test]
    fn gradient_hook_preserving_shape() {
        let device = default_device();
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(values, vec![2, 2]),
            &device,
        );
        let cast = GradientHook::cast_gradients_to_e5m2_preserving_shape(&tensor);
        assert_eq!(cast.shape().dims::<2>(), [2, 2]);
    }
}
