use burn::tensor::{Tensor, TensorData, backend::Backend};
use mt_core::precision::{Fp8Format, Fp8Tensor, from_fp8, to_fp8};

pub fn fp8_to_burn<B: Backend, const D: usize>(
    fp8_tensor: &Fp8Tensor,
    device: &B::Device,
) -> Tensor<B, D> {
    let f32_values = from_fp8(fp8_tensor);
    let numel = f32_values.len();
    let dims: Vec<usize> = if D == 1 {
        vec![numel]
    } else {
        let last = numel.ilog(D);
        let mut d = vec![D; D - 1];
        d.push(numel / D.pow(last as u32).max(1));
        d
    };
    let data = TensorData::new(f32_values, dims);
    Tensor::from_data(data, device)
}

pub fn burn_to_fp8<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    format: Fp8Format,
) -> Fp8Tensor {
    let data = tensor.into_data();
    let f32_values: Vec<f32> = data.to_vec().expect("dtype conversion to f32");
    to_fp8(&f32_values, format)
}

pub fn fp8_to_burn_1d<B: Backend>(
    fp8_tensor: &Fp8Tensor,
    device: &B::Device,
) -> Tensor<B, 1> {
    let f32_values = from_fp8(fp8_tensor);
    let data = TensorData::new(f32_values, vec![fp8_tensor.len()]);
    Tensor::from_data(data, device)
}

pub fn fp8_to_burn_2d<B: Backend>(
    fp8_tensor: &Fp8Tensor,
    rows: usize,
    cols: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let f32_values = from_fp8(fp8_tensor);
    assert_eq!(f32_values.len(), rows * cols);
    let data = TensorData::new(f32_values, vec![rows, cols]);
    Tensor::from_data(data, device)
}

pub fn burn_1d_to_fp8<B: Backend>(tensor: Tensor<B, 1>, format: Fp8Format) -> Fp8Tensor {
    let data = tensor.into_data();
    let f32_values: Vec<f32> = data.to_vec().expect("dtype conversion to f32");
    to_fp8(&f32_values, format)
}

pub fn burn_2d_to_fp8<B: Backend>(tensor: Tensor<B, 2>, format: Fp8Format) -> Fp8Tensor {
    let data = tensor.into_data();
    let f32_values: Vec<f32> = data.to_vec().expect("dtype conversion to f32");
    to_fp8(&f32_values, format)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use burn_ndarray::NdArrayDevice;
    use rstest::rstest;

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

    #[rstest]
    #[case::e4m3(Fp8Format::E4M3, 0.07)]
    #[case::e5m2(Fp8Format::E5M2, 0.15)]
    fn roundtrip_1d(#[case] format: Fp8Format, #[case] tolerance: f32) {
        let device = default_device();
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 0.5, -1.0];
        let data = TensorData::new(values.clone(), vec![values.len()]);
        let tensor: Tensor<TestBackend, 1> = Tensor::from_data(data, &device);

        let fp8 = burn_1d_to_fp8(tensor.clone(), format);
        let recovered: Tensor<TestBackend, 1> = fp8_to_burn_1d(&fp8, &device);

        let recovered_data = recovered.into_data();
        let recovered_values: Vec<f32> = recovered_data.to_vec().expect("f32 conversion");

        for (i, (orig, rec)) in values.iter().zip(recovered_values.iter()).enumerate() {
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
    fn roundtrip_2d(#[case] format: Fp8Format, #[case] tolerance: f32) {
        let device = default_device();
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data = TensorData::new(values.clone(), vec![2, 3]);
        let tensor: Tensor<TestBackend, 2> = Tensor::from_data(data, &device);

        let fp8 = burn_2d_to_fp8(tensor.clone(), format);
        let recovered: Tensor<TestBackend, 2> = fp8_to_burn_2d(&fp8, 2, 3, &device);

        let recovered_data = recovered.into_data();
        let recovered_values: Vec<f32> = recovered_data.to_vec().expect("f32 conversion");

        for (i, (orig, rec)) in values.iter().zip(recovered_values.iter()).enumerate() {
            let err = relative_error(*orig, *rec);
            assert!(
                err < tolerance,
                "index {i}: orig={orig}, rec={rec}, rel_err={err:.4}, tol={tolerance}"
            );
        }
    }

    #[test]
    fn fp8_preserves_format() {
        let device = default_device();
        let values = vec![1.0f32, 2.0, 3.0];
        let data = TensorData::new(values, vec![3]);
        let tensor: Tensor<TestBackend, 1> = Tensor::from_data(data, &device);

        let fp8 = burn_1d_to_fp8(tensor, Fp8Format::E4M3);
        assert_eq!(fp8.format, Fp8Format::E4M3);

        let fp8 = burn_1d_to_fp8(fp8_to_burn_1d::<TestBackend>(&fp8, &device), Fp8Format::E5M2);
        assert_eq!(fp8.format, Fp8Format::E5M2);
    }

    #[test]
    fn fp8_preserves_element_count() {
        let device = default_device();
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let data = TensorData::new(values, vec![5]);
        let tensor: Tensor<TestBackend, 1> = Tensor::from_data(data, &device);

        let fp8 = burn_1d_to_fp8(tensor.clone(), Fp8Format::E4M3);
        assert_eq!(fp8.len(), 5);
    }

    #[test]
    fn zeros_roundtrip() {
        let device = default_device();
        let values = vec![0.0f32; 4];
        let data = TensorData::new(values.clone(), vec![4]);
        let tensor: Tensor<TestBackend, 1> = Tensor::from_data(data, &device);

        let fp8 = burn_1d_to_fp8(tensor, Fp8Format::E4M3);
        let recovered: Tensor<TestBackend, 1> = fp8_to_burn_1d(&fp8, &device);

        let recovered_data = recovered.into_data();
        let recovered_values: Vec<f32> = recovered_data.to_vec().expect("f32 conversion");

        for (i, rec) in recovered_values.iter().enumerate() {
            assert!(
                rec.abs() < 1e-6,
                "index {i}: expected ~0.0, got {rec}"
            );
        }
    }
}
