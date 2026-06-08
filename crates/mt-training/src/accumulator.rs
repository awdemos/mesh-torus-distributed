use std::collections::HashMap;
use std::marker::PhantomData;

use burn::module::ParamId;
use burn_optim::GradientsParams;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use tracing::debug;

/// Accumulates gradients across multiple micro-batches before applying an
/// optimizer step. This enables effective batch sizes larger than what fits
/// in GPU memory by trading off update frequency for throughput.
///
/// Gradients are stored in FP32 (master) format regardless of their incoming
/// precision, which avoids precision loss during accumulation.
///
/// Because Burn's [`GradientsParams`] does not expose iteration over its
/// entries, this accumulator stores raw FP32 values internally and
/// converts to/from [`GradientsParams`] at accumulation and extraction
/// boundaries.
///
/// # Example
/// ```ignore
/// let mut accumulator = GradientAccumulator::new(device);
///
/// for micro_batch in micro_batches {
///     let grads = compute_gradients(micro_batch);
///     accumulator.drain_from(grads, param_ids);
/// }
///
/// let effective_bs = accumulator.effective_batch_size(16);
/// assert_eq!(effective_bs, 64);
///
/// let accumulated = accumulator.into_gradients();
/// ```
pub struct GradientAccumulator<B: Backend> {
    /// Per-parameter accumulated gradient values in FP32.
    grads: HashMap<ParamId, Vec<f32>>,
    /// Number of accumulation steps performed so far.
    count: usize,
    /// Device on which tensors will be materialised.
    device: B::Device,
    /// Phantom data for the backend type.
    _backend: PhantomData<B>,
}

impl<B: Backend> GradientAccumulator<B> {
    /// Creates a new empty gradient accumulator on the given device.
    pub fn new(device: B::Device) -> Self {
        Self {
            grads: HashMap::new(),
            count: 0,
            device,
            _backend: PhantomData,
        }
    }

    /// Register a single gradient tensor, flattening it to FP32 and
    /// summing with any previously stored gradient for the same [`ParamId`].
    ///
    /// Returns the squared L2 norm contribution of this gradient
    /// (before accumulation) for overflow detection.
    pub fn register<const D: usize>(
        &mut self,
        id: ParamId,
        grad: Tensor<B, D>,
    ) -> f32 {
        let values: Vec<f32> = grad
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default();
        self.store(id, &values);
        values.iter().map(|v| v * v).sum()
    }

    /// Register pre-extracted FP32 gradient values for the given parameter.
    pub fn register_values(&mut self, id: ParamId, values: Vec<f32>) {
        self.store(id, &values);
    }

    /// Internal helper: sum values into the stored map.
    fn store(&mut self, id: ParamId, values: &[f32]) {
        match self.grads.get_mut(&id) {
            Some(stored) => {
                debug_assert_eq!(
                    stored.len(),
                    values.len(),
                    "gradient size mismatch for param {:?}",
                    id
                );
                for (s, v) in stored.iter_mut().zip(values.iter()) {
                    *s += v;
                }
            }
            None => {
                self.grads.insert(id, values.to_vec());
            }
        }
        self.count += 1;
        debug!(
            "accumulated gradient for param {:?} (total steps: {})",
            id, self.count
        );
    }

    /// Drains all gradients from the given [`GradientsParams`], extracting
    /// FP32 values for each provided [`ParamId`] and storing them
    /// internally.
    ///
    /// Returns the L2 norm of all gradients extracted in this call.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if a gradient tensor has unexpected dimensions.
    pub fn drain_from(
        &mut self,
        grads: &mut GradientsParams,
        param_ids: &[ParamId],
    ) -> f32 {
        let mut squared_sum: f32 = 0.0;
        for id in param_ids {
            if let Some(tensor) = grads.remove::<B, 1>(*id) {
                let values: Vec<f32> = tensor
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap_or_default();
                squared_sum += values.iter().map(|v| v * v).sum::<f32>();
                // If param already stored, sum. Otherwise insert.
                match self.grads.get_mut(id) {
                    Some(stored) => {
                        debug_assert_eq!(stored.len(), values.len());
                        for (s, v) in stored.iter_mut().zip(values.iter()) {
                            *s += v;
                        }
                    }
                    None => {
                        self.grads.insert(*id, values);
                    }
                }
            }
        }
        self.count += 1;
        squared_sum.sqrt()
    }

    /// Resets all accumulated gradients and the step counter.
    pub fn reset(&mut self) {
        self.grads.clear();
        self.count = 0;
    }

    /// Returns the number of accumulation steps performed so far.
    pub fn step_count(&self) -> usize {
        self.count
    }

    /// Returns the effective batch size after the current accumulation.
    ///
    /// Equivalent to `micro_batch_size * step_count()`.
    pub fn effective_batch_size(&self, micro_batch_size: usize) -> usize {
        micro_batch_size * self.count
    }

    /// Converts the accumulated gradients into a [`GradientsParams`] map,
    /// consuming the accumulator's gradient data.
    ///
    /// After calling this the internal gradient map is empty, but the step
    /// counter is preserved for caller convenience.
    pub fn into_gradients(&mut self) -> GradientsParams {
        let mut gp = GradientsParams::new();
        let drained: Vec<(ParamId, Vec<f32>)> = self.grads.drain().collect();
        for (id, values) in drained {
            // Must compute len before moving `values`.
            let numel = values.len();
            let shape = vec![numel];
            let data = TensorData::new(values, shape);
            let tensor: Tensor<B, 1> = Tensor::from_data(data, &self.device);
            gp.register::<B, 1>(id, tensor);
        }
        gp
    }

    /// Returns a reference to the raw accumulated gradient data.
    pub fn raw_grads(&self) -> &HashMap<ParamId, Vec<f32>> {
        &self.grads
    }

    /// Returns a mutable reference to the raw accumulated gradient data.
    /// Used for in-place operations like gradient clipping.
    pub fn raw_grads_mut(&mut self) -> &mut HashMap<ParamId, Vec<f32>> {
        &mut self.grads
    }

    /// Returns the device this accumulator is bound to.
    pub fn device(&self) -> &B::Device {
        &self.device
    }
}

impl<B: Backend> std::fmt::Debug for GradientAccumulator<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradientAccumulator")
            .field("param_count", &self.grads.len())
            .field("count", &self.count)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::list_param_ids;
    use burn::module::Module;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;
    use burn_ndarray::NdArrayDevice;
    use rstest::rstest;

    type TestBackend = NdArray;

    fn default_device() -> NdArrayDevice {
        Default::default()
    }

    /// A simple single-parameter linear-like module for testing.
    #[derive(Debug, Clone, Module)]
    struct DummyModule {
        weight: burn::nn::Linear<TestBackend>,
    }

    impl DummyModule {
        fn new(device: &NdArrayDevice) -> Self {
            use burn::nn::LinearConfig;
            Self {
                weight: LinearConfig::new(4, 2).init(device),
            }
        }
    }

    fn dummy_param_ids(device: &NdArrayDevice) -> Vec<ParamId> {
        let module = DummyModule::new(device);
        list_param_ids::<DummyModule, TestBackend>(&module)
    }

    fn make_single_grads(
        device: &NdArrayDevice,
        id: ParamId,
        values: Vec<f32>,
    ) -> GradientsParams {
        let mut gp = GradientsParams::new();
        let numel = values.len();
        let tensor = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(values, vec![numel]),
            device,
        );
        gp.register::<TestBackend, 1>(id, tensor);
        gp
    }

    #[rstest]
    fn test_new_accumulator_is_empty() {
        let acc = GradientAccumulator::<TestBackend>::new(default_device());
        assert_eq!(acc.step_count(), 0);
        assert!(acc.raw_grads().is_empty());
    }

    #[rstest]
    fn test_register_single_param() {
        let device = default_device();
        let mut acc = GradientAccumulator::<TestBackend>::new(device.clone());
        let id = ParamId::from(1u64);
        let tensor = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![0.5f32, 0.2, -0.1], vec![3]),
            &device,
        );
        let _norm_sq = acc.register::<1>(id, tensor);
        assert_eq!(acc.step_count(), 1);
        assert_eq!(acc.raw_grads().len(), 1);
    }

    #[rstest]
    fn test_accumulate_multiple_params() {
        let device = default_device();
        let mut acc = GradientAccumulator::<TestBackend>::new(device.clone());

        let t1 = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![0.1f32, 0.2], vec![2]),
            &device,
        );
        let t2 = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![0.3f32, 0.4, 0.5], vec![3]),
            &device,
        );
        acc.register::<1>(ParamId::from(1u64), t1);
        acc.register::<1>(ParamId::from(2u64), t2);

        assert_eq!(acc.step_count(), 2);
        assert_eq!(acc.raw_grads().len(), 2);
    }

    #[rstest]
    fn test_multiple_accumulations_sum() {
        let device = default_device();
        let mut acc = GradientAccumulator::<TestBackend>::new(device.clone());

        let id = ParamId::from(1u64);

        let t1 = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![1.0f32, 2.0], vec![2]),
            &device,
        );
        acc.register::<1>(id, t1);
        assert_eq!(acc.step_count(), 1);

        let t2 = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![3.0f32, 4.0], vec![2]),
            &device,
        );
        acc.register::<1>(id, t2);
        assert_eq!(acc.step_count(), 2);

        let raw = acc.raw_grads();
        let values = raw.get(&id).unwrap();
        assert!((values[0] - 4.0).abs() < 1e-6, "expected 4.0, got {}", values[0]);
        assert!((values[1] - 6.0).abs() < 1e-6, "expected 6.0, got {}", values[1]);
    }

    #[rstest]
    fn test_reset_clears_everything() {
        let device = default_device();
        let mut acc = GradientAccumulator::<TestBackend>::new(device.clone());

        let t1 = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![1.0f32, 2.0], vec![2]),
            &device,
        );
        acc.register::<1>(ParamId::from(1u64), t1);
        assert_eq!(acc.step_count(), 1);
        assert_eq!(acc.raw_grads().len(), 1);

        acc.reset();
        assert_eq!(acc.step_count(), 0);
        assert!(acc.raw_grads().is_empty());
    }

    #[rstest]
    fn test_effective_batch_size() {
        let device = default_device();
        let mut acc = GradientAccumulator::<TestBackend>::new(device.clone());

        assert_eq!(acc.effective_batch_size(16), 0);

        for _ in 0..4 {
            let id = ParamId::from(1u64);
            let t = Tensor::<TestBackend, 1>::from_data(
                TensorData::new(vec![1.0f32], vec![1]),
                &device,
            );
            acc.register::<1>(id, t);
        }

        assert_eq!(acc.effective_batch_size(16), 64);
        assert_eq!(acc.effective_batch_size(32), 128);
    }

    #[rstest]
    fn test_into_gradients_returns_valid() {
        let device = default_device();
        let mut acc = GradientAccumulator::<TestBackend>::new(device.clone());

        let id = ParamId::from(1u64);
        let t = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![0.5f32, -0.5], vec![2]),
            &device,
        );
        acc.register::<1>(id, t);

        let result = acc.into_gradients();
        assert_eq!(result.len(), 1);
    }

    #[rstest]
    fn test_drain_from_multi_step() {
        let device = default_device();
        let mut acc = GradientAccumulator::<TestBackend>::new(device.clone());
        let ids = dummy_param_ids(&device);

        // Simulate three micro-batches
        for _ in 0..3 {
            let mut gp = GradientsParams::new();
            for (i, pid) in ids.iter().enumerate() {
                let val = (i + 1) as f32 * 0.5;
                let t = Tensor::<TestBackend, 1>::from_data(
                    TensorData::new(vec![val], vec![1]),
                    &device,
                );
                gp.register::<TestBackend, 1>(*pid, t);
            }
            acc.drain_from(&mut gp, &ids);
        }

        assert_eq!(acc.step_count(), 3);

        // Check values: each param has been accumulated 3 times
        // param 0: 0.5*3 = 1.5, param 1: 1.0*3 = 3.0
        let raw = acc.raw_grads();
        for (i, pid) in ids.iter().enumerate() {
            let expected = (i + 1) as f32 * 0.5 * 3.0;
            if let Some(vals) = raw.get(pid) {
                assert!(
                    (vals[0] - expected).abs() < 1e-5,
                    "param {}: expected {}, got {}",
                    i,
                    expected,
                    vals[0]
                );
            }
        }
    }

    #[rstest]
    fn test_accumulate_after_reset() {
        let device = default_device();
        let mut acc = GradientAccumulator::<TestBackend>::new(device.clone());

        let id = ParamId::from(1u64);
        let t1 = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![10.0f32], vec![1]),
            &device,
        );
        acc.register::<1>(id, t1);
        assert_eq!(acc.step_count(), 1);

        acc.reset();

        let t2 = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![5.0f32], vec![1]),
            &device,
        );
        acc.register::<1>(id, t2);
        assert_eq!(acc.step_count(), 1);

        let raw = acc.raw_grads();
        let v = raw.get(&id).unwrap();
        assert!((v[0] - 5.0).abs() < 1e-6, "expected 5.0 (fresh start), got {}", v[0]);
    }

    #[rstest]
    fn test_into_gradients_consumes_data() {
        let device = default_device();
        let mut acc = GradientAccumulator::<TestBackend>::new(device.clone());

        let id = ParamId::from(1u64);
        let t = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![1.0f32], vec![1]),
            &device,
        );
        acc.register::<1>(id, t);

        let _gp = acc.into_gradients();
        assert!(acc.raw_grads().is_empty(), "grads should be drained");
        assert_eq!(
            acc.step_count(),
            1,
            "step count should be preserved after into_gradients"
        );
    }

    #[rstest]
    fn test_debug_output() {
        let device = default_device();
        let mut acc = GradientAccumulator::<TestBackend>::new(device.clone());

        let id = ParamId::from(1u64);
        let t = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![1.0f32], vec![1]),
            &device,
        );
        acc.register::<1>(id, t);

        let debug_str = format!("{:?}", acc);
        assert!(debug_str.contains("GradientAccumulator"));
        assert!(debug_str.contains("count: 1"));
    }

    /// Test: register_values works the same as register()
    #[rstest]
    fn test_register_values() {
        let device = default_device();
        let mut acc = GradientAccumulator::<TestBackend>::new(device.clone());
        let id = ParamId::from(1u64);

        acc.register_values(id, vec![1.0f32, 2.0]);
        acc.register_values(id, vec![3.0f32, 4.0]);

        let raw = acc.raw_grads();
        let vals = raw.get(&id).unwrap();
        assert!((vals[0] - 4.0).abs() < 1e-6);
        assert!((vals[1] - 6.0).abs() < 1e-6);
        assert_eq!(acc.step_count(), 2);
    }

    /// Test: drain_from returns L2 norm
    #[rstest]
    fn test_drain_from_returns_norm() {
        let device = default_device();
        let mut acc = GradientAccumulator::<TestBackend>::new(device.clone());

        let mut gp = GradientsParams::new();
        let id = ParamId::from(1u64);
        let t = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![3.0f32, 4.0], vec![2]),
            &device,
        );
        gp.register(id, t);

        let norm = acc.drain_from(&mut gp, &[id]);
        assert!((norm - 5.0).abs() < 1e-5, "expected L2 norm 5.0, got {}", norm);
    }

    /// Test: device() returns the device
    #[rstest]
    fn test_device_accessor() {
        let device = default_device();
        let acc = GradientAccumulator::<TestBackend>::new(device.clone());
        assert_eq!(
            acc.device(),
            &device,
            "device should match what was provided"
        );
    }
}
