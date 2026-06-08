use std::time::Instant;

use burn::module::list_param_ids;
use burn::module::Module;
use burn::module::ParamId;
use burn_optim::GradientsParams;
use burn::tensor::backend::Backend;
use mt_core::precision::LossScaler;
use thiserror::Error;
use tracing::{debug, info};

use crate::accumulator::GradientAccumulator;
use crate::config::TrainingConfig;

/// Metrics collected during a single training step.
#[derive(Debug, Clone)]
pub struct StepMetrics {
    pub loss: f32,
    pub learning_rate: f64,
    pub step: usize,
    pub overflow: bool,
    pub grad_norm: f32,
}

/// Errors that can occur during a training step.
#[derive(Debug, Error)]
pub enum StepError {
    #[error("gradient overflow detected, skipping step")]
    Overflow,
    #[error("gradient shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("step error: {0}")]
    Internal(String),
}

/// Aggregate training metrics after a complete training loop.
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub avg_loss: f32,
    pub total_steps: usize,
    pub total_time_secs: f64,
    pub overflow_count: usize,
}

/// Errors that can occur during the training loop.
#[derive(Debug, Error)]
pub enum TrainingError {
    #[error("step error at step {step}: {source}")]
    StepError { step: usize, #[source] source: StepError },
    #[error("dataloader is empty")]
    EmptyDataloader,
}

fn compute_grad_norm(values: &[f32]) -> f32 {
    let sum_sq: f32 = values.iter().map(|v| v * v).sum();
    sum_sq.sqrt()
}

/// Performs gradient clipping in-place on a slice of values.
/// Returns the norm before clipping.
pub fn clip_grad_norm(grads: &mut [f32], max_norm: f32) -> f32 {
    let norm = compute_grad_norm(grads);
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        for g in grads.iter_mut() {
            *g *= scale;
        }
    }
    norm
}

/// Checks all accumulated values for NaN/Inf overflow.
pub fn check_overflow_values(raw: &std::collections::HashMap<ParamId, Vec<f32>>) -> (f32, bool) {
    let mut all_values = Vec::new();
    for values in raw.values() {
        all_values.extend(values);
    }
    let norm = compute_grad_norm(&all_values);
    let overflow = norm.is_nan() || norm.is_infinite();
    (norm, overflow)
}

/// Applies gradient clipping to all values stored in the accumulator.
pub fn apply_clip_norm(acc: &mut GradientAccumulator<impl Backend>, max_norm: f32) {
    let mut total_norm_sq: f32 = 0.0;
    for values in acc.raw_grads().values() {
        total_norm_sq += values.iter().map(|v| v * v).sum::<f32>();
    }
    let total_norm = total_norm_sq.sqrt();
    if total_norm > max_norm && total_norm > 0.0 {
        let scale = max_norm / total_norm;
        for values in acc.raw_grads_mut().values_mut() {
            for v in values.iter_mut() {
                *v *= scale;
            }
        }
    }
}

/// Executes one training step: drains the provided [`GradientsParams`] into
/// the accumulator, handles overflow detection, and when the accumulation
/// target is met, applies gradient clipping.
///
/// The caller is responsible for:
/// - Forward and backward passes
/// - Calling [`Optimizer::step`] using [`GradientAccumulator::into_gradients`]
///   when the returned [`StepMetrics`] indicates the step count advanced.
///
/// # Returns
///
/// Returns [`StepError::Overflow`] if NaN/Inf gradients are detected.
/// Otherwise returns [`StepMetrics`] — the caller should check
/// `metrics.overs` to detect overflow and skip the optimizer step.
pub fn train_step<B, M>(
    mut grads: GradientsParams,
    loss_value: f32,
    config: &TrainingConfig,
    accumulator: &mut GradientAccumulator<B>,
    loss_scaler: &mut LossScaler,
    global_step: &mut usize,
    module: &M,
) -> Result<StepMetrics, StepError>
where
    B: Backend,
    M: Module<B>,
{
    let param_ids = list_param_ids(module);
    let raw_norm = accumulator.drain_from(&mut grads, &param_ids);
    let overflow = raw_norm.is_nan() || raw_norm.is_infinite();

    if overflow {
        loss_scaler.update(true);
        debug!(
            "overflow detected, loss scaler halved to {}",
            loss_scaler.scale()
        );
        accumulator.reset();
        return Err(StepError::Overflow);
    }

    if accumulator.step_count() >= config.accumulation_steps {
        if config.clip_grad_norm > 0.0 {
            apply_clip_norm(accumulator, config.clip_grad_norm);
        }

        let lr = if *global_step < config.warmup_steps && config.warmup_steps > 0 {
            config.learning_rate * (*global_step as f64) / (config.warmup_steps as f64)
        } else {
            config.learning_rate
        };

        loss_scaler.update(false);
        *global_step += 1;

        let metrics = StepMetrics {
            loss: loss_value,
            learning_rate: lr,
            step: *global_step,
            overflow,
            grad_norm: raw_norm,
        };

        if *global_step % config.log_interval == 0 {
            info!(
                "step {}: loss={:.6}, lr={:.2e}, grad_norm={:.4}",
                metrics.step, metrics.loss, metrics.learning_rate, metrics.grad_norm
            );
        }

        Ok(metrics)
    } else {
        Ok(StepMetrics {
            loss: loss_value,
            learning_rate: config.learning_rate,
            step: *global_step,
            overflow,
            grad_norm: raw_norm,
        })
    }
}

/// Runs a training loop over the dataloader.
///
/// The `dataloader` yields `(input, target)` pairs.
/// This is a scaffold — in production the forward and backward passes
/// happen inside the loop using an autodiff-capable backend.
pub fn training_loop<B, M, D>(
    _model: &mut M,
    dataloader: D,
    _config: &TrainingConfig,
) -> Result<TrainingMetrics, TrainingError>
where
    B: Backend,
    M: Module<B>,
    D: IntoIterator<Item = (burn::tensor::Tensor<B, 2>, burn::tensor::Tensor<B, 2>)>,
{
    let start_time = Instant::now();
    let mut accumulator = GradientAccumulator::<B>::new(B::Device::default());
    let _loss_scaler = LossScaler::new_static(1.0);
    let mut global_step: usize = 0;
    let overflow_count: usize = 0;

    let mut data_iter = dataloader.into_iter();
    let _first = data_iter.next().ok_or(TrainingError::EmptyDataloader)?;

    for (_input, _target) in std::iter::once(_first).chain(data_iter) {
        // In production with B: AutodiffBackend:
        //   1. let output = model.forward(input);
        //   2. let loss = loss_fn(output, target);
        //   3. let loss_val: f32 = loss.clone().into_scalar();
        //   4. loss.backward();
        //   5. let grads = GradientsParams::from_grads(loss.backward(), &model);
        //   6. let result = train_step(grads, loss_val, config, &mut accumulator,
        //                               &mut loss_scaler, &mut global_step, model);
        //   7. match result {
        //          Ok(metrics) if metrics.step > old_step => {
        //              let accumulated = accumulator.into_gradients();
        //              // model = optimizer.step(metrics.learning_rate, model, accumulated);
        //          }
        //          ...
        //      }

        // Placeholder: count as one accumulation step
        let _ = _input;
        let _ = _target;
        accumulator.reset();
        global_step += 1;
    }

    let elapsed = start_time.elapsed().as_secs_f64();

    info!(
        "training loop completed: steps={}, time={:.2}s, overflows={}",
        global_step, elapsed, overflow_count
    );

    Ok(TrainingMetrics {
        avg_loss: 0.0,
        total_steps: global_step,
        total_time_secs: elapsed,
        overflow_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::nn::{Linear, LinearConfig};
    use burn::tensor::{Tensor, TensorData};
    use burn_ndarray::{NdArray, NdArrayDevice};
    use rstest::rstest;

    type TestBackend = NdArray;

    fn default_device() -> NdArrayDevice {
        Default::default()
    }

    fn dummy_module(device: &NdArrayDevice) -> Linear<TestBackend> {
        LinearConfig::new(4, 2).init(device)
    }

    fn dummy_grads(device: &NdArrayDevice, module: &Linear<TestBackend>) -> GradientsParams {
        // Since we can't iterate through GradientsParams, we construct
        // one manually using register() with known param IDs.
        let mut gp = GradientsParams::new();
        let ids = list_param_ids(module);
        for id in ids {
            let t = Tensor::<TestBackend, 1>::from_data(
                TensorData::new(vec![0.1f32, 0.2, 0.3], vec![3]),
                device,
            );
            gp.register(id, t);
        }
        gp
    }

    fn nan_grads(device: &NdArrayDevice, module: &Linear<TestBackend>) -> GradientsParams {
        let mut gp = GradientsParams::new();
        let ids = list_param_ids(module);
        for id in ids {
            let t = Tensor::<TestBackend, 1>::from_data(
                TensorData::new(vec![f32::NAN], vec![1]),
                device,
            );
            gp.register(id, t);
        }
        gp
    }

    #[rstest]
    fn test_compute_grad_norm_basic() {
        let grads = vec![3.0, 4.0];
        let norm = compute_grad_norm(&grads);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[rstest]
    fn test_compute_grad_norm_zero() {
        let norm = compute_grad_norm(&[0.0, 0.0, 0.0]);
        assert!((norm).abs() < 1e-6);
    }

    #[rstest]
    fn test_compute_grad_norm_negative() {
        let norm = compute_grad_norm(&[-3.0, -4.0]);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[rstest]
    fn test_compute_grad_norm_empty() {
        let norm = compute_grad_norm(&[]);
        assert!((norm).abs() < 1e-6);
    }

    #[rstest]
    fn test_clip_below_threshold() {
        let mut g = vec![1.0, 2.0, 3.0];
        let n = clip_grad_norm(&mut g, 10.0);
        assert!((n - 3.741657).abs() < 1e-4);
        assert!((g[0] - 1.0).abs() < 1e-6);
    }

    #[rstest]
    fn test_clip_above_threshold() {
        let mut g = vec![10.0, 0.0, 0.0];
        let n = clip_grad_norm(&mut g, 5.0);
        assert!((n - 10.0).abs() < 1e-6);
        assert!((g[0] - 5.0).abs() < 1e-6);
    }

    #[rstest]
    fn test_clip_zero() {
        let mut g = vec![0.0, 0.0];
        let n = clip_grad_norm(&mut g, 1.0);
        assert!((n).abs() < 1e-6);
    }

    #[rstest]
    fn test_clip_exact_threshold() {
        let mut g = vec![6.0, 8.0];
        let n = clip_grad_norm(&mut g, 10.0);
        assert!((n - 10.0).abs() < 1e-6);
        assert!((g[0] - 6.0).abs() < 1e-6);
    }

    #[rstest]
    fn test_clip_mixed_sign() {
        let mut g = vec![3.0, -4.0, 0.0];
        let n = clip_grad_norm(&mut g, 2.5);
        assert!((n - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
        assert!((g[1] + 2.0).abs() < 1e-6);
    }

    #[rstest]
    fn test_check_overflow_clean() {
        let mut map = std::collections::HashMap::new();
        map.insert(ParamId::from(1u64), vec![0.1f32, 0.2, 0.3]);
        map.insert(ParamId::from(2u64), vec![0.4f32]);
        let (norm, overflow) = check_overflow_values(&map);
        assert!(!overflow);
        assert!(norm > 0.0);
    }

    #[rstest]
    fn test_check_overflow_nan() {
        let mut map = std::collections::HashMap::new();
        map.insert(ParamId::from(1u64), vec![f32::NAN]);
        let (_, overflow) = check_overflow_values(&map);
        assert!(overflow);
    }

    #[rstest]
    fn test_step_metrics_struct() {
        let m = StepMetrics {
            loss: 1.0,
            learning_rate: 1e-4,
            step: 0,
            overflow: false,
            grad_norm: 0.0,
        };
        assert!((m.loss - 1.0).abs() < 1e-6);
        assert!((m.learning_rate - 1e-4).abs() < 1e-10);
        assert_eq!(m.step, 0);
    }

    #[rstest]
    fn test_train_step_overflow_error() {
        let device = default_device();
        let module = dummy_module(&device);
        let config = TrainingConfig::default();
        let mut acc = GradientAccumulator::<TestBackend>::new(device.clone());
        let mut scaler = LossScaler::new_dynamic(65536.0, 2000);
        let mut gs = 0usize;
        let grads = nan_grads(&device, &module);

        let result = train_step::<TestBackend, Linear<TestBackend>>(
            grads, 1.0, &config, &mut acc, &mut scaler, &mut gs, &module,
        );
        assert!(result.is_err());
        assert!((scaler.scale() - 32768.0).abs() < 1e-6);
    }

    #[rstest]
    fn test_train_step_no_opt_step_early() {
        let device = default_device();
        let module = dummy_module(&device);
        let config = TrainingConfig {
            accumulation_steps: 4,
            ..Default::default()
        };
        let mut acc = GradientAccumulator::<TestBackend>::new(device.clone());
        let mut scaler = LossScaler::new_static(1.0);
        let mut gs = 0usize;
        let grads = dummy_grads(&device, &module);

        let result = train_step::<TestBackend, Linear<TestBackend>>(
            grads, 1.5, &config, &mut acc, &mut scaler, &mut gs, &module,
        );
        assert!(result.is_ok());
        let metrics = result.unwrap();
        assert!((metrics.loss - 1.5).abs() < 1e-6);
        assert_eq!(gs, 0);
        assert_eq!(acc.step_count(), 1);
    }

    #[rstest]
    fn test_training_metrics_struct() {
        let m = TrainingMetrics {
            avg_loss: 2.5,
            total_steps: 100,
            total_time_secs: 42.0,
            overflow_count: 0,
        };
        assert!((m.avg_loss - 2.5).abs() < 1e-6);
        assert_eq!(m.total_steps, 100);
    }

    #[rstest]
    fn test_step_error_display() {
        assert_eq!(
            StepError::Overflow.to_string(),
            "gradient overflow detected, skipping step"
        );
        assert_eq!(
            StepError::ShapeMismatch("bad".into()).to_string(),
            "gradient shape mismatch: bad"
        );
        assert_eq!(
            StepError::Internal("err".into()).to_string(),
            "step error: err"
        );
    }

    #[rstest]
    fn test_training_error_display() {
        assert_eq!(
            TrainingError::EmptyDataloader.to_string(),
            "dataloader is empty"
        );
        let e = TrainingError::StepError {
            step: 5,
            source: StepError::Overflow,
        };
        assert!(e.to_string().contains("step 5"));
    }

    #[rstest]
    fn test_apply_clip_norm_fn() {
        let device = default_device();
        let mut acc = GradientAccumulator::<TestBackend>::new(device.clone());

        acc.register_values(ParamId::from(1u64), vec![10.0f32, 0.0]);
        apply_clip_norm(&mut acc, 5.0);

        let raw = acc.raw_grads();
        let vals = raw.get(&ParamId::from(1u64)).unwrap();
        assert!((vals[0] - 5.0).abs() < 1e-6);
    }

    #[rstest]
    fn test_lr_warmup() {
        let cfg = TrainingConfig {
            learning_rate: 1e-3,
            warmup_steps: 100,
            ..Default::default()
        };

        // Simulate warmup logic
        let compute_lr = |step: usize| -> f64 {
            if step < cfg.warmup_steps && cfg.warmup_steps > 0 {
                cfg.learning_rate * step as f64 / cfg.warmup_steps as f64
            } else {
                cfg.learning_rate
            }
        };

        assert!((compute_lr(0)).abs() < 1e-10);
        assert!((compute_lr(50) - 5e-4).abs() < 1e-10);
        assert!((compute_lr(100) - 1e-3).abs() < 1e-10);
    }

    #[rstest]
    fn test_no_warmup_constant_lr() {
        let cfg = TrainingConfig {
            learning_rate: 1e-4,
            warmup_steps: 0,
            ..Default::default()
        };
        for s in [0usize, 50, 100] {
            let lr = if s < cfg.warmup_steps && cfg.warmup_steps > 0 {
                cfg.learning_rate * s as f64 / cfg.warmup_steps as f64
            } else {
                cfg.learning_rate
            };
            assert!((lr - 1e-4).abs() < 1e-10, "step {}: {}", s, lr);
        }
    }

    /// Test that drain_from inside train_step correctly accumulates
    /// and that the caller can extract gradients via into_gradients.
    #[rstest]
    fn test_train_step_accumulates_correctly() {
        let device = default_device();
        let module = dummy_module(&device);
        let config = TrainingConfig {
            accumulation_steps: 3,
            ..Default::default()
        };
        let mut acc = GradientAccumulator::<TestBackend>::new(device.clone());
        let mut scaler = LossScaler::new_static(1.0);
        let mut gs = 0usize;

        // Three micro-batches
        for _ in 0..3 {
            let grads = dummy_grads(&device, &module);
            let result = train_step::<TestBackend, Linear<TestBackend>>(
                grads, 1.0, &config, &mut acc, &mut scaler, &mut gs, &module,
            );
            assert!(result.is_ok());
        }

        // After 3 accumulations, step should have advanced
        assert_eq!(gs, 1);

        // The caller extracts gradients and resets the accumulator
        // (into_gradients drains the stored values but preserves step_count)
        let _accumulated = acc.into_gradients();
        assert!(acc.raw_grads().is_empty());
        // step_count reflects the total number of accumulations so far
        assert_eq!(acc.step_count(), 3);

        // The caller resets the accumulator for the next cycle
        acc.reset();
        assert_eq!(acc.step_count(), 0);
    }
}
