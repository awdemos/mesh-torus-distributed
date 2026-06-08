use mt_core::precision::MixedPrecisionConfig;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Schedule strategy for pipeline-parallel execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineScheduleConfig {
    /// GPipe schedule (forward pass for all micro-batches before backward).
    GPipe,
    /// 1F1B (one-forward-one-backward) schedule with the given number of pipeline stages.
    OneF1B {
        /// Number of pipeline stages.
        num_stages: usize,
    },
}

/// Topology configuration for the distributed environment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyConfig {
    /// Manually specified torus dimensions.
    Manual {
        /// Width, height, depth of the torus.
        dims: [usize; 3],
    },
    /// Auto-discover topology from hardware.
    Auto,
}

/// Central training configuration holding all hyper-parameters and
/// distributed training settings.
///
/// Defaults are suitable for small-scale experiments. Tune for production.
///
/// # Example
/// ```ignore
/// let config = TrainingConfig {
///     batch_size: 64,
///     micro_batch_size: 16,
///     accumulation_steps: 4,
///     ..Default::default()
/// };
/// config.validate().expect("invalid config");
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Total logical batch size across all devices.
    pub batch_size: usize,
    /// Batch size per micro-batch in gradient accumulation.
    pub micro_batch_size: usize,
    /// Number of micro-batches over which to accumulate gradients.
    pub accumulation_steps: usize,
    /// Base learning rate.
    pub learning_rate: f64,
    /// Number of training epochs.
    pub epochs: usize,
    /// Number of linear warmup steps for the learning rate.
    pub warmup_steps: usize,
    /// Configuration for FP8 / mixed-precision training.
    pub mixed_precision: MixedPrecisionConfig,
    /// Pipeline schedule configuration.
    pub pipeline_schedule: PipelineScheduleConfig,
    /// Topology configuration (manual or auto-discovered).
    pub topology_config: TopologyConfig,
    /// Interval (in steps) at which to save checkpoints.
    pub checkpoint_interval: usize,
    /// Interval (in steps) at which to log training metrics.
    pub log_interval: usize,
    /// Maximum gradient L2 norm for gradient clipping.
    /// Set to 0.0 to disable clipping.
    pub clip_grad_norm: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            micro_batch_size: 16,
            accumulation_steps: 2,
            learning_rate: 1e-4,
            epochs: 10,
            warmup_steps: 0,
            mixed_precision: MixedPrecisionConfig::default(),
            pipeline_schedule: PipelineScheduleConfig::GPipe,
            topology_config: TopologyConfig::Auto,
            checkpoint_interval: 1000,
            log_interval: 10,
            clip_grad_norm: 0.0,
        }
    }
}

/// Errors that can arise during [`TrainingConfig`] validation.
#[derive(Debug, Clone, Error, PartialEq)]
pub enum ValidationError {
    /// `accumulation_steps` must be at least 1.
    #[error("accumulation_steps must be >= 1, got {0}")]
    AccumulationStepsZero(usize),

    /// `batch_size` must be >= `micro_batch_size`.
    #[error("batch_size ({0}) must be >= micro_batch_size ({1})")]
    BatchSizeTooSmall(usize, usize),

    /// `batch_size` must be divisible by `micro_batch_size`.
    #[error("batch_size ({0}) must be divisible by micro_batch_size ({1})")]
    BatchSizeNotDivisible(usize, usize),

    /// `epochs` must be at least 1.
    #[error("epochs must be >= 1, got {0}")]
    EpochsZero(usize),

    /// `learning_rate` must be positive.
    #[error("learning_rate must be positive, got {0}")]
    LearningRateNonPositive(f64),

    /// `micro_batch_size` must be at least 1.
    #[error("micro_batch_size must be >= 1, got {0}")]
    MicroBatchSizeZero(usize),
}

impl TrainingConfig {
    /// Validates the configuration and returns an error if any constraint
    /// is violated.
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.micro_batch_size == 0 {
            return Err(ValidationError::MicroBatchSizeZero(self.micro_batch_size));
        }
        if self.accumulation_steps == 0 {
            return Err(ValidationError::AccumulationStepsZero(
                self.accumulation_steps,
            ));
        }
        if self.batch_size < self.micro_batch_size {
            return Err(ValidationError::BatchSizeTooSmall(
                self.batch_size,
                self.micro_batch_size,
            ));
        }
        if self.batch_size % self.micro_batch_size != 0 {
            return Err(ValidationError::BatchSizeNotDivisible(
                self.batch_size,
                self.micro_batch_size,
            ));
        }
        if self.epochs == 0 {
            return Err(ValidationError::EpochsZero(self.epochs));
        }
        if self.learning_rate <= 0.0 {
            return Err(ValidationError::LearningRateNonPositive(
                self.learning_rate,
            ));
        }
        Ok(())
    }

    /// Returns the effective batch size after gradient accumulation.
    pub fn effective_batch_size(&self) -> usize {
        self.micro_batch_size * self.accumulation_steps
    }

    /// Returns the number of optimizer updates per epoch.
    pub fn steps_per_epoch(&self) -> usize {
        let samples_per_epoch = self.batch_size; // per-step samples
        let effective = self.effective_batch_size();
        if effective == 0 {
            return 0;
        }
        // This is simplified: in a real setup it would be dataset_size / batch_size.
        // Here we return the number of optimizer steps per epoch assuming
        // each step consumes `batch_size` samples.
        // For the effective batch view:
        samples_per_epoch.div_ceil(effective)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    fn test_default_config_creation() {
        let config = TrainingConfig::default();
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.micro_batch_size, 16);
        assert_eq!(config.accumulation_steps, 2);
        assert!((config.learning_rate - 1e-4).abs() < 1e-10);
        assert_eq!(config.epochs, 10);
        assert_eq!(config.warmup_steps, 0);
        assert_eq!(config.pipeline_schedule, PipelineScheduleConfig::GPipe);
        assert_eq!(config.topology_config, TopologyConfig::Auto);
    }

    #[rstest]
    fn test_default_config_valid() {
        let config = TrainingConfig::default();
        assert!(config.validate().is_ok());
    }

    #[rstest]
    fn test_accumulation_steps_zero_invalid() {
        let config = TrainingConfig {
            accumulation_steps: 0,
            ..Default::default()
        };
        assert_eq!(
            config.validate(),
            Err(ValidationError::AccumulationStepsZero(0))
        );
    }

    #[rstest]
    fn test_micro_batch_size_zero_invalid() {
        let config = TrainingConfig {
            micro_batch_size: 0,
            ..Default::default()
        };
        assert_eq!(
            config.validate(),
            Err(ValidationError::MicroBatchSizeZero(0))
        );
    }

    #[rstest]
    fn test_batch_size_smaller_than_micro_batch_invalid() {
        let config = TrainingConfig {
            batch_size: 8,
            micro_batch_size: 16,
            ..Default::default()
        };
        assert_eq!(
            config.validate(),
            Err(ValidationError::BatchSizeTooSmall(8, 16))
        );
    }

    #[rstest]
    fn test_batch_size_not_divisible_invalid() {
        let config = TrainingConfig {
            batch_size: 30,
            micro_batch_size: 16,
            ..Default::default()
        };
        assert_eq!(
            config.validate(),
            Err(ValidationError::BatchSizeNotDivisible(30, 16))
        );
    }

    #[rstest]
    fn test_learning_rate_non_positive_invalid() {
        let config = TrainingConfig {
            learning_rate: 0.0,
            ..Default::default()
        };
        assert_eq!(
            config.validate(),
            Err(ValidationError::LearningRateNonPositive(0.0))
        );
    }

    #[rstest]
    fn test_epochs_zero_invalid() {
        let config = TrainingConfig {
            epochs: 0,
            ..Default::default()
        };
        assert_eq!(
            config.validate(),
            Err(ValidationError::EpochsZero(0))
        );
    }

    #[rstest]
    fn test_effective_batch_size_computation() {
        let config = TrainingConfig {
            micro_batch_size: 8,
            accumulation_steps: 4,
            ..Default::default()
        };
        assert_eq!(config.effective_batch_size(), 32);
    }

    #[rstest]
    fn test_steps_per_epoch_with_default() {
        let config = TrainingConfig::default();
        // batch_size=32, effective=16*2=32 => steps_per_epoch = 32.div_ceil(32) = 1
        assert_eq!(config.steps_per_epoch(), 1);
    }

    #[rstest]
    fn test_custom_valid_config() {
        let config = TrainingConfig {
            batch_size: 128,
            micro_batch_size: 32,
            accumulation_steps: 4,
            learning_rate: 1e-3,
            epochs: 50,
            warmup_steps: 100,
            pipeline_schedule: PipelineScheduleConfig::OneF1B { num_stages: 4 },
            topology_config: TopologyConfig::Manual {
                dims: [2, 2, 1],
            },
            ..Default::default()
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.effective_batch_size(), 128);
    }

    #[rstest]
    fn test_serialization_roundtrip() {
        let config = TrainingConfig::default();
        let toml_str = toml::to_string(&config).expect("serialization failed");
        let deserialized: TrainingConfig = toml::from_str(&toml_str).expect("deserialization failed");
        assert_eq!(config, deserialized);
    }

    #[rstest]
    fn test_serialization_custom() {
        let config = TrainingConfig {
            batch_size: 256,
            micro_batch_size: 64,
            accumulation_steps: 4,
            learning_rate: 5e-4,
            epochs: 100,
            ..Default::default()
        };
        let toml_str = toml::to_string(&config).expect("serialization failed");
        let deserialized: TrainingConfig = toml::from_str(&toml_str).expect("deserialization failed");
        assert_eq!(config, deserialized);
        assert_eq!(deserialized.batch_size, 256);
        assert_eq!(deserialized.epochs, 100);
    }

    #[rstest]
    fn test_validation_passes_with_manual_topology() {
        let config = TrainingConfig {
            topology_config: TopologyConfig::Manual {
                dims: [4, 4, 1],
            },
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[rstest]
    fn test_pipeline_schedule_config_equality() {
        assert_eq!(
            PipelineScheduleConfig::GPipe,
            PipelineScheduleConfig::GPipe
        );
        assert_ne!(
            PipelineScheduleConfig::GPipe,
            PipelineScheduleConfig::OneF1B { num_stages: 4 }
        );
        assert_eq!(
            PipelineScheduleConfig::OneF1B { num_stages: 4 },
            PipelineScheduleConfig::OneF1B { num_stages: 4 }
        );
        assert_ne!(
            PipelineScheduleConfig::OneF1B { num_stages: 4 },
            PipelineScheduleConfig::OneF1B { num_stages: 8 }
        );
    }

    #[rstest]
    fn test_topology_config_equality() {
        assert_eq!(TopologyConfig::Auto, TopologyConfig::Auto);
        assert_eq!(
            TopologyConfig::Manual { dims: [2, 2, 1] },
            TopologyConfig::Manual { dims: [2, 2, 1] }
        );
        assert_ne!(
            TopologyConfig::Auto,
            TopologyConfig::Manual { dims: [2, 2, 1] }
        );
    }
}
