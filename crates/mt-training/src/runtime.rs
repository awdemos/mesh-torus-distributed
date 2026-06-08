use std::sync::Arc;

use mt_comm::Communicator;
use mt_core::precision::{LossScaler, MixedPrecisionConfig};
use mt_core::topology::MeshTorusHybrid;
use parking_lot::RwLock;
use thiserror::Error;
use tracing::info;

use crate::config::TrainingConfig;

/// Errors that can occur during distributed runtime operations.
#[derive(Debug, Error)]
pub enum RuntimeError {
    /// The topology configuration is invalid or incomplete.
    #[error("topology error: {0}")]
    Topology(String),

    /// Communication layer failed to initialise.
    #[error("communication error: {0}")]
    Communication(String),

    /// Pipeline initialisation failed.
    #[error("pipeline error: {0}")]
    Pipeline(String),

    /// The configuration is invalid.
    #[error("configuration error: {0}")]
    Config(String),
}

/// Creates a [`LossScaler`] from a [`MixedPrecisionConfig`].
///
/// This is a free function rather than a `From` impl because both types
/// are defined in the external `mt_core` crate, which would violate the
/// orphan rule.
fn loss_scaler_from_config(config: &MixedPrecisionConfig) -> LossScaler {
    match config.scaling_strategy {
        mt_core::precision::ScalingStrategy::Dynamic => {
            LossScaler::new_dynamic(65536.0, 2000)
        }
        mt_core::precision::ScalingStrategy::Delayed { .. } => {
            LossScaler::new_static(1.0)
        }
    }
}

/// The central orchestrator for distributed training.
///
/// Holds the topology, communicator, precision configuration, and training
/// configuration.  Created via the builder pattern.
///
/// # Example
/// ```ignore
/// let runtime = DistributedRuntime::builder()
///     .with_topology(topo)
///     .with_communicator(comm)
///     .with_config(config)
///     .build()
///     .expect("runtime creation failed");
/// ```
pub struct DistributedRuntime {
    /// The mesh-torus hybrid topology.
    pub topology: MeshTorusHybrid,
    /// The communicator for collective operations.
    pub communicator: Arc<dyn Communicator>,
    /// Mixed-precision configuration.
    pub precision_config: MixedPrecisionConfig,
    /// Training configuration.
    pub training_config: TrainingConfig,
    /// Loss scaler for FP8 / mixed-precision training.
    pub loss_scaler: Arc<RwLock<LossScaler>>,
}

impl DistributedRuntime {
    /// Creates a new [`DistributedRuntimeBuilder`].
    pub fn builder() -> DistributedRuntimeBuilder {
        DistributedRuntimeBuilder::new()
    }

    /// Returns the rank of this process in the distributed world.
    pub fn rank(&self) -> usize {
        self.communicator.rank()
    }

    /// Returns the total number of processes in the distributed world.
    pub fn world_size(&self) -> usize {
        self.communicator.world_size()
    }

    /// Returns a shared reference to the loss scaler.
    pub fn loss_scaler(&self) -> &Arc<RwLock<LossScaler>> {
        &self.loss_scaler
    }
}

impl std::fmt::Debug for DistributedRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistributedRuntime")
            .field("topology", &self.topology)
            .field("rank", &self.rank())
            .field("world_size", &self.world_size())
            .field("precision_config", &self.precision_config)
            .field("training_config", &self.training_config)
            .finish()
    }
}

/// Builder for [`DistributedRuntime`].
///
/// Each `with_*` method sets a field.  The builder validates that all
/// required fields are provided before constructing the runtime.
pub struct DistributedRuntimeBuilder {
    topology: Option<MeshTorusHybrid>,
    communicator: Option<Arc<dyn Communicator>>,
    precision_config: Option<MixedPrecisionConfig>,
    training_config: Option<TrainingConfig>,
}

impl DistributedRuntimeBuilder {
    /// Creates a new empty builder.
    pub fn new() -> Self {
        Self {
            topology: None,
            communicator: None,
            precision_config: None,
            training_config: None,
        }
    }

    /// Sets the mesh-torus hybrid topology.
    pub fn with_topology(mut self, topology: MeshTorusHybrid) -> Self {
        self.topology = Some(topology);
        self
    }

    /// Sets the communicator for collective operations.
    pub fn with_communicator(mut self, communicator: Arc<dyn Communicator>) -> Self {
        self.communicator = Some(communicator);
        self
    }

    /// Sets the communicator from a boxed implementation.
    pub fn with_boxed_communicator(self, communicator: Box<dyn Communicator>) -> Self {
        self.with_communicator(Arc::from(communicator))
    }

    /// Sets the mixed-precision configuration.
    pub fn with_precision_config(mut self, config: MixedPrecisionConfig) -> Self {
        self.precision_config = Some(config);
        self
    }

    /// Sets the training configuration.
    pub fn with_config(mut self, config: TrainingConfig) -> Self {
        self.training_config = Some(config);
        self
    }

    /// Consumes the builder and returns a [`DistributedRuntime`].
    ///
    /// Returns a [`RuntimeError`] if required fields are missing or if
    /// the training configuration is invalid.
    pub fn build(self) -> Result<DistributedRuntime, RuntimeError> {
        let topology = self
            .topology
            .ok_or_else(|| RuntimeError::Topology("topology is required".into()))?;

        let communicator = self
            .communicator
            .ok_or_else(|| RuntimeError::Communication("communicator is required".into()))?;

        let precision_config = self.precision_config.unwrap_or_default();
        let training_config = self.training_config.unwrap_or_default();

        training_config
            .validate()
            .map_err(|e| RuntimeError::Config(e.to_string()))?;

        let loss_scaler = loss_scaler_from_config(&precision_config);

        info!(
            "DistributedRuntime initialised: rank={}, world_size={}, nodes={}",
            communicator.rank(),
            communicator.world_size(),
            topology.total_nodes(),
        );

        Ok(DistributedRuntime {
            topology,
            communicator,
            precision_config,
            training_config,
            loss_scaler: Arc::new(RwLock::new(loss_scaler)),
        })
    }
}

impl Default for DistributedRuntimeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mt_comm::MockCommunicator;

    fn sample_topology() -> MeshTorusHybrid {
        MeshTorusHybrid::new_2d(2, 2, 2).unwrap()
    }

    fn sample_config() -> TrainingConfig {
        TrainingConfig {
            batch_size: 32,
            micro_batch_size: 16,
            accumulation_steps: 2,
            ..Default::default()
        }
    }

    fn sample_communicator() -> Arc<dyn Communicator> {
        let comms = MockCommunicator::create_world(4);
        Arc::new(comms.into_iter().next().unwrap())
    }

    #[test]
    fn test_builder_new() {
        let builder = DistributedRuntimeBuilder::new();
        assert!(builder.topology.is_none());
        assert!(builder.communicator.is_none());
    }

    #[test]
    fn test_builder_with_topology() {
        let topo = sample_topology();
        let builder = DistributedRuntimeBuilder::new().with_topology(topo.clone());
        assert_eq!(builder.topology.unwrap().total_nodes(), 4);
    }

    #[test]
    fn test_build_success() {
        let runtime = DistributedRuntime::builder()
            .with_topology(sample_topology())
            .with_communicator(sample_communicator())
            .with_config(sample_config())
            .build()
            .expect("build should succeed");
        assert_eq!(runtime.rank(), 0);
        assert_eq!(runtime.world_size(), 4);
    }

    #[test]
    fn test_build_missing_topology() {
        let result = DistributedRuntime::builder()
            .with_communicator(sample_communicator())
            .with_config(sample_config())
            .build();
        assert!(result.is_err());
        match result {
            Err(RuntimeError::Topology(_)) => {} // expected
            _ => panic!("expected Topology error"),
        }
    }

    #[test]
    fn test_build_missing_communicator() {
        let result = DistributedRuntime::builder()
            .with_topology(sample_topology())
            .with_config(sample_config())
            .build();
        assert!(result.is_err());
        match result {
            Err(RuntimeError::Communication(_)) => {} // expected
            _ => panic!("expected Communication error"),
        }
    }

    #[test]
    fn test_build_invalid_config() {
        let invalid_config = TrainingConfig {
            accumulation_steps: 0,
            ..Default::default()
        };
        let result = DistributedRuntime::builder()
            .with_topology(sample_topology())
            .with_communicator(sample_communicator())
            .with_config(invalid_config)
            .build();
        assert!(result.is_err());
        match result {
            Err(RuntimeError::Config(_)) => {} // expected
            _ => panic!("expected Config error"),
        }
    }

    #[test]
    fn test_build_world_size_matches_communicator() {
        let comms = MockCommunicator::create_world(8);
        let comm = Arc::new(comms.into_iter().next().unwrap());
        let runtime = DistributedRuntime::builder()
            .with_topology(sample_topology())
            .with_communicator(comm)
            .with_config(sample_config())
            .build()
            .expect("build should succeed");
        assert_eq!(runtime.world_size(), 8);
        assert_eq!(runtime.rank(), 0);
    }

    #[test]
    fn test_build_with_boxed_communicator() {
        let comms = MockCommunicator::create_world(2);
        let comm: Box<dyn Communicator> = Box::new(comms.into_iter().next().unwrap());
        let runtime = DistributedRuntime::builder()
            .with_topology(sample_topology())
            .with_boxed_communicator(comm)
            .with_config(sample_config())
            .build()
            .expect("build should succeed");
        assert_eq!(runtime.world_size(), 2);
    }

    #[test]
    fn test_build_default_precision_config() {
        let runtime = DistributedRuntime::builder()
            .with_topology(sample_topology())
            .with_communicator(sample_communicator())
            .with_config(sample_config())
            .build()
            .expect("build should succeed");
        assert_eq!(
            runtime.precision_config,
            MixedPrecisionConfig::default()
        );
    }

    #[test]
    fn test_build_custom_precision_config() {
        let precision = MixedPrecisionConfig {
            forward_format: mt_core::precision::Fp8Format::E5M2,
            backward_format: mt_core::precision::Fp8Format::E4M3,
            ..Default::default()
        };
        let runtime = DistributedRuntime::builder()
            .with_topology(sample_topology())
            .with_communicator(sample_communicator())
            .with_precision_config(precision.clone())
            .with_config(sample_config())
            .build()
            .expect("build should succeed");
        assert_eq!(runtime.precision_config.forward_format, mt_core::precision::Fp8Format::E5M2);
    }

    #[test]
    fn test_loss_scaler_initialized() {
        let runtime = DistributedRuntime::builder()
            .with_topology(sample_topology())
            .with_communicator(sample_communicator())
            .with_config(sample_config())
            .build()
            .expect("build should succeed");
        let scaler = runtime.loss_scaler.read();
        // Default MixedPrecisionConfig uses Delayed scaling → static scaler with scale=1.0
        assert!((scaler.scale() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_debug_output() {
        let runtime = DistributedRuntime::builder()
            .with_topology(sample_topology())
            .with_communicator(sample_communicator())
            .with_config(sample_config())
            .build()
            .expect("build should succeed");
        let debug = format!("{:?}", runtime);
        assert!(debug.contains("DistributedRuntime"));
        assert!(debug.contains("rank: 0"));
        assert!(debug.contains("world_size: 4"));
    }

    #[test]
    fn test_loss_scaler_is_shared() {
        let runtime = DistributedRuntime::builder()
            .with_topology(sample_topology())
            .with_communicator(sample_communicator())
            .with_config(sample_config())
            .build()
            .expect("build should succeed");
        // Verify the Arc can be cloned for shared access
        let scaler1 = runtime.loss_scaler.clone();
        let scaler2 = runtime.loss_scaler.clone();
        assert_eq!(scaler1.read().scale(), scaler2.read().scale());
    }

    #[test]
    fn test_multiple_builders_independent() {
        let runtime1 = DistributedRuntime::builder()
            .with_topology(sample_topology())
            .with_communicator(sample_communicator())
            .with_config(sample_config())
            .build()
            .expect("build should succeed");

        let topo2 = MeshTorusHybrid::new_2d(3, 3, 2).unwrap();
        let comms2 = MockCommunicator::create_world(2);
        let comm2 = Arc::new(comms2.into_iter().next().unwrap());
        let runtime2 = DistributedRuntime::builder()
            .with_topology(topo2)
            .with_communicator(comm2)
            .with_config(sample_config())
            .build()
            .expect("build should succeed");

        assert_eq!(runtime1.world_size(), 4);
        assert_eq!(runtime2.world_size(), 2);
    }
}
