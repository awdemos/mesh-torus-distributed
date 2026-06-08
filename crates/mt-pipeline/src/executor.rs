use mt_comm::Communicator;

use crate::checkpoint::CheckpointStrategy;
use crate::communicator::PipelineComm;
use crate::schedule::{Schedule, ScheduleKind};
use crate::stage::PipelineStage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleType {
    Gpipe,
    OneF1B,
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub num_stages: usize,
    pub num_microbatches: usize,
    pub checkpoint_strategy: CheckpointStrategy,
    pub schedule_type: ScheduleType,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            num_stages: 1,
            num_microbatches: 1,
            checkpoint_strategy: CheckpointStrategy::None,
            schedule_type: ScheduleType::Gpipe,
        }
    }
}

pub struct PipelineExecutor<C: Communicator> {
    pub stages: Vec<PipelineStage>,
    schedule: ScheduleKind,
    pub comm: Vec<PipelineComm<C>>,
    pub config: PipelineConfig,
}

impl<C: Communicator> PipelineExecutor<C> {
    pub fn new(
        stages: Vec<PipelineStage>,
        comm: Vec<PipelineComm<C>>,
        config: PipelineConfig,
    ) -> Self {
        let schedule = match config.schedule_type {
            ScheduleType::Gpipe => ScheduleKind::Gpipe(crate::schedule::GpipeSchedule),
            ScheduleType::OneF1B => ScheduleKind::OneF1B(crate::schedule::OneF1BSchedule),
        };
        Self {
            stages,
            schedule,
            comm,
            config,
        }
    }

    pub fn run(&mut self) -> anyhow::Result<()> {
        self.schedule
            .execute(&mut self.stages, &self.comm, self.config.num_microbatches)
    }

    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage::{StageData, StageForward};
    use mt_comm::MockCommunicator;
    use rstest::*;

    struct IdentityLayer;

    impl StageForward for IdentityLayer {
        fn forward(&self, input: StageData) -> StageData {
            input
        }
        fn backward(&self, _input: StageData, grad_output: StageData) -> StageData {
            grad_output
        }
    }

    #[rstest]
    fn test_executor_gpipe() {
        let comms = MockCommunicator::create_world(2);
        let pipe_comms: Vec<_> = comms.into_iter().map(PipelineComm::new).collect();
        let stages = vec![
            PipelineStage::new(0, vec![Box::new(IdentityLayer)]),
            PipelineStage::new(1, vec![Box::new(IdentityLayer)]),
        ];
        let config = PipelineConfig {
            num_stages: 2,
            num_microbatches: 2,
            checkpoint_strategy: CheckpointStrategy::None,
            schedule_type: ScheduleType::Gpipe,
        };
        let mut executor = PipelineExecutor::new(stages, pipe_comms, config);
        assert!(executor.run().is_ok());
    }

    #[rstest]
    fn test_executor_onef1b() {
        let comms = MockCommunicator::create_world(2);
        let pipe_comms: Vec<_> = comms.into_iter().map(PipelineComm::new).collect();
        let stages = vec![
            PipelineStage::new(0, vec![Box::new(IdentityLayer)]),
            PipelineStage::new(1, vec![Box::new(IdentityLayer)]),
        ];
        let config = PipelineConfig {
            num_stages: 2,
            num_microbatches: 4,
            checkpoint_strategy: CheckpointStrategy::None,
            schedule_type: ScheduleType::OneF1B,
        };
        let mut executor = PipelineExecutor::new(stages, pipe_comms, config);
        assert!(executor.run().is_ok());
    }

    #[rstest]
    fn test_executor_num_stages() {
        let comms = MockCommunicator::create_world(3);
        let pipe_comms: Vec<_> = comms.into_iter().map(PipelineComm::new).collect();
        let stages = vec![
            PipelineStage::new(0, vec![Box::new(IdentityLayer)]),
            PipelineStage::new(1, vec![Box::new(IdentityLayer)]),
            PipelineStage::new(2, vec![Box::new(IdentityLayer)]),
        ];
        let config = PipelineConfig::default();
        let executor = PipelineExecutor::new(stages, pipe_comms, config);
        assert_eq!(executor.num_stages(), 3);
    }

    #[rstest]
    fn test_default_config() {
        let config = PipelineConfig::default();
        assert_eq!(config.num_stages, 1);
        assert_eq!(config.num_microbatches, 1);
        assert_eq!(config.checkpoint_strategy, CheckpointStrategy::None);
        assert_eq!(config.schedule_type, ScheduleType::Gpipe);
    }

    #[rstest]
    fn test_schedule_type_equality() {
        assert_eq!(ScheduleType::Gpipe, ScheduleType::Gpipe);
        assert_ne!(ScheduleType::Gpipe, ScheduleType::OneF1B);
    }
}
