pub mod gpipe;
pub mod onef1b;

pub use gpipe::GpipeSchedule;
pub use onef1b::OneF1BSchedule;

use crate::communicator::PipelineComm;
use crate::stage::PipelineStage;
use mt_comm::Communicator;

pub trait Schedule: Send + Sync {
    fn execute<C: Communicator>(
        &self,
        stages: &mut [PipelineStage],
        comm: &[PipelineComm<C>],
        num_microbatches: usize,
    ) -> anyhow::Result<()>;
}

pub enum ScheduleKind {
    Gpipe(GpipeSchedule),
    OneF1B(OneF1BSchedule),
}

impl Schedule for ScheduleKind {
    fn execute<C: Communicator>(
        &self,
        stages: &mut [PipelineStage],
        comm: &[PipelineComm<C>],
        num_microbatches: usize,
    ) -> anyhow::Result<()> {
        match self {
            ScheduleKind::Gpipe(s) => s.execute(stages, comm, num_microbatches),
            ScheduleKind::OneF1B(s) => s.execute(stages, comm, num_microbatches),
        }
    }
}
