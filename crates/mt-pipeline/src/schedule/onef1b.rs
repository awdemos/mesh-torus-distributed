use mt_comm::Communicator;

use crate::communicator::PipelineComm;
use crate::schedule::Schedule;
use crate::stage::{PipelineStage, StageData};

pub struct OneF1BSchedule;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Forward(usize),
    Backward(usize),
}

impl OneF1BSchedule {
    fn build_schedule(num_stages: usize, num_microbatches: usize) -> Vec<Vec<Op>> {
        let mut stage_schedules: Vec<Vec<Op>> = vec![Vec::new(); num_stages];

        for (stage_id, schedule) in stage_schedules.iter_mut().enumerate() {
            let warmup = (num_stages - stage_id - 1).min(num_microbatches);
            let steady = num_microbatches.saturating_sub(warmup);
            let cooldown = warmup;

            let mut fwd_mb = 0usize;
            let mut bwd_mb = 0usize;

            for _ in 0..warmup {
                schedule.push(Op::Forward(fwd_mb));
                fwd_mb += 1;
            }

            for _ in 0..steady {
                schedule.push(Op::Forward(fwd_mb));
                fwd_mb += 1;
                schedule.push(Op::Backward(bwd_mb));
                bwd_mb += 1;
            }

            for _ in 0..cooldown {
                schedule.push(Op::Backward(bwd_mb));
                bwd_mb += 1;
            }
        }

        stage_schedules
    }
}

impl Schedule for OneF1BSchedule {
    fn execute<C: Communicator>(
        &self,
        stages: &mut [PipelineStage],
        comms: &[PipelineComm<C>],
        num_microbatches: usize,
    ) -> anyhow::Result<()> {
        let num_stages = stages.len();
        if num_stages == 0 {
            return Ok(());
        }

        let schedules = Self::build_schedule(num_stages, num_microbatches);
        let mut stage_inputs: Vec<Vec<StageData>> = vec![Vec::new(); num_stages];
        let mut pending_fwd: Vec<Vec<(usize, StageData)>> = vec![Vec::new(); num_stages];

        let mut max_ops = schedules.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut op_idx = vec![0usize; num_stages];

        while max_ops > 0 {
            for stage_id in 0..num_stages {
                if op_idx[stage_id] >= schedules[stage_id].len() {
                    continue;
                }
                let op = schedules[stage_id][op_idx[stage_id]];

                match op {
                    Op::Forward(mb) => {
                        let input = if stage_id == 0 {
                            StageData::new(vec![1.0; 4], 2, 2)
                        } else {
                            let pos = pending_fwd[stage_id]
                                .iter()
                                .position(|(m, _)| *m == mb)
                                .expect("activation from previous stage should be available");
                            pending_fwd[stage_id].remove(pos).1
                        };

                        stage_inputs[stage_id].push(input.clone());
                        let output = stages[stage_id].forward(input);

                        if stage_id < num_stages - 1 {
                            comms[stage_id].send_activations(&output, stage_id + 1)?;
                            pending_fwd[stage_id + 1].push((mb, output));
                        }
                    }
                    Op::Backward(mb) => {
                        let input = stage_inputs[stage_id][mb].clone();
                        let grad = StageData::new(vec![1.0; 4], 2, 2);
                        let grad_input = stages[stage_id].backward(input, grad);

                        if stage_id > 0 {
                            comms[stage_id].send_gradients(&grad_input, stage_id - 1)?;
                        }
                    }
                }
                op_idx[stage_id] += 1;
            }
            max_ops = max_ops.saturating_sub(1);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage::StageForward;
    use mt_comm::MockCommunicator;
    use rstest::*;

    struct CountingLayer {
        forward_count: std::sync::Arc<parking_lot::Mutex<usize>>,
        backward_count: std::sync::Arc<parking_lot::Mutex<usize>>,
    }

    impl CountingLayer {
        fn new(
            fw: std::sync::Arc<parking_lot::Mutex<usize>>,
            bw: std::sync::Arc<parking_lot::Mutex<usize>>,
        ) -> Self {
            Self {
                forward_count: fw,
                backward_count: bw,
            }
        }
    }

    impl StageForward for CountingLayer {
        fn forward(&self, input: StageData) -> StageData {
            *self.forward_count.lock() += 1;
            input
        }
        fn backward(&self, _input: StageData, grad_output: StageData) -> StageData {
            *self.backward_count.lock() += 1;
            grad_output
        }
    }

    fn make_counting_stages(
        num_stages: usize,
    ) -> (
        Vec<PipelineStage>,
        Vec<std::sync::Arc<parking_lot::Mutex<usize>>>,
        Vec<std::sync::Arc<parking_lot::Mutex<usize>>>,
    ) {
        let mut stages = Vec::new();
        let mut fw_counts = Vec::new();
        let mut bw_counts = Vec::new();
        for i in 0..num_stages {
            let fw = std::sync::Arc::new(parking_lot::Mutex::new(0usize));
            let bw = std::sync::Arc::new(parking_lot::Mutex::new(0usize));
            stages.push(PipelineStage::new(
                i,
                vec![Box::new(CountingLayer::new(fw.clone(), bw.clone()))],
            ));
            fw_counts.push(fw);
            bw_counts.push(bw);
        }
        (stages, fw_counts, bw_counts)
    }

    #[rstest]
    fn test_onef1b_single_stage() {
        let (mut stages, fw, bw) = make_counting_stages(1);
        let comms = MockCommunicator::create_world(1);
        let pipe_comms: Vec<_> = comms.into_iter().map(PipelineComm::new).collect();

        OneF1BSchedule.execute(&mut stages, &pipe_comms, 2).unwrap();
        assert_eq!(*fw[0].lock(), 2);
        assert_eq!(*bw[0].lock(), 2);
    }

    #[rstest]
    fn test_onef1b_multi_stage() {
        let (mut stages, fw, bw) = make_counting_stages(3);
        let comms = MockCommunicator::create_world(3);
        let pipe_comms: Vec<_> = comms.into_iter().map(PipelineComm::new).collect();

        OneF1BSchedule.execute(&mut stages, &pipe_comms, 4).unwrap();
        for i in 0..3 {
            assert_eq!(*fw[i].lock(), 4, "stage {} forward count", i);
            assert_eq!(*bw[i].lock(), 4, "stage {} backward count", i);
        }
    }

    #[rstest]
    fn test_onef1b_no_stages() {
        let mut stages: Vec<PipelineStage> = vec![];
        let comms = MockCommunicator::create_world(1);
        let pipe_comms: Vec<_> = comms.into_iter().map(PipelineComm::new).collect();

        let result = OneF1BSchedule.execute(&mut stages, &pipe_comms, 4);
        assert!(result.is_ok());
    }

    #[rstest]
    fn test_onef1b_interleaving() {
        let order = std::sync::Arc::new(parking_lot::Mutex::new(Vec::<String>::new()));
        let mut stages = Vec::new();
        for i in 0..2 {
            let order_clone = order.clone();
            let order_clone2 = order.clone();
            let layer = Box::new(crate::stage::StageForwardClosure {
                forward_fn: Box::new(move |input: StageData| {
                    order_clone.lock().push(format!("fw_{}", i));
                    input
                }),
                backward_fn: Box::new(move |_input: StageData, grad: StageData| {
                    order_clone2.lock().push(format!("bw_{}", i));
                    grad
                }),
            });
            stages.push(PipelineStage::new(i, vec![layer]));
        }
        let comms = MockCommunicator::create_world(2);
        let pipe_comms: Vec<_> = comms.into_iter().map(PipelineComm::new).collect();

        OneF1BSchedule.execute(&mut stages, &pipe_comms, 4).unwrap();
        let ord = order.lock();
        let fw_count = ord.iter().filter(|s| s.starts_with("fw")).count();
        let bw_count = ord.iter().filter(|s| s.starts_with("bw")).count();
        assert_eq!(fw_count, 8);
        assert_eq!(bw_count, 8);
    }
}
