use mt_comm::Communicator;

use crate::communicator::PipelineComm;
use crate::schedule::Schedule;
use crate::stage::{PipelineStage, StageData};

pub struct GpipeSchedule;

impl Schedule for GpipeSchedule {
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

        let mut all_inputs: Vec<Vec<StageData>> = vec![Vec::new(); num_stages];
        let mut all_activations: Vec<Vec<StageData>> = vec![Vec::new(); num_stages];

        for _mb in 0..num_microbatches {
            let input = StageData::new(vec![1.0; 4], 2, 2);
            let mut activation = input;

            for (stage_idx, stage) in stages.iter_mut().enumerate() {
                all_inputs[stage_idx].push(activation.clone());
                let output = stage.forward(activation);
                activation = output;

                if stage_idx < num_stages - 1 {
                    comms[stage_idx].send_activations(&activation, stage_idx + 1)?;
                    activation = comms[stage_idx + 1].recv_activations(stage_idx)?;
                }
            }
            all_activations[num_stages - 1].push(activation);
        }

        #[allow(clippy::needless_range_loop)]
        for mb in 0..num_microbatches {
            let mut grad_output = StageData::new(vec![1.0; 4], 2, 2);

            for stage_idx in (0..num_stages).rev() {
                let input = all_inputs[stage_idx][mb].clone();
                let grad_input = stages[stage_idx].backward(input, grad_output.clone());

                if stage_idx > 0 {
                    comms[stage_idx].send_gradients(&grad_input, stage_idx - 1)?;
                    grad_output = comms[stage_idx - 1].recv_gradients(stage_idx)?;
                }
            }
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
            forward_count: std::sync::Arc<parking_lot::Mutex<usize>>,
            backward_count: std::sync::Arc<parking_lot::Mutex<usize>>,
        ) -> Self {
            Self {
                forward_count,
                backward_count,
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
    fn test_gpipe_single_stage_single_microbatch() {
        let (mut stages, fw, bw) = make_counting_stages(1);
        let comms = MockCommunicator::create_world(1);
        let pipe_comms: Vec<_> = comms.into_iter().map(PipelineComm::new).collect();

        GpipeSchedule.execute(&mut stages, &pipe_comms, 1).unwrap();
        assert_eq!(*fw[0].lock(), 1);
        assert_eq!(*bw[0].lock(), 1);
    }

    #[rstest]
    fn test_gpipe_multi_stage() {
        let (mut stages, fw, bw) = make_counting_stages(3);
        let comms = MockCommunicator::create_world(3);
        let pipe_comms: Vec<_> = comms.into_iter().map(PipelineComm::new).collect();

        GpipeSchedule.execute(&mut stages, &pipe_comms, 2).unwrap();
        for i in 0..3 {
            assert_eq!(*fw[i].lock(), 2, "stage {} forward count", i);
            assert_eq!(*bw[i].lock(), 2, "stage {} backward count", i);
        }
    }

    #[rstest]
    fn test_gpipe_no_stages() {
        let mut stages: Vec<PipelineStage> = vec![];
        let comms = MockCommunicator::create_world(1);
        let pipe_comms: Vec<_> = comms.into_iter().map(PipelineComm::new).collect();

        let result = GpipeSchedule.execute(&mut stages, &pipe_comms, 4);
        assert!(result.is_ok());
    }

    #[rstest]
    fn test_gpipe_order_all_forward_then_backward() {
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

        GpipeSchedule.execute(&mut stages, &pipe_comms, 2).unwrap();
        let ord = order.lock();
        let fw_count = ord.iter().filter(|s| s.starts_with("fw")).count();
        let bw_count = ord.iter().filter(|s| s.starts_with("bw")).count();
        assert_eq!(fw_count, 4);
        assert_eq!(bw_count, 4);
        let last_fw_idx = ord.iter().rposition(|s| s.starts_with("fw")).unwrap();
        let first_bw_idx = ord.iter().position(|s| s.starts_with("bw")).unwrap();
        assert!(
            last_fw_idx < first_bw_idx,
            "all forwards must complete before any backward"
        );
    }
}
