use crate::stage::{PipelineStage, StageData, StageForward};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointStrategy {
    Full,
    Selective,
    None,
}

#[derive(Debug, Clone)]
pub struct CheckpointHandle {
    input: StageData,
}

pub struct ActivationCheckpoint;

impl ActivationCheckpoint {
    pub fn save_input(input: StageData) -> CheckpointHandle {
        CheckpointHandle { input }
    }

    pub fn recompute_forward(
        handle: CheckpointHandle,
        layers: &[Box<dyn StageForward>],
    ) -> (StageData, StageData) {
        let mut output = handle.input.clone();
        for layer in layers {
            output = layer.forward(output);
        }
        (handle.input, output)
    }
}

pub fn apply_checkpoint_forward(
    stage: &mut PipelineStage,
    input: StageData,
    strategy: CheckpointStrategy,
) -> StageData {
    match strategy {
        CheckpointStrategy::None => stage.forward(input),
        CheckpointStrategy::Full => {
            stage.input_cache = Some(input.clone());
            stage.forward(input)
        }
        CheckpointStrategy::Selective => {
            if stage.layers.len() > 1 {
                let split_point = stage.layers.len() / 2;
                let mut output = input.clone();
                for layer in &stage.layers[..split_point] {
                    output = layer.forward(output);
                }
                let checkpoint_output = output.clone();
                for layer in &stage.layers[split_point..] {
                    output = layer.forward(output);
                }
                stage.input_cache = Some(input);
                let _ = checkpoint_output;
                output
            } else {
                stage.input_cache = Some(input.clone());
                stage.forward(input)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    struct DoublerLayer;

    impl StageForward for DoublerLayer {
        fn forward(&self, input: StageData) -> StageData {
            StageData::new(
                input.values.iter().map(|v| v * 2.0).collect(),
                input.rows,
                input.cols,
            )
        }
        fn backward(&self, _input: StageData, grad_output: StageData) -> StageData {
            StageData::new(
                grad_output.values.iter().map(|v| v * 2.0).collect(),
                grad_output.rows,
                grad_output.cols,
            )
        }
    }

    #[rstest]
    fn test_save_input() {
        let data = StageData::new(vec![1.0, 2.0], 1, 2);
        let handle = ActivationCheckpoint::save_input(data.clone());
        assert_eq!(handle.input.values, data.values);
    }

    #[rstest]
    fn test_recompute_forward() {
        let layers: Vec<Box<dyn StageForward>> = vec![Box::new(DoublerLayer)];
        let input = StageData::new(vec![3.0], 1, 1);
        let handle = ActivationCheckpoint::save_input(input);
        let (input_out, output) = ActivationCheckpoint::recompute_forward(handle, &layers);
        assert_eq!(input_out.values, vec![3.0]);
        assert_eq!(output.values, vec![6.0]);
    }

    #[rstest]
    fn test_checkpoint_strategy_none() {
        let mut stage = PipelineStage::new(0, vec![Box::new(DoublerLayer)]);
        let input = StageData::new(vec![1.0], 1, 1);
        let output = apply_checkpoint_forward(&mut stage, input, CheckpointStrategy::None);
        assert_eq!(output.values, vec![2.0]);
        assert!(stage.input_cache.is_none());
    }

    #[rstest]
    fn test_checkpoint_strategy_full() {
        let mut stage = PipelineStage::new(0, vec![Box::new(DoublerLayer)]);
        let input = StageData::new(vec![1.0], 1, 1);
        let output = apply_checkpoint_forward(&mut stage, input.clone(), CheckpointStrategy::Full);
        assert_eq!(output.values, vec![2.0]);
        assert!(stage.input_cache.is_some());
        assert_eq!(stage.input_cache.unwrap().values, input.values);
    }

    #[rstest]
    fn test_checkpoint_strategy_selective() {
        let mut stage = PipelineStage::new(0, vec![Box::new(DoublerLayer), Box::new(DoublerLayer)]);
        let input = StageData::new(vec![1.0], 1, 1);
        let output =
            apply_checkpoint_forward(&mut stage, input.clone(), CheckpointStrategy::Selective);
        assert_eq!(output.values, vec![4.0]);
        assert!(stage.input_cache.is_some());
    }

    #[rstest]
    fn test_checkpoint_strategy_variants() {
        assert_ne!(CheckpointStrategy::Full, CheckpointStrategy::None);
        assert_ne!(CheckpointStrategy::Selective, CheckpointStrategy::Full);
        assert_eq!(CheckpointStrategy::Full, CheckpointStrategy::Full);
    }
}
