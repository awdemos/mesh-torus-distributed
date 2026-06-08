use std::collections::VecDeque;

pub trait StageForward: Send + Sync {
    fn forward(&self, input: StageData) -> StageData;
    fn backward(&self, input: StageData, grad_output: StageData) -> StageData;
}

#[derive(Debug, Clone)]
pub struct StageData {
    pub values: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl StageData {
    pub fn new(values: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(values.len(), rows * cols, "data length mismatch");
        Self { values, rows, cols }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            values: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 + 8 + self.values.len() * 4);
        bytes.extend_from_slice(&(self.rows as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.cols as u64).to_le_bytes());
        for v in &self.values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let rows = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let cols = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let mut values = Vec::with_capacity(rows * cols);
        let mut offset = 16;
        for _ in 0..rows * cols {
            let v = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            values.push(v);
            offset += 4;
        }
        Self { values, rows, cols }
    }
}

pub struct StageForwardClosure {
    pub forward_fn: Box<dyn Fn(StageData) -> StageData + Send + Sync>,
    pub backward_fn: Box<dyn Fn(StageData, StageData) -> StageData + Send + Sync>,
}

impl StageForward for StageForwardClosure {
    fn forward(&self, input: StageData) -> StageData {
        (self.forward_fn)(input)
    }
    fn backward(&self, input: StageData, grad_output: StageData) -> StageData {
        (self.backward_fn)(input, grad_output)
    }
}

pub struct PipelineStage {
    pub stage_id: usize,
    pub layers: Vec<Box<dyn StageForward>>,
    pub input_cache: Option<StageData>,
    pub activation_buffer: VecDeque<StageData>,
}

impl PipelineStage {
    pub fn new(stage_id: usize, layers: Vec<Box<dyn StageForward>>) -> Self {
        Self {
            stage_id,
            layers,
            input_cache: None,
            activation_buffer: VecDeque::new(),
        }
    }

    pub fn forward(&mut self, input: StageData) -> StageData {
        let mut output = input;
        for layer in &self.layers {
            output = layer.forward(output);
        }
        output
    }

    pub fn backward(&mut self, input: StageData, grad_output: StageData) -> StageData {
        let mut grad = grad_output;
        for layer in self.layers.iter().rev() {
            grad = layer.backward(input.clone(), grad);
        }
        grad
    }

    pub fn checkpoint_forward(&mut self, input: StageData) -> StageData {
        self.input_cache = Some(input.clone());
        self.forward(input)
    }

    pub fn recompute_and_backward(&mut self, grad_output: StageData) -> StageData {
        let input = self
            .input_cache
            .take()
            .expect("no cached input for recomputation");
        let output = self.forward(input.clone());
        let _ = output;
        self.backward(input, grad_output)
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    struct ScaleLayer {
        factor: f32,
    }

    impl StageForward for ScaleLayer {
        fn forward(&self, input: StageData) -> StageData {
            StageData::new(
                input.values.iter().map(|v| v * self.factor).collect(),
                input.rows,
                input.cols,
            )
        }
        fn backward(&self, _input: StageData, grad_output: StageData) -> StageData {
            StageData::new(
                grad_output.values.iter().map(|v| v * self.factor).collect(),
                grad_output.rows,
                grad_output.cols,
            )
        }
    }

    #[rstest]
    fn test_identity_stage() {
        let mut stage = PipelineStage::new(0, vec![Box::new(IdentityLayer)]);
        let input = StageData::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let output = stage.forward(input.clone());
        assert_eq!(output.values, input.values);
    }

    #[rstest]
    fn test_scale_stage() {
        let mut stage = PipelineStage::new(0, vec![Box::new(ScaleLayer { factor: 2.0 })]);
        let input = StageData::new(vec![1.0, 2.0], 1, 2);
        let output = stage.forward(input);
        assert_eq!(output.values, vec![2.0, 4.0]);
    }

    #[rstest]
    fn test_multi_layer_stage() {
        let mut stage = PipelineStage::new(
            0,
            vec![
                Box::new(ScaleLayer { factor: 2.0 }),
                Box::new(ScaleLayer { factor: 3.0 }),
            ],
        );
        let input = StageData::new(vec![1.0], 1, 1);
        let output = stage.forward(input);
        assert_eq!(output.values, vec![6.0]);
    }

    #[rstest]
    fn test_backward() {
        let mut stage = PipelineStage::new(0, vec![Box::new(ScaleLayer { factor: 2.0 })]);
        let input = StageData::new(vec![1.0], 1, 1);
        let grad = StageData::new(vec![1.0], 1, 1);
        let grad_input = stage.backward(input, grad);
        assert_eq!(grad_input.values, vec![2.0]);
    }

    #[rstest]
    fn test_checkpoint_forward() {
        let mut stage = PipelineStage::new(0, vec![Box::new(IdentityLayer)]);
        let input = StageData::new(vec![1.0, 2.0], 1, 2);
        let _ = stage.checkpoint_forward(input.clone());
        assert!(stage.input_cache.is_some());
        let cached = stage.input_cache.as_ref().unwrap();
        assert_eq!(cached.values, input.values);
    }

    #[rstest]
    fn test_stage_data_to_from_bytes() {
        let data = StageData::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let bytes = data.to_bytes();
        let recovered = StageData::from_bytes(&bytes);
        assert_eq!(recovered.rows, 2);
        assert_eq!(recovered.cols, 2);
        assert_eq!(recovered.values, data.values);
    }

    #[rstest]
    fn test_stage_data_zeros() {
        let data = StageData::zeros(3, 4);
        assert_eq!(data.rows, 3);
        assert_eq!(data.cols, 4);
        assert_eq!(data.values.len(), 12);
        assert!(data.values.iter().all(|&v| v == 0.0));
    }

    #[rstest]
    fn test_num_layers() {
        let stage = PipelineStage::new(
            0,
            vec![
                Box::new(IdentityLayer),
                Box::new(IdentityLayer),
                Box::new(IdentityLayer),
            ],
        );
        assert_eq!(stage.num_layers(), 3);
    }
}
