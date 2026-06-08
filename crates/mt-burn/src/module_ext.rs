use std::boxed::Box;
use std::string::String;
use std::vec::Vec;
use burn::tensor::{Tensor, backend::Backend};
use burn::module::Module;

pub trait StageForward<B: Backend> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2>;
}

pub struct PipelineStage<B: Backend> {
    name: String,
    inner: Box<dyn StageForward<B>>,
}

impl<B: Backend> PipelineStage<B> {
    pub fn new(name: impl Into<String>, inner: Box<dyn StageForward<B>>) -> Self {
        Self {
            name: name.into(),
            inner,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.inner.forward(input)
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

pub trait PipelineModule<B: Backend>: Module<B> {
    fn split_into_stages(self, num_stages: usize) -> Vec<PipelineStage<B>>;
}

pub struct SequentialStages<B: Backend> {
    stages: Vec<PipelineStage<B>>,
}

impl<B: Backend> SequentialStages<B> {
    pub fn new(stages: Vec<PipelineStage<B>>) -> Self {
        Self { stages }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut output = input;
        for stage in &self.stages {
            output = stage.forward(output);
        }
        output
    }

    pub fn stage(&self, index: usize) -> Option<&PipelineStage<B>> {
        self.stages.get(index)
    }

    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }
}

pub fn split_evenly(count: usize, num_stages: usize) -> Vec<(usize, usize)> {
    if num_stages == 0 || count == 0 {
        return Vec::new();
    }
    let actual_stages = num_stages.min(count);
    let base = count / actual_stages;
    let remainder = count % actual_stages;
    let mut ranges = Vec::with_capacity(actual_stages);
    let mut offset = 0;
    for i in 0..actual_stages {
        let extra = if i < remainder { 1 } else { 0 };
        let len = base + extra;
        ranges.push((offset, offset + len));
        offset += len;
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, backend::Backend};
    use burn_ndarray::NdArray;
    use burn_ndarray::NdArrayDevice;
    use burn::module::Module;

    type TestBackend = NdArray;

    fn default_device() -> NdArrayDevice {
        Default::default()
    }

    #[derive(Module, Debug)]
    struct IdentityStage<B: Backend> {
        _phantom: core::marker::PhantomData<B>,
    }

    impl<B: Backend> IdentityStage<B> {
        fn new() -> Self {
            Self {
                _phantom: core::marker::PhantomData,
            }
        }
    }

    impl<B: Backend> StageForward<B> for IdentityStage<B> {
        fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            input
        }
    }

    #[test]
    fn split_evenly_basic() {
        let ranges = split_evenly(10, 3);
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges, vec![(0, 4), (4, 7), (7, 10)]);
    }

    #[test]
    fn split_evenly_exact() {
        let ranges = split_evenly(6, 3);
        assert_eq!(ranges, vec![(0, 2), (2, 4), (4, 6)]);
    }

    #[test]
    fn split_evenly_more_stages_than_items() {
        let ranges = split_evenly(2, 5);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges, vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn split_evenly_single_stage() {
        let ranges = split_evenly(5, 1);
        assert_eq!(ranges, vec![(0, 5)]);
    }

    #[test]
    fn split_evenly_zero_count() {
        let ranges = split_evenly(0, 3);
        assert!(ranges.is_empty());
    }

    #[test]
    fn split_evenly_zero_stages() {
        let ranges = split_evenly(5, 0);
        assert!(ranges.is_empty());
    }

    #[test]
    fn pipeline_stage_forward() {
        let device = default_device();
        let stage: PipelineStage<TestBackend> = PipelineStage::new(
            "identity",
            Box::new(IdentityStage::<TestBackend>::new()),
        );

        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]),
            &device,
        );
        let output = stage.forward(input.clone());
        let input_data = input.into_data();
        let output_data = output.into_data();
        assert_eq!(input_data, output_data);
    }

    #[test]
    fn sequential_stages_forward() {
        let device = default_device();
        let stages: Vec<PipelineStage<TestBackend>> = (0..3)
            .map(|i| {
                PipelineStage::new(
                    format!("stage_{i}"),
                    Box::new(IdentityStage::<TestBackend>::new()),
                )
            })
            .collect();

        let pipeline = SequentialStages::new(stages);
        assert_eq!(pipeline.num_stages(), 3);

        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]),
            &device,
        );
        let output = pipeline.forward(input.clone());
        let input_data = input.into_data();
        let output_data = output.into_data();
        assert_eq!(input_data, output_data);
    }

    #[test]
    fn stage_name() {
        let stage: PipelineStage<TestBackend> = PipelineStage::new(
            "my_stage",
            Box::new(IdentityStage::<TestBackend>::new()),
        );
        assert_eq!(stage.name(), "my_stage");
    }

    #[test]
    fn sequential_stages_access() {
        let stages: Vec<PipelineStage<TestBackend>> = (0..3)
            .map(|i| {
                PipelineStage::new(
                    format!("stage_{i}"),
                    Box::new(IdentityStage::<TestBackend>::new()),
                )
            })
            .collect();

        let pipeline = SequentialStages::new(stages);
        assert!(pipeline.stage(0).is_some());
        assert!(pipeline.stage(2).is_some());
        assert!(pipeline.stage(3).is_none());
        assert_eq!(pipeline.stage(1).unwrap().name(), "stage_1");
    }
}
