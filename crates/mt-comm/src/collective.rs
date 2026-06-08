use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Product,
    Min,
    Max,
    Mean,
}

impl fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReduceOp::Sum => write!(f, "Sum"),
            ReduceOp::Product => write!(f, "Product"),
            ReduceOp::Min => write!(f, "Min"),
            ReduceOp::Max => write!(f, "Max"),
            ReduceOp::Mean => write!(f, "Mean"),
        }
    }
}

pub trait Communicator: Send + Sync {
    fn send(&self, data: Vec<u8>, dst: usize) -> anyhow::Result<()>;
    fn recv(&self, src: usize) -> anyhow::Result<Vec<u8>>;
    fn all_reduce(&self, data: Vec<u8>, op: ReduceOp) -> anyhow::Result<Vec<u8>>;
    fn broadcast(&self, data: Vec<u8>, root: usize) -> anyhow::Result<Vec<u8>>;
    fn all_gather(&self, data: Vec<u8>) -> anyhow::Result<Vec<Vec<u8>>>;
    fn rank(&self) -> usize;
    fn world_size(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    #[case(ReduceOp::Sum, "Sum")]
    #[case(ReduceOp::Product, "Product")]
    #[case(ReduceOp::Min, "Min")]
    #[case(ReduceOp::Max, "Max")]
    #[case(ReduceOp::Mean, "Mean")]
    fn test_reduce_op_display(#[case] op: ReduceOp, #[case] expected: &str) {
        assert_eq!(op.to_string(), expected);
    }

    #[rstest]
    fn test_reduce_op_equality() {
        assert_eq!(ReduceOp::Sum, ReduceOp::Sum);
        assert_ne!(ReduceOp::Sum, ReduceOp::Max);
    }
}
