use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CollectiveOp {
    AllReduce,
    Broadcast,
    AllGather,
    ReduceScatter,
}

impl fmt::Display for CollectiveOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CollectiveOp::AllReduce => write!(f, "AllReduce"),
            CollectiveOp::Broadcast => write!(f, "Broadcast"),
            CollectiveOp::AllGather => write!(f, "AllGather"),
            CollectiveOp::ReduceScatter => write!(f, "ReduceScatter"),
        }
    }
}

pub trait TopologyAwareCommunicator: Send + Sync {
    fn nearest_neighbor_exchange(&self, data: Vec<u8>, axis: usize) -> Vec<Vec<u8>>;
    fn all_reduce_mesh(&self, data: Vec<u8>, group: &[usize]) -> Vec<u8>;
    fn pipeline_stage_sendrecv(&self, data: Vec<u8>, stage_rank: usize) -> Vec<u8>;
    fn route_collective(&self, op: CollectiveOp, data: Vec<u8>) -> Vec<u8>;
}

pub struct DummyCommunicator {
    pub rank: usize,
    pub world_size: usize,
}

impl DummyCommunicator {
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self { rank, world_size }
    }
}

impl TopologyAwareCommunicator for DummyCommunicator {
    fn nearest_neighbor_exchange(&self, data: Vec<u8>, _axis: usize) -> Vec<Vec<u8>> {
        vec![data.clone(); self.world_size.min(4)]
    }

    fn all_reduce_mesh(&self, data: Vec<u8>, _group: &[usize]) -> Vec<u8> {
        data
    }

    fn pipeline_stage_sendrecv(&self, data: Vec<u8>, _stage_rank: usize) -> Vec<u8> {
        data
    }

    fn route_collective(&self, _op: CollectiveOp, data: Vec<u8>) -> Vec<u8> {
        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    fn test_collective_op_display() {
        assert_eq!(CollectiveOp::AllReduce.to_string(), "AllReduce");
        assert_eq!(CollectiveOp::Broadcast.to_string(), "Broadcast");
        assert_eq!(CollectiveOp::AllGather.to_string(), "AllGather");
        assert_eq!(CollectiveOp::ReduceScatter.to_string(), "ReduceScatter");
    }

    #[rstest]
    fn test_collective_op_equality() {
        assert_eq!(CollectiveOp::AllReduce, CollectiveOp::AllReduce);
        assert_ne!(CollectiveOp::AllReduce, CollectiveOp::Broadcast);
    }

    #[rstest]
    fn test_dummy_communicator() {
        let comm = DummyCommunicator::new(0, 4);
        let data = vec![1u8, 2, 3, 4];

        let result = comm.nearest_neighbor_exchange(data.clone(), 0);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], data);

        let reduced = comm.all_reduce_mesh(data.clone(), &[0, 1, 2, 3]);
        assert_eq!(reduced, data);

        let pipelined = comm.pipeline_stage_sendrecv(data.clone(), 1);
        assert_eq!(pipelined, data);

        let routed = comm.route_collective(CollectiveOp::AllReduce, data.clone());
        assert_eq!(routed, data);
    }

    #[rstest]
    #[case(CollectiveOp::AllReduce)]
    #[case(CollectiveOp::Broadcast)]
    #[case(CollectiveOp::AllGather)]
    #[case(CollectiveOp::ReduceScatter)]
    fn test_route_collective_variants(#[case] op: CollectiveOp) {
        let comm = DummyCommunicator::new(0, 2);
        let data = vec![42u8];
        let result = comm.route_collective(op, data.clone());
        assert_eq!(result, data);
    }

    #[rstest]
    fn test_dummy_world_size_clamp() {
        let comm = DummyCommunicator::new(0, 2);
        let result = comm.nearest_neighbor_exchange(vec![1u8], 0);
        assert_eq!(result.len(), 2);
    }
}
