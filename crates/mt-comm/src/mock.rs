use parking_lot::RwLock;
use std::sync::Arc;

use crate::collective::{Communicator, ReduceOp};

#[derive(Debug)]
pub struct MockCommunicator {
    rank: usize,
    world_size: usize,
    mailboxes: Arc<RwLock<Vec<Vec<Vec<u8>>>>>,
}

impl MockCommunicator {
    pub fn new(rank: usize, world_size: usize) -> Self {
        let mailboxes = vec![vec![Vec::new(); world_size]; world_size];
        Self {
            rank,
            world_size,
            mailboxes: Arc::new(RwLock::new(mailboxes)),
        }
    }

    pub fn create_world(world_size: usize) -> Vec<Self> {
        let mailboxes = Arc::new(RwLock::new(vec![vec![Vec::new(); world_size]; world_size]));
        (0..world_size)
            .map(|rank| Self {
                rank,
                world_size,
                mailboxes: mailboxes.clone(),
            })
            .collect()
    }

    fn apply_reduce(op: ReduceOp, chunks: &[Vec<u8>]) -> Vec<u8> {
        if chunks.is_empty() {
            return Vec::new();
        }
        let elem_size = 8;
        let n_elems = chunks[0].len() / elem_size;
        if n_elems == 0 {
            return Vec::new();
        }
        let mut result = vec![0u8; n_elems * elem_size];
        for i in 0..n_elems {
            let values: Vec<u64> = chunks
                .iter()
                .map(|c| {
                    let start = i * elem_size;
                    let bytes: [u8; 8] = c[start..start + elem_size].try_into().unwrap_or([0; 8]);
                    u64::from_le_bytes(bytes)
                })
                .collect();
            let reduced = match op {
                ReduceOp::Sum => values.iter().sum(),
                ReduceOp::Product => values.iter().product(),
                ReduceOp::Min => values.iter().copied().min().unwrap_or(0),
                ReduceOp::Max => values.iter().copied().max().unwrap_or(0),
                ReduceOp::Mean => values.iter().sum::<u64>() / values.len() as u64,
            };
            let start = i * elem_size;
            result[start..start + elem_size].copy_from_slice(&reduced.to_le_bytes());
        }
        result
    }
}

impl Communicator for MockCommunicator {
    fn send(&self, data: Vec<u8>, dst: usize) -> anyhow::Result<()> {
        if dst >= self.world_size {
            anyhow::bail!("destination {} >= world_size {}", dst, self.world_size);
        }
        self.mailboxes.write()[self.rank][dst] = data;
        Ok(())
    }

    fn recv(&self, src: usize) -> anyhow::Result<Vec<u8>> {
        if src >= self.world_size {
            anyhow::bail!("source {} >= world_size {}", src, self.world_size);
        }
        let mailboxes = self.mailboxes.read();
        Ok(mailboxes[src][self.rank].clone())
    }

    fn all_reduce(&self, data: Vec<u8>, op: ReduceOp) -> anyhow::Result<Vec<u8>> {
        let all_data = vec![data; self.world_size];
        Ok(Self::apply_reduce(op, &all_data))
    }

    fn broadcast(&self, data: Vec<u8>, root: usize) -> anyhow::Result<Vec<u8>> {
        if root >= self.world_size {
            anyhow::bail!("root {} >= world_size {}", root, self.world_size);
        }
        Ok(data)
    }

    fn all_gather(&self, data: Vec<u8>) -> anyhow::Result<Vec<Vec<u8>>> {
        Ok(vec![data; self.world_size])
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    fn to_bytes(val: u64) -> Vec<u8> {
        val.to_le_bytes().to_vec()
    }

    fn from_bytes(data: &[u8]) -> u64 {
        let bytes: [u8; 8] = data.try_into().unwrap();
        u64::from_le_bytes(bytes)
    }

    #[rstest]
    fn test_mock_all_reduce_sum() {
        let comms = MockCommunicator::create_world(4);
        let input = to_bytes(10);
        let result = comms[0].all_reduce(input.clone(), ReduceOp::Sum).unwrap();
        assert_eq!(from_bytes(&result), 40);
    }

    #[rstest]
    fn test_mock_all_reduce_sum_multi_element() {
        let comms = MockCommunicator::create_world(4);
        let mut input = Vec::new();
        input.extend_from_slice(&1u64.to_le_bytes());
        input.extend_from_slice(&2u64.to_le_bytes());
        input.extend_from_slice(&3u64.to_le_bytes());
        let result = comms[0].all_reduce(input.clone(), ReduceOp::Sum).unwrap();
        assert_eq!(result.len(), 24);
        let v0 = from_bytes(&result[0..8]);
        let v1 = from_bytes(&result[8..16]);
        let v2 = from_bytes(&result[16..24]);
        assert_eq!(v0, 4);
        assert_eq!(v1, 8);
        assert_eq!(v2, 12);
    }

    #[rstest]
    fn test_mock_all_reduce_product() {
        let comms = MockCommunicator::create_world(3);
        let input = to_bytes(3);
        let result = comms[0].all_reduce(input, ReduceOp::Product).unwrap();
        assert_eq!(from_bytes(&result), 27);
    }

    #[rstest]
    fn test_mock_all_reduce_min() {
        let comms = MockCommunicator::create_world(4);
        let input = to_bytes(10);
        let result = comms[0].all_reduce(input, ReduceOp::Min).unwrap();
        assert_eq!(from_bytes(&result), 10);
    }

    #[rstest]
    fn test_mock_all_reduce_max() {
        let comms = MockCommunicator::create_world(4);
        let input = to_bytes(10);
        let result = comms[0].all_reduce(input, ReduceOp::Max).unwrap();
        assert_eq!(from_bytes(&result), 10);
    }

    #[rstest]
    fn test_mock_all_reduce_mean() {
        let comms = MockCommunicator::create_world(4);
        let input = to_bytes(10);
        let result = comms[0].all_reduce(input, ReduceOp::Mean).unwrap();
        assert_eq!(from_bytes(&result), 10);
    }

    #[rstest]
    fn test_mock_broadcast_from_rank_0() {
        let comms = MockCommunicator::create_world(4);
        let data = vec![42u8, 43, 44];
        for comm in &comms {
            let result = comm.broadcast(data.clone(), 0).unwrap();
            assert_eq!(result, data);
        }
    }

    #[rstest]
    fn test_mock_send_recv() {
        let comms = MockCommunicator::create_world(2);
        let data = vec![1u8, 2, 3, 4];
        comms[0].send(data.clone(), 1).unwrap();
        let received = comms[1].recv(0).unwrap();
        assert_eq!(received, data);
    }

    #[rstest]
    fn test_mock_all_gather() {
        let comms = MockCommunicator::create_world(3);
        let data = vec![99u8];
        let result = comms[0].all_gather(data.clone()).unwrap();
        assert_eq!(result.len(), 3);
        for chunk in &result {
            assert_eq!(*chunk, data);
        }
    }

    #[rstest]
    fn test_mock_rank_and_world_size() {
        let comms = MockCommunicator::create_world(4);
        assert_eq!(comms[0].rank(), 0);
        assert_eq!(comms[3].rank(), 3);
        assert_eq!(comms[0].world_size(), 4);
    }

    #[rstest]
    fn test_mock_send_invalid_dst() {
        let comms = MockCommunicator::create_world(2);
        assert!(comms[0].send(vec![1], 5).is_err());
    }

    #[rstest]
    fn test_mock_recv_invalid_src() {
        let comms = MockCommunicator::create_world(2);
        assert!(comms[0].recv(5).is_err());
    }

    #[rstest]
    fn test_mock_broadcast_invalid_root() {
        let comms = MockCommunicator::create_world(2);
        assert!(comms[0].broadcast(vec![1], 5).is_err());
    }
}
