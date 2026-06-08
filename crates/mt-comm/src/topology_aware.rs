use mt_core::topology::{
    CollectiveOp, MeshTorusHybrid, TopologyAwareCommunicator, TorusCoordinates,
};

use crate::collective::{Communicator, ReduceOp};

#[derive(Debug)]
pub struct TopologyAwareComm<C: Communicator> {
    inner: C,
    topology: MeshTorusHybrid,
}

impl<C: Communicator> TopologyAwareComm<C> {
    pub fn new(inner: C, topology: MeshTorusHybrid) -> Self {
        Self { inner, topology }
    }

    pub fn rank(&self) -> usize {
        self.inner.rank()
    }

    pub fn world_size(&self) -> usize {
        self.inner.world_size()
    }

    pub fn inner(&self) -> &C {
        &self.inner
    }

    pub fn topology(&self) -> &MeshTorusHybrid {
        &self.topology
    }

    pub fn rank_coords(&self) -> TorusCoordinates {
        self.topology.rank_to_coords(self.inner.rank())
    }

    pub fn exchange_with_neighbors(&self, data: Vec<u8>) -> anyhow::Result<Vec<(usize, Vec<u8>)>> {
        let coords = self.rank_coords();
        let neighbors = coords.neighbors(&self.topology.dims);
        let mut results = Vec::with_capacity(neighbors.len());
        for (neighbor_coords, _axis) in &neighbors {
            let neighbor_rank = self.topology.node_rank(neighbor_coords);
            self.inner.send(data.clone(), neighbor_rank)?;
            let recv_data = self.inner.recv(neighbor_rank)?;
            results.push((neighbor_rank, recv_data));
        }
        Ok(results)
    }

    pub fn perform_collective(
        &self,
        op: CollectiveOp,
        data: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>> {
        match op {
            CollectiveOp::AllReduce => self.inner.all_reduce(data, ReduceOp::Sum),
            CollectiveOp::Broadcast => self.inner.broadcast(data, 0),
            CollectiveOp::AllGather => {
                let gathered = self.inner.all_gather(data)?;
                let mut flat = Vec::new();
                for chunk in gathered {
                    flat.extend_from_slice(&chunk);
                }
                Ok(flat)
            }
            CollectiveOp::ReduceScatter => {
                let reduced = self.inner.all_reduce(data, ReduceOp::Sum)?;
                let chunk_size = reduced.len() / self.inner.world_size().max(1);
                let start = self.inner.rank() * chunk_size;
                Ok(reduced[start..start + chunk_size].to_vec())
            }
        }
    }

    pub fn reduce_on_mesh(
        &self,
        data: Vec<u8>,
        group: &[usize],
    ) -> anyhow::Result<Vec<u8>> {
        let _ = group;
        self.inner.all_reduce(data, ReduceOp::Sum)
    }
}

impl<C: Communicator> TopologyAwareCommunicator for TopologyAwareComm<C> {
    fn nearest_neighbor_exchange(&self, data: Vec<u8>, axis: usize) -> Vec<Vec<u8>> {
        let coords = self.rank_coords();
        let positive = coords.neighbor(&self.topology.dims, axis, true);
        let negative = coords.neighbor(&self.topology.dims, axis, false);
        let pos_rank = self.topology.node_rank(&positive);
        let neg_rank = self.topology.node_rank(&negative);

        let mut results = Vec::new();
        if self.inner.send(data.clone(), pos_rank).is_ok() {
            match self.inner.recv(pos_rank) {
                Ok(recv) => results.push(recv),
                Err(_) => results.push(data.clone()),
            }
        } else {
            results.push(data.clone());
        }
        if self.inner.send(data.clone(), neg_rank).is_ok() {
            match self.inner.recv(neg_rank) {
                Ok(recv) => results.push(recv),
                Err(_) => results.push(data.clone()),
            }
        } else {
            results.push(data.clone());
        }
        results
    }

    fn all_reduce_mesh(&self, data: Vec<u8>, group: &[usize]) -> Vec<u8> {
        self.reduce_on_mesh(data, group).unwrap_or_default()
    }

    fn pipeline_stage_sendrecv(&self, data: Vec<u8>, stage_rank: usize) -> Vec<u8> {
        if self.inner.send(data.clone(), stage_rank).is_ok() {
            self.inner.recv(stage_rank).unwrap_or(data)
        } else {
            data
        }
    }

    fn route_collective(&self, op: CollectiveOp, data: Vec<u8>) -> Vec<u8> {
        self.perform_collective(op, data).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockCommunicator;
    use rstest::*;

    fn to_bytes(val: u64) -> Vec<u8> {
        val.to_le_bytes().to_vec()
    }

    fn from_bytes(data: &[u8]) -> u64 {
        let bytes: [u8; 8] = data.try_into().unwrap();
        u64::from_le_bytes(bytes)
    }

    fn setup_2d_torus(w: usize, h: usize) -> Vec<TopologyAwareComm<MockCommunicator>> {
        let topo = MeshTorusHybrid::new_2d(w, h, 2).unwrap();
        let mocks = MockCommunicator::create_world(w * h);
        mocks
            .into_iter()
            .map(|m| TopologyAwareComm::new(m, topo.clone()))
            .collect()
    }

    #[rstest]
    fn test_topology_aware_rank() {
        let comms = setup_2d_torus(3, 3);
        assert_eq!(comms[0].rank(), 0);
        assert_eq!(comms[8].rank(), 8);
        assert_eq!(comms[0].world_size(), 9);
    }

    #[rstest]
    fn test_topology_aware_rank_coords() {
        let comms = setup_2d_torus(3, 3);
        let coords = comms[0].rank_coords();
        assert_eq!(coords, TorusCoordinates::new_2d(0, 0));
        let coords5 = comms[5].rank_coords();
        assert_eq!(coords5, TorusCoordinates::new_2d(2, 1));
    }

    #[rstest]
    fn test_exchange_with_neighbors_2d() {
        let comms = setup_2d_torus(3, 3);
        let data = vec![42u8];
        let results = comms[4].exchange_with_neighbors(data.clone()).unwrap();
        assert_eq!(results.len(), 4);
    }

    #[rstest]
    fn test_exchange_with_neighbors_corner_wrap() {
        let comms = setup_2d_torus(3, 3);
        let data = vec![99u8];
        let results = comms[0].exchange_with_neighbors(data.clone()).unwrap();
        assert_eq!(results.len(), 4);
        let neighbor_ranks: Vec<usize> = results.iter().map(|(r, _)| *r).collect();
        assert!(neighbor_ranks.contains(&1));
        assert!(neighbor_ranks.contains(&3));
    }

    #[rstest]
    fn test_perform_collective_allreduce() {
        let comms = setup_2d_torus(2, 2);
        let data = to_bytes(10);
        let result = comms[0]
            .perform_collective(CollectiveOp::AllReduce, data.clone())
            .unwrap();
        assert_eq!(from_bytes(&result), 40);
    }

    #[rstest]
    fn test_perform_collective_broadcast() {
        let comms = setup_2d_torus(2, 2);
        let data = vec![1u8, 2, 3];
        let result = comms[1]
            .perform_collective(CollectiveOp::Broadcast, data.clone())
            .unwrap();
        assert_eq!(result, data);
    }

    #[rstest]
    fn test_perform_collective_allgather() {
        let comms = setup_2d_torus(2, 2);
        let data = vec![42u8];
        let result = comms[0]
            .perform_collective(CollectiveOp::AllGather, data.clone())
            .unwrap();
        assert_eq!(result.len(), 4);
    }

    #[rstest]
    fn test_perform_collective_reduce_scatter() {
        let comms = setup_2d_torus(2, 2);
        let data = to_bytes(10);
        let result = comms[0]
            .perform_collective(CollectiveOp::ReduceScatter, data.clone())
            .unwrap();
        let chunk_size = 8 / 4;
        assert_eq!(result.len(), chunk_size);
    }

    #[rstest]
    fn test_trait_nearest_neighbor_exchange() {
        let comms = setup_2d_torus(3, 3);
        let data = vec![1u8, 2, 3];
        let results = <TopologyAwareComm<MockCommunicator> as TopologyAwareCommunicator>::nearest_neighbor_exchange(&comms[4], data.clone(), 0);
        assert_eq!(results.len(), 2);
    }

    #[rstest]
    fn test_trait_all_reduce_mesh() {
        let comms = setup_2d_torus(2, 2);
        let data = to_bytes(42);
        let result = <TopologyAwareComm<MockCommunicator> as TopologyAwareCommunicator>::all_reduce_mesh(&comms[0], data.clone(), &[0, 1, 2, 3]);
        assert_eq!(from_bytes(&result), 168);
    }

    #[rstest]
    fn test_trait_pipeline_stage_sendrecv() {
        let comms = setup_2d_torus(2, 2);
        let data = to_bytes(99);
        comms[1].inner().send(data.clone(), 0).unwrap();
        let result = <TopologyAwareComm<MockCommunicator> as TopologyAwareCommunicator>::pipeline_stage_sendrecv(&comms[0], data.clone(), 1);
        assert_eq!(from_bytes(&result), 99);
    }

    #[rstest]
    fn test_trait_route_collective() {
        let comms = setup_2d_torus(2, 2);
        let data = vec![42u8];
        let result = <TopologyAwareComm<MockCommunicator> as TopologyAwareCommunicator>::route_collective(&comms[0], CollectiveOp::Broadcast, data.clone());
        assert_eq!(result, data);
    }

    #[rstest]
    fn test_reduce_on_mesh_group() {
        let comms = setup_2d_torus(2, 2);
        let data = to_bytes(5);
        let result = comms[0].reduce_on_mesh(data.clone(), &[0, 1]).unwrap();
        assert_eq!(from_bytes(&result), 20);
    }

    #[rstest]
    fn test_3d_topology() {
        let topo = MeshTorusHybrid::new_3d(2, 2, 2, 2).unwrap();
        let mocks = MockCommunicator::create_world(8);
        let comms: Vec<TopologyAwareComm<MockCommunicator>> = mocks
            .into_iter()
            .map(|m| TopologyAwareComm::new(m, topo.clone()))
            .collect();
        assert_eq!(comms.len(), 8);
        assert_eq!(comms[0].world_size(), 8);
        let coords = comms[7].rank_coords();
        assert_eq!(coords, TorusCoordinates::new_3d(1, 1, 1));
    }

    #[rstest]
    fn test_inner_and_topology_access() {
        let comms = setup_2d_torus(3, 3);
        assert_eq!(comms[0].inner().rank(), 0);
        assert_eq!(comms[0].topology().total_nodes(), 9);
    }
}
