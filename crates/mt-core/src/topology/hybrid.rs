use crate::topology::mesh::IntraNodeMesh;
use crate::topology::torus::{TorusCoordinates, TorusDimensions};

#[derive(Debug, Clone)]
pub struct MeshTorusHybrid {
    pub dims: TorusDimensions,
    pub nodes: Vec<IntraNodeMesh>,
}

impl MeshTorusHybrid {
    pub fn new_2d(width: usize, height: usize, devices_per_node: usize) -> anyhow::Result<Self> {
        if width == 0 || height == 0 || devices_per_node == 0 {
            anyhow::bail!("width, height, and devices_per_node must be > 0");
        }
        let dims = TorusDimensions::new_2d(width, height);
        let nodes = (0..dims.total_nodes())
            .map(|_| IntraNodeMesh::full_mesh(devices_per_node))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { dims, nodes })
    }

    pub fn new_3d(
        width: usize,
        height: usize,
        depth: usize,
        devices_per_node: usize,
    ) -> anyhow::Result<Self> {
        if width == 0 || height == 0 || depth == 0 || devices_per_node == 0 {
            anyhow::bail!("width, height, depth, and devices_per_node must be > 0");
        }
        let dims = TorusDimensions::new_3d(width, height, depth);
        let nodes = (0..dims.total_nodes())
            .map(|_| IntraNodeMesh::full_mesh(devices_per_node))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { dims, nodes })
    }

    pub fn total_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn total_devices(&self) -> usize {
        self.nodes.iter().map(|n| n.device_count()).sum()
    }

    pub fn devices_per_node(&self) -> usize {
        self.nodes.first().map(|n| n.device_count()).unwrap_or(0)
    }

    pub fn node_at(&self, coords: &TorusCoordinates) -> Option<&IntraNodeMesh> {
        let idx = coords.linear_index(&self.dims);
        self.nodes.get(idx)
    }

    pub fn node_at_mut(&mut self, coords: &TorusCoordinates) -> Option<&mut IntraNodeMesh> {
        let idx = coords.linear_index(&self.dims);
        self.nodes.get_mut(idx)
    }

    pub fn global_device_id(&self, node_coords: &TorusCoordinates, local_id: usize) -> usize {
        let node_idx = node_coords.linear_index(&self.dims);
        node_idx * self.devices_per_node() + local_id
    }

    pub fn node_neighbors(&self, coords: &TorusCoordinates) -> Vec<(TorusCoordinates, IntraNodeMesh)> {
        coords
            .neighbors(&self.dims)
            .into_iter()
            .filter_map(|(neighbor, _)| {
                self.node_at(&neighbor).map(|mesh| (neighbor, mesh.clone()))
            })
            .collect()
    }

    pub fn node_rank(&self, coords: &TorusCoordinates) -> usize {
        coords.linear_index(&self.dims)
    }

    pub fn rank_to_coords(&self, rank: usize) -> TorusCoordinates {
        TorusCoordinates::from_linear(rank, &self.dims)
    }
}

impl PartialEq for MeshTorusHybrid {
    fn eq(&self, other: &Self) -> bool {
        self.dims == other.dims && self.nodes == other.nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    fn test_2d_hybrid() {
        let hybrid = MeshTorusHybrid::new_2d(3, 3, 4).unwrap();
        assert_eq!(hybrid.total_nodes(), 9);
        assert_eq!(hybrid.devices_per_node(), 4);
        assert_eq!(hybrid.total_devices(), 36);
    }

    #[rstest]
    fn test_3d_hybrid() {
        let hybrid = MeshTorusHybrid::new_3d(2, 2, 2, 8).unwrap();
        assert_eq!(hybrid.total_nodes(), 8);
        assert_eq!(hybrid.devices_per_node(), 8);
        assert_eq!(hybrid.total_devices(), 64);
    }

    #[rstest]
    fn test_2d_zero_fails() {
        assert!(MeshTorusHybrid::new_2d(0, 3, 4).is_err());
        assert!(MeshTorusHybrid::new_2d(3, 0, 4).is_err());
        assert!(MeshTorusHybrid::new_2d(3, 3, 0).is_err());
    }

    #[rstest]
    fn test_3d_zero_fails() {
        assert!(MeshTorusHybrid::new_3d(0, 2, 2, 4).is_err());
        assert!(MeshTorusHybrid::new_3d(2, 0, 2, 4).is_err());
        assert!(MeshTorusHybrid::new_3d(2, 2, 0, 4).is_err());
        assert!(MeshTorusHybrid::new_3d(2, 2, 2, 0).is_err());
    }

    #[rstest]
    fn test_node_at_2d() {
        let hybrid = MeshTorusHybrid::new_2d(3, 3, 4).unwrap();
        let coords = TorusCoordinates::new_2d(1, 2);
        let node = hybrid.node_at(&coords).unwrap();
        assert_eq!(node.device_count(), 4);
    }

    #[rstest]
    fn test_node_at_3d() {
        let hybrid = MeshTorusHybrid::new_3d(2, 3, 2, 4).unwrap();
        let coords = TorusCoordinates::new_3d(1, 2, 1);
        let node = hybrid.node_at(&coords).unwrap();
        assert_eq!(node.device_count(), 4);
    }

    #[rstest]
    fn test_global_device_id() {
        let hybrid = MeshTorusHybrid::new_2d(3, 3, 4).unwrap();
        let coords = TorusCoordinates::new_2d(1, 1);
        assert_eq!(hybrid.global_device_id(&coords, 0), 16);
        assert_eq!(hybrid.global_device_id(&coords, 3), 19);
    }

    #[rstest]
    fn test_node_neighbors_2d() {
        let hybrid = MeshTorusHybrid::new_2d(3, 3, 4).unwrap();
        let coords = TorusCoordinates::new_2d(1, 1);
        let neighbors = hybrid.node_neighbors(&coords);
        assert_eq!(neighbors.len(), 4);
        let neighbor_coords: Vec<_> = neighbors.iter().map(|(c, _)| *c).collect();
        assert!(neighbor_coords.contains(&TorusCoordinates::new_2d(2, 1)));
        assert!(neighbor_coords.contains(&TorusCoordinates::new_2d(0, 1)));
        assert!(neighbor_coords.contains(&TorusCoordinates::new_2d(1, 2)));
        assert!(neighbor_coords.contains(&TorusCoordinates::new_2d(1, 0)));
    }

    #[rstest]
    fn test_node_neighbors_3d() {
        let hybrid = MeshTorusHybrid::new_3d(3, 3, 3, 4).unwrap();
        let coords = TorusCoordinates::new_3d(1, 1, 1);
        let neighbors = hybrid.node_neighbors(&coords);
        assert_eq!(neighbors.len(), 6);
    }

    #[rstest]
    fn test_rank_roundtrip_2d() {
        let hybrid = MeshTorusHybrid::new_2d(4, 3, 2).unwrap();
        for rank in 0..hybrid.total_nodes() {
            let coords = hybrid.rank_to_coords(rank);
            assert_eq!(hybrid.node_rank(&coords), rank);
        }
    }

    #[rstest]
    fn test_rank_roundtrip_3d() {
        let hybrid = MeshTorusHybrid::new_3d(3, 4, 2, 2).unwrap();
        for rank in 0..hybrid.total_nodes() {
            let coords = hybrid.rank_to_coords(rank);
            assert_eq!(hybrid.node_rank(&coords), rank);
        }
    }

    #[rstest]
    fn test_2d_corner_neighbors_wrap() {
        let hybrid = MeshTorusHybrid::new_2d(3, 3, 2).unwrap();
        let corner = TorusCoordinates::new_2d(0, 0);
        let neighbors = hybrid.node_neighbors(&corner);
        assert_eq!(neighbors.len(), 4);
        let neighbor_coords: Vec<_> = neighbors.iter().map(|(c, _)| *c).collect();
        assert!(neighbor_coords.contains(&TorusCoordinates::new_2d(2, 0)));
        assert!(neighbor_coords.contains(&TorusCoordinates::new_2d(0, 2)));
    }
}
