use mt_core::topology::{MeshTorusHybrid, TorusCoordinates};

pub fn topology_aware_placement(
    num_stages: usize,
    topology: &MeshTorusHybrid,
) -> Vec<(usize, TorusCoordinates)> {
    let mut placement = Vec::with_capacity(num_stages);
    for stage_id in 0..num_stages {
        let coords = topology.rank_to_coords(stage_id);
        placement.push((stage_id, coords));
    }
    placement
}

pub fn compute_hop_distance(
    from: &TorusCoordinates,
    to: &TorusCoordinates,
    topology: &MeshTorusHybrid,
) -> usize {
    let x_dist = torus_distance(from.x, to.x, topology.dims.width);
    let y_dist = torus_distance(from.y, to.y, topology.dims.height);
    let z_dist = match (from.z, to.z, topology.dims.depth) {
        (Some(fz), Some(tz), Some(depth)) => torus_distance(fz, tz, depth),
        _ => 0,
    };
    x_dist + y_dist + z_dist
}

fn torus_distance(a: usize, b: usize, size: usize) -> usize {
    let forward = b.abs_diff(a);
    let backward = size - forward;
    forward.min(backward)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    fn test_placement_linear_2d() {
        let topo = MeshTorusHybrid::new_2d(4, 1, 2).unwrap();
        let placement = topology_aware_placement(4, &topo);
        assert_eq!(placement.len(), 4);
        for (i, (stage_id, _coords)) in placement.iter().enumerate() {
            assert_eq!(*stage_id, i);
        }
    }

    #[rstest]
    fn test_placement_3d() {
        let topo = MeshTorusHybrid::new_3d(2, 2, 2, 4).unwrap();
        let placement = topology_aware_placement(8, &topo);
        assert_eq!(placement.len(), 8);
    }

    #[rstest]
    fn test_placement_adjacent_stages_adjacent_nodes() {
        let topo = MeshTorusHybrid::new_2d(4, 1, 2).unwrap();
        let placement = topology_aware_placement(4, &topo);
        for i in 0..placement.len() - 1 {
            let (_, from) = &placement[i];
            let (_, to) = &placement[i + 1];
            let dist = compute_hop_distance(from, to, &topo);
            assert_eq!(
                dist, 1,
                "adjacent stages should be 1 hop apart, got {} for stage {}",
                dist, i
            );
        }
    }

    #[rstest]
    fn test_torus_distance_same() {
        assert_eq!(torus_distance(2, 2, 5), 0);
    }

    #[rstest]
    fn test_torus_distance_forward() {
        assert_eq!(torus_distance(0, 2, 5), 2);
    }

    #[rstest]
    fn test_torus_distance_wrap() {
        assert_eq!(torus_distance(0, 4, 5), 1);
    }

    #[rstest]
    fn test_compute_hop_distance_2d() {
        let topo = MeshTorusHybrid::new_2d(4, 4, 2).unwrap();
        let a = TorusCoordinates::new_2d(0, 0);
        let b = TorusCoordinates::new_2d(1, 1);
        assert_eq!(compute_hop_distance(&a, &b, &topo), 2);
    }

    #[rstest]
    fn test_compute_hop_distance_3d() {
        let topo = MeshTorusHybrid::new_3d(4, 4, 4, 2).unwrap();
        let a = TorusCoordinates::new_3d(0, 0, 0);
        let b = TorusCoordinates::new_3d(1, 1, 1);
        assert_eq!(compute_hop_distance(&a, &b, &topo), 3);
    }

    #[rstest]
    fn test_placement_fewer_stages_than_nodes() {
        let topo = MeshTorusHybrid::new_2d(4, 4, 2).unwrap();
        let placement = topology_aware_placement(3, &topo);
        assert_eq!(placement.len(), 3);
    }
}
