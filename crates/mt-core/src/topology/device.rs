#[derive(Debug, Clone, PartialEq)]
pub struct GPUDevice {
    pub id: usize,
    pub numa_node: Option<usize>,
    pub pci_bus_id: String,
    pub nvlink_peers: Vec<usize>,
}

impl GPUDevice {
    pub fn new(id: usize, numa_node: Option<usize>, pci_bus_id: impl Into<String>, nvlink_peers: Vec<usize>) -> Self {
        Self {
            id,
            numa_node,
            pci_bus_id: pci_bus_id.into(),
            nvlink_peers,
        }
    }

    pub fn simple(id: usize, peer_count: usize) -> Self {
        let nvlink_peers = (0..peer_count).filter(|&p| p != id).collect();
        Self {
            id,
            numa_node: None,
            pci_bus_id: format!("0000:0{}:00.0", id),
            nvlink_peers,
        }
    }

    pub fn is_nvlink_peer(&self, other_id: usize) -> bool {
        self.nvlink_peers.contains(&other_id)
    }

    pub fn peer_count(&self) -> usize {
        self.nvlink_peers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    fn test_device_new() {
        let dev = GPUDevice::new(0, Some(1), "0000:01:00.0", vec![1, 2, 3]);
        assert_eq!(dev.id, 0);
        assert_eq!(dev.numa_node, Some(1));
        assert_eq!(dev.pci_bus_id, "0000:01:00.0");
        assert_eq!(dev.nvlink_peers, vec![1, 2, 3]);
    }

    #[rstest]
    fn test_device_simple() {
        let dev = GPUDevice::simple(2, 4);
        assert_eq!(dev.id, 2);
        assert_eq!(dev.nvlink_peers, vec![0, 1, 3]);
        assert!(dev.numa_node.is_none());
    }

    #[rstest]
    #[case(0, 1, true)]
    #[case(0, 3, true)]
    #[case(0, 4, false)]
    fn test_is_nvlink_peer(#[case] id: usize, #[case] peer: usize, #[case] expected: bool) {
        let dev = GPUDevice::new(id, None, "0000:00:00.0", vec![1, 2, 3]);
        assert_eq!(dev.is_nvlink_peer(peer), expected);
    }

    #[rstest]
    fn test_peer_count() {
        let dev = GPUDevice::new(0, None, "0000:00:00.0", vec![1, 2, 3]);
        assert_eq!(dev.peer_count(), 3);
    }
}
