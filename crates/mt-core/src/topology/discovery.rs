use crate::topology::device::GPUDevice;
use crate::topology::mesh::IntraNodeMesh;

pub struct DiscoveryConfig {
    pub assume_full_mesh: bool,
    pub num_devices_override: Option<usize>,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            assume_full_mesh: true,
            num_devices_override: None,
        }
    }
}

pub fn discover_devices(config: &DiscoveryConfig) -> anyhow::Result<Vec<GPUDevice>> {
    #[cfg(feature = "discovery")]
    {
        discover_nvml(config)
    }
    #[cfg(not(feature = "discovery"))]
    {
        discover_manual(config)
    }
}

pub fn discover_mesh(config: &DiscoveryConfig) -> anyhow::Result<IntraNodeMesh> {
    let devices = discover_devices(config)?;
    if config.assume_full_mesh {
        let count = devices.len();
        let full_mesh_devices: Vec<GPUDevice> = devices
            .into_iter()
            .map(|mut dev| {
                dev.nvlink_peers = (0..count).filter(|&p| p != dev.id).collect();
                dev
            })
            .collect();
        IntraNodeMesh::new(full_mesh_devices)
    } else {
        IntraNodeMesh::new(devices)
    }
}

#[cfg(feature = "discovery")]
fn discover_nvml(config: &DiscoveryConfig) -> anyhow::Result<Vec<GPUDevice>> {
    use nvml_wrapper::Nvml;
    let nvml = Nvml::init().map_err(|e| anyhow::anyhow!("NVML init failed: {}", e))?;
    let count = match config.num_devices_override {
        Some(n) => n,
        None => nvml.device_count().map_err(|e| anyhow::anyhow!("NVML device count failed: {}", e))? as usize,
    };
    let mut devices = Vec::with_capacity(count);
    for i in 0..count {
        let device = nvml.device_by_index(i as u32).map_err(|e| anyhow::anyhow!("NVML device {} failed: {}", i, e))?;
        let pci_info = device.pci_info().map_err(|e| anyhow::anyhow!("NVML pci_info failed: {}", e))?;
        let bus_id = pci_info.bus_id;
        let mut peers = Vec::new();
        if let Ok(links) = device.nvlink_utilization_counter(0, true, true) {
            let _ = links;
        }
        for j in 0..count {
            if j != i {
                peers.push(j);
            }
        }
        devices.push(GPUDevice::new(i, None, bus_id, peers));
    }
    Ok(devices)
}

fn discover_manual(config: &DiscoveryConfig) -> anyhow::Result<Vec<GPUDevice>> {
    let count = config.num_devices_override.unwrap_or(8);
    let devices: Vec<GPUDevice> = (0..count)
        .map(|id| {
            let peers: Vec<usize> = (0..count).filter(|&p| p != id).collect();
            GPUDevice::new(
                id,
                Some(id / (count / 2).max(1)),
                format!("0000:{:02x}:00.0", id + 1),
                peers,
            )
        })
        .collect();
    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    fn test_manual_discovery_default() {
        let config = DiscoveryConfig::default();
        let devices = discover_manual(&config).unwrap();
        assert_eq!(devices.len(), 8);
        assert_eq!(devices[0].id, 0);
        assert_eq!(devices[7].id, 7);
    }

    #[rstest]
    fn test_manual_discovery_custom_count() {
        let config = DiscoveryConfig {
            num_devices_override: Some(4),
            ..Default::default()
        };
        let devices = discover_manual(&config).unwrap();
        assert_eq!(devices.len(), 4);
        assert_eq!(devices[0].nvlink_peers, vec![1, 2, 3]);
    }

    #[rstest]
    fn test_discover_mesh_full() {
        let config = DiscoveryConfig {
            num_devices_override: Some(4),
            assume_full_mesh: true,
        };
        let mesh = discover_mesh(&config).unwrap();
        assert_eq!(mesh.device_count(), 4);
        assert!(mesh.all_pairs_nvlinked());
    }

    #[rstest]
    fn test_discover_mesh_custom_peers() {
        let config = DiscoveryConfig {
            num_devices_override: Some(3),
            assume_full_mesh: false,
        };
        let mesh = discover_mesh(&config).unwrap();
        assert_eq!(mesh.device_count(), 3);
    }

    #[rstest]
    fn test_discover_devices_without_nvml() {
        let config = DiscoveryConfig {
            num_devices_override: Some(2),
            ..Default::default()
        };
        let devices = discover_devices(&config).unwrap();
        assert_eq!(devices.len(), 2);
    }

    #[rstest]
    fn test_default_config() {
        let config = DiscoveryConfig::default();
        assert!(config.assume_full_mesh);
        assert!(config.num_devices_override.is_none());
    }

    #[rstest]
    fn test_numa_assignment() {
        let config = DiscoveryConfig {
            num_devices_override: Some(8),
            ..Default::default()
        };
        let devices = discover_manual(&config).unwrap();
        assert_eq!(devices[0].numa_node, Some(0));
        assert_eq!(devices[3].numa_node, Some(0));
        assert_eq!(devices[4].numa_node, Some(1));
        assert_eq!(devices[7].numa_node, Some(1));
    }
}
