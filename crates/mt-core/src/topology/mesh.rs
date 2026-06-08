use crate::topology::device::GPUDevice;

#[derive(Debug, Clone, PartialEq)]
pub struct IntraNodeMesh {
    pub devices: Vec<GPUDevice>,
}

impl IntraNodeMesh {
    pub fn new(devices: Vec<GPUDevice>) -> anyhow::Result<Self> {
        let device_count = devices.len();
        if device_count == 0 {
            anyhow::bail!("IntraNodeMesh requires at least one device");
        }
        for dev in &devices {
            if dev.id >= device_count {
                anyhow::bail!(
                    "Device id {} exceeds device count {}",
                    dev.id,
                    device_count
                );
            }
        }
        Ok(Self { devices })
    }

    pub fn full_mesh(device_count: usize) -> anyhow::Result<Self> {
        if device_count == 0 {
            anyhow::bail!("IntraNodeMesh requires at least one device");
        }
        let devices: Vec<GPUDevice> = (0..device_count)
            .map(|id| {
                let peers: Vec<usize> = (0..device_count).filter(|&p| p != id).collect();
                GPUDevice::new(id, None, format!("0000:0{}:00.0", id), peers)
            })
            .collect();
        Ok(Self { devices })
    }

    pub fn device(&self, id: usize) -> Option<&GPUDevice> {
        self.devices.iter().find(|d| d.id == id)
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    pub fn are_nvlinked(&self, a: usize, b: usize) -> bool {
        match self.device(a) {
            Some(dev) => dev.is_nvlink_peer(b),
            None => false,
        }
    }

    pub fn all_pairs_nvlinked(&self) -> bool {
        for i in 0..self.devices.len() {
            for j in 0..self.devices.len() {
                if i != j && !self.are_nvlinked(i, j) {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    fn test_full_mesh_4() {
        let mesh = IntraNodeMesh::full_mesh(4).unwrap();
        assert_eq!(mesh.device_count(), 4);
        assert!(mesh.all_pairs_nvlinked());
    }

    #[rstest]
    fn test_full_mesh_8() {
        let mesh = IntraNodeMesh::full_mesh(8).unwrap();
        assert_eq!(mesh.device_count(), 8);
        assert!(mesh.all_pairs_nvlinked());
    }

    #[rstest]
    fn test_full_mesh_1() {
        let mesh = IntraNodeMesh::full_mesh(1).unwrap();
        assert_eq!(mesh.device_count(), 1);
        assert!(mesh.all_pairs_nvlinked());
    }

    #[rstest]
    fn test_empty_mesh_fails() {
        assert!(IntraNodeMesh::full_mesh(0).is_err());
    }

    #[rstest]
    fn test_custom_mesh() {
        let devices = vec![
            GPUDevice::new(0, Some(0), "0000:01:00.0", vec![1, 2]),
            GPUDevice::new(1, Some(0), "0000:02:00.0", vec![0, 2]),
            GPUDevice::new(2, Some(1), "0000:03:00.0", vec![0, 1]),
        ];
        let mesh = IntraNodeMesh::new(devices).unwrap();
        assert_eq!(mesh.device_count(), 3);
        assert!(mesh.are_nvlinked(0, 1));
        assert!(mesh.are_nvlinked(1, 2));
        assert!(mesh.are_nvlinked(0, 2));
    }

    #[rstest]
    fn test_device_lookup() {
        let mesh = IntraNodeMesh::full_mesh(4).unwrap();
        let dev = mesh.device(2).unwrap();
        assert_eq!(dev.id, 2);
        assert_eq!(dev.nvlink_peers, vec![0, 1, 3]);
    }

    #[rstest]
    fn test_device_not_found() {
        let mesh = IntraNodeMesh::full_mesh(2).unwrap();
        assert!(mesh.device(5).is_none());
    }

    #[rstest]
    fn test_invalid_device_id_fails() {
        let devices = vec![
            GPUDevice::new(0, None, "0000:00:00.0", vec![1]),
            GPUDevice::new(5, None, "0000:01:00.0", vec![0]),
        ];
        assert!(IntraNodeMesh::new(devices).is_err());
    }
}
