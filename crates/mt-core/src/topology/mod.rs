pub mod coordinator;
pub mod device;
pub mod discovery;
pub mod hybrid;
pub mod mesh;
pub mod torus;

pub use coordinator::{CollectiveOp, DummyCommunicator, TopologyAwareCommunicator};
pub use device::GPUDevice;
pub use discovery::{DiscoveryConfig, discover_devices, discover_mesh};
pub use hybrid::MeshTorusHybrid;
pub use mesh::IntraNodeMesh;
pub use torus::{TorusCoordinates, TorusDimensions};
