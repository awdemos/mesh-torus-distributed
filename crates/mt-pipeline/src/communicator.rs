use mt_comm::Communicator;
use mt_core::precision::{from_fp8, to_fp8, Fp8Format};

use crate::stage::StageData;

pub struct PipelineComm<C: Communicator> {
    comm: C,
    fp8_format: Fp8Format,
}

impl<C: Communicator> PipelineComm<C> {
    pub fn new(comm: C) -> Self {
        Self {
            comm,
            fp8_format: Fp8Format::E4M3,
        }
    }

    pub fn with_fp8_format(mut self, format: Fp8Format) -> Self {
        self.fp8_format = format;
        self
    }

    pub fn send_activations(&self, data: &StageData, dst_stage: usize) -> anyhow::Result<()> {
        let bytes = self.compress_stage_data(data);
        self.comm.send(bytes, dst_stage)
    }

    pub fn recv_activations(&self, src_stage: usize) -> anyhow::Result<StageData> {
        let bytes = self.comm.recv(src_stage)?;
        Ok(self.decompress_stage_data(&bytes))
    }

    pub fn send_gradients(&self, data: &StageData, dst_stage: usize) -> anyhow::Result<()> {
        let bytes = self.compress_stage_data(data);
        self.comm.send(bytes, dst_stage)
    }

    pub fn recv_gradients(&self, src_stage: usize) -> anyhow::Result<StageData> {
        let bytes = self.comm.recv(src_stage)?;
        Ok(self.decompress_stage_data(&bytes))
    }

    fn compress_stage_data(&self, data: &StageData) -> Vec<u8> {
        let fp8_tensor = to_fp8(&data.values, self.fp8_format);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(data.rows as u64).to_le_bytes());
        bytes.extend_from_slice(&(data.cols as u64).to_le_bytes());
        bytes.push(self.fp8_format as u8);
        let scale_bytes = fp8_tensor.scale.to_le_bytes();
        bytes.extend_from_slice(&scale_bytes);
        bytes.extend_from_slice(&(fp8_tensor.data.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&fp8_tensor.data);
        bytes
    }

    fn decompress_stage_data(&self, bytes: &[u8]) -> StageData {
        let rows = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let cols = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let format_byte = bytes[16];
        let format = if format_byte == 1 {
            Fp8Format::E5M2
        } else {
            Fp8Format::E4M3
        };
        let scale = f32::from_le_bytes(bytes[17..21].try_into().unwrap());
        let data_len = u64::from_le_bytes(bytes[21..29].try_into().unwrap()) as usize;
        let data = bytes[29..29 + data_len].to_vec();

        let fp8_tensor = mt_core::precision::Fp8Tensor::new(data, format, scale);
        let values = from_fp8(&fp8_tensor);
        StageData { values, rows, cols }
    }

    pub fn rank(&self) -> usize {
        self.comm.rank()
    }

    pub fn world_size(&self) -> usize {
        self.comm.world_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mt_comm::MockCommunicator;
    use rstest::*;

    fn make_pipe_comms(n: usize) -> Vec<PipelineComm<MockCommunicator>> {
        MockCommunicator::create_world(n)
            .into_iter()
            .map(PipelineComm::new)
            .collect()
    }

    #[rstest]
    fn test_send_recv_activations() {
        let pipe_comms = make_pipe_comms(2);

        let data = StageData::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        pipe_comms[0].send_activations(&data, 1).unwrap();
        let received = pipe_comms[1].recv_activations(0).unwrap();

        assert_eq!(received.rows, 2);
        assert_eq!(received.cols, 2);
        for (orig, rec) in data.values.iter().zip(received.values.iter()) {
            assert!((orig - rec).abs() < 0.2, "orig={orig}, rec={rec}");
        }
    }

    #[rstest]
    fn test_send_recv_gradients() {
        let pipe_comms = make_pipe_comms(2);

        let data = StageData::new(vec![0.5, -0.5], 1, 2);
        pipe_comms[0].send_gradients(&data, 1).unwrap();
        let received = pipe_comms[1].recv_gradients(0).unwrap();

        assert_eq!(received.rows, 1);
        assert_eq!(received.cols, 2);
        for (orig, rec) in data.values.iter().zip(received.values.iter()) {
            assert!((orig - rec).abs() < 0.2, "orig={orig}, rec={rec}");
        }
    }

    #[rstest]
    fn test_fp8_format_config() {
        let comms = MockCommunicator::create_world(1);
        let pipe_comm =
            PipelineComm::new(comms.into_iter().next().unwrap()).with_fp8_format(Fp8Format::E5M2);
        assert_eq!(pipe_comm.fp8_format, Fp8Format::E5M2);
    }

    #[rstest]
    fn test_rank_and_world_size() {
        let comms = MockCommunicator::create_world(4);
        let pipe_comm = PipelineComm::new(comms.into_iter().nth(2).unwrap());
        assert_eq!(pipe_comm.rank(), 2);
        assert_eq!(pipe_comm.world_size(), 4);
    }

    #[rstest]
    fn test_compression_decompression_roundtrip() {
        let comms = MockCommunicator::create_world(1);
        let pipe_comm = PipelineComm::new(comms.into_iter().next().unwrap());

        let data = StageData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let compressed = pipe_comm.compress_stage_data(&data);
        let decompressed = pipe_comm.decompress_stage_data(&compressed);

        assert_eq!(decompressed.rows, 2);
        assert_eq!(decompressed.cols, 3);
        assert_eq!(decompressed.values.len(), 6);
    }
}
