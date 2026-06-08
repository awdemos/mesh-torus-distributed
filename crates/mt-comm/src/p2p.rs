use std::sync::Arc;

use parking_lot::Mutex;

#[derive(Debug)]
pub struct P2PHandle {
    inner: Arc<Mutex<Option<Vec<u8>>>>,
}

impl Default for P2PHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl P2PHandle {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(None)),
        }
    }

    pub fn set_result(&self, data: Vec<u8>) {
        *self.inner.lock() = Some(data);
    }

    pub fn wait(self) -> Vec<u8> {
        loop {
            {
                let guard = self.inner.lock();
                if guard.is_some() {
                    return guard.clone().unwrap();
                }
            }
            std::thread::yield_now();
        }
    }

    pub fn try_get(&self) -> Option<Vec<u8>> {
        self.inner.lock().clone()
    }
}

#[derive(Debug)]
pub struct P2PChannel {
    rank: usize,
    world_size: usize,
    buffers: Arc<Mutex<Vec<Vec<Vec<u8>>>>>,
}

impl P2PChannel {
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self {
            rank,
            world_size,
            buffers: Arc::new(Mutex::new(vec![vec![Vec::new(); world_size]; world_size])),
        }
    }

    pub fn create_group(world_size: usize) -> Vec<Self> {
        let buffers = Arc::new(Mutex::new(vec![vec![Vec::new(); world_size]; world_size]));
        (0..world_size)
            .map(|rank| Self {
                rank,
                world_size,
                buffers: buffers.clone(),
            })
            .collect()
    }

    pub fn isend(&self, data: Vec<u8>, dst: usize) -> P2PHandle {
        let handle = P2PHandle::new();
        {
            let mut buffers = self.buffers.lock();
            buffers[self.rank][dst] = data.clone();
        }
        handle.set_result(data);
        handle
    }

    pub fn irecv(&self, src: usize) -> P2PHandle {
        let handle = P2PHandle::new();
        let data = {
            let buffers = self.buffers.lock();
            buffers[src][self.rank].clone()
        };
        handle.set_result(data);
        handle
    }

    pub fn sendrecv(&self, send_data: Vec<u8>, dst: usize, src: usize) -> (P2PHandle, P2PHandle) {
        let send_handle = self.isend(send_data, dst);
        let recv_handle = self.irecv(src);
        (send_handle, recv_handle)
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }
}

#[derive(Debug)]
pub struct AsyncP2PChannel {
    rank: usize,
    world_size: usize,
    buffers: Arc<tokio::sync::Mutex<Vec<Vec<Vec<u8>>>>>,
}

impl AsyncP2PChannel {
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self {
            rank,
            world_size,
            buffers: Arc::new(tokio::sync::Mutex::new(vec![
                vec![Vec::new(); world_size];
                world_size
            ])),
        }
    }

    pub fn create_group(world_size: usize) -> Vec<Self> {
        let buffers = Arc::new(tokio::sync::Mutex::new(vec![
            vec![Vec::new(); world_size];
            world_size
        ]));
        (0..world_size)
            .map(|rank| Self {
                rank,
                world_size,
                buffers: buffers.clone(),
            })
            .collect()
    }

    pub async fn isend(&self, data: Vec<u8>, dst: usize) -> P2PHandle {
        {
            let mut buffers = self.buffers.lock().await;
            buffers[self.rank][dst] = data.clone();
        }
        let handle = P2PHandle::new();
        handle.set_result(data);
        handle
    }

    pub async fn irecv(&self, src: usize) -> P2PHandle {
        let data = {
            let buffers = self.buffers.lock().await;
            buffers[src][self.rank].clone()
        };
        let handle = P2PHandle::new();
        handle.set_result(data);
        handle
    }

    pub async fn sendrecv(
        &self,
        send_data: Vec<u8>,
        dst: usize,
        src: usize,
    ) -> (P2PHandle, P2PHandle) {
        let send_handle = self.isend(send_data, dst).await;
        let recv_handle = self.irecv(src).await;
        (send_handle, recv_handle)
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    fn test_p2p_isend_irecv() {
        let channels = P2PChannel::create_group(2);
        let data = vec![1u8, 2, 3, 4];
        channels[0].isend(data.clone(), 1);
        let handle = channels[1].irecv(0);
        let result = handle.wait();
        assert_eq!(result, data);
    }

    #[rstest]
    fn test_p2p_sendrecv() {
        let channels = P2PChannel::create_group(2);
        let data_a = vec![10u8, 20];
        let data_b = vec![30u8, 40];
        channels[0].isend(data_a.clone(), 1);
        channels[1].isend(data_b.clone(), 0);
        let recv_a = channels[0].irecv(1).wait();
        let recv_b = channels[1].irecv(0).wait();
        assert_eq!(recv_a, data_b);
        assert_eq!(recv_b, data_a);
    }

    #[rstest]
    fn test_p2p_multi_rank() {
        let channels = P2PChannel::create_group(4);
        let data = vec![42u8];
        channels[0].isend(data.clone(), 2);
        let result = channels[2].irecv(0).wait();
        assert_eq!(result, data);
    }

    #[rstest]
    fn test_p2p_handle_try_get() {
        let handle = P2PHandle::new();
        assert!(handle.try_get().is_none());
        handle.set_result(vec![1u8, 2, 3]);
        assert_eq!(handle.try_get(), Some(vec![1u8, 2, 3]));
    }

    #[rstest]
    fn test_p2p_rank_world_size() {
        let channels = P2PChannel::create_group(4);
        assert_eq!(channels[0].rank(), 0);
        assert_eq!(channels[3].rank(), 3);
        assert_eq!(channels[0].world_size(), 4);
    }

    #[tokio::test]
    async fn test_async_p2p_isend_irecv() {
        let channels = AsyncP2PChannel::create_group(2);
        let data = vec![5u8, 6, 7, 8];
        channels[0].isend(data.clone(), 1).await;
        let handle = channels[1].irecv(0).await;
        let result = handle.wait();
        assert_eq!(result, data);
    }

    #[tokio::test]
    async fn test_async_p2p_sendrecv() {
        let channels = AsyncP2PChannel::create_group(2);
        let data_a = vec![100u8];
        let data_b = vec![200u8];
        channels[0].isend(data_a.clone(), 1).await;
        channels[1].isend(data_b.clone(), 0).await;
        let recv_a = channels[0].irecv(1).await.wait();
        let recv_b = channels[1].irecv(0).await.wait();
        assert_eq!(recv_a, data_b);
        assert_eq!(recv_b, data_a);
    }

    #[tokio::test]
    async fn test_async_p2p_rank() {
        let channels = AsyncP2PChannel::create_group(4);
        assert_eq!(channels[0].rank(), 0);
        assert_eq!(channels[0].world_size(), 4);
    }
}
