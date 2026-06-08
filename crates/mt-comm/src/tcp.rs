use std::collections::HashMap;
use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

use crate::collective::{Communicator, ReduceOp};

const LEN_PREFIX_SIZE: usize = 8;

#[derive(Debug)]
pub struct TcpCommunicator {
    rank: usize,
    world_size: usize,
    connections: Arc<Mutex<HashMap<usize, TcpStream>>>,
    listener_port: u16,
}

impl TcpCommunicator {
    pub async fn new(rank: usize, addresses: Vec<String>) -> anyhow::Result<Self> {
        let world_size = addresses.len();
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let listener_port = listener.local_addr()?.port();

        let connections: Arc<Mutex<HashMap<usize, TcpStream>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let conns_clone = connections.clone();
        tokio::spawn(async move {
            loop {
                let (mut stream, _) = match listener.accept().await {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let mut rank_buf = [0u8; 8];
                if stream.read_exact(&mut rank_buf).await.is_err() {
                    continue;
                }
                let src_rank = u64::from_le_bytes(rank_buf) as usize;
                let mut conns = conns_clone.lock().await;
                conns.insert(src_rank, stream);
            }
        });

        let comm = Self {
            rank,
            world_size,
            connections,
            listener_port,
        };

        for (dst, addr) in addresses.iter().enumerate() {
            if dst == rank {
                continue;
            }
            if dst > rank {
                if let Ok(mut stream) = TcpStream::connect(addr).await {
                    let rank_bytes = (rank as u64).to_le_bytes();
                    let _ = stream.write_all(&rank_bytes).await;
                    comm.connections.lock().await.insert(dst, stream);
                }
            }
        }

        Ok(comm)
    }

    pub fn port(&self) -> u16 {
        self.listener_port
    }

    pub async fn send_async(&self, data: Vec<u8>, dst: usize) -> anyhow::Result<()> {
        let mut conns = self.connections.lock().await;
        let stream = conns
            .get_mut(&dst)
            .ok_or_else(|| anyhow::anyhow!("no connection to rank {}", dst))?;
        send_frame(stream, &data).await
    }

    pub async fn recv_async(&self, src: usize) -> anyhow::Result<Vec<u8>> {
        let mut conns = self.connections.lock().await;
        let stream = conns
            .get_mut(&src)
            .ok_or_else(|| anyhow::anyhow!("no connection from rank {}", src))?;
        recv_frame(stream).await
    }

    pub async fn connect_to(&self, dst: usize, addr: &str) -> anyhow::Result<()> {
        let stream = TcpStream::connect(addr).await?;
        self.connections.lock().await.insert(dst, stream);
        Ok(())
    }
}

async fn send_frame(stream: &mut TcpStream, data: &[u8]) -> anyhow::Result<()> {
    let len = (data.len() as u64).to_le_bytes();
    stream.write_all(&len).await?;
    stream.write_all(data).await?;
    stream.flush().await?;
    Ok(())
}

pub async fn recv_frame(stream: &mut TcpStream) -> anyhow::Result<Vec<u8>> {
    let mut len_buf = [0u8; LEN_PREFIX_SIZE];
    stream.read_exact(&mut len_buf).await?;
    let len = u64::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).await?;
    Ok(buf)
}

impl Communicator for TcpCommunicator {
    fn send(&self, _data: Vec<u8>, _dst: usize) -> anyhow::Result<()> {
        anyhow::bail!("TcpCommunicator requires async runtime; use send_async/recv_async")
    }

    fn recv(&self, _src: usize) -> anyhow::Result<Vec<u8>> {
        anyhow::bail!("TcpCommunicator requires async runtime; use send_async/recv_async")
    }

    fn all_reduce(&self, _data: Vec<u8>, _op: ReduceOp) -> anyhow::Result<Vec<u8>> {
        anyhow::bail!("TcpCommunicator all_reduce requires async coordination")
    }

    fn broadcast(&self, _data: Vec<u8>, _root: usize) -> anyhow::Result<Vec<u8>> {
        anyhow::bail!("TcpCommunicator broadcast requires async coordination")
    }

    fn all_gather(&self, _data: Vec<u8>) -> anyhow::Result<Vec<Vec<u8>>> {
        anyhow::bail!("TcpCommunicator all_gather requires async coordination")
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
    use tokio::net::{TcpListener, TcpStream};

    #[tokio::test]
    async fn test_tcp_roundtrip() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let data = recv_frame(&mut stream).await.unwrap();
            send_frame(&mut stream, &data).await.unwrap();
        });

        let client = tokio::spawn(async move {
            let mut stream = TcpStream::connect(server_addr).await.unwrap();
            let msg = b"hello world".to_vec();
            send_frame(&mut stream, &msg).await.unwrap();
            let reply = recv_frame(&mut stream).await.unwrap();
            assert_eq!(reply, msg);
        });

        server.await.unwrap();
        client.await.unwrap();
    }

    #[tokio::test]
    async fn test_tcp_large_payload() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let data = recv_frame(&mut stream).await.unwrap();
            send_frame(&mut stream, &data).await.unwrap();
        });

        let client = tokio::spawn(async move {
            let mut stream = TcpStream::connect(server_addr).await.unwrap();
            let msg: Vec<u8> = (0..65_536).map(|i| (i % 256) as u8).collect();
            send_frame(&mut stream, &msg).await.unwrap();
            let reply = recv_frame(&mut stream).await.unwrap();
            assert_eq!(reply.len(), 65_536);
            assert_eq!(reply, msg);
        });

        server.await.unwrap();
        client.await.unwrap();
    }

    #[tokio::test]
    async fn test_tcp_multiple_messages() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            for _ in 0..5 {
                let data = recv_frame(&mut stream).await.unwrap();
                let reply: Vec<u8> = data.iter().map(|&b| b.wrapping_add(1)).collect();
                send_frame(&mut stream, &reply).await.unwrap();
            }
        });

        let client = tokio::spawn(async move {
            let mut stream = TcpStream::connect(server_addr).await.unwrap();
            for i in 0..5u8 {
                let msg = vec![i; 10];
                send_frame(&mut stream, &msg).await.unwrap();
                let reply = recv_frame(&mut stream).await.unwrap();
                let expected: Vec<u8> = vec![i.wrapping_add(1); 10];
                assert_eq!(reply, expected);
            }
        });

        server.await.unwrap();
        client.await.unwrap();
    }

    #[tokio::test]
    async fn test_tcp_communicator_rank() {
        let comm = TcpCommunicator::new(
            2,
            vec![
                "127.0.0.1:0".to_string(),
                "127.0.0.1:0".to_string(),
                "127.0.0.1:0".to_string(),
            ],
        )
        .await
        .unwrap();
        assert_eq!(comm.rank(), 2);
        assert_eq!(comm.world_size(), 3);
    }

    #[tokio::test]
    async fn test_tcp_empty_payload() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let data = recv_frame(&mut stream).await.unwrap();
            assert!(data.is_empty());
            send_frame(&mut stream, &data).await.unwrap();
        });

        let client = tokio::spawn(async move {
            let mut stream = TcpStream::connect(server_addr).await.unwrap();
            let msg = Vec::new();
            send_frame(&mut stream, &msg).await.unwrap();
            let reply = recv_frame(&mut stream).await.unwrap();
            assert!(reply.is_empty());
        });

        server.await.unwrap();
        client.await.unwrap();
    }
}
