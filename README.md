# Mesh-Torus Hybrid Distributed Training

A Rust framework for distributed deep learning using mesh-torus hybrid topologies,
built on top of the [Burn](https://github.com/tracel-ai/burn) deep learning framework.

## Features

- **Mesh-Torus Hybrid Topology** — Combines intra-node NVLink full-mesh with
  inter-node torus routing for optimal communication patterns.
- **FP8 Mixed Precision** — E4M3 forward pass, E5M2 backward pass with delayed
  scaling, dynamic/static loss scaling, and overflow detection.
- **Pipeline Parallelism** — GPipe and 1F1B schedules with activation
  checkpointing and topology-aware stage placement.
- **Topology-Aware Communication** — Nearest-neighbor exchange, collective
  operations (all-reduce, broadcast, all-gather) with mock, TCP, and NCCL backends.
- **Burn Integration** — Seamless integration with Burn's `Module`, `Optimizer`,
  and `Autodiff` traits.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Orchestration                   │
│                   (mt-training crate)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Distributed  │  │   Gradient   │  │   Training Step  │  │
│  │   Runtime    │  │  Accumulator │  │      Logic       │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Pipeline Parallelism                       │
│                   (mt-pipeline crate)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Pipeline   │  │   Pipeline   │  │   GPipe / 1F1B   │  │
│  │    Stage     │  │   Executor   │  │     Schedule     │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Communication Layer                         │
│                    (mt-comm crate)                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Mock/TCP   │  │  Collectives │  │ Topology-Aware   │  │
│  │ Communicator │  │  (all-reduce)│  │   Exchange       │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Core Primitives                         │
│                     (mt-core crate)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   FP8 Tensor │  │ Mesh/Torus   │  │   Loss Scaling   │  │
│  │   (E4M3/E5M2)│  │   Topology   │  │   & Quantization │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Workspace Crates

| Crate | Description | Tests |
|-------|-------------|-------|
| `mt-core` | Topology (MeshTorusHybrid, TorusCoordinates, GPUDevice) + FP8 precision (Fp8Tensor, DelayedScaling, LossScaler) | 109 |
| `mt-comm` | Communication primitives (Communicator trait, MockCommunicator, TcpCommunicator, collectives) | 47 |
| `mt-pipeline` | Pipeline parallelism (PipelineStage, GPipe/1F1B schedules, checkpointing, placement) | 41 |
| `mt-burn` | Burn integration (Fp8Optimizer, PipelineModule, fp8_to_burn/burn_to_fp8) | 29 |
| `mt-training` | Training orchestration (GradientAccumulator, TrainingConfig, DistributedRuntime, train_step) | 65 |
| `gpt-1b` | Example: GPT-1B model with distributed training loop | 22 |

**Total: 313 tests passing**

## Quick Start

### Run the GPT-1B Example

```sh
cargo run --bin gpt-1b
```

This runs a small GPT model (2 layers, d_model=128) on synthetic data with the
mesh-torus framework in single-process mock mode.

### Run All Tests

```sh
cargo test --workspace
```

### Run a Specific Crate's Tests

```sh
cargo test -p mt-core      # FP8 + topology
cargo test -p mt-comm      # Communication
cargo test -p mt-pipeline  # Pipeline parallelism
cargo test -p mt-training  # Training orchestration
cargo test -p mt-burn      # Burn integration
cargo test -p gpt-1b       # Example
```

## Crate Overview

### mt-core — Core Primitives

```rust
use mt_core::precision::{Fp8Tensor, Fp8Format, MixedPrecisionConfig, LossScaler};
use mt_core::topology::{MeshTorusHybrid, GPUDevice, TorusCoordinates};

// FP8 quantization
let config = MixedPrecisionConfig::default();
let scaler = LossScaler::new(config.loss_scale_config);

// Topology construction
let topology = MeshTorusHybrid::new_2d_torus(4, 4);
let neighbors = topology.nearest_neighbors(&TorusCoordinates::new(1, 2, 0));
```

### mt-comm — Communication Layer

```rust
use mt_comm::{MockCommunicator, all_reduce, CollectiveOp};

// Mock communicator for testing
let comm = MockCommunicator::new(4, 0); // 4 ranks, rank 0
let mut data = vec![1.0, 2.0, 3.0];
all_reduce(&comm, &mut data, CollectiveOp::Sum).unwrap();
```

### mt-pipeline — Pipeline Parallelism

```rust
use mt_pipeline::{PipelineStage, PipelineExecutor, Schedule};
use mt_pipeline::schedule::OneF1B;

// Create pipeline with 1F1B schedule
let schedule = Schedule::OneF1B;
let executor = PipelineExecutor::new(stages, schedule, communicator);
```

### mt-training — Training Orchestration

```rust
use mt_training::{TrainingConfig, DistributedRuntime, GradientAccumulator};

// Build distributed runtime
let runtime = DistributedRuntime::builder()
    .with_topology(topology)
    .with_communicator(communicator)
    .with_config(config)
    .build()?;

// Accumulate gradients across micro-batches
let mut accumulator = GradientAccumulator::new();
accumulator.accumulate(grads);
```

### mt-burn — Burn Integration

```rust
use mt_burn::{Fp8Optimizer, PipelineModule};
use burn::optim::AdamConfig;

// FP8-aware optimizer wrapping Adam
let inner = AdamConfig::new().init::<B, M>();
let optimizer = Fp8Optimizer::new(inner, fp8_config);
```

## Configuration

Training is configured via `TrainingConfig` (TOML-serializable):

```toml
batch_size = 32
micro_batch_size = 4
accumulation_steps = 8
learning_rate = 0.0001
epochs = 10
warmup_steps = 100

[mixed_precision]
forward_format = "E4M3"
backward_format = "E5M2"
loss_scale = { Dynamic = { initial = 128.0, growth_factor = 2.0, backoff_factor = 0.5, growth_interval = 2000 } }

[pipeline_schedule]
OneF1B = { num_stages = 8 }
```

## Development

### Prerequisites

- Rust 1.78+
- (Optional) CUDA for GPU backend
- (Optional) NVML for GPU topology auto-discovery

### Building

```sh
cargo check --workspace     # Fast check
cargo build --workspace     # Full build
cargo test --workspace      # Run all tests
cargo clippy --workspace    # Linting
```

## License

MIT OR Apache-2.0
