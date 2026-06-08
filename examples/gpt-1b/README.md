# GPT-1B Distributed Training Example

A GPT-style language model trained with synthetic data using the
Mesh-Torus Distributed Training framework.

## Architecture

The model is a standard GPT decoder-only transformer:

```
Input tokens [batch, seq]
    │
    ├── Token Embedding (vocab_size → d_model)
    ├── Position Embedding (max_seq_len → d_model)
    ├── Dropout
    │
    ├── TransformerBlock × N
    │   ├── LayerNorm → MultiHeadAttention → residual
    │   └── LayerNorm → FeedForward (GELU) → residual
    │
    ├── Final LayerNorm
    └── LM Head (d_model → vocab_size) → Logits [batch, seq, vocab]
```

### ~1B Parameter Configuration

| Parameter    | Value  |
|-------------|--------|
| vocab_size  | 50257  |
| d_model     | 2048   |
| num_layers  | 24     |
| num_heads   | 32     |
| d_ff        | 8192   |
| max_seq_len | 2048   |
| dropout     | 0.1    |

The binary uses a **tiny test configuration** (d_model=128, 2 layers, 4 heads)
for fast CPU execution.

## How to Run

```sh
cargo run --bin gpt-1b
```

Expected output (10 training steps with decreasing loss):

```
GPT model created: 2 layers, 4 heads, d_model=128, vocab=1000
Step   1 | loss = 7.025931
Step   2 | loss = 6.891234
...
Step  10 | loss = 5.123456
Training complete!
```

## Running Tests

```sh
cargo test -p gpt-1b
```

## File Layout

- `src/main.rs` — Training loop entry point
- `src/transformer.rs` — MultiHeadAttention, FeedForward, TransformerBlock
- `src/model.rs` — GptConfig, GptModel
- `src/synthetic_data.rs` — SyntheticDataLoader

## Key Components Used

- **mt-training** — `TrainingConfig`, `DistributedRuntime`
- **mt-comm** — `MockCommunicator`
- **mt-core** — `MeshTorusHybrid` topology, `MixedPrecisionConfig`
- **mt-burn** — Burn integration utilities
- **burn-ndarray** — CPU backend
- **burn-autodiff** — Automatic differentiation
- **burn-optim** — Adam optimizer
