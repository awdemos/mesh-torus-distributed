# mesh-torus-distributed

## OVERVIEW

Rust workspace for distributed deep learning using mesh-torus hybrid topologies, built on the Burn framework.

## STRUCTURE

```
crates/mt-core/       Topology (MeshTorusHybrid, TorusCoordinates, GPUDevice) + FP8 precision
crates/mt-comm/       Communication primitives (Communicator trait, Mock/TCP/NCCL backends, collectives)
crates/mt-pipeline/   Pipeline parallelism (GPipe/1F1B schedules, checkpointing, topology-aware placement)
crates/mt-burn/       Burn integration (Fp8Optimizer, PipelineModule, fp8 conversion)
crates/mt-training/   Training orchestration (GradientAccumulator, DistributedRuntime, train_step)
examples/gpt-1b/      Example: GPT-1B model with distributed training loop
tests/integration/    Integration tests
```

## COMMANDS

```bash
cargo build    # Build workspace
cargo test     # Run all tests (313 passing)
cargo run --bin gpt-1b   # Run the GPT-1B example
```

## SETUP

- Install deps: `cargo build`
- Run tests: `cargo test`
- Rust edition 2021, rust-version 1.78

## CODE STYLE

- Rust strict mode
- Use `cargo fmt` before commits
- Prefer functional patterns where possible
- Avoid `unwrap()` in production code; use `?` or `anyhow::Result`

## DEPLOYMENT

No Dagger module or recognized deployment configuration was found.
General redeploy process:

1. Commit and push changes to the default branch.
2. Trigger the relevant CI/CD pipeline or run the documented deploy command.
3. If the project is served via GitHub Pages, the site redeploys automatically after the push.
