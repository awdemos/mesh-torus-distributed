mod model;
mod synthetic_data;
mod transformer;

use std::sync::Arc;

use anyhow::Result;
use burn::tensor::activation::log_softmax;
use burn::tensor::Tensor;
use burn_autodiff::Autodiff;
use burn_ndarray::{NdArray, NdArrayDevice};
use burn_optim::{AdamConfig, GradientsParams, Optimizer};
use mt_comm::MockCommunicator;
use mt_core::precision::MixedPrecisionConfig;
use mt_core::topology::MeshTorusHybrid;
use mt_training::config::{PipelineScheduleConfig, TopologyConfig, TrainingConfig};
use mt_training::runtime::DistributedRuntime;
use tracing_subscriber::EnvFilter;

use crate::model::{GptConfig, GptModel};
use crate::synthetic_data::SyntheticDataLoader;

/// Autodiff-capable CPU backend.
type B = Autodiff<NdArray>;

/// Manual cross-entropy loss for integer targets.
///
/// `logits`: `[N, vocab]` — pre-softmax logits
/// `targets`: `[N]` — integer class indices in `[0, vocab)`
///
/// Returns a scalar loss tensor.
fn cross_entropy_loss<Br: burn::tensor::backend::Backend>(
    logits: Tensor<Br, 2>,
    targets: Tensor<Br, 1, burn::tensor::Int>,
) -> Tensor<Br, 1> {
    let [n, _vocab] = logits.dims();
    let log_probs = log_softmax(logits, 1);
    let targets_2d = targets.reshape([n, 1]);
    let nll = log_probs.gather(1, targets_2d).squeeze::<1>();
    -nll.mean()
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .init();

    let device = NdArrayDevice::default();

    // Small config for fast CPU execution
    let config = GptConfig::test_config();
    let mut model = GptModel::<B>::new(&config, &device);

    println!(
        "GPT model: {} layers, {} heads, d_model={}, vocab={}",
        config.num_layers, config.num_heads, config.d_model, config.vocab_size
    );

    // Adam optimizer via burn-optim
    let mut optimizer = AdamConfig::new().init::<B, GptModel<B>>();

    // Distributed runtime with mock communicator
    let comms = MockCommunicator::create_world(1);
    let comm = Arc::new(comms.into_iter().next().unwrap());
    let topology = MeshTorusHybrid::new_2d(1, 1, 1)?;
    let training_config = TrainingConfig {
        batch_size: 2,
        micro_batch_size: 2,
        accumulation_steps: 1,
        learning_rate: 3e-4,
        epochs: 1,
        warmup_steps: 0,
        mixed_precision: MixedPrecisionConfig::default(),
        pipeline_schedule: PipelineScheduleConfig::GPipe,
        topology_config: TopologyConfig::Manual {
            dims: [1, 1, 1],
        },
        checkpoint_interval: 1000,
        log_interval: 1,
        clip_grad_norm: 0.0,
    };
    let _runtime = DistributedRuntime::builder()
        .with_topology(topology)
        .with_communicator(comm)
        .with_config(training_config)
        .build()?;

    // Synthetic data: 10 batches
    let num_steps = 10;
    let mut dataloader = SyntheticDataLoader::new(
        2, 32, config.vocab_size, num_steps, &device,
    );

    for step in 0..num_steps {
        let (input, target) = match dataloader.next() {
            Some(batch) => batch,
            None => break,
        };

        let logits = model.forward(input);

        let [batch, seq, vocab] = logits.dims();
        let logits_2d = logits.reshape([batch * seq, vocab]);
        let target_1d = target.reshape([batch * seq]);

        let loss = cross_entropy_loss(logits_2d, target_1d);
        let loss_val: f32 = loss.clone().into_scalar();

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(3e-4, model, grads);

        println!("Step {:3} | loss = {:.6}", step + 1, loss_val);
    }

    println!("Training complete!");
    Ok(())
}
