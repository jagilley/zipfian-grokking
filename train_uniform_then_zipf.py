"""
Uniform Pretrain → Zipfian Fine-tune Experiment

Tests Dhruv Batra's hypothesis: if a model first groks on uniformly-sampled data
(reaching the Fourier solution), does it remain stable when we switch to Zipfian
weighting? Or does the Fourier solution collapse because it's a saddle point
under Zipfian loss?

Protocol:
  Phase 1: Train with uniform weights (zipf_exponent=0) until 100% test accuracy
  Phase 2: Switch to Zipfian weights (zipf_exponent=s) and continue training

If Dhruv is right: the model stays at the Fourier solution (no collapse).
If the saddle point theory is right: small perturbations in θ_M grow exponentially
even starting from the converged Fourier solution, causing collapse.

Based on train_inverse_primary.py baseline (no ICM, just task loss).
Forward/inverse models removed since their weights are zero.
"""

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.training import train_state
import optax
import pickle
import argparse
from tqdm import tqdm
import os
from typing import Dict
from functools import partial


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class Encoder(nn.Module):
    """Shared encoder that maps (a, b) one-hot to representation z."""
    hidden_dim: int = 128
    n_layers: int = 2

    @nn.compact
    def __call__(self, x):
        for i in range(self.n_layers):
            x = nn.Dense(self.hidden_dim, name=f'hidden_{i}')(x)
            x = nn.relu(x)
        return x


class TaskHead(nn.Module):
    """Single linear layer from representation to task logits."""
    output_dim: int = 97

    @nn.compact
    def __call__(self, z):
        return nn.Dense(self.output_dim, name='output')(z)


# =============================================================================
# DATA GENERATION
# =============================================================================

def create_onehot_input(a: int, b: int, p: int) -> np.ndarray:
    """Create one-hot encoded input for (a, b) pair."""
    x = np.zeros(2 * p, dtype=np.float32)
    x[a] = 1.0
    x[p + b] = 1.0
    return x


def create_dataset(p: int, train_fraction: float = 0.3, seed: int = 42):
    """Create train/test split for modular addition."""
    rng = np.random.default_rng(seed)

    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    n_total = len(all_pairs)
    n_train = int(n_total * train_fraction)

    indices = rng.permutation(n_total)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    all_inputs = np.array([create_onehot_input(a, b, p) for a, b in all_pairs])
    all_labels = np.array([(a + b) % p for a, b in all_pairs])
    all_pairs = np.array(all_pairs)

    return {
        'all_inputs': all_inputs,
        'all_labels': all_labels,
        'all_pairs': all_pairs,
        'train_indices': train_indices,
        'test_indices': test_indices,
        'p': p,
    }


def zipf_weights(n: int, s: float = 1.0) -> np.ndarray:
    """Generate Zipf distribution weights for n items."""
    if s == 0.0:
        return np.ones(n) / n
    ranks = np.arange(1, n + 1)
    weights = 1.0 / np.power(ranks, s)
    return weights / weights.sum()


def compute_zipf_info(weights: np.ndarray) -> Dict[str, float]:
    """Compute statistics about the Zipfian distribution."""
    sorted_w = np.sort(weights)[::-1]
    n = len(weights)
    return {
        'weight_ratio': sorted_w[0] / sorted_w[-1] if sorted_w[-1] > 0 else float('inf'),
        'top_10pct_mass': sorted_w[:max(1, n // 10)].sum(),
        'bottom_50pct_mass': sorted_w[n // 2:].sum(),
        'effective_n': 1.0 / np.sum(weights ** 2),
    }


# =============================================================================
# TRAINING STATE CREATION
# =============================================================================

def create_train_states(
    rng: jax.random.PRNGKey,
    p: int,
    hidden_dim: int,
    n_encoder_layers: int,
    learning_rate: float,
    weight_decay: float,
) -> Dict[str, train_state.TrainState]:
    """Create training states for encoder and task head."""

    rng_enc, rng_head = jax.random.split(rng, 2)
    input_dim = 2 * p

    # Encoder (with weight decay)
    encoder = Encoder(hidden_dim=hidden_dim, n_layers=n_encoder_layers)
    encoder_params = encoder.init(rng_enc, jnp.ones((1, input_dim)))
    encoder_tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    encoder_state = train_state.TrainState.create(
        apply_fn=encoder.apply, params=encoder_params, tx=encoder_tx
    )

    # Task head (no weight decay - only encoder gets regularization)
    task_head = TaskHead(output_dim=p)
    task_params = task_head.init(rng_head, jnp.ones((1, hidden_dim)))
    task_tx = optax.adam(learning_rate=learning_rate)
    task_state = train_state.TrainState.create(
        apply_fn=task_head.apply, params=task_params, tx=task_tx
    )

    return {
        'encoder': encoder_state,
        'task_head': task_state,
    }


# =============================================================================
# TRAINING STEPS
# =============================================================================

@partial(jax.jit, static_argnames=['p'])
def train_step(
    encoder_state: train_state.TrainState,
    task_state: train_state.TrainState,
    inputs: jnp.ndarray,
    labels: jnp.ndarray,
    weights: jnp.ndarray,
    p: int,
):
    """Single training step: weighted cross-entropy on task loss only."""

    def loss_fn(enc_params, task_params):
        z = encoder_state.apply_fn(enc_params, inputs)
        logits = task_state.apply_fn(task_params, z)

        one_hot = jax.nn.one_hot(labels, p)
        per_sample_loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1)
        task_loss = jnp.sum(per_sample_loss * weights)

        return task_loss, {'task_loss': task_loss}

    (loss, metrics), grads = jax.value_and_grad(
        loss_fn, argnums=(0, 1), has_aux=True
    )(encoder_state.params, task_state.params)

    enc_grads, task_grads = grads

    new_encoder_state = encoder_state.apply_gradients(grads=enc_grads)
    new_task_state = task_state.apply_gradients(grads=task_grads)

    return new_encoder_state, new_task_state, metrics


@partial(jax.jit, static_argnames=['p', 'n_steps'])
def train_multiple_steps(
    encoder_state: train_state.TrainState,
    task_state: train_state.TrainState,
    inputs: jnp.ndarray,
    labels: jnp.ndarray,
    weights: jnp.ndarray,
    p: int,
    n_steps: int,
):
    """
    Run multiple training steps in a single JIT-compiled function.
    Reduces Python loop overhead significantly.
    """
    def body_fn(carry, _):
        enc_state, task_state = carry
        new_enc, new_task, metrics = train_step(
            enc_state, task_state,
            inputs, labels, weights, p
        )
        return (new_enc, new_task), metrics

    (new_enc, new_task), all_metrics = jax.lax.scan(
        body_fn,
        (encoder_state, task_state),
        None, length=n_steps
    )

    last_metrics = jax.tree_util.tree_map(lambda x: x[-1], all_metrics)

    return new_enc, new_task, last_metrics


@partial(jax.jit, static_argnames=['p'])
def eval_step(
    encoder_state: train_state.TrainState,
    task_state: train_state.TrainState,
    inputs: jnp.ndarray,
    labels: jnp.ndarray,
    p: int
):
    """Evaluate accuracy and loss on a dataset."""
    z = encoder_state.apply_fn(encoder_state.params, inputs)
    logits = task_state.apply_fn(task_state.params, z)
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)

    one_hot = jax.nn.one_hot(labels, p)
    loss = -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

    return {'accuracy': accuracy, 'loss': loss}


# =============================================================================
# FOURIER METRICS
# =============================================================================

def compute_fourier_metrics(
    encoder_state: train_state.TrainState,
    task_state: train_state.TrainState,
    p: int,
    n_keys: int = 5
):
    """Compute Fourier-based progress measures."""
    all_inputs = np.zeros((p * p, 2 * p), dtype=np.float32)
    for a in range(p):
        for b in range(p):
            idx = a * p + b
            all_inputs[idx, a] = 1.0
            all_inputs[idx, p + b] = 1.0

    z = encoder_state.apply_fn(encoder_state.params, jnp.array(all_inputs))
    logits = np.array(task_state.apply_fn(task_state.params, z))
    logits_tensor = logits.reshape(p, p, p)

    fourier_logits = np.fft.fft2(logits_tensor, axes=(0, 1))

    energy = np.sum(np.abs(fourier_logits) ** 2, axis=2)
    total_energy = np.sum(energy)

    diagonal_energy = []
    for k in range(p):
        k_neg = (-k) % p
        diagonal_energy.append((k, energy[k, k_neg]))

    diagonal_energy = sorted(diagonal_energy[1:], key=lambda x: x[1], reverse=True)
    key_freqs = [k for k, e in diagonal_energy[:n_keys]]

    key_energy = sum(energy[k, (-k) % p] for k in key_freqs)
    energy_ratio = key_energy / total_energy if total_energy > 0 else 0.0

    return {
        'energy_ratio': float(energy_ratio),
        'key_frequencies': key_freqs,
        'total_energy': float(total_energy),
    }


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train(args):
    """Main training function with two-phase protocol."""

    rng = jax.random.PRNGKey(args.seed)

    # Create dataset
    print(f"Creating dataset with p={args.p}, train_fraction={args.train_fraction}")
    data = create_dataset(args.p, args.train_fraction, args.seed)

    # Phase 1: uniform weights; Phase 2: Zipfian weights
    n_train = len(data['train_indices'])
    uniform_weights = zipf_weights(n_train, 0.0)  # Phase 1
    zipf_weights_phase2 = zipf_weights(n_train, args.zipf_exponent_phase2)  # Phase 2
    zipf_info = compute_zipf_info(zipf_weights_phase2)
    print(f"Phase 1: Uniform weights (zipf_exponent=0.0) until {args.grok_threshold*100:.1f}% test accuracy")
    print(f"Phase 2: Zipfian weights (zipf_exponent={args.zipf_exponent_phase2})")
    print(f"  Phase 2 weight ratio: {zipf_info['weight_ratio']:.2f}")
    print(f"  Phase 2 top 10% mass: {zipf_info['top_10pct_mass']:.4f}")
    print(f"  Phase 2 effective N: {zipf_info['effective_n']:.2f}")

    # Create training states
    rng, rng_init = jax.random.split(rng)
    states = create_train_states(
        rng_init, args.p, args.hidden_dim, args.n_encoder_layers,
        args.learning_rate, args.weight_decay,
    )

    # Prepare data arrays
    train_inputs = jnp.array(data['all_inputs'][data['train_indices']])
    train_labels = jnp.array(data['all_labels'][data['train_indices']])
    test_inputs = jnp.array(data['all_inputs'][data['test_indices']])
    test_labels = jnp.array(data['all_labels'][data['test_indices']])

    # Start in Phase 1 (uniform)
    phase = 1
    weights_jax = jnp.array(uniform_weights)
    phase_switch_epoch = None

    # History tracking
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'task_loss': [],
        'energy_ratio': [],
        'phase': [],
    }

    # Training loop with multi-step JIT compilation for speed
    steps_per_update = args.steps_per_update
    n_updates = args.n_epochs // steps_per_update
    log_every_n_updates = max(1, args.log_interval // steps_per_update)
    fourier_every_n_updates = max(1, args.fourier_interval // steps_per_update)

    print(f"\nStarting training for {args.n_epochs} epochs...")
    print(f"  Using {steps_per_update} steps per JIT-compiled update ({n_updates} updates total)")
    print(f"Two-phase protocol: uniform → zipf_exponent={args.zipf_exponent_phase2}\n")

    pbar = tqdm(range(n_updates), desc="Phase 1 (uniform)", unit="update")
    for update_idx in pbar:
        epoch = update_idx * steps_per_update

        # Run multiple training steps in one JIT-compiled call
        states['encoder'], states['task_head'], metrics = train_multiple_steps(
            states['encoder'], states['task_head'],
            train_inputs, train_labels, weights_jax,
            args.p, steps_per_update
        )

        # Logging
        if update_idx % log_every_n_updates == 0:
            train_eval = eval_step(states['encoder'], states['task_head'], train_inputs, train_labels, args.p)
            test_eval = eval_step(states['encoder'], states['task_head'], test_inputs, test_labels, args.p)

            test_acc = float(test_eval['accuracy'])

            history['epoch'].append(epoch)
            history['train_loss'].append(float(train_eval['loss']))
            history['train_acc'].append(float(train_eval['accuracy']))
            history['test_loss'].append(float(test_eval['loss']))
            history['test_acc'].append(test_acc)
            history['task_loss'].append(float(metrics['task_loss']))
            history['phase'].append(phase)

            # Phase transition: switch to Zipfian weights when grokking is achieved
            if phase == 1 and test_acc >= args.grok_threshold:
                phase = 2
                phase_switch_epoch = epoch
                weights_jax = jnp.array(zipf_weights_phase2)
                pbar.set_description(f"Phase 2 (zipf={args.zipf_exponent_phase2})")
                print(f"\n{'='*60}")
                print(f"PHASE TRANSITION at epoch {epoch}")
                print(f"  Test accuracy reached {test_acc:.4f} >= {args.grok_threshold}")
                print(f"  Switching from uniform to Zipf exponent {args.zipf_exponent_phase2}")
                print(f"{'='*60}\n")

            # Fourier metrics (less frequent)
            if update_idx % fourier_every_n_updates == 0:
                fourier_metrics = compute_fourier_metrics(states['encoder'], states['task_head'], args.p)
                history['energy_ratio'].append(fourier_metrics['energy_ratio'])
            else:
                history['energy_ratio'].append(history['energy_ratio'][-1] if history['energy_ratio'] else 0.0)

            pbar.set_postfix({
                'epoch': epoch,
                'phase': phase,
                'train': f"{train_eval['accuracy']:.3f}",
                'test': f"{test_eval['accuracy']:.3f}",
                'E': f"{history['energy_ratio'][-1]:.3f}"
            })

    # Final evaluation
    train_eval = eval_step(states['encoder'], states['task_head'], train_inputs, train_labels, args.p)
    test_eval = eval_step(states['encoder'], states['task_head'], test_inputs, test_labels, args.p)
    fourier_metrics = compute_fourier_metrics(states['encoder'], states['task_head'], args.p)

    print(f"\nFinal Results:")
    print(f"  Train accuracy: {train_eval['accuracy']:.4f}")
    print(f"  Test accuracy: {test_eval['accuracy']:.4f}")
    print(f"  Energy ratio: {fourier_metrics['energy_ratio']:.4f}")
    print(f"  Phase switch epoch: {phase_switch_epoch}")
    if phase == 1:
        print(f"  WARNING: Model never reached grok threshold ({args.grok_threshold}) - stayed in Phase 1")

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)

    # Add phase info to history
    history['phase_switch_epoch'] = phase_switch_epoch
    history['zipf_exponent_phase2'] = args.zipf_exponent_phase2
    history['grok_threshold'] = args.grok_threshold

    with open(os.path.join(args.save_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    with open(os.path.join(args.save_dir, 'params.pkl'), 'wb') as f:
        pickle.dump({
            'encoder': states['encoder'].params,
            'task_head': states['task_head'].params,
        }, f)

    with open(os.path.join(args.save_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(vars(args), f)

    with open(os.path.join(args.save_dir, 'zipf_info.pkl'), 'wb') as f:
        pickle.dump(zipf_info, f)

    print(f"\nResults saved to {args.save_dir}")

    return history, states


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Uniform Pretrain → Zipfian Fine-tune Experiment')

    # Data parameters
    parser.add_argument('--p', type=int, default=97, help='Modulus for arithmetic')
    parser.add_argument('--train_fraction', type=float, default=0.3, help='Fraction of data for training')
    parser.add_argument('--zipf_exponent_phase2', type=float, default=1.5,
                        help='Zipf exponent for Phase 2 (after grokking)')
    parser.add_argument('--grok_threshold', type=float, default=1.0,
                        help='Test accuracy threshold to trigger phase switch (1.0 = 100%%)')

    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--n_encoder_layers', type=int, default=2, help='Number of encoder layers')

    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=100000, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1.0, help='Weight decay (encoder only)')

    # Logging parameters
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval (in epochs)')
    parser.add_argument('--fourier_interval', type=int, default=1000, help='Fourier metrics interval')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Performance parameters
    parser.add_argument('--steps_per_update', type=int, default=100,
                        help='Number of training steps per JIT-compiled update (reduces Python overhead)')

    args = parser.parse_args()
    train(args)
