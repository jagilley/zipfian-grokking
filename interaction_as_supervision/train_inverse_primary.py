"""
ICM for Zipfian Grokking (Separate Encoder Architecture)

This experiment tests whether ICM can filter out Zipfian weighting bias when:
1. The encoder is separate from the task head (ICM shapes encoder more directly)
2. ICM losses use Zipfian weighting (realistic - no unfair advantage)

Key insight: The original ICM paper used the inverse model to shape encoder
features by filtering transformation-irrelevant noise. In Zipfian grokking,
the "noise" is the biased weighting that pushes toward memorization.

Critical design decision: ICM losses use Zipfian weighting too (realistic).
The filtering should work because Fourier structure is GLOBAL (same basis for
all samples), so learning Fourier features for high-weight samples automatically
gives Fourier features for ALL samples.

Architecture:
- Shared Encoder E: (a, b) one-hot -> z in R^d
- Task Head H: z -> 97 class logits (single linear layer)
- Inverse Model I: (z, z') -> predicted k
- Forward Model F: (z, k) -> predicted z' (MLP by default)

Loss:
    L_total = lambda_task * L_task + beta_inverse * L_inverse + beta_forward * L_forward

    All losses are Zipfian-weighted (except k sampling is uniform per sample).
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
from typing import Tuple, Dict, Any
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


class ForwardModelMLP(nn.Module):
    """
    MLP forward model: predicts z' from (z, T) using arbitrary nonlinear mapping.

    This is expressive - it can learn to predict z' even for memorization-based
    representations. Included to test whether inverse-primary training makes
    the linear constraint unnecessary.
    """
    hidden_dim: int = 128
    output_dim: int = 128
    n_transformations: int = 97

    @nn.compact
    def __call__(self, z, t_onehot):
        x = jnp.concatenate([z, t_onehot], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x


class ForwardModelLinear(nn.Module):
    """
    Linear forward model: z' = W_t @ z + b_t

    For Fourier representations, translations are LINEAR transformations.
    For memorization-based, translations are nonlinear permutations.
    """
    output_dim: int = 128
    n_transformations: int = 97
    mode: str = 'low_rank'  # 'full', 'low_rank', 'diagonal'
    rank: int = 32

    @nn.compact
    def __call__(self, z, t_onehot):
        if self.mode == 'diagonal':
            scale = nn.Dense(self.output_dim, name='scale')(t_onehot)
            shift = nn.Dense(self.output_dim, name='shift')(t_onehot)
            return z * scale + shift

        elif self.mode == 'low_rank':
            # z' = A @ (B_t @ z) + b_t where W_t = A @ B_t is low-rank
            A = self.param('A', nn.initializers.lecun_normal(),
                          (self.output_dim, self.rank))
            B_flat = nn.Dense(self.rank * self.output_dim, name='B')(t_onehot)
            B = B_flat.reshape(-1, self.rank, self.output_dim)
            b = nn.Dense(self.output_dim, name='b')(t_onehot)

            z_expanded = z[..., None]
            Bz = jnp.squeeze(B @ z_expanded, axis=-1)
            z_transformed = Bz @ A.T
            return z_transformed + b

        else:  # 'full'
            W_flat = nn.Dense(self.output_dim * self.output_dim, name='W')(t_onehot)
            W = W_flat.reshape(-1, self.output_dim, self.output_dim)
            b = nn.Dense(self.output_dim, name='b')(t_onehot)

            z_expanded = z[..., None]
            z_transformed = jnp.squeeze(W @ z_expanded, axis=-1)
            return z_transformed + b


class InverseModel(nn.Module):
    """
    Predicts transformation T from (z(a,b), z(T(a,b))).

    In inverse-primary training, this is the main driver of representation
    learning. It forces representations to encode transformation structure.
    """
    hidden_dim: int = 128
    n_transformations: int = 97

    @nn.compact
    def __call__(self, z1, z2):
        x = jnp.concatenate([z1, z2], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_transformations)(x)
        return x


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
    icm_learning_rate: float,
    forward_model_type: str = 'linear',
    forward_model_mode: str = 'low_rank',
    forward_model_rank: int = 32,
) -> Dict[str, train_state.TrainState]:
    """Create training states for encoder, task head, and ICM components."""

    rng_enc, rng_head, rng_forward, rng_inverse = jax.random.split(rng, 4)
    input_dim = 2 * p

    # Encoder
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

    # Forward model
    if forward_model_type == 'linear':
        forward_model = ForwardModelLinear(
            output_dim=hidden_dim,
            n_transformations=p,
            mode=forward_model_mode,
            rank=forward_model_rank
        )
        print(f"Using LINEAR forward model (mode={forward_model_mode}, rank={forward_model_rank})")
    else:
        forward_model = ForwardModelMLP(
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_transformations=p
        )
        print(f"Using MLP forward model")

    forward_params = forward_model.init(rng_forward, jnp.ones((1, hidden_dim)), jnp.ones((1, p)))
    forward_tx = optax.adam(learning_rate=icm_learning_rate)
    forward_state = train_state.TrainState.create(
        apply_fn=forward_model.apply, params=forward_params, tx=forward_tx
    )

    n_forward_params = sum(x.size for x in jax.tree_util.tree_leaves(forward_params))
    print(f"Forward model parameters: {n_forward_params:,}")

    # Inverse model
    inverse_model = InverseModel(hidden_dim=hidden_dim, n_transformations=p)
    inverse_params = inverse_model.init(rng_inverse, jnp.ones((1, hidden_dim)), jnp.ones((1, hidden_dim)))
    inverse_tx = optax.adam(learning_rate=icm_learning_rate)
    inverse_state = train_state.TrainState.create(
        apply_fn=inverse_model.apply, params=inverse_params, tx=inverse_tx
    )

    return {
        'encoder': encoder_state,
        'task_head': task_state,
        'forward': forward_state,
        'inverse': inverse_state,
    }


# =============================================================================
# TRAINING STEPS
# =============================================================================

@partial(jax.jit, static_argnames=['p'])
def train_step(
    encoder_state: train_state.TrainState,
    task_state: train_state.TrainState,
    forward_state: train_state.TrainState,
    inverse_state: train_state.TrainState,
    inputs: jnp.ndarray,
    labels: jnp.ndarray,
    pairs: jnp.ndarray,
    weights: jnp.ndarray,
    all_inputs: jnp.ndarray,
    rng: jax.random.PRNGKey,
    p: int,
    lambda_task: float,
    beta_forward: float,
    beta_inverse: float,
):
    """
    Training step with ICM.

    Critical: All losses are Zipfian-weighted (realistic test).
    The filtering works because Fourier structure is global.

    Loss = lambda_task * L_task + beta_inverse * L_inverse + beta_forward * L_forward
    """
    batch_size = inputs.shape[0]

    # Sample transformations (k is uniform for each sample)
    k = jax.random.randint(rng, (batch_size,), 0, p)
    k_onehot = jax.nn.one_hot(k, p)

    # Apply transformations to get new (a', b') pairs
    a = pairs[:, 0]
    b = pairs[:, 1]
    a_new = (a + k) % p
    b_new = (b + k) % p

    # Get indices for transformed pairs
    transformed_indices = a_new * p + b_new
    inputs_transformed = all_inputs[transformed_indices]

    def combined_loss_fn(enc_params, task_params, forward_params, inverse_params):
        # Get representations for original inputs
        z = encoder_state.apply_fn(enc_params, inputs)
        logits = task_state.apply_fn(task_params, z)

        # Get representations for transformed inputs (stop gradient to avoid double counting)
        z_transformed = encoder_state.apply_fn(enc_params, inputs_transformed)
        z_transformed_sg = jax.lax.stop_gradient(z_transformed)

        # Task loss (Zipfian-weighted)
        one_hot = jax.nn.one_hot(labels, p)
        per_sample_task_loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1)
        task_loss = jnp.sum(per_sample_task_loss * weights)

        # Forward model loss (Zipfian-weighted!)
        # This is realistic: high-weight samples dominate the forward loss too
        # Note: We always compute this (even if beta_forward=0) for JAX tracing
        z_pred = forward_state.apply_fn(forward_params, z, k_onehot)
        per_sample_forward_loss = jnp.sum((z_pred - z_transformed_sg) ** 2, axis=-1)
        forward_loss = jnp.sum(per_sample_forward_loss * weights)

        # Inverse model loss (Zipfian-weighted!)
        # Key insight: Even with Zipfian weighting, learning Fourier features for
        # high-weight samples gives Fourier features for ALL samples (global structure)
        t_logits = inverse_state.apply_fn(inverse_params, z, z_transformed_sg)
        per_sample_inverse_loss = -jnp.sum(k_onehot * jax.nn.log_softmax(t_logits), axis=-1)
        inverse_loss = jnp.sum(per_sample_inverse_loss * weights)

        # Combined loss
        total_loss = lambda_task * task_loss + beta_inverse * inverse_loss + beta_forward * forward_loss

        return total_loss, {
            'task_loss': task_loss,
            'forward_loss': forward_loss,
            'inverse_loss': inverse_loss,
            'total_loss': total_loss,
        }

    # Compute gradients for all parameters
    (loss, metrics), grads = jax.value_and_grad(
        combined_loss_fn, argnums=(0, 1, 2, 3), has_aux=True
    )(encoder_state.params, task_state.params, forward_state.params, inverse_state.params)

    enc_grads, task_grads, forward_grads, inverse_grads = grads

    # Update states
    new_encoder_state = encoder_state.apply_gradients(grads=enc_grads)
    new_task_state = task_state.apply_gradients(grads=task_grads)
    new_forward_state = forward_state.apply_gradients(grads=forward_grads)
    new_inverse_state = inverse_state.apply_gradients(grads=inverse_grads)

    # Compute inverse accuracy using the predictions already computed in the loss
    # (avoid redundant forward pass)
    z = encoder_state.apply_fn(encoder_state.params, inputs)
    z_transformed = encoder_state.apply_fn(encoder_state.params, inputs_transformed)
    t_logits = inverse_state.apply_fn(inverse_state.params, z, jax.lax.stop_gradient(z_transformed))
    inverse_acc = jnp.mean(jnp.argmax(t_logits, axis=-1) == k)
    metrics['inverse_acc'] = inverse_acc

    return new_encoder_state, new_task_state, new_forward_state, new_inverse_state, metrics


@partial(jax.jit, static_argnames=['p', 'n_steps'])
def train_multiple_steps(
    encoder_state: train_state.TrainState,
    task_state: train_state.TrainState,
    forward_state: train_state.TrainState,
    inverse_state: train_state.TrainState,
    inputs: jnp.ndarray,
    labels: jnp.ndarray,
    pairs: jnp.ndarray,
    weights: jnp.ndarray,
    all_inputs: jnp.ndarray,
    rng: jax.random.PRNGKey,
    p: int,
    lambda_task: float,
    beta_forward: float,
    beta_inverse: float,
    n_steps: int,
):
    """
    Run multiple training steps in a single JIT-compiled function.
    Reduces Python loop overhead significantly.
    """
    def body_fn(carry, rng_i):
        enc_state, task_state, fwd_state, inv_state = carry
        new_enc, new_task, new_fwd, new_inv, metrics = train_step(
            enc_state, task_state, fwd_state, inv_state,
            inputs, labels, pairs, weights, all_inputs,
            rng_i, p, lambda_task, beta_forward, beta_inverse
        )
        return (new_enc, new_task, new_fwd, new_inv), metrics

    # Split RNG for all steps
    rngs = jax.random.split(rng, n_steps)

    # Run n_steps training iterations
    (new_enc, new_task, new_fwd, new_inv), all_metrics = jax.lax.scan(
        body_fn,
        (encoder_state, task_state, forward_state, inverse_state),
        rngs
    )

    # Return last metrics (or could average)
    last_metrics = jax.tree_util.tree_map(lambda x: x[-1], all_metrics)

    return new_enc, new_task, new_fwd, new_inv, last_metrics


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
    # Generate all inputs
    all_inputs = np.zeros((p * p, 2 * p), dtype=np.float32)
    for a in range(p):
        for b in range(p):
            idx = a * p + b
            all_inputs[idx, a] = 1.0
            all_inputs[idx, p + b] = 1.0

    # Get logits for all pairs
    z = encoder_state.apply_fn(encoder_state.params, jnp.array(all_inputs))
    logits = np.array(task_state.apply_fn(task_state.params, z))
    logits_tensor = logits.reshape(p, p, p)

    # 2D FFT over (a, b) dimensions
    fourier_logits = np.fft.fft2(logits_tensor, axes=(0, 1))

    # Compute energy per frequency
    energy = np.sum(np.abs(fourier_logits) ** 2, axis=2)
    total_energy = np.sum(energy)

    # Find key diagonal frequencies (where k1 + k2 = 0 mod p)
    diagonal_energy = []
    for k in range(p):
        k_neg = (-k) % p
        diagonal_energy.append((k, energy[k, k_neg]))

    # Sort by energy (exclude DC component k=0)
    diagonal_energy = sorted(diagonal_energy[1:], key=lambda x: x[1], reverse=True)
    key_freqs = [k for k, e in diagonal_energy[:n_keys]]

    # Compute energy ratio in key frequencies
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
    """Main training function."""

    rng = jax.random.PRNGKey(args.seed)

    # Create dataset
    print(f"Creating dataset with p={args.p}, train_fraction={args.train_fraction}")
    data = create_dataset(args.p, args.train_fraction, args.seed)

    # Create Zipfian weights
    n_train = len(data['train_indices'])
    weights = zipf_weights(n_train, args.zipf_exponent)
    zipf_info = compute_zipf_info(weights)
    print(f"Zipf exponent: {args.zipf_exponent}")
    print(f"  Weight ratio: {zipf_info['weight_ratio']:.2f}")
    print(f"  Top 10% mass: {zipf_info['top_10pct_mass']:.4f}")
    print(f"  Effective N: {zipf_info['effective_n']:.2f}")

    # Create training states
    rng, rng_init = jax.random.split(rng)
    states = create_train_states(
        rng_init, args.p, args.hidden_dim, args.n_encoder_layers,
        args.learning_rate, args.weight_decay, args.icm_learning_rate,
        forward_model_type=args.forward_model_type,
        forward_model_mode=args.forward_model_mode,
        forward_model_rank=args.forward_model_rank,
    )

    # Prepare data arrays
    train_inputs = jnp.array(data['all_inputs'][data['train_indices']])
    train_labels = jnp.array(data['all_labels'][data['train_indices']])
    train_pairs = jnp.array(data['all_pairs'][data['train_indices']])
    test_inputs = jnp.array(data['all_inputs'][data['test_indices']])
    test_labels = jnp.array(data['all_labels'][data['test_indices']])
    all_inputs = jnp.array(data['all_inputs'])
    weights_jax = jnp.array(weights)

    # History tracking
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'task_loss': [],
        'forward_loss': [],
        'inverse_loss': [],
        'inverse_acc': [],
        'energy_ratio': [],
    }

    # Training loop with multi-step JIT compilation for speed
    steps_per_update = args.steps_per_update
    n_updates = args.n_epochs // steps_per_update
    log_every_n_updates = max(1, args.log_interval // steps_per_update)
    fourier_every_n_updates = max(1, args.fourier_interval // steps_per_update)

    print(f"\nStarting training for {args.n_epochs} epochs...")
    print(f"  Using {steps_per_update} steps per JIT-compiled update ({n_updates} updates total)")
    print(f"Loss weights: lambda_task={args.lambda_task}, beta_forward={args.beta_forward}, beta_inverse={args.beta_inverse}")
    print(f"ICM losses use Zipfian weighting (realistic)\n")

    pbar = tqdm(range(n_updates), desc="Training", unit="update")
    for update_idx in pbar:
        rng, rng_step = jax.random.split(rng)
        epoch = update_idx * steps_per_update

        # Run multiple training steps in one JIT-compiled call
        states['encoder'], states['task_head'], states['forward'], states['inverse'], metrics = train_multiple_steps(
            states['encoder'], states['task_head'], states['forward'], states['inverse'],
            train_inputs, train_labels, train_pairs, weights_jax,
            all_inputs, rng_step, args.p,
            args.lambda_task, args.beta_forward, args.beta_inverse,
            steps_per_update
        )

        # Logging
        if update_idx % log_every_n_updates == 0:
            train_eval = eval_step(states['encoder'], states['task_head'], train_inputs, train_labels, args.p)
            test_eval = eval_step(states['encoder'], states['task_head'], test_inputs, test_labels, args.p)

            history['epoch'].append(epoch)
            history['train_loss'].append(float(train_eval['loss']))
            history['train_acc'].append(float(train_eval['accuracy']))
            history['test_loss'].append(float(test_eval['loss']))
            history['test_acc'].append(float(test_eval['accuracy']))
            history['task_loss'].append(float(metrics['task_loss']))
            history['forward_loss'].append(float(metrics['forward_loss']))
            history['inverse_loss'].append(float(metrics['inverse_loss']))
            history['inverse_acc'].append(float(metrics['inverse_acc']))

            # Fourier metrics (less frequent)
            if update_idx % fourier_every_n_updates == 0:
                fourier_metrics = compute_fourier_metrics(states['encoder'], states['task_head'], args.p)
                history['energy_ratio'].append(fourier_metrics['energy_ratio'])
            else:
                history['energy_ratio'].append(history['energy_ratio'][-1] if history['energy_ratio'] else 0.0)

            pbar.set_postfix({
                'epoch': epoch,
                'train': f"{train_eval['accuracy']:.3f}",
                'test': f"{test_eval['accuracy']:.3f}",
                'inv': f"{metrics['inverse_acc']:.3f}",
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

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)

    with open(os.path.join(args.save_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    with open(os.path.join(args.save_dir, 'params.pkl'), 'wb') as f:
        pickle.dump({
            'encoder': states['encoder'].params,
            'task_head': states['task_head'].params,
            'forward': states['forward'].params,
            'inverse': states['inverse'].params,
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
    parser = argparse.ArgumentParser(description='Inverse-Primary ICM for Zipfian Grokking')

    # Data parameters
    parser.add_argument('--p', type=int, default=97, help='Modulus for arithmetic')
    parser.add_argument('--train_fraction', type=float, default=0.3, help='Fraction of data for training')
    parser.add_argument('--zipf_exponent', type=float, default=1.5, help='Zipf distribution exponent')

    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--n_encoder_layers', type=int, default=2, help='Number of encoder layers')

    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=100000, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1.0, help='Weight decay (encoder only)')
    parser.add_argument('--icm_learning_rate', type=float, default=1e-3, help='Learning rate for ICM models')

    # Loss weights
    parser.add_argument('--lambda_task', type=float, default=1.0,
                        help='Weight for task loss')
    parser.add_argument('--beta_forward', type=float, default=0.1,
                        help='Weight for forward model loss')
    parser.add_argument('--beta_inverse', type=float, default=0.1,
                        help='Weight for inverse model loss')

    # Forward model architecture
    parser.add_argument('--forward_model_type', type=str, default='mlp',
                        choices=['mlp', 'linear'],
                        help='Forward model type')
    parser.add_argument('--forward_model_mode', type=str, default='low_rank',
                        choices=['full', 'low_rank', 'diagonal'],
                        help='For linear forward model: full, low_rank, or diagonal')
    parser.add_argument('--forward_model_rank', type=int, default=32,
                        help='Rank for low_rank forward model mode')

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
