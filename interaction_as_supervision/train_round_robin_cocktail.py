"""
Round-Robin Cocktail follow-up for zipfian grokking.

This keeps the inverse-primary training recipe as fixed as possible while swapping
out the transformation distribution. Instead of training on translation only, we
train on a fixed 4-way mixture derived from the final round-robin tournament
policy masses.

The recipe source is the round-robin tournament's summary.json file.
Run the tournament first, then pass the path via --round_robin_summary.

Transform definitions intentionally match the tournament / learned-transformations
setup:
- translation: (a + k, b + k) mod p, k in {0, ..., p-1}
- scaling: (alpha a, alpha b) mod p, alpha in {1, ..., p-1}
- quadratic: (k a^2, k b^2) mod p, k in {0, ..., p-2}
- random: fixed random permutations
"""

import argparse
import json
import os
import pickle
from functools import partial
from typing import Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tqdm import tqdm


TYPE_NAMES = ["translation", "scaling", "quadratic", "random"]
TRANSFORM_TRANSLATION = 0
TRANSFORM_SCALING = 1
TRANSFORM_QUADRATIC = 2
TRANSFORM_RANDOM = 3


# =============================================================================
# MODEL DEFINITIONS (kept close to grokking_icm_inverse_primary)
# =============================================================================

class Encoder(nn.Module):
    hidden_dim: int = 128
    n_layers: int = 2

    @nn.compact
    def __call__(self, x):
        for i in range(self.n_layers):
            x = nn.Dense(self.hidden_dim, name=f'hidden_{i}')(x)
            x = nn.relu(x)
        return x


class TaskHead(nn.Module):
    output_dim: int = 97

    @nn.compact
    def __call__(self, z):
        return nn.Dense(self.output_dim, name='output')(z)


class ForwardModelMLP(nn.Module):
    hidden_dim: int = 128
    output_dim: int = 128
    n_actions: int = 385

    @nn.compact
    def __call__(self, z, action_onehot):
        x = jnp.concatenate([z, action_onehot], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x


class InverseModel(nn.Module):
    hidden_dim: int = 128
    n_actions: int = 385

    @nn.compact
    def __call__(self, z1, z2):
        x = jnp.concatenate([z1, z2], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        return x


# =============================================================================
# DATA
# =============================================================================


def create_onehot_input(a: int, b: int, p: int) -> np.ndarray:
    x = np.zeros(2 * p, dtype=np.float32)
    x[a] = 1.0
    x[p + b] = 1.0
    return x



def create_dataset(p: int, train_fraction: float = 0.3, seed: int = 42):
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
    if s == 0.0:
        return np.ones(n) / n
    ranks = np.arange(1, n + 1)
    weights = 1.0 / np.power(ranks, s)
    return weights / weights.sum()



def compute_zipf_info(weights: np.ndarray) -> Dict[str, float]:
    sorted_w = np.sort(weights)[::-1]
    n = len(weights)
    return {
        'weight_ratio': sorted_w[0] / sorted_w[-1] if sorted_w[-1] > 0 else float('inf'),
        'top_10pct_mass': float(sorted_w[:max(1, n // 10)].sum()),
        'bottom_50pct_mass': float(sorted_w[n // 2:].sum()),
        'effective_n': float(1.0 / np.sum(weights ** 2)),
    }


# =============================================================================
# TRANSFORMS / COCKTAIL RECIPE
# =============================================================================


def load_round_robin_cocktail(summary_path: str) -> Dict[str, Dict[str, float]]:
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    raw_masses = {name: 0.0 for name in TYPE_NAMES}
    for row in summary['ranking']:
        raw_masses[row['transform']] = float(row['mean_final_policy_mass'])

    total_mass = sum(raw_masses.values())
    normalized = {k: (v / total_mass if total_mass > 0 else 0.0) for k, v in raw_masses.items()}

    return {
        'source_summary_path': summary_path,
        'source_results_dir': summary.get('results_dir', ''),
        'raw_mean_final_policy_mass': raw_masses,
        'raw_mass_sum': total_mass,
        'normalized_type_probs': normalized,
    }



def get_type_sizes(p: int) -> jnp.ndarray:
    return jnp.array([p, p - 1, p - 1, p - 1], dtype=jnp.int32)



def get_type_offsets_full_action_space(p: int) -> jnp.ndarray:
    # translation starts at 0, scaling at p, quadratic at p + (p-1), random at p + 2*(p-1)
    return jnp.array([0, p, p + (p - 1), p + 2 * (p - 1)], dtype=jnp.int32)



def get_param_offsets_for_sampling() -> jnp.ndarray:
    # scaling params start at 1, all others start at 0
    return jnp.array([0, 1, 0, 0], dtype=jnp.int32)



def generate_random_permutations(rng: jax.random.PRNGKey, p: int, n_random: int) -> jnp.ndarray:
    perms = []
    for _ in range(n_random):
        rng, subkey = jax.random.split(rng)
        perms.append(jax.random.permutation(subkey, p))
    return jnp.stack(perms)



def sample_transform_types_and_params(
    rng: jax.random.PRNGKey,
    batch_size: int,
    type_probs: jnp.ndarray,
    p: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    rng_type, rng_param = jax.random.split(rng)
    transform_types = jax.random.choice(rng_type, 4, shape=(batch_size,), p=type_probs)

    type_sizes = get_type_sizes(p)
    param_offsets = get_param_offsets_for_sampling()
    raw_params = jax.random.randint(rng_param, (batch_size,), 0, p)

    sizes = type_sizes[transform_types]
    offsets = param_offsets[transform_types]
    params = (raw_params % sizes) + offsets
    return transform_types.astype(jnp.int32), params.astype(jnp.int32)



def type_and_param_to_action_id(transform_types: jnp.ndarray, params: jnp.ndarray, p: int) -> jnp.ndarray:
    type_starts = get_type_offsets_full_action_space(p)
    adjusted_param = jnp.where(transform_types == TRANSFORM_SCALING, params - 1, params)
    return (type_starts[transform_types] + adjusted_param).astype(jnp.int32)



def apply_transformation_batch(
    a: jnp.ndarray,
    b: jnp.ndarray,
    transform_types: jnp.ndarray,
    params: jnp.ndarray,
    p: int,
    random_perms: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    a_translate = (a + params) % p
    b_translate = (b + params) % p

    a_scale = (a * params) % p
    b_scale = (b * params) % p

    a_quadratic = (a * a * params) % p
    b_quadratic = (b * b * params) % p

    safe_perm_idx = jnp.clip(params, 0, random_perms.shape[0] - 1)
    a_random = random_perms[safe_perm_idx, a]
    b_random = random_perms[safe_perm_idx, b]

    is_t = transform_types == TRANSFORM_TRANSLATION
    is_s = transform_types == TRANSFORM_SCALING
    is_q = transform_types == TRANSFORM_QUADRATIC
    is_r = transform_types == TRANSFORM_RANDOM

    a_new = a_translate * is_t + a_scale * is_s + a_quadratic * is_q + a_random * is_r
    b_new = b_translate * is_t + b_scale * is_s + b_quadratic * is_q + b_random * is_r
    return a_new, b_new


# =============================================================================
# TRAINING STATE
# =============================================================================


def create_train_states(
    rng: jax.random.PRNGKey,
    p: int,
    hidden_dim: int,
    n_encoder_layers: int,
    learning_rate: float,
    weight_decay: float,
    icm_learning_rate: float,
) -> Dict[str, train_state.TrainState]:
    n_actions = p + 3 * (p - 1)
    input_dim = 2 * p

    rng_enc, rng_head, rng_forward, rng_inverse = jax.random.split(rng, 4)

    encoder = Encoder(hidden_dim=hidden_dim, n_layers=n_encoder_layers)
    encoder_params = encoder.init(rng_enc, jnp.ones((1, input_dim)))
    encoder_state = train_state.TrainState.create(
        apply_fn=encoder.apply,
        params=encoder_params,
        tx=optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )

    task_head = TaskHead(output_dim=p)
    task_params = task_head.init(rng_head, jnp.ones((1, hidden_dim)))
    task_state = train_state.TrainState.create(
        apply_fn=task_head.apply,
        params=task_params,
        tx=optax.adam(learning_rate=learning_rate),
    )

    forward_model = ForwardModelMLP(hidden_dim=hidden_dim, output_dim=hidden_dim, n_actions=n_actions)
    forward_params = forward_model.init(rng_forward, jnp.ones((1, hidden_dim)), jnp.ones((1, n_actions)))
    forward_state = train_state.TrainState.create(
        apply_fn=forward_model.apply,
        params=forward_params,
        tx=optax.adam(learning_rate=icm_learning_rate),
    )

    inverse_model = InverseModel(hidden_dim=hidden_dim, n_actions=n_actions)
    inverse_params = inverse_model.init(rng_inverse, jnp.ones((1, hidden_dim)), jnp.ones((1, hidden_dim)))
    inverse_state = train_state.TrainState.create(
        apply_fn=inverse_model.apply,
        params=inverse_params,
        tx=optax.adam(learning_rate=icm_learning_rate),
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
    random_perms: jnp.ndarray,
    type_probs: jnp.ndarray,
    rng: jax.random.PRNGKey,
    p: int,
    lambda_task: float,
    beta_forward: float,
    beta_inverse: float,
):
    batch_size = inputs.shape[0]
    transform_types, params = sample_transform_types_and_params(rng, batch_size, type_probs, p)
    action_ids = type_and_param_to_action_id(transform_types, params, p)
    n_actions = p + 3 * (p - 1)
    action_onehot = jax.nn.one_hot(action_ids, n_actions)

    a = pairs[:, 0]
    b = pairs[:, 1]
    a_new, b_new = apply_transformation_batch(a, b, transform_types, params, p, random_perms)
    transformed_indices = a_new * p + b_new
    inputs_transformed = all_inputs[transformed_indices]

    def combined_loss_fn(enc_params, task_params, forward_params, inverse_params):
        z = encoder_state.apply_fn(enc_params, inputs)
        logits = task_state.apply_fn(task_params, z)

        z_transformed = encoder_state.apply_fn(enc_params, inputs_transformed)
        z_transformed_sg = jax.lax.stop_gradient(z_transformed)

        one_hot = jax.nn.one_hot(labels, p)
        per_sample_task_loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1)
        task_loss = jnp.sum(per_sample_task_loss * weights)

        z_pred = forward_state.apply_fn(forward_params, z, action_onehot)
        per_sample_forward_loss = jnp.sum((z_pred - z_transformed_sg) ** 2, axis=-1)
        forward_loss = jnp.sum(per_sample_forward_loss * weights)

        t_logits = inverse_state.apply_fn(inverse_params, z, z_transformed_sg)
        per_sample_inverse_loss = -jnp.sum(action_onehot * jax.nn.log_softmax(t_logits), axis=-1)
        inverse_loss = jnp.sum(per_sample_inverse_loss * weights)

        total_loss = lambda_task * task_loss + beta_inverse * inverse_loss + beta_forward * forward_loss
        return total_loss, {
            'task_loss': task_loss,
            'forward_loss': forward_loss,
            'inverse_loss': inverse_loss,
            'total_loss': total_loss,
        }

    (loss, metrics), grads = jax.value_and_grad(
        combined_loss_fn, argnums=(0, 1, 2, 3), has_aux=True
    )(encoder_state.params, task_state.params, forward_state.params, inverse_state.params)

    enc_grads, task_grads, forward_grads, inverse_grads = grads

    new_encoder_state = encoder_state.apply_gradients(grads=enc_grads)
    new_task_state = task_state.apply_gradients(grads=task_grads)
    new_forward_state = forward_state.apply_gradients(grads=forward_grads)
    new_inverse_state = inverse_state.apply_gradients(grads=inverse_grads)

    z = encoder_state.apply_fn(encoder_state.params, inputs)
    z_transformed = encoder_state.apply_fn(encoder_state.params, inputs_transformed)
    t_logits = inverse_state.apply_fn(inverse_state.params, z, jax.lax.stop_gradient(z_transformed))
    predictions = jnp.argmax(t_logits, axis=-1)
    metrics['inverse_acc'] = jnp.mean(predictions == action_ids)
    metrics['sampled_translation'] = jnp.mean(transform_types == TRANSFORM_TRANSLATION)
    metrics['sampled_scaling'] = jnp.mean(transform_types == TRANSFORM_SCALING)
    metrics['sampled_quadratic'] = jnp.mean(transform_types == TRANSFORM_QUADRATIC)
    metrics['sampled_random'] = jnp.mean(transform_types == TRANSFORM_RANDOM)

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
    random_perms: jnp.ndarray,
    type_probs: jnp.ndarray,
    rng: jax.random.PRNGKey,
    p: int,
    lambda_task: float,
    beta_forward: float,
    beta_inverse: float,
    n_steps: int,
):
    def body_fn(carry, rng_i):
        enc_state, task_state, fwd_state, inv_state = carry
        new_enc, new_task, new_fwd, new_inv, metrics = train_step(
            enc_state, task_state, fwd_state, inv_state,
            inputs, labels, pairs, weights, all_inputs,
            random_perms, type_probs, rng_i, p,
            lambda_task, beta_forward, beta_inverse,
        )
        return (new_enc, new_task, new_fwd, new_inv), metrics

    rngs = jax.random.split(rng, n_steps)
    (new_enc, new_task, new_fwd, new_inv), all_metrics = jax.lax.scan(
        body_fn,
        (encoder_state, task_state, forward_state, inverse_state),
        rngs,
    )
    last_metrics = jax.tree_util.tree_map(lambda x: x[-1], all_metrics)
    return new_enc, new_task, new_fwd, new_inv, last_metrics


@partial(jax.jit, static_argnames=['p'])
def eval_step(
    encoder_state: train_state.TrainState,
    task_state: train_state.TrainState,
    inputs: jnp.ndarray,
    labels: jnp.ndarray,
    p: int,
):
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
    n_keys: int = 5,
):
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
    key_freqs = [k for k, _ in diagonal_energy[:n_keys]]
    key_energy = sum(energy[k, (-k) % p] for k in key_freqs)
    energy_ratio = key_energy / total_energy if total_energy > 0 else 0.0

    return {
        'energy_ratio': float(energy_ratio),
        'key_frequencies': key_freqs,
        'total_energy': float(total_energy),
    }


# =============================================================================
# MAIN
# =============================================================================


def train(args):
    rng = jax.random.PRNGKey(args.seed)

    cocktail = load_round_robin_cocktail(args.round_robin_summary)
    type_probs_np = np.array([cocktail['normalized_type_probs'][name] for name in TYPE_NAMES], dtype=np.float32)
    type_probs = jnp.array(type_probs_np)

    print("Round-Robin Cocktail")
    print("=" * 60)
    print(f"Source summary: {args.round_robin_summary}")
    print("Raw mean final policy masses:")
    for name in TYPE_NAMES:
        print(f"  {name:>11s}: {cocktail['raw_mean_final_policy_mass'][name]:.6f}")
    print(f"  {'sum':>11s}: {cocktail['raw_mass_sum']:.6f}")
    print("Normalized type probabilities:")
    for name, prob in zip(TYPE_NAMES, type_probs_np):
        print(f"  {name:>11s}: {prob:.4%}")

    data = create_dataset(args.p, args.train_fraction, args.seed)
    weights = zipf_weights(len(data['train_indices']), args.zipf_exponent)
    zipf_info = compute_zipf_info(weights)

    print(f"\nZipf exponent: {args.zipf_exponent}")
    print(f"  Weight ratio: {zipf_info['weight_ratio']:.2f}")
    print(f"  Top 10% mass: {zipf_info['top_10pct_mass']:.4f}")
    print(f"  Effective N: {zipf_info['effective_n']:.2f}")

    rng, rng_init, rng_perms = jax.random.split(rng, 3)
    states = create_train_states(
        rng_init,
        args.p,
        args.hidden_dim,
        args.n_encoder_layers,
        args.learning_rate,
        args.weight_decay,
        args.icm_learning_rate,
    )

    random_perms = generate_random_permutations(rng_perms, args.p, args.p - 1)

    train_inputs = jnp.array(data['all_inputs'][data['train_indices']])
    train_labels = jnp.array(data['all_labels'][data['train_indices']])
    train_pairs = jnp.array(data['all_pairs'][data['train_indices']])
    test_inputs = jnp.array(data['all_inputs'][data['test_indices']])
    test_labels = jnp.array(data['all_labels'][data['test_indices']])
    all_inputs = jnp.array(data['all_inputs'])
    weights_jax = jnp.array(weights)

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
        'sampled_translation': [],
        'sampled_scaling': [],
        'sampled_quadratic': [],
        'sampled_random': [],
        'target_translation': [],
        'target_scaling': [],
        'target_quadratic': [],
        'target_random': [],
    }

    steps_per_update = args.steps_per_update
    n_updates = args.n_epochs // steps_per_update
    log_every_n_updates = max(1, args.log_interval // steps_per_update)
    fourier_every_n_updates = max(1, args.fourier_interval // steps_per_update)

    print(f"\nStarting training for {args.n_epochs} epochs")
    print(f"  steps_per_update={steps_per_update} ({n_updates} updates total)")
    print(f"  lambda_task={args.lambda_task}, beta_forward={args.beta_forward}, beta_inverse={args.beta_inverse}\n")

    pbar = tqdm(range(n_updates), desc="Cocktail training", unit="update")
    for update_idx in pbar:
        epoch = update_idx * steps_per_update
        rng, rng_step = jax.random.split(rng)

        states['encoder'], states['task_head'], states['forward'], states['inverse'], metrics = train_multiple_steps(
            states['encoder'], states['task_head'], states['forward'], states['inverse'],
            train_inputs, train_labels, train_pairs, weights_jax,
            all_inputs, random_perms, type_probs, rng_step,
            args.p, args.lambda_task, args.beta_forward, args.beta_inverse,
            steps_per_update,
        )

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
            history['sampled_translation'].append(float(metrics['sampled_translation']))
            history['sampled_scaling'].append(float(metrics['sampled_scaling']))
            history['sampled_quadratic'].append(float(metrics['sampled_quadratic']))
            history['sampled_random'].append(float(metrics['sampled_random']))
            history['target_translation'].append(float(type_probs_np[0]))
            history['target_scaling'].append(float(type_probs_np[1]))
            history['target_quadratic'].append(float(type_probs_np[2]))
            history['target_random'].append(float(type_probs_np[3]))

            if update_idx % fourier_every_n_updates == 0:
                fourier_metrics = compute_fourier_metrics(states['encoder'], states['task_head'], args.p)
                history['energy_ratio'].append(fourier_metrics['energy_ratio'])
            else:
                history['energy_ratio'].append(history['energy_ratio'][-1] if history['energy_ratio'] else 0.0)

            pbar.set_postfix({
                'epoch': epoch,
                'test': f"{test_eval['accuracy']:.3f}",
                'inv': f"{metrics['inverse_acc']:.3f}",
                'T': f"{metrics['sampled_translation']:.2f}",
                'Q': f"{metrics['sampled_quadratic']:.2f}",
            })

    train_eval = eval_step(states['encoder'], states['task_head'], train_inputs, train_labels, args.p)
    test_eval = eval_step(states['encoder'], states['task_head'], test_inputs, test_labels, args.p)
    fourier_metrics = compute_fourier_metrics(states['encoder'], states['task_head'], args.p)

    peak_test_acc = max(history['test_acc'])
    peak_idx = history['test_acc'].index(peak_test_acc)
    peak_epoch = history['epoch'][peak_idx]

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Train accuracy: {float(train_eval['accuracy']):.4f}")
    print(f"  Test accuracy:  {float(test_eval['accuracy']):.4f}")
    print(f"  Peak test acc:  {peak_test_acc:.4f} at epoch {peak_epoch}")
    print(f"  Energy ratio:   {fourier_metrics['energy_ratio']:.4f}")

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
    with open(os.path.join(args.save_dir, 'random_perms.pkl'), 'wb') as f:
        pickle.dump(np.array(random_perms), f)
    with open(os.path.join(args.save_dir, 'cocktail_recipe.json'), 'w') as f:
        json.dump(cocktail, f, indent=2)
    with open(os.path.join(args.save_dir, 'final_metrics.json'), 'w') as f:
        json.dump({
            'final_train_acc': float(train_eval['accuracy']),
            'final_test_acc': float(test_eval['accuracy']),
            'peak_test_acc': float(peak_test_acc),
            'peak_epoch': int(peak_epoch),
            'energy_ratio': float(fourier_metrics['energy_ratio']),
            'n_actions': int(args.p + 3 * (args.p - 1)),
        }, f, indent=2)

    print(f"\nResults saved to {args.save_dir}")
    return history, states


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train inverse-only model on round-robin cocktail mixture')

    parser.add_argument('--round_robin_summary', type=str, required=True,
                        help='Path to the round-robin tournament summary.json file')
    parser.add_argument('--p', type=int, default=97)
    parser.add_argument('--train_fraction', type=float, default=0.3)
    parser.add_argument('--zipf_exponent', type=float, default=1.5)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_encoder_layers', type=int, default=2)
    parser.add_argument('--n_epochs', type=int, default=1000000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1.0)
    parser.add_argument('--icm_learning_rate', type=float, default=1e-3)
    parser.add_argument('--lambda_task', type=float, default=1.0)
    parser.add_argument('--beta_forward', type=float, default=0.0)
    parser.add_argument('--beta_inverse', type=float, default=0.1)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--fourier_interval', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps_per_update', type=int, default=100)

    args = parser.parse_args()
    train(args)
