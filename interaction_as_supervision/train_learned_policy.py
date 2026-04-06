"""
Learned Transformation Selection via Policy Gradients

This experiment tests whether policy gradients with INVERSE MODEL LEARNABILITY as reward
can discover which transformation types help learn Fourier representations,
avoiding the LP Trap that plagued learning progress approaches.

Key insight: The inverse model directly measures "can this transform be learned?"
- Translation has systematic structure (Fourier phase rotation) → low inverse loss
- Random permutations are arbitrary → high inverse loss (can't be compressed)

Principled design choice: The reward signal is Δ(inverse_loss), grounded in S₀.
The inverse model asks "what action caused this transition?" - transforms with
learnable structure (translation, scaling) will have lower inverse loss than
transforms without structure (random). This is the "patient instructor"
from questions.md - selecting which transforms have learnable structure.

Transformation space (4 types):
- Translation (a+k, b+k): 97 actions (k ∈ {0..96}) - HELPFUL (Fourier phase shift)
- Scaling (αa, αb): 96 actions (α ∈ {1..96}) - HELPFUL (Fourier frequency permutation)
- Quadratic (a²k, b²k): 96 actions (k ∈ {1..96}) - AMBIGUOUS (has modified Fourier structure)
- Random permutation (π(a), π(b)): 96 actions - UNHELPFUL (noisy TV - no structure)

Policy: 4-way unconditional policy over transform TYPES (not individual actions).
Reward: Δ(inverse_loss) measured every N batches (positive when inverse loss decreases).

See SPEC.md for full details.
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
from typing import Tuple, Dict, Any, Optional
from functools import partial


# =============================================================================
# TRANSFORMATION SPACE DEFINITIONS
# =============================================================================

# Constants for transformation types
TRANSFORM_TRANSLATION = 0
TRANSFORM_SCALING = 1
TRANSFORM_QUADRATIC = 2  # Quadratic: (a², b²) mod p - has structure but ambiguously helpful
TRANSFORM_RANDOM = 3

# Type names for logging
TYPE_NAMES = ['translation', 'scaling', 'quadratic', 'random']


def get_n_actions(n_translation: int, n_scaling: int, n_quadratic: int, n_random: int) -> int:
    """Total number of actions across all transformation types."""
    return n_translation + n_scaling + n_quadratic + n_random


def get_type_ranges(n_translation: int, n_scaling: int, n_quadratic: int, n_random: int) -> Dict[str, Tuple[int, int]]:
    """Get action ID ranges for each transformation type."""
    t_end = n_translation
    s_end = t_end + n_scaling
    q_end = s_end + n_quadratic
    r_end = q_end + n_random
    return {
        'translation': (0, t_end),
        'scaling': (t_end, s_end),
        'quadratic': (s_end, q_end),
        'random': (q_end, r_end),
    }


def get_n_enabled_actions(
    enabled_transforms: list,
    n_translation: int,
    n_scaling: int,
    n_quadratic: int,
    n_random: int,
) -> int:
    """
    Get number of actions for enabled transforms only.

    This is the correct output dimension for the inverse model when
    only a subset of transforms are enabled.
    """
    type_sizes = [n_translation, n_scaling, n_quadratic, n_random]
    return sum(type_sizes[t] for t in enabled_transforms)


def get_enabled_type_offsets(
    enabled_transforms: list,
    n_translation: int,
    n_scaling: int,
    n_quadratic: int,
    n_random: int,
) -> Dict[int, int]:
    """
    Get the starting offset for each enabled transform type in the
    contiguous enabled action space.

    Returns:
        Dict mapping transform_type -> offset in enabled action space
    """
    type_sizes = [n_translation, n_scaling, n_quadratic, n_random]
    offsets = {}
    current_offset = 0
    for t in sorted(enabled_transforms):
        offsets[t] = current_offset
        current_offset += type_sizes[t]
    return offsets


def generate_random_permutations(rng: jax.random.PRNGKey, p: int, n_random: int = 96) -> jnp.ndarray:
    """Generate n_random fixed random permutations of {0, ..., p-1}."""
    perms = []
    for i in range(n_random):
        rng, subkey = jax.random.split(rng)
        perm = jax.random.permutation(subkey, p)
        perms.append(perm)
    return jnp.stack(perms)  # Shape: (n_random, p)


def apply_transformation_batch(
    a: jnp.ndarray,
    b: jnp.ndarray,
    transform_types: jnp.ndarray,
    params: jnp.ndarray,
    p: int,
    random_perms: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply transformations to batched input pairs (a, b).

    Fully vectorized implementation for efficiency.

    Args:
        a, b: Input values, shape (batch_size,), values in {0, ..., p-1}
        transform_types: Shape (batch_size,), values in {0, 1, 2, 3}
        params: Shape (batch_size,), transformation parameters
        p: Prime modulus (97)
        random_perms: Precomputed random permutations, shape (n_random, p)

    Returns:
        (a', b'): Transformed pairs, each shape (batch_size,)
    """
    # Compute all transformation results (fully vectorized)

    # Translation: (a + param) % p, (b + param) % p
    a_translate = (a + params) % p
    b_translate = (b + params) % p

    # Scaling: (a * param) % p, (b * param) % p
    a_scale = (a * params) % p
    b_scale = (b * params) % p

    # Quadratic: (a² * param) % p, (b² * param) % p
    # Has modified Fourier structure but ambiguously helpful
    a_quadratic = (a * a * params) % p
    b_quadratic = (b * b * params) % p

    # Random permutation: (perm[param][a], perm[param][b])
    n_random = random_perms.shape[0]
    safe_perm_idx = jnp.clip(params, 0, n_random - 1)
    a_random = random_perms[safe_perm_idx, a]
    b_random = random_perms[safe_perm_idx, b]

    # Create masks for each type
    is_translation = (transform_types == TRANSFORM_TRANSLATION)
    is_scaling = (transform_types == TRANSFORM_SCALING)
    is_quadratic = (transform_types == TRANSFORM_QUADRATIC)
    is_random = (transform_types == TRANSFORM_RANDOM)

    # Select based on type using masks
    a_new = (
        a_translate * is_translation +
        a_scale * is_scaling +
        a_quadratic * is_quadratic +
        a_random * is_random
    )
    b_new = (
        b_translate * is_translation +
        b_scale * is_scaling +
        b_quadratic * is_quadratic +
        b_random * is_random
    )

    return a_new, b_new


# =============================================================================
# POLICY FOR TRANSFORM TYPE SELECTION
# =============================================================================

class TransformPolicy(nn.Module):
    """
    4-way unconditional policy over transform types.

    Learns which transform types help the task via policy gradients.
    Unconditional because we want to discover general transform usefulness,
    not context-dependent selection.
    """
    n_types: int = 4

    @nn.compact
    def __call__(self):
        # Learnable logits for each transform type
        logits = self.param('logits', nn.initializers.zeros, (self.n_types,))
        return logits  # [translation, scaling, quadratic, random]


def sample_transform_types_and_params(
    policy_logits: jnp.ndarray,
    rng: jax.random.PRNGKey,
    batch_size: int,
    n_translation: int,
    n_scaling: int,
    n_quadratic: int,
    n_random: int,
    temperature: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Sample transform types from policy, then sample params uniformly within type.

    Returns:
        transform_types: Shape (batch_size,), values in {0, 1, 2, 3}
        params: Shape (batch_size,), transformation parameters
        log_probs: Shape (batch_size,), log probabilities for REINFORCE
    """
    rng1, rng2, rng3 = jax.random.split(rng, 3)

    # Sample types with temperature
    scaled_logits = policy_logits / temperature
    log_probs_all = jax.nn.log_softmax(scaled_logits)

    # Sample transform types
    transform_types = jax.random.categorical(rng1, scaled_logits, shape=(batch_size,))

    # Get log probability of sampled types
    log_probs = log_probs_all[transform_types]

    # Sample params uniformly within each type
    # Generate uniform random values for each type
    type_sizes = jnp.array([n_translation, n_scaling, n_quadratic, n_random])
    type_offsets = jnp.array([0, 1, 0, 0])  # Scaling starts at 1, not 0

    max_size = type_sizes.max()
    raw_params = jax.random.randint(rng2, (batch_size,), 0, max_size)

    # Clamp to valid range for each type
    sizes = type_sizes[transform_types]
    offsets = type_offsets[transform_types]
    params = (raw_params % sizes) + offsets

    return transform_types, params.astype(jnp.int32), log_probs


def sample_single_type_from_policy(
    policy_logits: jnp.ndarray,
    rng: jax.random.PRNGKey,
    temperature: float = 1.0,
) -> Tuple[int, float]:
    """
    Sample a SINGLE transform type from the policy (for bandit-style learning).

    Returns:
        transform_type: Single int in {0, 1, 2, 3}
        log_prob: Log probability of the sampled type
    """
    scaled_logits = policy_logits / temperature
    log_probs = jax.nn.log_softmax(scaled_logits)

    transform_type = jax.random.categorical(rng, scaled_logits)
    log_prob = log_probs[transform_type]

    return int(transform_type), float(log_prob)


def sample_params_for_type(
    rng: jax.random.PRNGKey,
    transform_type: int,
    batch_size: int,
    n_translation: int,
    n_scaling: int,
    n_quadratic: int,
    n_random: int,
) -> jnp.ndarray:
    """
    Sample transformation parameters for a FIXED transform type.

    Args:
        rng: Random key
        transform_type: The transform type (0=translation, 1=scaling, 2=quadratic, 3=random)
        batch_size: Number of params to sample
        n_translation, n_scaling, n_quadratic, n_random: Number of actions per type

    Returns:
        params: Shape (batch_size,), transformation parameters for this type
    """
    type_sizes = [n_translation, n_scaling, n_quadratic, n_random]
    type_offsets = [0, 1, 0, 0]  # Scaling starts at 1, not 0

    size = type_sizes[transform_type]
    offset = type_offsets[transform_type]

    raw_params = jax.random.randint(rng, (batch_size,), 0, size)
    params = raw_params + offset

    return params.astype(jnp.int32)


def compute_policy_gradient_loss(
    policy_params: Dict,
    policy_apply_fn,
    transform_types_sampled: jnp.ndarray,
    rewards: jnp.ndarray,
    temperature: float = 1.0,
    entropy_weight: float = 0.1,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    REINFORCE with baseline subtraction and entropy regularization.

    Args:
        policy_params: Policy parameters
        policy_apply_fn: Policy apply function
        transform_types_sampled: Array of transform types sampled during this window
        rewards: Corresponding rewards (all same value for this window)
        temperature: Softmax temperature
        entropy_weight: Weight for entropy regularization

    Returns:
        loss: Scalar loss value
        metrics: Dict with entropy and mean advantage
    """
    logits = policy_apply_fn(policy_params)
    log_probs = jax.nn.log_softmax(logits / temperature)
    probs = jax.nn.softmax(logits / temperature)

    # Log probability of sampled types
    log_prob_sampled = log_probs[transform_types_sampled]

    # Baseline subtraction (simple: mean reward)
    baseline = jnp.mean(rewards)
    advantage = rewards - baseline

    # REINFORCE: minimize -E[advantage * log_prob]
    policy_loss = -jnp.mean(advantage * log_prob_sampled)

    # Entropy regularization to encourage exploration
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-8))

    # Total loss: policy loss - entropy bonus
    total_loss = policy_loss - entropy_weight * entropy

    metrics = {
        'entropy': entropy,
        'mean_advantage': jnp.mean(advantage),
        'policy_loss': policy_loss,
    }

    return total_loss, metrics


def compute_bandit_policy_gradient_loss(
    policy_params: Dict,
    policy_apply_fn,
    sampled_type: int,
    reward: float,
    baseline: float,
    temperature: float = 1.0,
    entropy_weight: float = 0.1,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Bandit-style REINFORCE: single (type, reward) update with per-type baseline.

    This provides clean credit assignment: each reward is directly attributable
    to one transform type.

    Args:
        policy_params: Policy parameters
        policy_apply_fn: Policy apply function
        sampled_type: The single transform type used this window (0-3)
        reward: The reward for this window (Δ train loss)
        baseline: Per-type baseline for the sampled type
        temperature: Softmax temperature
        entropy_weight: Weight for entropy regularization

    Returns:
        loss: Scalar loss value
        metrics: Dict with entropy and advantage
    """
    logits = policy_apply_fn(policy_params)
    log_probs = jax.nn.log_softmax(logits / temperature)
    probs = jax.nn.softmax(logits / temperature)

    # Log probability of the sampled type
    log_prob_sampled = log_probs[sampled_type]

    # Advantage with per-type baseline subtraction
    advantage = reward - baseline

    # REINFORCE: minimize -advantage * log_prob
    policy_loss = -advantage * log_prob_sampled

    # Entropy regularization to encourage exploration
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-8))

    # Total loss: policy loss - entropy bonus
    total_loss = policy_loss - entropy_weight * entropy

    metrics = {
        'entropy': entropy,
        'advantage': advantage,
        'policy_loss': policy_loss,
    }

    return total_loss, metrics


# =============================================================================
# MODEL DEFINITIONS (same as train_multi_transform.py)
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
    """MLP forward model: predicts z' from (z, action_id)."""
    hidden_dim: int = 128
    output_dim: int = 128
    n_actions: int = 386

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
    """Predicts action_id from (z(a,b), z(T(a,b)))."""
    hidden_dim: int = 128
    n_actions: int = 386

    @nn.compact
    def __call__(self, z1, z2):
        x = jnp.concatenate([z1, z2], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
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
    n_enabled_actions: int,
    hidden_dim: int,
    n_encoder_layers: int,
    learning_rate: float,
    weight_decay: float,
    icm_learning_rate: float,
    policy_learning_rate: float,
    policy_adam_beta1: float = 0.0,
    icm_hidden_dim: Optional[int] = None,
    n_policy_types: int = 4,
) -> Dict[str, train_state.TrainState]:
    """Create training states for encoder, task head, ICM, and policy.

    Args:
        n_enabled_actions: Number of actions for ENABLED transforms only.
                          This is the output dimension of the inverse model.
    """

    if icm_hidden_dim is None:
        icm_hidden_dim = hidden_dim

    rng_enc, rng_head, rng_forward, rng_inverse, rng_policy = jax.random.split(rng, 5)
    input_dim = 2 * p

    # Encoder
    encoder = Encoder(hidden_dim=hidden_dim, n_layers=n_encoder_layers)
    encoder_params = encoder.init(rng_enc, jnp.ones((1, input_dim)))
    encoder_tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    encoder_state = train_state.TrainState.create(
        apply_fn=encoder.apply, params=encoder_params, tx=encoder_tx
    )

    # Task head (no weight decay)
    task_head = TaskHead(output_dim=p)
    task_params = task_head.init(rng_head, jnp.ones((1, hidden_dim)))
    task_tx = optax.adam(learning_rate=learning_rate)
    task_state = train_state.TrainState.create(
        apply_fn=task_head.apply, params=task_params, tx=task_tx
    )

    # Forward model (MLP) - uses n_enabled_actions for action one-hot input
    forward_model = ForwardModelMLP(
        hidden_dim=icm_hidden_dim,
        output_dim=hidden_dim,
        n_actions=n_enabled_actions
    )
    forward_params = forward_model.init(rng_forward, jnp.ones((1, hidden_dim)), jnp.ones((1, n_enabled_actions)))
    forward_tx = optax.adam(learning_rate=icm_learning_rate)
    forward_state = train_state.TrainState.create(
        apply_fn=forward_model.apply, params=forward_params, tx=forward_tx
    )

    # Inverse model - uses n_enabled_actions for output dimension
    inverse_model = InverseModel(hidden_dim=icm_hidden_dim, n_actions=n_enabled_actions)
    inverse_params = inverse_model.init(rng_inverse, jnp.ones((1, hidden_dim)), jnp.ones((1, hidden_dim)))
    inverse_tx = optax.adam(learning_rate=icm_learning_rate)
    inverse_state = train_state.TrainState.create(
        apply_fn=inverse_model.apply, params=inverse_params, tx=inverse_tx
    )

    # Transform policy (n_policy_types-way unconditional)
    # Use Adam with configurable beta1 (default 0 = no momentum, like RMSprop)
    # This prevents momentum from overpowering small gradient signals when advantage changes sign
    policy = TransformPolicy(n_types=n_policy_types)
    policy_params = policy.init(rng_policy)
    policy_tx = optax.adam(learning_rate=policy_learning_rate, b1=policy_adam_beta1)
    policy_state = train_state.TrainState.create(
        apply_fn=policy.apply, params=policy_params, tx=policy_tx
    )

    return {
        'encoder': encoder_state,
        'task_head': task_state,
        'forward': forward_state,
        'inverse': inverse_state,
        'policy': policy_state,
    }


# =============================================================================
# TRAINING STEPS
# =============================================================================

def type_and_param_to_action_id(
    transform_type: jnp.ndarray,
    param: jnp.ndarray,
    n_translation: int,
    n_scaling: int,
    n_quadratic: int,
    n_random: int,
) -> jnp.ndarray:
    """Convert (transform_type, param) to action_id in FULL action space."""
    type_starts = jnp.array([0, n_translation, n_translation + n_scaling,
                            n_translation + n_scaling + n_quadratic])
    offsets = jnp.array([0, 1, 0, 0])  # Scaling params start at 1

    # For scaling, adjust param since it starts at 1 but index starts at 0
    adjusted_param = jnp.where(transform_type == TRANSFORM_SCALING, param - 1, param)

    action_id = type_starts[transform_type] + adjusted_param
    return action_id.astype(jnp.int32)


def type_and_param_to_enabled_action_id(
    transform_type: jnp.ndarray,
    param: jnp.ndarray,
    enabled_type_starts: jnp.ndarray,
) -> jnp.ndarray:
    """
    Convert (transform_type, param) to action_id in ENABLED-ONLY action space.

    This maps actions to a contiguous space containing only enabled transforms,
    which is the correct target for the inverse model when using a subset of transforms.

    Args:
        transform_type: Shape (batch_size,), values in {0, 1, 2, 3}
        param: Shape (batch_size,), transformation parameters
        enabled_type_starts: Shape (4,), start offset for each type in enabled space
                            (-1 for disabled types)

    Returns:
        action_ids: Shape (batch_size,), contiguous IDs in enabled action space
    """
    # For scaling, adjust param since it starts at 1 but index starts at 0
    adjusted_param = jnp.where(transform_type == TRANSFORM_SCALING, param - 1, param)

    action_id = enabled_type_starts[transform_type] + adjusted_param
    return action_id.astype(jnp.int32)


def compute_enabled_type_starts(
    enabled_transforms: list,
    n_translation: int,
    n_scaling: int,
    n_quadratic: int,
    n_random: int,
) -> jnp.ndarray:
    """
    Compute the starting offset for each transform type in the enabled action space.

    Returns array of shape (4,) where:
    - enabled_type_starts[t] = starting offset for type t if enabled
    - enabled_type_starts[t] = -1 if type t is disabled (should never be indexed)

    This is a JAX-friendly representation of get_enabled_type_offsets().
    """
    type_sizes = [n_translation, n_scaling, n_quadratic, n_random]
    enabled_type_starts = [-1, -1, -1, -1]  # -1 = disabled

    current_offset = 0
    for t in sorted(enabled_transforms):
        enabled_type_starts[t] = current_offset
        current_offset += type_sizes[t]

    return jnp.array(enabled_type_starts, dtype=jnp.int32)


@partial(jax.jit, static_argnames=['p', 'n_enabled_actions'])
def train_step_with_types(
    encoder_state: train_state.TrainState,
    task_state: train_state.TrainState,
    forward_state: train_state.TrainState,
    inverse_state: train_state.TrainState,
    inputs: jnp.ndarray,
    labels: jnp.ndarray,
    pairs: jnp.ndarray,
    weights: jnp.ndarray,
    all_inputs: jnp.ndarray,
    transform_types: jnp.ndarray,
    params: jnp.ndarray,
    random_perms: jnp.ndarray,
    enabled_type_starts: jnp.ndarray,
    p: int,
    n_enabled_actions: int,
    lambda_task: float,
    beta_forward: float,
    beta_inverse: float,
):
    """
    Training step with pre-sampled transform types and params.

    Args:
        enabled_type_starts: Shape (4,), start offset for each type in enabled action space
        n_enabled_actions: Number of actions for enabled transforms only
    """
    batch_size = inputs.shape[0]

    # Convert to action IDs in ENABLED action space (not full space)
    action_ids = type_and_param_to_enabled_action_id(
        transform_types, params, enabled_type_starts
    )
    action_onehot = jax.nn.one_hot(action_ids, n_enabled_actions)

    # Apply transformations to get (a', b')
    a = pairs[:, 0]
    b = pairs[:, 1]
    a_new, b_new = apply_transformation_batch(a, b, transform_types, params, p, random_perms)

    # Get transformed inputs
    transformed_indices = a_new * p + b_new
    inputs_transformed = all_inputs[transformed_indices]

    def combined_loss_fn(enc_params, task_params, forward_params, inverse_params):
        z = encoder_state.apply_fn(enc_params, inputs)
        logits = task_state.apply_fn(task_params, z)
        z_transformed = encoder_state.apply_fn(enc_params, inputs_transformed)
        z_transformed_sg = jax.lax.stop_gradient(z_transformed)

        # Task loss (Zipfian-weighted)
        one_hot = jax.nn.one_hot(labels, p)
        per_sample_task_loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1)
        task_loss = jnp.sum(per_sample_task_loss * weights)

        # Forward model loss (Zipfian-weighted)
        z_pred = forward_state.apply_fn(forward_params, z, action_onehot)
        per_sample_forward_loss = jnp.sum((z_pred - z_transformed_sg) ** 2, axis=-1)
        forward_loss = jnp.sum(per_sample_forward_loss * weights)

        # Inverse model loss (Zipfian-weighted)
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

    # Compute gradients
    (loss, metrics), grads = jax.value_and_grad(
        combined_loss_fn, argnums=(0, 1, 2, 3), has_aux=True
    )(encoder_state.params, task_state.params, forward_state.params, inverse_state.params)

    enc_grads, task_grads, forward_grads, inverse_grads = grads

    # Update states
    new_encoder_state = encoder_state.apply_gradients(grads=enc_grads)
    new_task_state = task_state.apply_gradients(grads=task_grads)
    new_forward_state = forward_state.apply_gradients(grads=forward_grads)
    new_inverse_state = inverse_state.apply_gradients(grads=inverse_grads)

    # Compute inverse accuracy
    z = encoder_state.apply_fn(encoder_state.params, inputs)
    z_transformed = encoder_state.apply_fn(encoder_state.params, inputs_transformed)
    t_logits = inverse_state.apply_fn(inverse_state.params, z, jax.lax.stop_gradient(z_transformed))
    inverse_acc = jnp.mean(jnp.argmax(t_logits, axis=-1) == action_ids)
    metrics['inverse_acc'] = inverse_acc

    return new_encoder_state, new_task_state, new_forward_state, new_inverse_state, metrics


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
# COUNTERFACTUAL EVALUATION
# =============================================================================

@partial(jax.jit, static_argnames=['p', 'n_enabled_actions', 'n_translation', 'n_scaling', 'n_quadratic', 'n_random', 'transform_type'])
def compute_counterfactual_inverse_loss(
    encoder_state: train_state.TrainState,
    inverse_state: train_state.TrainState,
    eval_inputs: jnp.ndarray,
    eval_pairs: jnp.ndarray,
    all_inputs: jnp.ndarray,
    random_perms: jnp.ndarray,
    enabled_type_starts: jnp.ndarray,
    rng: jax.random.PRNGKey,
    p: int,
    n_enabled_actions: int,
    n_translation: int,
    n_scaling: int,
    n_quadratic: int,
    n_random: int,
    transform_type: int,
):
    """
    Compute inverse loss for a specific transform type on a held-out evaluation set.

    This is used for counterfactual policy evaluation: we compare the inverse loss
    for different transform types on the SAME encoder state and evaluation set,
    giving a clean signal about which type is currently more learnable.

    No gradients are computed; this is evaluation only.

    Args:
        encoder_state: Current encoder state
        inverse_state: Current inverse model state
        eval_inputs: One-hot encoded inputs for evaluation pairs, shape (n_eval, 2*p)
        eval_pairs: (a, b) pairs for evaluation, shape (n_eval, 2)
        all_inputs: All possible one-hot encoded inputs, shape (p*p, 2*p)
        random_perms: Precomputed random permutations, shape (n_random, p)
        enabled_type_starts: Shape (4,), start offset for each type in enabled action space
        rng: Random key for sampling transform parameters
        p: Prime modulus
        n_enabled_actions: Number of actions for enabled transforms only
        n_translation, n_scaling, n_quadratic, n_random: Action counts per type
        transform_type: Which transform type to evaluate (0=T, 1=S, 2=Q, 3=R)

    Returns:
        mean_inverse_loss: Mean inverse loss (cross-entropy) for this transform type
    """
    batch_size = eval_inputs.shape[0]

    # Sample parameters for this transform type
    transform_types = jnp.full(batch_size, transform_type, dtype=jnp.int32)
    params = sample_params_for_type(
        rng, transform_type, batch_size,
        n_translation, n_scaling, n_quadratic, n_random
    )

    # Convert to action IDs in ENABLED action space
    action_ids = type_and_param_to_enabled_action_id(
        transform_types, params, enabled_type_starts
    )
    action_onehot = jax.nn.one_hot(action_ids, n_enabled_actions)

    # Apply transformations to get (a', b')
    a = eval_pairs[:, 0]
    b = eval_pairs[:, 1]
    a_new, b_new = apply_transformation_batch(a, b, transform_types, params, p, random_perms)

    # Get transformed inputs
    transformed_indices = a_new * p + b_new
    inputs_transformed = all_inputs[transformed_indices]

    # Compute encoder representations
    z = encoder_state.apply_fn(encoder_state.params, eval_inputs)
    z_transformed = encoder_state.apply_fn(encoder_state.params, inputs_transformed)

    # Compute inverse model prediction
    t_logits = inverse_state.apply_fn(inverse_state.params, z, z_transformed)

    # Compute cross-entropy loss (no weighting - uniform over eval set)
    per_sample_loss = -jnp.sum(action_onehot * jax.nn.log_softmax(t_logits), axis=-1)
    mean_loss = jnp.mean(per_sample_loss)

    # Also compute accuracy for logging
    accuracy = jnp.mean(jnp.argmax(t_logits, axis=-1) == action_ids)

    return mean_loss, accuracy


def create_counterfactual_eval_set(
    p: int,
    n_eval: int,
    seed: int,
    train_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a held-out evaluation set for counterfactual policy evaluation.

    The eval set is sampled from pairs NOT in the training set, ensuring
    the counterfactual evaluation doesn't leak training data.

    Args:
        p: Prime modulus
        n_eval: Number of evaluation pairs
        seed: Random seed
        train_indices: Indices of training pairs (to exclude)

    Returns:
        eval_inputs: One-hot encoded inputs, shape (n_eval, 2*p)
        eval_pairs: (a, b) pairs, shape (n_eval, 2)
        eval_indices: Indices into the full p*p array
    """
    rng = np.random.default_rng(seed + 12345)  # Different seed from training

    # Get all possible indices, excluding training
    all_indices = set(range(p * p))
    train_set = set(train_indices.tolist())
    available_indices = list(all_indices - train_set)

    # Sample from available indices
    n_eval = min(n_eval, len(available_indices))
    eval_indices = rng.choice(available_indices, size=n_eval, replace=False)

    # Create inputs and pairs
    eval_inputs = np.zeros((n_eval, 2 * p), dtype=np.float32)
    eval_pairs = np.zeros((n_eval, 2), dtype=np.int32)

    for i, idx in enumerate(eval_indices):
        a = idx // p
        b = idx % p
        eval_inputs[i, a] = 1.0
        eval_inputs[i, p + b] = 1.0
        eval_pairs[i] = [a, b]

    return eval_inputs, eval_pairs, eval_indices


@partial(jax.jit, static_argnames=['n_batches', 'transform_type', 'p', 'n_enabled_actions', 'n_translation', 'n_scaling', 'n_quadratic', 'n_random'])
def evaluate_inverse_loss_no_train(
    encoder_state: train_state.TrainState,
    inverse_state: train_state.TrainState,
    eval_inputs: jnp.ndarray,
    eval_pairs: jnp.ndarray,
    all_inputs: jnp.ndarray,
    random_perms: jnp.ndarray,
    enabled_type_starts: jnp.ndarray,
    rng: jax.random.PRNGKey,
    n_batches: int,
    transform_type: int,
    p: int,
    n_enabled_actions: int,
    n_translation: int,
    n_scaling: int,
    n_quadratic: int,
    n_random: int,
):
    """
    Evaluate inverse loss for a transform type over multiple batches WITHOUT training.

    This is the evaluation-only counterpart to run_calibration_window_scan.
    Used for policy-guided training where we need to measure both types' performance
    without training on them.

    Returns: (avg_inverse_loss, avg_accuracy, updated_rng)
    """
    batch_size = eval_inputs.shape[0]

    # Pre-generate all RNG keys for the batches
    rngs = jax.random.split(rng, n_batches + 1)
    rng_out = rngs[-1]

    def eval_single_batch(rng_batch):
        # Sample parameters for this transform type
        transform_types = jnp.full(batch_size, transform_type, dtype=jnp.int32)
        params = sample_params_for_type(
            rng_batch, transform_type, batch_size,
            n_translation, n_scaling, n_quadratic, n_random
        )

        # Convert to action IDs in enabled space
        action_ids = type_and_param_to_enabled_action_id(
            transform_types, params, enabled_type_starts
        )
        action_onehot = jax.nn.one_hot(action_ids, n_enabled_actions)

        # Apply transformations
        a = eval_pairs[:, 0]
        b = eval_pairs[:, 1]
        a_new, b_new = apply_transformation_batch(a, b, transform_types, params, p, random_perms)
        transformed_indices = a_new * p + b_new
        inputs_transformed = all_inputs[transformed_indices]

        # Compute representations (no gradients)
        z = encoder_state.apply_fn(encoder_state.params, eval_inputs)
        z_transformed = encoder_state.apply_fn(encoder_state.params, inputs_transformed)

        # Compute inverse loss
        t_logits = inverse_state.apply_fn(inverse_state.params, z, z_transformed)
        per_sample_loss = -jnp.sum(action_onehot * jax.nn.log_softmax(t_logits), axis=-1)
        mean_loss = jnp.mean(per_sample_loss)
        accuracy = jnp.mean(jnp.argmax(t_logits, axis=-1) == action_ids)

        return mean_loss, accuracy

    # Evaluate over all batches using vmap
    losses, accuracies = jax.vmap(eval_single_batch)(rngs[:n_batches])

    return jnp.mean(losses), jnp.mean(accuracies), rng_out


def evaluate_all_types_no_train(
    states: Dict[str, train_state.TrainState],
    enabled_transforms: list,
    n_batches: int,
    eval_inputs: jnp.ndarray,
    eval_pairs: jnp.ndarray,
    all_inputs: jnp.ndarray,
    random_perms: jnp.ndarray,
    enabled_type_starts: jnp.ndarray,
    rng: jax.random.PRNGKey,
    p: int,
    n_enabled_actions: int,
    n_translation: int,
    n_scaling: int,
    n_quadratic: int,
    n_random: int,
) -> Tuple[Dict[int, float], Dict[int, float], jax.random.PRNGKey]:
    """
    Evaluate inverse loss for ALL enabled transform types WITHOUT training.

    This provides a clean calibration signal for policy updates without
    corrupting the representations by training on all types.

    Returns:
        eval_losses: Dict mapping transform_type -> avg_inverse_loss
        eval_accuracies: Dict mapping transform_type -> avg_inverse_accuracy
        rng: Updated RNG key
    """
    eval_losses = {}
    eval_accuracies = {}

    for transform_type in enabled_transforms:
        avg_loss, avg_acc, rng = evaluate_inverse_loss_no_train(
            states['encoder'], states['inverse'],
            eval_inputs, eval_pairs, all_inputs, random_perms,
            enabled_type_starts, rng, n_batches, transform_type,
            p, n_enabled_actions, n_translation, n_scaling, n_quadratic, n_random,
        )
        eval_losses[transform_type] = float(avg_loss)
        eval_accuracies[transform_type] = float(avg_acc)

    return eval_losses, eval_accuracies, rng


@partial(jax.jit, static_argnames=['n_batches', 'transform_type', 'p', 'n_enabled_actions', 'n_translation', 'n_scaling', 'n_quadratic', 'n_random'])
def run_calibration_window_scan(
    encoder_state: train_state.TrainState,
    task_state: train_state.TrainState,
    forward_state: train_state.TrainState,
    inverse_state: train_state.TrainState,
    train_inputs: jnp.ndarray,
    train_labels: jnp.ndarray,
    train_pairs: jnp.ndarray,
    weights: jnp.ndarray,
    all_inputs: jnp.ndarray,
    random_perms: jnp.ndarray,
    enabled_type_starts: jnp.ndarray,
    rng: jax.random.PRNGKey,
    n_batches: int,
    transform_type: int,
    p: int,
    n_enabled_actions: int,
    n_translation: int,
    n_scaling: int,
    n_quadratic: int,
    n_random: int,
    lambda_task: float,
    beta_forward: float,
    beta_inverse: float,
):
    """
    Run a calibration window using jax.lax.scan for maximum efficiency.

    After JIT compilation is amortized, this is ~5.6x faster than the Python loop
    because the entire n_batches loop is compiled into a single optimized kernel.

    Returns: (updated_states_tuple, avg_inverse_loss, updated_rng)
    """
    batch_size = train_inputs.shape[0]

    # Pre-generate all RNG keys for the window
    rngs = jax.random.split(rng, n_batches + 1)
    rng_out = rngs[-1]

    # Pre-compute type-specific sampling info
    type_sizes = jnp.array([n_translation, n_scaling, n_quadratic, n_random])
    type_offsets = jnp.array([0, 1, 0, 0])
    size = type_sizes[transform_type]
    offset = type_offsets[transform_type]

    # Pre-generate all params for all batches (vectorized)
    all_raw_params = jax.vmap(
        lambda r: jax.random.randint(r, (batch_size,), 0, size)
    )(rngs[:n_batches])
    all_params = (all_raw_params + offset).astype(jnp.int32)

    # Create transform_types array (same for all batches)
    transform_types = jnp.full(batch_size, transform_type, dtype=jnp.int32)

    def scan_body(carry, batch_params):
        enc_state, task_state, fwd_state, inv_state = carry

        new_enc, new_task, new_fwd, new_inv, metrics = train_step_with_types(
            enc_state, task_state, fwd_state, inv_state,
            train_inputs, train_labels, train_pairs, weights, all_inputs,
            transform_types, batch_params, random_perms,
            enabled_type_starts,
            p, n_enabled_actions,
            lambda_task, beta_forward, beta_inverse,
        )

        return (new_enc, new_task, new_fwd, new_inv), metrics['inverse_loss']

    # Run the scan - entire loop compiled into single kernel
    init_carry = (encoder_state, task_state, forward_state, inverse_state)
    (final_enc, final_task, final_fwd, final_inv), inverse_losses = jax.lax.scan(
        scan_body, init_carry, all_params
    )

    # Compute average loss
    avg_loss = jnp.mean(inverse_losses)

    return (final_enc, final_task, final_fwd, final_inv), avg_loss, rng_out


def run_calibration_window(
    states: Dict[str, train_state.TrainState],
    transform_type: int,
    n_batches: int,
    train_inputs: jnp.ndarray,
    train_labels: jnp.ndarray,
    train_pairs: jnp.ndarray,
    weights: jnp.ndarray,
    all_inputs: jnp.ndarray,
    random_perms: jnp.ndarray,
    enabled_type_starts: jnp.ndarray,
    rng: jax.random.PRNGKey,
    p: int,
    n_enabled_actions: int,
    n_translation: int,
    n_scaling: int,
    n_quadratic: int,
    n_random: int,
    lambda_task: float,
    beta_forward: float,
    beta_inverse: float,
) -> Tuple[Dict[str, train_state.TrainState], float, jax.random.PRNGKey]:
    """
    Run a calibration window: train for n_batches with a specific transform type.

    This is a wrapper around run_calibration_window_scan which uses jax.lax.scan
    to compile the entire loop into a single optimized kernel (~5.6x faster than
    Python loop after compilation is amortized).

    Returns the updated states, the final average inverse loss, and the updated rng.
    """
    (final_enc, final_task, final_fwd, final_inv), avg_loss, rng = run_calibration_window_scan(
        states['encoder'], states['task_head'], states['forward'], states['inverse'],
        train_inputs, train_labels, train_pairs, weights, all_inputs, random_perms,
        enabled_type_starts,
        rng, n_batches, transform_type,
        p, n_enabled_actions, n_translation, n_scaling, n_quadratic, n_random,
        lambda_task, beta_forward, beta_inverse,
    )

    # Update states dict
    states['encoder'] = final_enc
    states['task_head'] = final_task
    states['forward'] = final_fwd
    states['inverse'] = final_inv

    return states, float(avg_loss), rng


@partial(jax.jit, static_argnames=['p', 'n_enabled_actions', 'n_types'])
def interleaved_train_step(
    encoder_state: train_state.TrainState,
    task_state: train_state.TrainState,
    forward_state: train_state.TrainState,
    inverse_state: train_state.TrainState,
    inputs: jnp.ndarray,
    labels: jnp.ndarray,
    pairs: jnp.ndarray,
    weights: jnp.ndarray,
    all_inputs: jnp.ndarray,
    transform_types: jnp.ndarray,  # Shape: (batch_size * n_types,) - interleaved T/R/T/R/...
    params: jnp.ndarray,  # Shape: (batch_size * n_types,)
    type_masks: jnp.ndarray,  # Shape: (n_types, batch_size * n_types) - for loss separation
    random_perms: jnp.ndarray,
    enabled_type_starts: jnp.ndarray,
    p: int,
    n_enabled_actions: int,
    lambda_task: float,
    beta_forward: float,
    beta_inverse: float,
    n_types: int,
):
    """
    Training step with interleaved transform types for parallel GPU computation.

    Instead of training T then R sequentially, we train on a batch containing
    BOTH T and R samples simultaneously. This doubles GPU utilization.

    The batch is: [T_sample_1, R_sample_1, T_sample_2, R_sample_2, ...]

    Returns per-type inverse losses for fair comparison.
    """
    mega_batch_size = transform_types.shape[0]

    # Tile inputs/labels/pairs/weights to match the mega batch
    n_repeats = mega_batch_size // inputs.shape[0]
    inputs_tiled = jnp.tile(inputs, (n_repeats, 1))
    labels_tiled = jnp.tile(labels, (n_repeats,))
    pairs_tiled = jnp.tile(pairs, (n_repeats, 1))
    weights_tiled = jnp.tile(weights, (n_repeats,)) / n_repeats  # Normalize weights

    # Convert to action IDs in ENABLED action space
    action_ids = type_and_param_to_enabled_action_id(
        transform_types, params, enabled_type_starts
    )
    action_onehot = jax.nn.one_hot(action_ids, n_enabled_actions)

    # Apply transformations
    a = pairs_tiled[:, 0]
    b = pairs_tiled[:, 1]
    a_new, b_new = apply_transformation_batch(a, b, transform_types, params, p, random_perms)
    transformed_indices = a_new * p + b_new
    inputs_transformed = all_inputs[transformed_indices]

    def combined_loss_fn(enc_params, task_params, fwd_params, inv_params):
        z = encoder_state.apply_fn(enc_params, inputs_tiled)
        logits = task_state.apply_fn(task_params, z)
        z_transformed = encoder_state.apply_fn(enc_params, inputs_transformed)
        z_transformed_sg = jax.lax.stop_gradient(z_transformed)

        one_hot = jax.nn.one_hot(labels_tiled, p)
        per_sample_task_loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1)
        task_loss = jnp.sum(per_sample_task_loss * weights_tiled)

        z_pred = forward_state.apply_fn(fwd_params, z, action_onehot)
        per_sample_forward_loss = jnp.sum((z_pred - z_transformed_sg) ** 2, axis=-1)
        forward_loss = jnp.sum(per_sample_forward_loss * weights_tiled)

        t_logits = inverse_state.apply_fn(inv_params, z, z_transformed_sg)
        per_sample_inverse_loss = -jnp.sum(action_onehot * jax.nn.log_softmax(t_logits), axis=-1)
        inverse_loss = jnp.sum(per_sample_inverse_loss * weights_tiled)

        # Compute per-type inverse losses for comparison (vectorized)
        # type_masks shape: (n_types, mega_batch_size)
        # per_sample_inverse_loss shape: (mega_batch_size,)
        masked_losses = type_masks * per_sample_inverse_loss[None, :]  # (n_types, mega_batch_size)
        per_type_inverse_losses = jnp.sum(masked_losses, axis=1) / jnp.sum(type_masks, axis=1)

        total_loss = lambda_task * task_loss + beta_inverse * inverse_loss + beta_forward * forward_loss

        return total_loss, {
            'task_loss': task_loss,
            'forward_loss': forward_loss,
            'inverse_loss': inverse_loss,
            'total_loss': total_loss,
            'per_type_inverse_losses': per_type_inverse_losses,
        }

    (loss, metrics), grads = jax.value_and_grad(
        combined_loss_fn, argnums=(0, 1, 2, 3), has_aux=True
    )(encoder_state.params, task_state.params, forward_state.params, inverse_state.params)

    enc_grads, task_grads, fwd_grads, inv_grads = grads

    new_encoder_state = encoder_state.apply_gradients(grads=enc_grads)
    new_task_state = task_state.apply_gradients(grads=task_grads)
    new_forward_state = forward_state.apply_gradients(grads=fwd_grads)
    new_inverse_state = inverse_state.apply_gradients(grads=inv_grads)

    return new_encoder_state, new_task_state, new_forward_state, new_inverse_state, metrics


def run_interleaved_calibration(
    states: Dict[str, train_state.TrainState],
    enabled_transforms: list,
    n_batches: int,
    train_inputs: jnp.ndarray,
    train_labels: jnp.ndarray,
    train_pairs: jnp.ndarray,
    weights: jnp.ndarray,
    all_inputs: jnp.ndarray,
    random_perms: jnp.ndarray,
    enabled_type_starts: jnp.ndarray,
    rng: jax.random.PRNGKey,
    p: int,
    n_enabled_actions: int,
    n_translation: int,
    n_scaling: int,
    n_quadratic: int,
    n_random: int,
    lambda_task: float,
    beta_forward: float,
    beta_inverse: float,
) -> Tuple[Dict[str, train_state.TrainState], Dict[int, float], jax.random.PRNGKey]:
    """
    Run calibration with interleaved transform types for GPU parallelism.

    Instead of training 500 batches of T then 500 batches of R (sequential),
    we train 500 batches where each batch contains BOTH T and R samples.
    This utilizes GPU parallelism while maintaining fair comparison.

    Returns:
        states: Updated model states
        calibration_losses: Dict mapping transform_type -> avg_inverse_loss
        rng: Updated RNG key
    """
    n_types = len(enabled_transforms)
    batch_size = train_inputs.shape[0]
    mega_batch_size = batch_size * n_types

    # Pre-create type masks for loss separation (computed once)
    type_masks = jnp.zeros((n_types, mega_batch_size))
    for i in range(n_types):
        mask = jnp.zeros(mega_batch_size)
        mask = mask.at[i::n_types].set(1.0)
        type_masks = type_masks.at[i].set(mask)

    # Pre-create the interleaved transform type array (same for all batches)
    # Pattern: [T, R, T, R, ...] repeated batch_size times
    transform_types = jnp.array(
        [enabled_transforms[i % n_types] for i in range(mega_batch_size)],
        dtype=jnp.int32
    )

    # Pre-compute type-specific sampling info
    type_sizes = jnp.array([n_translation, n_scaling, n_quadratic, n_random])
    type_offsets = jnp.array([0, 1, 0, 0])
    max_size = max(n_translation, n_scaling, n_quadratic, n_random)

    # Accumulate per-type losses
    type_loss_accum = {t: [] for t in enabled_transforms}

    for batch_idx in range(n_batches):
        rng, rng_params = jax.random.split(rng)

        # Generate all parameters at once (vectorized)
        # Each position needs a param for its corresponding transform type
        sizes = type_sizes[transform_types]
        offsets = type_offsets[transform_types]
        raw_params = jax.random.randint(rng_params, (mega_batch_size,), 0, max_size)
        params = ((raw_params % sizes) + offsets).astype(jnp.int32)

        # Training step with interleaved batch
        states['encoder'], states['task_head'], states['forward'], states['inverse'], metrics = \
            interleaved_train_step(
                states['encoder'], states['task_head'], states['forward'], states['inverse'],
                train_inputs, train_labels, train_pairs, weights, all_inputs,
                transform_types, params, type_masks, random_perms,
                enabled_type_starts,
                p, n_enabled_actions,
                lambda_task, beta_forward, beta_inverse, n_types,
            )

        # Accumulate per-type losses
        per_type_losses = metrics['per_type_inverse_losses']
        for i, transform_type in enumerate(enabled_transforms):
            type_loss_accum[transform_type].append(float(per_type_losses[i]))

    # Compute average losses per type
    calibration_losses = {t: np.mean(losses) for t, losses in type_loss_accum.items()}

    return states, calibration_losses, rng


def run_alternating_then_compare(
    states: Dict[str, train_state.TrainState],
    enabled_transforms: list,
    n_batches: int,
    train_inputs: jnp.ndarray,
    train_labels: jnp.ndarray,
    train_pairs: jnp.ndarray,
    weights: jnp.ndarray,
    all_inputs: jnp.ndarray,
    random_perms: jnp.ndarray,
    enabled_type_starts: jnp.ndarray,
    rng: jax.random.PRNGKey,
    p: int,
    n_enabled_actions: int,
    n_translation: int,
    n_scaling: int,
    n_quadratic: int,
    n_random: int,
    lambda_task: float,
    beta_forward: float,
    beta_inverse: float,
) -> Tuple[Dict[str, train_state.TrainState], Dict[int, float], jax.random.PRNGKey]:
    """
    Run alternating calibration: train with each type sequentially, then return losses.

    This is the "fair A/B" approach - we train the actual model with each type
    in sequence, recording the inverse loss achieved by each.

    Returns:
        states: Updated model states (after training with all types)
        calibration_losses: Dict mapping transform_type -> avg_inverse_loss
        rng: Updated RNG key
    """
    n_types = len(enabled_transforms)

    # Randomize order to avoid bias
    rng, rng_order = jax.random.split(rng)
    order = list(range(n_types))
    np.random.shuffle(order)

    calibration_losses = {}
    for policy_idx in order:
        transform_type = enabled_transforms[policy_idx]
        states, avg_loss, rng = run_calibration_window(
            states, transform_type, n_batches,
            train_inputs, train_labels, train_pairs, weights, all_inputs,
            random_perms, enabled_type_starts, rng,
            p, n_enabled_actions, n_translation, n_scaling, n_quadratic, n_random,
            lambda_task, beta_forward, beta_inverse,
        )
        calibration_losses[transform_type] = avg_loss

    return states, calibration_losses, rng


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
    """Main training function with policy gradient transform selection."""

    rng = jax.random.PRNGKey(args.seed)

    # Compute action space sizes
    n_translation = args.n_translation if args.n_translation is not None else args.p
    n_scaling = args.n_scaling if args.n_scaling is not None else args.p - 1
    n_quadratic = args.n_quadratic if args.n_quadratic is not None else args.p
    n_random = args.n_random if args.n_random is not None else args.p - 1

    n_actions = get_n_actions(n_translation, n_scaling, n_quadratic, n_random)
    icm_hidden_dim = args.icm_hidden_dim if args.icm_hidden_dim is not None else args.hidden_dim

    # Parse enabled transforms
    enabled_transforms = [int(x) for x in args.enabled_transforms.split(',')]
    n_enabled_types = len(enabled_transforms)
    type_names_enabled = [TYPE_NAMES[i] for i in enabled_transforms]

    # Compute ENABLED action space size (for inverse model)
    # This fixes the overparameterization bug where the inverse model was sized
    # for all 385 actions but only trained on enabled transforms (e.g., 193 for T+R)
    n_enabled_actions = get_n_enabled_actions(
        enabled_transforms, n_translation, n_scaling, n_quadratic, n_random
    )
    enabled_type_starts = compute_enabled_type_starts(
        enabled_transforms, n_translation, n_scaling, n_quadratic, n_random
    )

    print(f"Learned Transformation Selection via Policy Gradients")
    print(f"=" * 60)
    print(f"Transformation space: {n_actions} total actions")
    print(f"  - Translation: {n_translation} actions (HELPFUL)")
    print(f"  - Scaling: {n_scaling} actions (HELPFUL)")
    print(f"  - Quadratic: {n_quadratic} actions (AMBIGUOUS)")
    print(f"  - Random: {n_random} actions (UNHELPFUL)")
    print(f"  - Helpful: {n_translation + n_scaling} | Ambiguous: {n_quadratic} | Unhelpful: {n_random}")
    print(f"\nEnabled transforms: {type_names_enabled} ({n_enabled_types} types)")
    print(f"Enabled action space: {n_enabled_actions} actions (inverse model output)")
    print(f"Policy mode: {args.policy_mode}")

    # Generate random permutations
    rng, rng_perms = jax.random.split(rng)
    random_perms = generate_random_permutations(rng_perms, args.p, n_random)
    print(f"\nGenerated {n_random} random permutations (seed={args.seed})")

    # Create dataset
    print(f"\nCreating dataset with p={args.p}, train_fraction={args.train_fraction}")
    data = create_dataset(args.p, args.train_fraction, args.seed)

    # Create Zipfian weights
    n_train = len(data['train_indices'])
    weights = zipf_weights(n_train, args.zipf_exponent)
    zipf_info = compute_zipf_info(weights)
    print(f"Zipf exponent: {args.zipf_exponent}")
    print(f"  Weight ratio: {zipf_info['weight_ratio']:.2f}")
    print(f"  Top 10% mass: {zipf_info['top_10pct_mass']:.4f}")
    print(f"  Effective N: {zipf_info['effective_n']:.2f}")

    # Create training states (policy has n_enabled_types logits, not always 4)
    # NOTE: We pass n_enabled_actions for the inverse/forward models, NOT n_actions
    rng, rng_init = jax.random.split(rng)
    states = create_train_states(
        rng_init, args.p, n_enabled_actions, args.hidden_dim, args.n_encoder_layers,
        args.learning_rate, args.weight_decay, args.icm_learning_rate,
        args.policy_learning_rate, policy_adam_beta1=args.policy_adam_beta1,
        icm_hidden_dim=icm_hidden_dim, n_policy_types=n_enabled_types,
    )

    # Prepare data arrays
    train_inputs = jnp.array(data['all_inputs'][data['train_indices']])
    train_labels = jnp.array(data['all_labels'][data['train_indices']])
    train_pairs = jnp.array(data['all_pairs'][data['train_indices']])
    test_inputs = jnp.array(data['all_inputs'][data['test_indices']])
    test_labels = jnp.array(data['all_labels'][data['test_indices']])
    all_inputs = jnp.array(data['all_inputs'])
    weights_jax = jnp.array(weights)

    # Create counterfactual evaluation set (if enabled)
    if args.counterfactual_eval:
        cf_eval_inputs, cf_eval_pairs, cf_eval_indices = create_counterfactual_eval_set(
            args.p, args.counterfactual_eval_size, args.seed, data['train_indices']
        )
        cf_eval_inputs = jnp.array(cf_eval_inputs)
        cf_eval_pairs = jnp.array(cf_eval_pairs)
        print(f"\nCounterfactual evaluation: ENABLED")
        print(f"  Eval set size: {len(cf_eval_indices)} pairs (held-out from test set)")
        print(f"  Samples per type: {args.counterfactual_n_samples}")
        print(f"  Reward = mean(loss_other_types) - loss_sampled_type")

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
        # Policy metrics
        'prob_translation': [],
        'prob_scaling': [],
        'prob_quadratic': [],
        'prob_random': [],
        'policy_entropy': [],
        'reward_signal': [],
    }

    # Policy history for detailed tracking
    policy_history = {
        'epoch': [],
        'logits': [],
        'probs': [],
        'reward': [],
        'entropy': [],
        'sampled_type': [],  # Which type was used this window (bandit-style)
        'baseline': [],  # Per-type baseline used for this update
        'all_baselines': [],  # All 4 per-type baselines at each update
        'type_counts': [],  # How many times each type has been sampled
        'cf_losses': [],  # Counterfactual losses per type (if enabled)
    }

    print(f"\nStarting training for {args.n_epochs} epochs...")
    print(f"Loss weights: lambda_task={args.lambda_task}, beta_forward={args.beta_forward}, beta_inverse={args.beta_inverse}")
    print(f"Policy: lr={args.policy_learning_rate}, adam_beta1={args.policy_adam_beta1}, temperature={args.policy_temperature}, entropy_weight={args.entropy_weight}")

    if args.policy_guided_calibration:
        print(f"Reward signal: POLICY-GUIDED CALIBRATION (RECOMMENDED)")
        print(f"  Training: Uses policy-sampled transforms ({args.calibration_window_size} batches per cycle)")
        print(f"  Evaluation: Tests ALL types without training ({args.eval_calibration_batches} batches each)")
        print(f"  Advantage: Policy guides training while calibration provides clean reward signal\n")
    elif args.alternating_calibration or args.interleaved_calibration:
        if args.interleaved_calibration:
            print(f"Reward signal: INTERLEAVED CALIBRATION - train T and R in parallel batches")
            print(f"  Window size: {args.calibration_window_size} batches (each contains both types)")
            print(f"  GPU parallelism: ENABLED (2x speedup)")
        else:
            print(f"Reward signal: ALTERNATING CALIBRATION - train both types, compare final losses")
            print(f"  Window size per type: {args.calibration_window_size} batches")
            print(f"  Total per policy update: {args.calibration_window_size * 2} batches")
        print(f"Credit assignment: A/B TEST (fair comparison, no recency bias)\n")
    elif args.counterfactual_eval:
        print(f"Policy update interval: {args.policy_update_interval} batches")
        print(f"Reward signal: COUNTERFACTUAL - loss_others - loss_sampled on held-out set")
        print(f"Credit assignment: BANDIT-STYLE (one type per window)")
        print(f"Transition filtering: N/A (counterfactual eval eliminates transition noise)\n")
    else:
        print(f"Policy update interval: {args.policy_update_interval} batches")
        print(f"Reward signal: Δ(inverse_loss) - measures transform learnability")
        print(f"Credit assignment: BANDIT-STYLE (one type per window)")
        if args.no_transition_filtering:
            print(f"Transition filtering: DISABLED (update policy on all windows)\n")
        else:
            print(f"Transition filtering: ENABLED (skip policy updates on type switches)\n")

    # Training loop
    batch_size = train_inputs.shape[0]
    inverse_loss_history = []  # Track inverse loss for reward signal
    inverse_loss_window = []  # Accumulate inverse loss within each window
    reward = 0.0  # Initialize reward for logging

    # Global baseline (running average of ALL rewards, regardless of type)
    # This stabilizes learning while preserving relative differences between types
    global_baseline = 0.0
    per_type_counts = [0] * n_enabled_types  # Track how many times each type was sampled
    baseline_decay = 0.95  # EMA decay - slightly slower to reduce variance

    # Bandit-style: sample ONE transform type per window
    current_window_type = None  # Will be sampled at start of each window (actual transform type)
    current_policy_idx = None  # Index into enabled_transforms (for policy gradient)
    previous_window_type = None  # Track previous window's type to filter transition rewards

    # =========================================================================
    # POLICY-GUIDED CALIBRATION TRAINING LOOP (RECOMMENDED)
    # =========================================================================
    if args.policy_guided_calibration:
        # This mode trains using policy-sampled transforms but evaluates ALL types
        # (without training) to compute clean calibration signals for policy updates.

        # Create evaluation set for calibration (use training data for evaluation)
        eval_inputs_cal = train_inputs
        eval_pairs_cal = train_pairs

        # Warmup JIT compilation for both training and evaluation
        print("Warming up JIT compilation...")
        for transform_type in enabled_transforms:
            # Warm up training
            _ = run_calibration_window_scan(
                states['encoder'], states['task_head'], states['forward'], states['inverse'],
                train_inputs, train_labels, train_pairs, weights_jax, all_inputs, random_perms,
                enabled_type_starts,
                rng, args.calibration_window_size, transform_type,
                args.p, n_enabled_actions, n_translation, n_scaling, n_quadratic, n_random,
                args.lambda_task, args.beta_forward, args.beta_inverse,
            )
            jax.block_until_ready(_[0][0].params)

            # Warm up evaluation
            _ = evaluate_inverse_loss_no_train(
                states['encoder'], states['inverse'],
                eval_inputs_cal, eval_pairs_cal, all_inputs, random_perms,
                enabled_type_starts, rng, args.eval_calibration_batches, transform_type,
                args.p, n_enabled_actions, n_translation, n_scaling, n_quadratic, n_random,
            )
            jax.block_until_ready(_[0])
        print("JIT warmup complete - starting training\n")

        epoch = 0
        n_cycles = args.n_epochs // args.calibration_window_size
        epochs_per_cycle = args.calibration_window_size

        pbar = tqdm(range(n_cycles), desc="Policy-guided training", unit="cycle")

        for cycle in pbar:
            # Sample transform type from policy for this training cycle
            policy_logits = states['policy'].apply_fn(states['policy'].params)
            rng, rng_sample = jax.random.split(rng)
            sampled_policy_idx, _ = sample_single_type_from_policy(
                policy_logits, rng_sample, args.policy_temperature
            )
            sampled_type = enabled_transforms[sampled_policy_idx]
            per_type_counts[sampled_policy_idx] += 1

            # TRAIN using the sampled transform type only
            states, train_loss, rng = run_calibration_window(
                states, sampled_type, args.calibration_window_size,
                train_inputs, train_labels, train_pairs, weights_jax, all_inputs,
                random_perms, enabled_type_starts, rng,
                args.p, n_enabled_actions, n_translation, n_scaling, n_quadratic, n_random,
                args.lambda_task, args.beta_forward, args.beta_inverse,
            )
            epoch += epochs_per_cycle

            # EVALUATE all types (without training) to get calibration signal
            eval_losses, eval_accs, rng = evaluate_all_types_no_train(
                states, enabled_transforms, args.eval_calibration_batches,
                eval_inputs_cal, eval_pairs_cal, all_inputs, random_perms,
                enabled_type_starts, rng,
                args.p, n_enabled_actions, n_translation, n_scaling, n_quadratic, n_random,
            )

            # Compute reward: positive when sampled type has lower loss than others
            sampled_loss = eval_losses[sampled_type]
            other_losses = [eval_losses[t] for t in enabled_transforms if t != sampled_type]
            if other_losses:
                reward = np.mean(other_losses) - sampled_loss
            else:
                reward = 0.0

            # Update global baseline
            global_baseline = baseline_decay * global_baseline + (1 - baseline_decay) * reward

            # Log policy state
            policy_probs = jax.nn.softmax(policy_logits / args.policy_temperature)
            policy_history['epoch'].append(epoch)
            policy_history['logits'].append(np.asarray(policy_logits).tolist())
            policy_history['probs'].append(np.asarray(policy_probs).tolist())
            policy_history['reward'].append(float(reward))
            policy_history['sampled_type'].append(sampled_type)
            policy_history['baseline'].append(float(global_baseline))
            policy_history['all_baselines'].append([float(global_baseline)] * 4)
            policy_history['type_counts'].append(list(per_type_counts))
            policy_history['cf_losses'].append(eval_losses)

            # Update policy using REINFORCE
            def policy_loss_fn(params):
                loss, metrics = compute_bandit_policy_gradient_loss(
                    params, states['policy'].apply_fn,
                    sampled_policy_idx, reward, global_baseline,
                    args.policy_temperature, args.entropy_weight
                )
                return loss, metrics

            (pol_loss, pol_metrics), pol_grads = jax.value_and_grad(
                policy_loss_fn, has_aux=True
            )(states['policy'].params)
            states['policy'] = states['policy'].apply_gradients(grads=pol_grads)

            # Periodic evaluation and logging
            if (cycle + 1) % max(1, n_cycles // 100) == 0 or cycle == 0:
                # Evaluate test accuracy
                test_metrics = eval_step(states['encoder'], states['task_head'],
                                         test_inputs, test_labels, args.p)

                # Log to history
                history['epoch'].append(epoch)
                history['test_acc'].append(float(test_metrics['accuracy']))
                history['test_loss'].append(float(test_metrics['loss']))
                history['train_acc'].append(0.0)  # Not computed every step
                history['train_loss'].append(float(train_loss))
                history['task_loss'].append(0.0)
                history['forward_loss'].append(0.0)
                history['inverse_loss'].append(float(sampled_loss))
                history['inverse_acc'].append(float(eval_accs.get(sampled_type, 0.0)))

                # Policy probs
                for i, t in enumerate([0, 1, 2, 3]):
                    key = f'prob_{TYPE_NAMES[t]}'
                    if t in enabled_transforms:
                        idx = enabled_transforms.index(t)
                        history[key].append(float(policy_probs[idx]))
                    else:
                        history[key].append(0.0)
                history['policy_entropy'].append(float(pol_metrics['entropy']))
                history['reward_signal'].append(float(reward))
                history['energy_ratio'].append(0.0)  # Computed less frequently

                # Format loss strings
                loss_strs = [f"{TYPE_NAMES[t][0].upper()}:{eval_losses[t]:.2f}" for t in enabled_transforms]
                loss_str = " ".join(loss_strs)
                prob_strs = [f"{TYPE_NAMES[t][0].upper()}:{policy_probs[enabled_transforms.index(t)]*100:.0f}"
                            for t in enabled_transforms]
                prob_str = " ".join(prob_strs)

                pbar.set_postfix({
                    'test': f"{test_metrics['accuracy']:.3f}",
                    'π': prob_str,
                    'loss': loss_str,
                    'trained': TYPE_NAMES[sampled_type][0].upper(),
                })

        # Save results
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)
        with open(os.path.join(args.save_dir, 'policy_history.pkl'), 'wb') as f:
            pickle.dump(policy_history, f)
        with open(os.path.join(args.save_dir, 'params.pkl'), 'wb') as f:
            pickle.dump({
                'encoder': states['encoder'].params,
                'task_head': states['task_head'].params,
                'forward': states['forward'].params,
                'inverse': states['inverse'].params,
                'policy': states['policy'].params,
            }, f)
        with open(os.path.join(args.save_dir, 'args.pkl'), 'wb') as f:
            pickle.dump(vars(args), f)
        with open(os.path.join(args.save_dir, 'random_perms.pkl'), 'wb') as f:
            pickle.dump(np.asarray(random_perms), f)
        with open(os.path.join(args.save_dir, 'zipf_info.pkl'), 'wb') as f:
            pickle.dump(zipf_info, f)

        # Print final results
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        final_test_metrics = eval_step(states['encoder'], states['task_head'],
                                       test_inputs, test_labels, args.p)
        final_train_metrics = eval_step(states['encoder'], states['task_head'],
                                        train_inputs, train_labels, args.p)
        print(f"  Train accuracy: {final_train_metrics['accuracy']:.4f}")
        print(f"  Test accuracy: {final_test_metrics['accuracy']:.4f}")

        final_policy_probs = jax.nn.softmax(states['policy'].apply_fn(states['policy'].params))
        print(f"\nFinal Policy Distribution (enabled transforms only):")
        for i, t in enumerate(enabled_transforms):
            print(f"  {TYPE_NAMES[t]}: {final_policy_probs[i]*100:.1f}%")

        # Peak test accuracy
        if history['test_acc']:
            peak_test_acc = max(history['test_acc'])
            peak_epoch = history['epoch'][history['test_acc'].index(peak_test_acc)]
            print(f"\n  Peak test accuracy: {peak_test_acc:.4f} at epoch {peak_epoch}")

        print(f"\nResults saved to {args.save_dir}")

        # Generate visualizations
        try:
            from visualize_results import generate_visualizations
            generate_visualizations(args.save_dir)
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")

        return

    # =========================================================================
    # ALTERNATING CALIBRATION TRAINING LOOP
    # =========================================================================
    elif args.alternating_calibration or args.interleaved_calibration:
        # Warmup JIT compilation with actual window size to amortize compilation cost
        # This ensures the first real cycle runs at full speed
        print("Warming up JIT compilation...")
        for transform_type in enabled_transforms:
            _ = run_calibration_window_scan(
                states['encoder'], states['task_head'], states['forward'], states['inverse'],
                train_inputs, train_labels, train_pairs, weights_jax, all_inputs, random_perms,
                enabled_type_starts,
                rng, args.calibration_window_size, transform_type,
                args.p, n_enabled_actions, n_translation, n_scaling, n_quadratic, n_random,
                args.lambda_task, args.beta_forward, args.beta_inverse,
            )
            # Force compilation to complete
            jax.block_until_ready(_[0][0].params)
        print("JIT warmup complete - starting training\n")

        epoch = 0
        if args.interleaved_calibration:
            # Interleaved: each batch trains on all types, so n_epochs = n_calibrations * window_size
            n_calibrations = args.n_epochs // args.calibration_window_size
            epochs_per_cycle = args.calibration_window_size
        else:
            # Alternating: train each type sequentially
            n_calibrations = args.n_epochs // (args.calibration_window_size * n_enabled_types)
            epochs_per_cycle = args.calibration_window_size * n_enabled_types

        pbar = tqdm(range(n_calibrations), desc="Calibration cycles", unit="cycle")

        for cycle in pbar:
            if args.interleaved_calibration:
                # Interleaved: train T and R in parallel within each batch
                states, calibration_losses, rng = run_interleaved_calibration(
                    states, enabled_transforms, args.calibration_window_size,
                    train_inputs, train_labels, train_pairs, weights_jax, all_inputs,
                    random_perms, enabled_type_starts, rng,
                    args.p, n_enabled_actions, n_translation, n_scaling, n_quadratic, n_random,
                    args.lambda_task, args.beta_forward, args.beta_inverse,
                )
            else:
                # Alternating: train each type sequentially
                states, calibration_losses, rng = run_alternating_then_compare(
                    states, enabled_transforms, args.calibration_window_size,
                    train_inputs, train_labels, train_pairs, weights_jax, all_inputs,
                    random_perms, enabled_type_starts, rng,
                    args.p, n_enabled_actions, n_translation, n_scaling, n_quadratic, n_random,
                    args.lambda_task, args.beta_forward, args.beta_inverse,
                )
            epoch += epochs_per_cycle

            # Sample action from policy (for REINFORCE gradient)
            policy_logits = states['policy'].apply_fn(states['policy'].params)
            rng, rng_sample = jax.random.split(rng)
            sampled_policy_idx, _ = sample_single_type_from_policy(
                policy_logits, rng_sample, args.policy_temperature
            )
            sampled_type = enabled_transforms[sampled_policy_idx]

            # Compute reward: positive when sampled type has lower loss
            sampled_loss = calibration_losses[sampled_type]
            other_losses = [calibration_losses[t] for t in enabled_transforms if t != sampled_type]
            if other_losses:
                reward = np.mean(other_losses) - sampled_loss
            else:
                reward = 0.0

            # Log policy state
            policy_probs = jax.nn.softmax(policy_logits / args.policy_temperature)
            policy_history['epoch'].append(epoch)
            policy_history['logits'].append(np.asarray(policy_logits).tolist())
            policy_history['probs'].append(np.asarray(policy_probs).tolist())
            policy_history['reward'].append(float(reward))
            policy_history['sampled_type'].append(sampled_type)
            policy_history['baseline'].append(float(global_baseline))
            policy_history['all_baselines'].append([float(global_baseline)] * 4)
            policy_history['type_counts'].append(list(per_type_counts))
            policy_history['cf_losses'].append({t: calibration_losses[t] for t in enabled_transforms})

            # Update policy
            def policy_loss_fn(params):
                loss, metrics = compute_bandit_policy_gradient_loss(
                    params, states['policy'].apply_fn,
                    sampled_policy_idx, reward, global_baseline,
                    args.policy_temperature, args.entropy_weight
                )
                return loss, metrics

            (loss, pg_metrics), policy_grads = jax.value_and_grad(
                policy_loss_fn, has_aux=True
            )(states['policy'].params)

            states['policy'] = states['policy'].apply_gradients(grads=policy_grads)
            per_type_counts[sampled_policy_idx] += 1
            global_baseline = baseline_decay * global_baseline + (1 - baseline_decay) * reward
            policy_history['entropy'].append(float(pg_metrics['entropy']))

            # Logging
            if cycle % max(1, n_calibrations // 100) == 0:
                train_eval = eval_step(states['encoder'], states['task_head'], train_inputs, train_labels, args.p)
                test_eval = eval_step(states['encoder'], states['task_head'], test_inputs, test_labels, args.p)

                history['epoch'].append(epoch)
                history['train_loss'].append(float(train_eval['loss']))
                history['train_acc'].append(float(train_eval['accuracy']))
                history['test_loss'].append(float(test_eval['loss']))
                history['test_acc'].append(float(test_eval['accuracy']))
                history['task_loss'].append(0.0)  # Not tracked per-batch in this mode
                history['forward_loss'].append(0.0)
                history['inverse_loss'].append(np.mean(list(calibration_losses.values())))
                history['inverse_acc'].append(0.0)

                # Policy distribution
                policy_probs_full = np.zeros(4)
                for i, t in enumerate(enabled_transforms):
                    policy_probs_full[t] = float(policy_probs[i])

                history['prob_translation'].append(float(policy_probs_full[0]))
                history['prob_scaling'].append(float(policy_probs_full[1]))
                history['prob_quadratic'].append(float(policy_probs_full[2]))
                history['prob_random'].append(float(policy_probs_full[3]))
                history['policy_entropy'].append(float(pg_metrics['entropy']))
                history['reward_signal'].append(float(reward))

                # Fourier metrics (less frequent)
                if cycle % max(1, n_calibrations // 10) == 0:
                    fourier_metrics = compute_fourier_metrics(states['encoder'], states['task_head'], args.p)
                    history['energy_ratio'].append(fourier_metrics['energy_ratio'])
                else:
                    history['energy_ratio'].append(history['energy_ratio'][-1] if history['energy_ratio'] else 0.0)

                # Progress bar
                type_abbrev = ['T', 'S', 'Q', 'R']
                policy_parts = [f"{type_abbrev[t]}:{policy_probs_full[t]*100:.0f}" for t in enabled_transforms]
                policy_str = " ".join(policy_parts)
                loss_parts = [f"{type_abbrev[t]}:{calibration_losses[t]:.2f}" for t in enabled_transforms]
                loss_str = " ".join(loss_parts)
                pbar.set_postfix({
                    'test': f"{test_eval['accuracy']:.3f}",
                    'π': policy_str,
                    'loss': loss_str,
                })

    # =========================================================================
    # ORIGINAL TRAINING LOOP (bandit-style)
    # =========================================================================
    else:
        pbar = tqdm(range(args.n_epochs), desc="Training", unit="epoch")
        for epoch in pbar:
            rng, rng_sample, rng_step, rng_type = jax.random.split(rng, 4)

            # At the START of each window, sample a single transform type (learned mode)
            if args.policy_mode == 'learned' and epoch % args.policy_update_interval == 0:
                policy_logits = states['policy'].apply_fn(states['policy'].params)
                # Sample from enabled types only, then map to actual transform type
                policy_idx, _ = sample_single_type_from_policy(
                    policy_logits, rng_type, args.policy_temperature
                )
                current_window_type = enabled_transforms[policy_idx]
                current_policy_idx = policy_idx  # Track which policy index was sampled

            # Sample transform types and params
            if args.policy_mode == 'learned':
                # Bandit: use the SAME type for all samples in this window
                transform_types = jnp.full(batch_size, current_window_type, dtype=jnp.int32)
                params = sample_params_for_type(
                    rng_sample, current_window_type, batch_size,
                    n_translation, n_scaling, n_quadratic, n_random
                )
            elif args.policy_mode == 'uniform_all':
                # Uniform over all 4 types (unchanged - for comparison)
                transform_types = jax.random.randint(rng_sample, (batch_size,), 0, 4)
                # Sample params within each type
                type_sizes = jnp.array([n_translation, n_scaling, n_quadratic, n_random])
                type_offsets = jnp.array([0, 1, 0, 0])
                sizes = type_sizes[transform_types]
                offsets = type_offsets[transform_types]
                rng_sample, rng_param = jax.random.split(rng_sample)
                params = (jax.random.randint(rng_param, (batch_size,), 0, sizes.max()) % sizes) + offsets
                params = params.astype(jnp.int32)
            elif args.policy_mode == 'uniform_translation':
                # Only use translation (baseline from grokking_icm_inverse_primary)
                transform_types = jnp.zeros(batch_size, dtype=jnp.int32)
                params = jax.random.randint(rng_sample, (batch_size,), 0, n_translation)
            else:
                raise ValueError(f"Unknown policy mode: {args.policy_mode}")

            # Training step
            states['encoder'], states['task_head'], states['forward'], states['inverse'], metrics = \
                train_step_with_types(
                    states['encoder'], states['task_head'], states['forward'], states['inverse'],
                    train_inputs, train_labels, train_pairs, weights_jax, all_inputs,
                    transform_types, params, random_perms,
                    enabled_type_starts,
                    args.p, n_enabled_actions,
                    args.lambda_task, args.beta_forward, args.beta_inverse,
                )

            # Accumulate inverse loss within the window
            inverse_loss_window.append(float(metrics['inverse_loss']))

            # At the END of each window: compute reward and update policy
            if (epoch + 1) % args.policy_update_interval == 0:
                # Compute average inverse loss for this window (for logging)
                current_inverse_loss = np.mean(inverse_loss_window)
                inverse_loss_history.append(current_inverse_loss)
                inverse_loss_window = []  # Reset for next window

                # Compute reward based on mode
                if args.counterfactual_eval and args.policy_mode == 'learned':
                    # COUNTERFACTUAL EVALUATION: Compare inverse loss for all enabled types
                    # on the same held-out evaluation set, at the current encoder/inverse state.
                    # This gives a clean, instantaneous signal about which type is more learnable.
                    #
                    # Reward = mean(loss_other_types) - loss_sampled_type
                    # Positive when sampled type has lower loss (more learnable) than alternatives.

                    cf_losses = {}
                    rng, *cf_rngs = jax.random.split(rng, 1 + args.counterfactual_n_samples * n_enabled_types)
                    cf_rng_idx = 0

                    for policy_idx, transform_type in enumerate(enabled_transforms):
                        # Average over multiple random parameter samples for stability
                        type_losses = []
                        for _ in range(args.counterfactual_n_samples):
                            loss, _ = compute_counterfactual_inverse_loss(
                                states['encoder'], states['inverse'],
                                cf_eval_inputs, cf_eval_pairs, all_inputs, random_perms,
                                enabled_type_starts,
                                cf_rngs[cf_rng_idx],
                                args.p, n_enabled_actions, n_translation, n_scaling, n_quadratic, n_random,
                                transform_type,
                            )
                            type_losses.append(float(loss))
                            cf_rng_idx += 1
                        cf_losses[transform_type] = np.mean(type_losses)

                    # Reward = mean(other types' losses) - sampled type's loss
                    sampled_loss = cf_losses[current_window_type]
                    other_losses = [cf_losses[t] for t in enabled_transforms if t != current_window_type]
                    if other_losses:
                        reward = np.mean(other_losses) - sampled_loss
                    else:
                        reward = 0.0  # Only one type enabled, no comparison possible

                    # No transition filtering needed with counterfactual eval
                    should_update = True

                else:
                    # LEGACY MODE: Δ(inverse_loss) from training window
                    # Compute reward: DECREASE in inverse loss (positive when inverse model improves)
                    if len(inverse_loss_history) >= 2:
                        reward = inverse_loss_history[-2] - inverse_loss_history[-1]
                    else:
                        reward = 0.0

                    # Transition filtering (legacy mode only)
                    # CRITICAL FIX: Only update on same-type windows to filter out transition noise
                    same_type_window = (previous_window_type is not None and
                                       current_window_type == previous_window_type)
                    should_update = same_type_window or args.no_transition_filtering

                if args.policy_mode == 'learned' and (len(inverse_loss_history) >= 2 or args.counterfactual_eval):
                    # Always log policy state (for analysis)
                    policy_logits = states['policy'].apply_fn(states['policy'].params)
                    policy_probs = jax.nn.softmax(policy_logits / args.policy_temperature)

                    policy_history['epoch'].append(epoch)
                    policy_history['logits'].append(np.asarray(policy_logits).tolist())
                    policy_history['probs'].append(np.asarray(policy_probs).tolist())
                    policy_history['reward'].append(float(reward))
                    policy_history['sampled_type'].append(current_window_type)
                    policy_history['baseline'].append(float(global_baseline))
                    policy_history['all_baselines'].append([float(global_baseline)] * 4)
                    policy_history['type_counts'].append(list(per_type_counts))
                    # Log counterfactual losses (if available)
                    if args.counterfactual_eval:
                        policy_history['cf_losses'].append({t: cf_losses[t] for t in enabled_transforms})
                    else:
                        policy_history['cf_losses'].append(None)

                    if should_update:
                        # Update policy (filtered to same-type windows unless --no_transition_filtering)
                        # This ensures we're measuring the true reward of each transform type,
                        # not the cost of switching between types.

                        # Bandit-style update: single (policy_idx, reward) pair
                        def policy_loss_fn(params):
                            loss, metrics = compute_bandit_policy_gradient_loss(
                                params, states['policy'].apply_fn,
                                current_policy_idx, reward, global_baseline,
                                args.policy_temperature, args.entropy_weight
                            )
                            return loss, metrics

                        (loss, pg_metrics), policy_grads = jax.value_and_grad(
                            policy_loss_fn, has_aux=True
                        )(states['policy'].params)

                        # Update policy
                        states['policy'] = states['policy'].apply_gradients(grads=policy_grads)

                        # Update global baseline (EMA of rewards from same-type windows only)
                        per_type_counts[current_policy_idx] += 1
                        global_baseline = baseline_decay * global_baseline + (1 - baseline_decay) * reward

                        policy_history['entropy'].append(float(pg_metrics['entropy']))
                    else:
                        # Skip policy update on type switches (log entropy as NaN to mark skipped)
                        policy_history['entropy'].append(float('nan'))

                # Update previous_window_type for next iteration
                previous_window_type = current_window_type

            # Logging
            if epoch % args.log_interval == 0:
                train_eval = eval_step(states['encoder'], states['task_head'], train_inputs, train_labels, args.p)
                # Test accuracy is computed as VALIDATION metric only (not used for policy updates)
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

                # Policy distribution (over enabled transforms only)
                if args.policy_mode == 'learned':
                    policy_logits = states['policy'].apply_fn(states['policy'].params)
                    policy_probs_enabled = jax.nn.softmax(policy_logits / args.policy_temperature)
                    policy_probs_enabled = np.asarray(policy_probs_enabled)
                    entropy = -np.sum(policy_probs_enabled * np.log(policy_probs_enabled + 1e-8))
                    # Map back to full 4-type array for logging consistency
                    policy_probs = np.zeros(4)
                    for i, t in enumerate(enabled_transforms):
                        policy_probs[t] = policy_probs_enabled[i]
                elif args.policy_mode == 'uniform_all':
                    policy_probs = np.zeros(4)
                    for t in enabled_transforms:
                        policy_probs[t] = 1.0 / n_enabled_types
                    entropy = np.log(n_enabled_types)
                elif args.policy_mode == 'uniform_translation':
                    policy_probs = np.array([1.0, 0.0, 0.0, 0.0])
                    entropy = 0.0
                else:
                    policy_probs = np.zeros(4)
                    for t in enabled_transforms:
                        policy_probs[t] = 1.0 / n_enabled_types
                    entropy = np.log(n_enabled_types)

                history['prob_translation'].append(float(policy_probs[0]))
                history['prob_scaling'].append(float(policy_probs[1]))
                history['prob_quadratic'].append(float(policy_probs[2]))
                history['prob_random'].append(float(policy_probs[3]))
                history['policy_entropy'].append(float(entropy))
                history['reward_signal'].append(float(reward) if 'reward' in dir() else 0.0)

                # Fourier metrics (less frequent)
                if epoch % args.fourier_interval == 0:
                    fourier_metrics = compute_fourier_metrics(states['encoder'], states['task_head'], args.p)
                    history['energy_ratio'].append(fourier_metrics['energy_ratio'])
                else:
                    history['energy_ratio'].append(history['energy_ratio'][-1] if history['energy_ratio'] else 0.0)

                # Progress bar - show policy distribution for enabled transform types only
                type_abbrev = ['T', 'S', 'Q', 'R']
                current_type_str = type_abbrev[current_window_type] if current_window_type is not None else '?'
                # Compact policy distribution for enabled types only
                policy_parts = [f"{type_abbrev[t]}:{policy_probs[t]*100:.0f}" for t in enabled_transforms]
                policy_str = " ".join(policy_parts)
                pbar.set_postfix({
                    'test': f"{test_eval['accuracy']:.3f}",
                    'inv': f"{metrics['inverse_acc']:.3f}",
                    'π': policy_str,
                    'now': current_type_str,
                })

    # Final evaluation
    train_eval = eval_step(states['encoder'], states['task_head'], train_inputs, train_labels, args.p)
    test_eval = eval_step(states['encoder'], states['task_head'], test_inputs, test_labels, args.p)
    fourier_metrics = compute_fourier_metrics(states['encoder'], states['task_head'], args.p)

    print(f"\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Train accuracy: {train_eval['accuracy']:.4f}")
    print(f"  Test accuracy: {test_eval['accuracy']:.4f}")
    print(f"  Energy ratio: {fourier_metrics['energy_ratio']:.4f}")

    if args.policy_mode == 'learned':
        policy_logits = states['policy'].apply_fn(states['policy'].params)
        policy_probs_enabled = np.asarray(jax.nn.softmax(policy_logits / args.policy_temperature))
        print(f"\nFinal Policy Distribution (enabled transforms only):")
        for i, t in enumerate(enabled_transforms):
            print(f"  {TYPE_NAMES[t]}: {policy_probs_enabled[i]:.1%}")
        # Map to full array for helpful/unhelpful calculation
        policy_probs = np.zeros(4)
        for i, t in enumerate(enabled_transforms):
            policy_probs[t] = policy_probs_enabled[i]
        helpful = policy_probs[0] + policy_probs[1]
        print(f"  HELPFUL (trans+scale): {helpful:.1%}")
        print(f"  AMBIGUOUS+UNHELPFUL (quad+rand): {1 - helpful:.1%}")

    # Compute peak test accuracy
    peak_test_acc = max(history['test_acc'])
    peak_epoch = history['epoch'][history['test_acc'].index(peak_test_acc)]
    print(f"\n  Peak test accuracy: {peak_test_acc:.4f} at epoch {peak_epoch}")

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)

    with open(os.path.join(args.save_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    with open(os.path.join(args.save_dir, 'policy_history.pkl'), 'wb') as f:
        pickle.dump(policy_history, f)

    with open(os.path.join(args.save_dir, 'params.pkl'), 'wb') as f:
        pickle.dump({
            'encoder': states['encoder'].params,
            'task_head': states['task_head'].params,
            'forward': states['forward'].params,
            'inverse': states['inverse'].params,
            'policy': states['policy'].params,
        }, f)

    with open(os.path.join(args.save_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(vars(args), f)

    with open(os.path.join(args.save_dir, 'zipf_info.pkl'), 'wb') as f:
        pickle.dump(zipf_info, f)

    with open(os.path.join(args.save_dir, 'random_perms.pkl'), 'wb') as f:
        pickle.dump(np.array(random_perms), f)

    print(f"\nResults saved to {args.save_dir}")

    return history, states


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learned Transformation Selection via Policy Gradients')

    # Data parameters
    parser.add_argument('--p', type=int, default=97, help='Modulus for arithmetic')
    parser.add_argument('--train_fraction', type=float, default=0.3, help='Fraction of data for training')
    parser.add_argument('--zipf_exponent', type=float, default=1.5, help='Zipf distribution exponent')

    # Transformation space parameters
    parser.add_argument('--n_translation', type=int, default=None, help='Number of translation actions (default: p)')
    parser.add_argument('--n_scaling', type=int, default=None, help='Number of scaling actions (default: p-1)')
    parser.add_argument('--n_quadratic', type=int, default=None, help='Number of quadratic actions (default: p-1)')
    parser.add_argument('--n_random', type=int, default=None, help='Number of random permutations (default: p-1)')

    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for encoder')
    parser.add_argument('--icm_hidden_dim', type=int, default=None, help='Hidden dimension for ICM models')
    parser.add_argument('--n_encoder_layers', type=int, default=2, help='Number of encoder layers')

    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=1500000, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1.0, help='Weight decay (encoder only)')
    parser.add_argument('--icm_learning_rate', type=float, default=1e-3, help='Learning rate for ICM models')

    # Loss weights
    parser.add_argument('--lambda_task', type=float, default=1.0, help='Weight for task loss')
    parser.add_argument('--beta_forward', type=float, default=0.0, help='Weight for forward model loss')
    parser.add_argument('--beta_inverse', type=float, default=0.1, help='Weight for inverse model loss')

    # Policy parameters
    parser.add_argument('--policy_mode', type=str, default='learned',
                        choices=['learned', 'uniform_all', 'uniform_translation'],
                        help='Policy mode: learned (policy gradient), uniform_all (uniform over 4 types), '
                             'uniform_translation (translation only - baseline)')
    parser.add_argument('--enabled_transforms', type=str, default='0,1,2,3',
                        help='Comma-separated list of enabled transform types: 0=translation, 1=scaling, 2=quadratic, 3=random')
    parser.add_argument('--policy_learning_rate', type=float, default=1e-2, help='Learning rate for policy')
    parser.add_argument('--policy_adam_beta1', type=float, default=0.0,
                        help='Adam beta1 (momentum) for policy optimizer. Default 0 = no momentum (RMSprop-like). '
                             'Set to 0.9 for standard Adam.')
    parser.add_argument('--policy_temperature', type=float, default=1.0, help='Temperature for policy softmax')
    parser.add_argument('--entropy_weight', type=float, default=0.1, help='Entropy regularization weight')
    parser.add_argument('--policy_update_interval', type=int, default=1000,
                        help='Update policy every N batches (also computes train loss for reward)')
    parser.add_argument('--no_transition_filtering', action='store_true',
                        help='Disable transition filtering (update policy on all windows, not just same-type)')

    # Counterfactual evaluation parameters
    parser.add_argument('--counterfactual_eval', action='store_true',
                        help='Use counterfactual evaluation for policy gradient signal. '
                             'Instead of Δ(inverse_loss), compute loss_R - loss_T on held-out set.')
    parser.add_argument('--counterfactual_eval_size', type=int, default=500,
                        help='Number of pairs in counterfactual evaluation set')
    parser.add_argument('--counterfactual_n_samples', type=int, default=10,
                        help='Number of random parameter samples to average over for each type')

    # Alternating calibration parameters
    parser.add_argument('--alternating_calibration', action='store_true',
                        help='Use alternating calibration for policy gradient signal. '
                             'Train both types sequentially, compare final losses. '
                             'Gives fair A/B comparison without recency bias.')
    parser.add_argument('--interleaved_calibration', action='store_true',
                        help='DEPRECATED: Use interleaved calibration. '
                             'Testing showed this is not faster than alternating mode.')
    parser.add_argument('--policy_guided_calibration', action='store_true',
                        help='RECOMMENDED: Policy-guided training with evaluation-only calibration. '
                             'Trains using transforms sampled from the policy, but evaluates ALL types '
                             '(without training) to compute clean calibration signals for policy updates. '
                             'This avoids the problem of alternating_calibration which trains on all types.')
    parser.add_argument('--calibration_window_size', type=int, default=500,
                        help='Batches per type during alternating calibration (total = 2x this). '
                             'Smaller values (50-100) give faster policy updates but noisier signal. '
                             'window=100 gives ~5x more policy updates/hr than window=500.')
    parser.add_argument('--eval_calibration_batches', type=int, default=100,
                        help='Number of evaluation batches per type for policy_guided_calibration. '
                             'These are evaluation-only (no training) to measure inverse loss.')

    # Logging parameters
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval (in epochs)')
    parser.add_argument('--fourier_interval', type=int, default=1000, help='Fourier metrics interval')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    train(args)
