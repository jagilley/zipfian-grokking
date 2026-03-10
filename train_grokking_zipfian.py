"""
Grokking experiment on modular arithmetic with Zipfian data distributions.

This script extends the standard grokking experiment to explore how grokking
behaves when training examples are sampled according to Zipf's law, rather than
uniformly. The hypothesis is that grokking may take longer (or not occur at all)
when the data distribution is heavily skewed.

GPU-optimized version: No Hessian computation, JIT-compiled training loop.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
import pickle
import argparse
import os
from tqdm import tqdm
import time


# =============================================================================
# Model Definition
# =============================================================================

class MLP(nn.Module):
    """Simple MLP for modular arithmetic."""
    hidden_dims: tuple = (128, 128)
    output_dim: int = 97

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x


# =============================================================================
# Zipfian Distribution Utilities
# =============================================================================

def zipf_weights(n, s=1.0):
    """
    Generate Zipf distribution weights for n items.
    P(k) ∝ 1 / k^s for k = 1, 2, ..., n
    """
    ranks = np.arange(1, n + 1)
    weights = 1.0 / np.power(ranks, s)
    return weights / weights.sum()


def generate_zipfian_modular_addition_data(p, train_frac=0.3, zipf_exponent=1.0,
                                            zipf_mode="pair_rank", seed=42):
    """
    Generate modular addition dataset with Zipfian training distribution.
    Test set is always uniform (to measure true generalization).
    """
    rng = np.random.RandomState(seed)

    # Generate all possible (a, b) pairs
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    n_total = len(all_pairs)
    n_train = int(n_total * train_frac)

    # Shuffle and split
    rng.shuffle(all_pairs)
    train_pairs = all_pairs[:n_train]
    test_pairs = all_pairs[n_train:]

    # Assign Zipfian weights based on mode
    if zipf_mode == "pair_rank":
        train_weights = zipf_weights(n_train, zipf_exponent)
    elif zipf_mode == "independent":
        a_weights = zipf_weights(p, zipf_exponent)
        b_weights = zipf_weights(p, zipf_exponent)
        train_weights = np.zeros(n_train)
        for i, (a, b) in enumerate(train_pairs):
            train_weights[i] = a_weights[a] * b_weights[b]
        train_weights = train_weights / train_weights.sum()
    elif zipf_mode == "output_freq":
        output_weights = zipf_weights(p, zipf_exponent)
        train_weights = np.zeros(n_train)
        for i, (a, b) in enumerate(train_pairs):
            train_weights[i] = output_weights[(a + b) % p]
        train_weights = train_weights / train_weights.sum()
    else:
        raise ValueError(f"Unknown zipf_mode: {zipf_mode}")

    def pairs_to_dataset(pairs):
        inputs = np.zeros((len(pairs), 2 * p), dtype=np.float32)
        labels = np.zeros(len(pairs), dtype=np.int32)
        for i, (a, b) in enumerate(pairs):
            inputs[i, a] = 1.0
            inputs[i, p + b] = 1.0
            labels[i] = (a + b) % p
        return inputs, labels

    train_data = pairs_to_dataset(train_pairs)
    test_data = pairs_to_dataset(test_pairs)

    # Distribution statistics
    sorted_weights = np.sort(train_weights)[::-1]
    top_10_pct_mass = sorted_weights[:max(1, n_train // 10)].sum()
    bottom_50_pct_mass = sorted_weights[n_train // 2:].sum()

    zipf_info = {
        "zipf_exponent": zipf_exponent,
        "zipf_mode": zipf_mode,
        "n_train_unique": n_train,
        "n_test_unique": len(test_pairs),
        "train_weights": train_weights,
        "train_pairs": train_pairs,
        "weight_ratio": train_weights.max() / train_weights.min(),
        "top_10pct_mass": top_10_pct_mass,
        "bottom_50pct_mass": bottom_50_pct_mass,
        "effective_n": 1.0 / (train_weights ** 2).sum(),
    }

    print(f"Generated Zipfian modular addition data (mod {p}):")
    print(f"  Zipf mode: {zipf_mode}, exponent: {zipf_exponent}")
    print(f"  Train: {n_train} unique pairs, Test: {len(test_pairs)} unique pairs")
    print(f"  Weight ratio (max/min): {zipf_info['weight_ratio']:.1f}x")
    print(f"  Top 10% of pairs: {100*top_10_pct_mass:.1f}% of mass")
    print(f"  Effective sample size: {zipf_info['effective_n']:.1f} (of {n_train})")

    return train_data, test_data, zipf_info


# =============================================================================
# Fourier-based Progress Measures (from Nanda et al. 2023)
# =============================================================================

def compute_fourier_metrics(state, train_inputs, train_labels, p, n_keys=5):
    """Compute Fourier-based progress measures (fast, runs on GPU then CPU for FFT)."""
    # Generate all (a, b) pairs
    all_inputs = np.zeros((p * p, 2 * p), dtype=np.float32)
    for a in range(p):
        for b in range(p):
            idx = a * p + b
            all_inputs[idx, a] = 1.0
            all_inputs[idx, p + b] = 1.0

    # Get logits (GPU)
    logits = state.apply_fn(state.params, jnp.array(all_inputs))
    logits_tensor = np.array(logits).reshape(p, p, p)

    # 2D FFT over (a, b) dimensions (CPU - numpy)
    fourier_logits = np.fft.fft2(logits_tensor, axes=(0, 1))

    # Find key frequencies (diagonal, excluding DC)
    energy = np.sum(np.abs(fourier_logits) ** 2, axis=2)
    diagonal_energy = [(k, float(energy[k, k])) for k in range(1, p)]
    diagonal_energy.sort(key=lambda x: x[1], reverse=True)
    key_freqs = [(k, k) for k, _ in diagonal_energy[:n_keys]]

    # Build masks
    restricted_mask = np.zeros((p, p), dtype=bool)
    excluded_mask = np.ones((p, p), dtype=bool)
    restricted_mask[0, 0] = True

    for (ka, kb) in key_freqs:
        restricted_mask[ka, kb] = True
        excluded_mask[ka, kb] = False
        restricted_mask[(p - ka) % p, (p - kb) % p] = True
        excluded_mask[(p - ka) % p, (p - kb) % p] = False

    # Apply masks and inverse FFT
    restricted_fourier = fourier_logits * restricted_mask[:, :, None]
    restricted_logits = np.real(np.fft.ifft2(restricted_fourier, axes=(0, 1)))
    excluded_fourier = fourier_logits * excluded_mask[:, :, None]
    excluded_logits = np.real(np.fft.ifft2(excluded_fourier, axes=(0, 1)))

    # Extract logits for training pairs
    train_inputs_np = np.array(train_inputs)
    train_labels_np = np.array(train_labels)
    a_indices = np.argmax(train_inputs_np[:, :p], axis=1)
    b_indices = np.argmax(train_inputs_np[:, p:], axis=1)

    def compute_loss(logits_3d, a_idx, b_idx, labels):
        sample_logits = logits_3d[a_idx, b_idx, :]
        max_logits = np.max(sample_logits, axis=1, keepdims=True)
        shifted = sample_logits - max_logits
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_probs = shifted - log_sum_exp
        one_hot = np.zeros_like(sample_logits)
        one_hot[np.arange(len(labels)), labels] = 1.0
        return -np.mean(np.sum(one_hot * log_probs, axis=1))

    restricted_loss = compute_loss(restricted_logits, a_indices, b_indices, train_labels_np)
    excluded_loss = compute_loss(excluded_logits, a_indices, b_indices, train_labels_np)

    total_energy = np.sum(np.abs(fourier_logits) ** 2)
    key_energy = np.sum(np.abs(restricted_fourier) ** 2)
    energy_ratio = float(key_energy / (total_energy + 1e-10))

    return {
        "restricted_loss": float(restricted_loss),
        "excluded_loss": float(excluded_loss),
        "energy_ratio": energy_ratio,
    }


# =============================================================================
# Training Functions (GPU-optimized)
# =============================================================================

def create_train_state(rng, model, learning_rate, weight_decay, input_dim):
    """Create initial training state."""
    params = model.init(rng, jnp.ones((1, input_dim)))
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def make_train_step(sample_weights=None):
    """Create JIT-compiled training step."""
    if sample_weights is not None:
        weights_jax = jnp.array(sample_weights)

        @jax.jit
        def train_step(state, inputs, labels):
            def loss_fn(params):
                logits = state.apply_fn(params, inputs)
                one_hot = jax.nn.one_hot(labels, logits.shape[-1])
                per_sample_loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1)
                return jnp.sum(per_sample_loss * weights_jax)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            return state.apply_gradients(grads=grads), loss
    else:
        @jax.jit
        def train_step(state, inputs, labels):
            def loss_fn(params):
                logits = state.apply_fn(params, inputs)
                one_hot = jax.nn.one_hot(labels, logits.shape[-1])
                return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            return state.apply_gradients(grads=grads), loss

    return train_step


@jax.jit
def compute_metrics(state, inputs, labels):
    """JIT-compiled metric computation."""
    logits = state.apply_fn(state.params, inputs)
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    loss = -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return loss, accuracy


# =============================================================================
# Main Training Loop
# =============================================================================

def train(args):
    """Main training function."""
    print(f"\n{'='*60}")
    print("GROKKING EXPERIMENT: Modular Arithmetic with ZIPFIAN Data")
    print(f"{'='*60}")
    print(f"JAX devices: {jax.devices()}\n")

    os.makedirs(args.save_dir, exist_ok=True)

    # Generate data
    (train_inputs, train_labels), (test_inputs, test_labels), zipf_info = \
        generate_zipfian_modular_addition_data(
            p=args.modulus,
            train_frac=args.train_frac,
            zipf_exponent=args.zipf_exponent,
            zipf_mode=args.zipf_mode,
            seed=args.seed,
        )

    train_weights = zipf_info["train_weights"]

    # Convert to JAX arrays
    train_inputs = jnp.array(train_inputs)
    train_labels = jnp.array(train_labels)
    test_inputs = jnp.array(test_inputs)
    test_labels = jnp.array(test_labels)

    # Create model
    model = MLP(hidden_dims=tuple(args.hidden_dims), output_dim=args.modulus)
    rng = jax.random.PRNGKey(args.seed)
    state = create_train_state(rng, model, args.learning_rate, args.weight_decay, 2 * args.modulus)

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    print(f"\nModel: MLP {args.hidden_dims}, {n_params} params")
    print(f"LR: {args.learning_rate}, Weight decay: {args.weight_decay}")

    # Create training step
    if args.weighted_loss:
        print("Using WEIGHTED loss")
        train_step = make_train_step(train_weights)
    else:
        print("Using UNWEIGHTED loss")
        train_step = make_train_step(None)

    print(f"Training for {args.n_epochs} epochs...\n")

    # History
    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [],
        "metric_epochs": [], "restricted_loss": [], "excluded_loss": [], "energy_ratio": [],
    }

    # Warmup JIT
    state, _ = train_step(state, train_inputs, train_labels)
    _ = compute_metrics(state, train_inputs, train_labels)
    _ = compute_metrics(state, test_inputs, test_labels)

    start_time = time.time()
    pbar = tqdm(range(args.n_epochs), desc="Training", unit="epoch",
                mininterval=0.5, smoothing=0.1)

    for epoch in pbar:
        state, _ = train_step(state, train_inputs, train_labels)

        # Log every log_interval epochs
        if epoch % args.log_interval == 0 or epoch == args.n_epochs - 1:
            train_loss, train_acc = compute_metrics(state, train_inputs, train_labels)
            test_loss, test_acc = compute_metrics(state, test_inputs, test_labels)

            history["epoch"].append(epoch)
            history["train_loss"].append(float(train_loss))
            history["train_acc"].append(float(train_acc))
            history["test_loss"].append(float(test_loss))
            history["test_acc"].append(float(test_acc))

            pbar.set_postfix({
                "train": f"{train_acc:.3f}",
                "test": f"{test_acc:.3f}",
            })

        # Print detailed info periodically
        if epoch % args.print_interval == 0 and epoch > 0:
            elapsed = time.time() - start_time
            epochs_per_sec = epoch / elapsed
            tqdm.write(f"Epoch {epoch:5d} | Train: {train_acc:.4f} | Test: {test_acc:.4f} | "
                       f"{epochs_per_sec:.1f} epochs/s")

        # Compute Fourier metrics less frequently
        if epoch % args.fourier_interval == 0:
            fourier = compute_fourier_metrics(state, train_inputs, train_labels, args.modulus)
            history["metric_epochs"].append(epoch)
            history["restricted_loss"].append(fourier["restricted_loss"])
            history["excluded_loss"].append(fourier["excluded_loss"])
            history["energy_ratio"].append(fourier["energy_ratio"])

    # Final stats
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Train Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Test Accuracy:  {history['test_acc'][-1]:.4f}")
    print(f"Grokked? {'YES' if history['test_acc'][-1] > 0.95 else 'NO'}")
    print(f"Time: {elapsed:.1f}s ({args.n_epochs/elapsed:.1f} epochs/s)")

    # Save
    with open(os.path.join(args.save_dir, "history.pkl"), "wb") as f:
        pickle.dump(history, f)
    with open(os.path.join(args.save_dir, "params.pkl"), "wb") as f:
        pickle.dump(state.params, f)
    with open(os.path.join(args.save_dir, "args.pkl"), "wb") as f:
        pickle.dump(vars(args), f)
    with open(os.path.join(args.save_dir, "zipf_info.pkl"), "wb") as f:
        pickle.dump(zipf_info, f)

    print(f"\nResults saved to {args.save_dir}")
    generate_plots(history, zipf_info, args.save_dir)

    return history


def generate_plots(history, zipf_info, save_dir):
    """Generate visualization plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Accuracy
    ax = axes[0, 0]
    ax.plot(history["epoch"], history["train_acc"], label="Train", alpha=0.8)
    ax.plot(history["epoch"], history["test_acc"], label="Test", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy (Grokking = sudden test acc spike)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss
    ax = axes[0, 1]
    ax.semilogy(history["epoch"], history["train_loss"], label="Train", alpha=0.8)
    ax.semilogy(history["epoch"], history["test_loss"], label="Test", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log)")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Fourier metrics
    ax = axes[0, 2]
    if history["restricted_loss"]:
        ax.semilogy(history["metric_epochs"], history["restricted_loss"], 'o-',
                    label="Restricted", alpha=0.8, markersize=3)
        ax.semilogy(history["metric_epochs"], history["excluded_loss"], 's-',
                    label="Excluded", alpha=0.8, markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (log)")
        ax.set_title("Fourier Progress Measures")
        ax.legend()
    ax.grid(True, alpha=0.3)

    # Zipf distribution
    ax = axes[1, 0]
    weights = zipf_info["train_weights"]
    sorted_weights = np.sort(weights)[::-1]
    ax.loglog(range(1, len(sorted_weights) + 1), sorted_weights, 'b-', alpha=0.8)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Weight")
    ax.set_title(f"Zipf Distribution (exp={zipf_info['zipf_exponent']})")
    ax.grid(True, alpha=0.3)

    # Energy ratio
    ax = axes[1, 1]
    if history["energy_ratio"]:
        ax.plot(history["metric_epochs"], history["energy_ratio"], 'o-', color='teal', markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Energy Ratio")
        ax.set_title("Fourier Energy in Key Frequencies")
        ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Test acc vs excluded loss
    ax = axes[1, 2]
    if history["excluded_loss"]:
        ax2 = ax.twinx()
        # Interpolate test_acc to metric_epochs
        test_acc_at_metrics = np.interp(history["metric_epochs"], history["epoch"], history["test_acc"])
        ln1 = ax.plot(history["metric_epochs"], test_acc_at_metrics, 'b-', label="Test Acc", alpha=0.8)
        ln2 = ax2.semilogy(history["metric_epochs"], history["excluded_loss"], 'r-', label="Excluded Loss", alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test Accuracy", color='blue')
        ax2.set_ylabel("Excluded Loss", color='red')
        ax.set_title("Test Acc vs Excluded Loss")
        lns = ln1 + ln2
        ax.legend(lns, [l.get_label() for l in lns], loc='center right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "grokking_zipfian_metrics.png"), dpi=150)
    plt.close()
    print(f"Saved plots to {save_dir}/grokking_zipfian_metrics.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grokking with Zipfian data (GPU-optimized)")

    # Task
    parser.add_argument("--modulus", type=int, default=97)
    parser.add_argument("--train_frac", type=float, default=0.3)

    # Zipf
    parser.add_argument("--zipf_exponent", type=float, default=1.0)
    parser.add_argument("--zipf_mode", type=str, default="pair_rank",
                        choices=["pair_rank", "independent", "output_freq"])
    parser.add_argument("--weighted_loss", action="store_true")

    # Model
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 128])

    # Training
    parser.add_argument("--n_epochs", type=int, default=50000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)

    # Logging
    parser.add_argument("--log_interval", type=int, default=100, help="Log metrics every N epochs")
    parser.add_argument("--print_interval", type=int, default=5000, help="Print to console every N epochs")
    parser.add_argument("--fourier_interval", type=int, default=1000, help="Compute Fourier metrics every N epochs")

    # Other
    parser.add_argument("--save_dir", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)
