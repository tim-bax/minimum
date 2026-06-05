#!/usr/bin/env python3
"""Big Optuna hyperparameter search for the no-history SHD model.

This is the *large* search derived from no_history/optuna_shd.py. In addition to
the neuron/loss knobs, it also tunes the DATA GEOMETRY (`bin_size_ms`,
`collapse_factor`), the augmentation strength (`channel_shift_range`, in
collapsed units), and the ARCHITECTURE (1 hidden layer of 128 vs 2 hidden layers
of 64->42). Configs are selected on a speaker-held-out VALIDATION split (a few
entire train speakers held out) so val approximates the speaker-disjoint SHD
test set; the test set is touched exactly once, in the final confirmation step.

Why this differs from a vanilla Optuna setup:
  * The channel-shift regime converges slowly but generalizes better; a median
    pruner would kill those trials early, so pruning is OFF by default.
  * Selecting on test accuracy tunes hyperparameters against the test set, so we
    select on validation instead and confirm on test only once at the end.
  * `bin_size_ms` and `collapse_factor` reshape the data tensor (timesteps and
    input-channel count) and are NOT cached inside the loader, so they are tuned
    as small CATEGORICAL grids (4x5 = 20 combos) and each combo's split is cached
    in-process; re-binning happens at most once per combo.
  * `loss_count_bias` is a no-op (a constant added to every logit before a
    shift-invariant softmax) and is intentionally NOT tunable here.

The neuron/loss knobs (lr, temperature, label_smoothing, beta_s, beta_d, a_adapt,
b_adapt, and the static taus/scheduler) still accept 1, 2, or 3 values:
  --lr 1e-3              -> static (fixed at 1e-3)
  --lr 1e-5 1e-1         -> tune, uniform in [1e-5, 1e-1]
  --lr 1e-5 1e-1 log     -> tune, log-uniform in [1e-5, 1e-1]

If a flag is omitted, the built-in default is used.
"""
import argparse
import datetime
import os
import sys

import optuna


def _precision_from_argv(argv):
    default = "32"
    for i, arg in enumerate(argv):
        if arg.startswith("--precision="):
            return arg.split("=", 1)[1]
        if arg == "--precision" and i + 1 < len(argv):
            return argv[i + 1]
    return default


_PRECISION = _precision_from_argv(sys.argv[1:])
if _PRECISION not in ("32", "64"):
    raise ValueError(f"Invalid --precision '{_PRECISION}'. Expected '32' or '64'.")

import jax
jax.config.update("jax_enable_x64", _PRECISION == "64")
import jax.numpy as jnp
from jax import random
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from config import NeuronConfig
from network import Network
from data.shd_binned import load_shd_binned, apply_channel_shift


# 1x128 vs 2x(64->42): map the arch label to (n_hidden, n_hidden2).
_ARCH_GEOMETRY = {
    "one_layer": (128, 0),
    "two_layer": (64, 42),
}


def parse_param(name, values):
    """Parse 1–3 string values into a param spec.

    Returns (static_val, low, high, log_scale). Exactly one of
    static_val or (low, high) will be set.
    """
    if len(values) == 1:
        return float(values[0]), None, None, False
    if len(values) == 2:
        return None, float(values[0]), float(values[1]), False
    if len(values) == 3 and values[2].lower() == "log":
        low, high = float(values[0]), float(values[1])
        if low <= 0:
            raise ValueError(
                f"--{name}: log-uniform range requires low > 0, got {low}"
            )
        return None, low, high, True
    raise ValueError(
        f"--{name}: expected 1 value (static), 2 values (low high), or "
        f"3 values (low high log). Got: {values}"
    )


def suggest_or_static(trial, name, values):
    static_val, low, high, log_scale = parse_param(name, values)
    if static_val is not None:
        return static_val
    return trial.suggest_float(name, low, high, log=log_scale)


def apply_temporal_jitter(x_input, jitter_range: int):
    if jitter_range <= 0:
        return np.asarray(x_input)
    x_np = np.asarray(x_input)
    T = x_np.shape[0]
    shift = np.random.randint(-jitter_range, jitter_range + 1)
    shifted_t = np.clip(np.arange(T) + shift, 0, T - 1)
    out = np.zeros_like(x_np)
    np.add.at(out, shifted_t, x_np)
    return out


def augment_sample(x, args, channel_shift_range):
    """Apply enabled training-time augmentations to one sample (training only).

    Takes an explicit `channel_shift_range` (resolved per trial, already in
    collapsed units) instead of reading it off args.
    """
    if args.augment_jitter:
        x = apply_temporal_jitter(x, args.jitter_range)
    if args.augment_channel_shift:
        x = apply_channel_shift(x, channel_shift_range)
    return x


def evaluate(net, dataset, batch_size=1):
    n = len(dataset)
    if batch_size <= 1:
        correct = sum(1 for x, y in dataset if net.predict(x) == int(y))
        return 100.0 * correct / max(n, 1)

    correct = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = dataset[start:end]
        actual = len(batch)
        xs = [x for x, y in batch]
        ys = jnp.array([int(y) for x, y in batch])
        if actual < batch_size:
            xs += [xs[0]] * (batch_size - actual)
        preds = net.batch_predict(jnp.stack(xs))
        correct += int(jnp.sum(preds[:actual] == ys))

    return 100.0 * correct / max(n, 1)


def train_and_eval(params, args, train_set, eval_set, n_inputs, seed,
                   trial=None, eval_label="val", verbose=True):
    """Run the full training loop for one config; return (best_acc, net).

    `params` is a fully-resolved dict (no Optuna suggestions inside). It carries
    the per-trial data geometry (`bin_size_ms` -> NeuronConfig dt) and the
    per-trial architecture (`n_hidden`, `n_hidden2`). The LR scheduler and
    best-acc tracking key off `eval_set` accuracy. When `trial` is given,
    per-epoch accuracy is reported (used only for logging/pruning; the default
    study uses NopPruner so reporting never kills a trial).
    """
    config = NeuronConfig(
        dt=params["bin_size_ms"],
        tau_soma=args.tau_soma,
        tau_dend=args.tau_dend,
        tau_m=args.tau_m,
        tau_plat_min=params["tau_plat_min"],
        tau_plat_max=params["tau_plat_max"],
        tau_w=params["tau_w"],
        a_adapt=params["a_adapt"],
        b_adapt=params["b_adapt"],
        mu_th=params["mu_th"],
        v_th=args.v_th,
        gamma=params["gamma"],
        beta_s=params["beta_s"],
        beta_d=params["beta_d"],
        weight_scale=args.weight_scale,
        loss_temperature=params["loss_temperature"],
        loss_label_smoothing=params["loss_label_smoothing"],
        # loss_count_bias intentionally left at its default: it is a no-op
        # (a constant added to every logit before a shift-invariant softmax).
    )

    key = random.PRNGKey(seed)
    B = args.batch_size
    net = Network(
        key, n_inputs, params["n_hidden"], args.n_outputs, config,
        optimizer=args.optimizer,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_eps=args.adam_eps,
        dropout_rate=params["dropout"],
        weight_decay=params["weight_decay"],
        n_hidden2=params["n_hidden2"],
    )

    csr = params["channel_shift_range"]
    n_train = len(train_set)
    n_batches = n_train // B

    current_lr = params["lr"]
    best_acc = 0.0
    epochs_since_lr_drop = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        idx = np.random.permutation(n_train)

        for bi in range(n_batches):
            start = bi * B
            batch_idx = idx[start: start + B]

            if B == 1:
                x, y = train_set[int(batch_idx[0])]
                x = augment_sample(x, args, csr)
                net.train_step(
                    jnp.array(x), int(y), lr=current_lr, clip_value=args.gradient_clip,
                )
            else:
                x_batch_np = [
                    augment_sample(train_set[int(i)][0], args, csr)
                    for i in batch_idx
                ]
                x_batch = jnp.stack(x_batch_np)
                y_batch = jnp.array([int(train_set[int(i)][1]) for i in batch_idx])
                net.batch_train_step(
                    x_batch, y_batch, lr=current_lr, clip_value=args.gradient_clip,
                )

        acc = evaluate(net, eval_set, B)

        improved = acc > best_acc
        if improved:
            best_acc = acc
            epochs_since_lr_drop = 0
            epochs_without_improvement = 0
        else:
            epochs_since_lr_drop += 1
            epochs_without_improvement += 1

        if verbose:
            marker = "*" if improved else " "
            print(
                f"  Epoch {epoch:3d}/{args.epochs}  {eval_label}_acc={acc:.2f}%{marker}"
                f"  lr={current_lr:.2e}",
                flush=True,
            )

        if trial is not None:
            trial.report(acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        lr_factor = params["lr_factor"]
        lr_patience = int(round(params["lr_patience"]))
        if (lr_factor < 1.0
                and lr_patience > 0
                and current_lr > args.lr_min
                and epochs_since_lr_drop >= lr_patience):
            new_lr = max(current_lr * lr_factor, args.lr_min)
            if new_lr < current_lr:
                current_lr = new_lr
                epochs_since_lr_drop = 0

        if (args.early_stop_patience > 0
                and epochs_without_improvement >= args.early_stop_patience):
            break

    return best_acc, net


def resolve_params(trial, args):
    """Resolve all tunable parameters for one trial into a flat dict."""
    params = {
        "lr": suggest_or_static(trial, "lr", args.lr),
        "loss_temperature": suggest_or_static(trial, "loss_temperature", args.loss_temperature),
        "loss_label_smoothing": suggest_or_static(trial, "loss_label_smoothing", args.loss_label_smoothing),
        "beta_s": suggest_or_static(trial, "beta_s", args.beta_s),
        "beta_d": suggest_or_static(trial, "beta_d", args.beta_d),
        "tau_w": suggest_or_static(trial, "tau_w", args.tau_w),
        "a_adapt": suggest_or_static(trial, "a_adapt", args.a_adapt),
        "b_adapt": suggest_or_static(trial, "b_adapt", args.b_adapt),
        "dropout": suggest_or_static(trial, "dropout", args.dropout),
        "weight_decay": suggest_or_static(trial, "weight_decay", args.weight_decay),
        "mu_th": suggest_or_static(trial, "mu_th", args.mu_th),
        "gamma": suggest_or_static(trial, "gamma", args.gamma),
        "tau_plat_min": suggest_or_static(trial, "tau_plat_min", args.tau_plat_min),
        "tau_plat_max": suggest_or_static(trial, "tau_plat_max", args.tau_plat_max),
        "lr_factor": suggest_or_static(trial, "lr_factor", args.lr_factor),
        "lr_patience": suggest_or_static(trial, "lr_patience", args.lr_patience),
    }

    # --- newly-tuned data geometry / augmentation / architecture ---
    # bin_size and collapse are small categorical grids: each distinct combo
    # forces a one-time re-bin (cached in _SPLIT_CACHE), so keep them discrete.
    params["bin_size_ms"] = trial.suggest_categorical("bin_size_ms", args.bin_size_choices)
    params["collapse_factor"] = trial.suggest_categorical("collapse_factor", args.collapse_choices)
    # channel-shift directly in collapsed units (no collapse mapping).
    params["channel_shift_range"] = trial.suggest_int(
        "channel_shift_range", args.channel_shift_min, args.channel_shift_max)
    # architecture: 1 hidden layer of 128 vs 2 hidden layers of 64->42.
    arch = trial.suggest_categorical("arch", list(_ARCH_GEOMETRY.keys()))
    n_hidden, n_hidden2 = _ARCH_GEOMETRY[arch]
    params["arch"] = arch
    params["n_hidden"] = n_hidden
    params["n_hidden2"] = n_hidden2
    return params


def run_trial(trial, args, get_split_data, fixed_config):
    params = resolve_params(trial, args)
    train_subset, _train_pool, val_data, _test_data, n_inputs = get_split_data(
        params["bin_size_ms"], params["collapse_factor"]
    )
    trial.set_user_attr("params", params)
    # Record the fixed (non-tunable) geometry this trial ran under, so the trial
    # is self-documenting and survives study resumes with different settings.
    trial.set_user_attr("fixed", fixed_config)

    print(
        f"\n--- Trial {trial.number + 1}/{args.n_trials} ---"
        f"  bin={params['bin_size_ms']}ms  collapse={params['collapse_factor']}"
        f"  arch={params['arch']}(n_hidden={params['n_hidden']},n_hidden2={params['n_hidden2']})"
        f"  ch_shift={params['channel_shift_range']}  n_inputs={n_inputs}"
        f"\n     lr={params['lr']:.4g}  temp={params['loss_temperature']:.4g}"
        f"  smooth={params['loss_label_smoothing']:.4g}"
        f"  beta_s={params['beta_s']:.4g}  beta_d={params['beta_d']:.4g}"
        f"  a={params['a_adapt']:.4g}  b={params['b_adapt']:.4g}",
        flush=True,
    )

    best_acc, _ = train_and_eval(
        params, args, train_subset, val_data, n_inputs,
        seed=args.seed + trial.number, trial=trial, eval_label="val",
    )
    return best_acc


def build_fixed_config(args):
    """Capture every non-tunable run setting as a JSON-serializable dict.

    The per-trial geometry (bin_size_ms, collapse_factor, n_hidden, arch,
    channel_shift_range) is NOT here — it lives on each trial's `params`. Instead
    we record the SEARCH SPACE for those, so the DB is self-documenting and a
    resume with the same search space doesn't trigger a false "not
    apples-to-apples" warning.
    """
    return {
        "max_duration_ms": args.max_duration_ms,
        "binarize": bool(args.binarize),
        "input_scale": args.input_scale,
        "n_outputs": args.n_outputs,
        "weight_scale": args.weight_scale,
        "tau_soma": args.tau_soma,
        "tau_dend": args.tau_dend,
        "tau_m": args.tau_m,
        "v_th": args.v_th,
        "optimizer": args.optimizer,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "adam_eps": args.adam_eps,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "gradient_clip": args.gradient_clip,
        "lr_min": args.lr_min,
        "early_stop_patience": args.early_stop_patience,
        "augment_channel_shift": bool(args.augment_channel_shift),
        "augment_jitter": bool(args.augment_jitter),
        "jitter_range": args.jitter_range,
        # search-space records for the now-per-trial geometry:
        "bin_size_choices": list(args.bin_size_choices),
        "collapse_choices": list(args.collapse_choices),
        "channel_shift_range_bounds": [args.channel_shift_min, args.channel_shift_max],
        "arch_choices": {k: list(v) for k, v in _ARCH_GEOMETRY.items()},
        "train_fraction": args.train_fraction,
        "val_n_speakers": args.val_n_speakers,
        "val_speakers": list(args.val_speakers) if args.val_speakers is not None else None,
        "val_fraction": args.val_fraction,
        "val_seed": args.val_seed if args.val_seed is not None else args.seed,
        "seed": args.seed,
        "precision": args.precision,
    }


# Tunable parameter names whose value lists live on args under the same name
# (flag-driven via suggest_or_static; 1/2/3 values each).
_TUNABLE_NAMES = [
    "lr", "loss_temperature", "loss_label_smoothing", "beta_s", "beta_d",
    "tau_w", "a_adapt", "b_adapt", "dropout", "weight_decay",
    "mu_th", "gamma", "tau_plat_min", "tau_plat_max",
    "lr_factor", "lr_patience",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Big Optuna hyperparameter search for no-history SHD model "
                    "(tunes data geometry + architecture; validation-selected).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- flag-driven tunable parameters (1, 2, or 3 values each) ----
    tunable = p.add_argument_group(
        "tunable parameters",
        "Pass 1 value = static, 2 values = uniform range, 3 values (last='log') = log-uniform.",
    )
    tunable.add_argument("--lr", nargs="+", default=["1e-4", "1e-2", "log"], metavar="VAL",
                         help="Learning rate (default tune log-uniform [1e-4, 1e-2]).")
    tunable.add_argument("--loss_temperature", nargs="+", default=["1.0", "8.0"], metavar="VAL")
    tunable.add_argument("--loss_label_smoothing", nargs="+", default=["0.0", "0.3"], metavar="VAL")
    tunable.add_argument("--beta_s", nargs="+", default=["0.05", "5.0"], metavar="VAL",
                         help="Somatic surrogate gradient scale (default tune [0.05, 5.0]).")
    tunable.add_argument("--beta_d", nargs="+", default=["0.05", "5.0"], metavar="VAL",
                         help="Dendritic surrogate gradient scale (default tune [0.05, 5.0]).")
    tunable.add_argument("--a_adapt", nargs="+", default=["0.0", "0.5"], metavar="VAL",
                         help="Subthreshold adaptation coupling (default tune [0.0, 0.5]).")
    tunable.add_argument("--b_adapt", nargs="+", default=["0.0", "0.5"], metavar="VAL",
                         help="Spike-triggered adaptation jump (default tune [0.0, 0.5]).")
    tunable.add_argument("--tau_w", nargs="+", default=["100.0"], metavar="VAL",
                         help="Adaptation time constant (ms). Static by default.")
    tunable.add_argument("--dropout", nargs="+", default=["0.0"], metavar="VAL",
                         help="Hidden->readout dropout rate. Static by default.")
    tunable.add_argument("--weight_decay", nargs="+", default=["0.0"], metavar="VAL",
                         help="Decoupled weight decay (AdamW-style). Static by default.")
    tunable.add_argument("--mu_th", nargs="+", default=["1.0"], metavar="VAL",
                         help="Dendritic plateau threshold. Static by default.")
    tunable.add_argument("--gamma", nargs="+", default=["0.5"], metavar="VAL",
                         help="Plateau-induced threshold reduction. Static by default.")
    tunable.add_argument("--tau_plat_min", nargs="+", default=["100.0"], metavar="VAL",
                         help="Plateau duration min (ms). Static by default.")
    tunable.add_argument("--tau_plat_max", nargs="+", default=["350.0"], metavar="VAL",
                         help="Plateau duration max (ms). Static by default.")

    # ---- newly-searched geometry / augmentation / architecture grids ----
    geo = p.add_argument_group("searched geometry / architecture")
    geo.add_argument("--bin_size_choices", type=float, nargs="+", default=[1, 2, 3, 4],
                     metavar="MS", help="Categorical bin sizes (ms) to search.")
    geo.add_argument("--collapse_choices", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                     metavar="K", help="Categorical collapse factors to search.")
    geo.add_argument("--channel_shift_min", type=int, default=0,
                     help="Min channel-shift range (collapsed units).")
    geo.add_argument("--channel_shift_max", type=int, default=7,
                     help="Max channel-shift range (collapsed units).")

    # ---- Optuna study settings ----
    study = p.add_argument_group("optuna study")
    study.add_argument("--n_trials", type=int, default=300)
    study.add_argument("--n_jobs", type=int, default=1,
                       help="Parallel Optuna workers (use 1 with JAX).")
    study.add_argument("--study_name", default="shd_optuna_big")
    study.add_argument("--storage", default=None,
                       help="Optuna storage URL, e.g. sqlite:///shd_big.db. "
                            "Default: in-memory (not persistent).")
    study.add_argument("--prune", action="store_true",
                       help="Enable MedianPruner (OFF by default). Even then, warmup "
                            "defaults to most of the horizon so slow-but-better trials survive.")
    study.add_argument("--pruner_startup", type=int, default=5,
                       help="MedianPruner n_startup_trials (only used with --prune).")
    study.add_argument("--pruner_warmup", type=int, default=None,
                       help="MedianPruner n_warmup_steps (only used with --prune). "
                            "Default: max(epochs-5, 5).")
    study.add_argument("--final_topk", type=int, default=3,
                       help="After the study, re-validate this many top trials on the "
                            "FULL train pool, then confirm the single best on test once.")

    # ---- validation split / subset ----
    splitg = p.add_argument_group("validation split")
    splitg.add_argument("--val_n_speakers", type=int, default=2,
                        help="Hold out this many entire TRAIN speakers as the validation split, "
                             "so val measures unseen-speaker generalization like the SHD test set "
                             "(default 2). Set 0 to fall back to a random fraction split.")
    splitg.add_argument("--val_speakers", type=int, nargs="+", default=None,
                        help="Explicit speaker IDs to hold out as val (overrides --val_n_speakers).")
    splitg.add_argument("--val_fraction", type=float, default=0.2,
                        help="Random-split fallback fraction, used only when --val_n_speakers 0 "
                             "and no --val_speakers (default 0.2).")
    splitg.add_argument("--val_seed", type=int, default=None,
                        help="Seed for choosing which speakers to hold out / the random split "
                             "(default: --seed).")
    splitg.add_argument("--train_fraction", type=float, default=0.5,
                        help="Fraction of the train POOL used per trial (fixed seeded subset). "
                             "Top-K are re-validated on the FULL pool at the end.")

    # ---- fixed model / data settings ----
    p.add_argument("--max_duration_ms", type=float, default=1400.0)
    p.add_argument("--binarize", action="store_true")
    p.add_argument("--input_scale", type=float, default=1.0)
    p.add_argument("--n_outputs", type=int, default=20)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_clip", type=float, default=5.0)
    p.add_argument("--weight_scale", type=float, default=0.25)
    p.add_argument("--tau_soma", type=float, default=15.0)
    p.add_argument("--tau_dend", type=float, default=15.0)
    p.add_argument("--tau_m", type=float, default=20.0)
    p.add_argument("--v_th", type=float, default=1.0)
    # augmentation: channel-shift ON by default; jitter optional.
    p.add_argument("--augment_channel_shift", action=argparse.BooleanOptionalAction,
                   default=True, help="Channel-shift augmentation (default ON; "
                                      "use --no-augment_channel_shift for a control run).")
    p.add_argument("--augment_jitter", action="store_true")
    p.add_argument("--jitter_range", type=int, default=10)
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    # LR-scheduler knobs are flag-driven tunables (1/2/3 values), static by default.
    p.add_argument("--lr_factor", nargs="+", default=["0.7"], metavar="VAL",
                   help="ReduceLROnPlateau multiplier (lr := lr * factor). 1.0 disables.")
    p.add_argument("--lr_patience", nargs="+", default=["5"], metavar="VAL",
                   help="Epochs without val improvement before an LR drop (rounded to int).")
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="0 (default, recommended for long runs) disables early stopping.")
    p.add_argument("--precision", choices=["32", "64"], default=_PRECISION)

    return p.parse_args()


def _describe_param(name, values):
    try:
        static_val, low, high, log_scale = parse_param(name, values)
    except ValueError as e:
        raise SystemExit(f"Error: {e}") from e
    if static_val is not None:
        return f"static={static_val}"
    scale = "log-uniform" if log_scale else "uniform"
    return f"tune [{low}, {high}] {scale}"


def main():
    args = parse_args()
    if args.precision != _PRECISION:
        raise ValueError(
            f"--precision mismatch during startup ({args.precision} vs {_PRECISION})."
        )
    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError("--val_fraction must be in (0, 1).")
    if args.channel_shift_min < 0 or args.channel_shift_max < args.channel_shift_min:
        raise ValueError("--channel_shift_min/max must satisfy 0 <= min <= max.")

    # Validate all flag-driven tunable param specs early so we fail fast.
    all_specs = [(n, getattr(args, n)) for n in _TUNABLE_NAMES]
    for name, vals in all_specs:
        parse_param(name, vals)  # raises on bad input

    print("Parameter configuration:")
    for name, vals in all_specs:
        print(f"  {name:25s}  {_describe_param(name, vals)}")
    print(f"  {'bin_size_ms':25s}  categorical {list(args.bin_size_choices)} (ms; sets NeuronConfig dt)")
    print(f"  {'collapse_factor':25s}  categorical {list(args.collapse_choices)}")
    print(f"  {'channel_shift_range':25s}  int [{args.channel_shift_min}, {args.channel_shift_max}] (collapsed units)")
    print(f"  {'arch':25s}  categorical { {k: list(v) for k, v in _ARCH_GEOMETRY.items()} }")
    print(f"  {'loss_count_bias':25s}  EXCLUDED (no-op: constant added to all logits before softmax)")
    print(f"  {'augment_channel_shift':25s}  {args.augment_channel_shift}")
    print(flush=True)

    np.random.seed(args.seed)

    dtype = np.float64 if args.precision == "64" else np.float32
    use_speakers = args.val_n_speakers > 0 or args.val_speakers is not None
    val_seed = args.val_seed if args.val_seed is not None else args.seed

    # ------------------------------------------------------------------
    # Bootstrap load at the CHEAPEST binning (largest bin, largest collapse)
    # purely to (a) learn the stable sample order / count and speaker labels,
    # and (b) compute the binning-INDEPENDENT split index sets ONCE. The
    # bootstrap split is then seeded into the cache so it isn't re-binned.
    # ------------------------------------------------------------------
    boot_bin = max(args.bin_size_choices)
    boot_collapse = max(args.collapse_choices)
    print(f"Loading SHD data (bootstrap binning bin={boot_bin}ms collapse={boot_collapse}) ...",
          flush=True)

    def _load_combo(bin_size_ms, collapse_factor):
        """Load + bin SHD at one geometry; return (train_data, test_data, spk_tr)."""
        loaded = load_shd_binned(
            bin_size_ms=bin_size_ms,
            collapse_factor=collapse_factor,
            max_duration_ms=args.max_duration_ms,
            binarize=args.binarize,
            dtype=dtype,
            return_speakers=use_speakers,
        )
        if use_speakers:
            X_tr, y_tr, _, X_te, y_te, _, spk_tr, _spk_te = loaded
        else:
            X_tr, y_tr, _, X_te, y_te, _ = loaded
            spk_tr = None
        if args.input_scale != 1.0:
            X_tr = X_tr * args.input_scale
            X_te = X_te * args.input_scale
        train_data = [(X_tr[i], int(y_tr[i])) for i in range(len(y_tr))]
        test_data = [(X_te[i], int(y_te[i])) for i in range(len(y_te))]
        return train_data, test_data, spk_tr

    boot_train, boot_test, boot_spk = _load_combo(boot_bin, boot_collapse)
    n_samples = len(boot_train)

    # --- compute the (binning-independent) validation split indices once ---
    if use_speakers:
        all_speakers = sorted(set(int(s) for s in boot_spk))
        if args.val_speakers is not None:
            held = [int(s) for s in args.val_speakers]
            missing = [s for s in held if s not in all_speakers]
            if missing:
                raise ValueError(f"--val_speakers {missing} not in train speakers {all_speakers}")
        else:
            shuf = list(all_speakers)
            np.random.RandomState(val_seed).shuffle(shuf)
            held = sorted(shuf[: args.val_n_speakers])
        held_set = set(held)
        val_idx = [i for i in range(n_samples) if int(boot_spk[i]) in held_set]
        pool_idx = [i for i in range(n_samples) if int(boot_spk[i]) not in held_set]
        pool_speakers = sorted(set(int(boot_spk[i]) for i in pool_idx))
        print(f"Speaker-held-out val: held speakers {held} (of {all_speakers}); "
              f"train fits on {pool_speakers}", flush=True)
    else:
        split_rng = np.random.RandomState(val_seed)
        perm = split_rng.permutation(n_samples)
        n_val = max(1, int(round(n_samples * args.val_fraction)))
        val_idx = [int(i) for i in perm[:n_val]]
        pool_idx = [int(i) for i in perm[n_val:]]
        print(f"Random val split: {len(val_idx)} held out by fraction "
              f"{args.val_fraction} (WARNING: seen-speaker only, weak test proxy)", flush=True)

    # --- per-trial training subset of the pool (fixed seeded subset) ---
    if args.train_fraction < 1.0:
        n_sub = max(1, int(round(len(pool_idx) * args.train_fraction)))
        sub_pos = np.random.RandomState(args.seed).permutation(len(pool_idx))[:n_sub]
        train_subset_idx = [pool_idx[int(j)] for j in sub_pos]
    else:
        train_subset_idx = list(pool_idx)

    print(
        f"Splits  ->  train_pool: {len(pool_idx)}  "
        f"(trial subset: {len(train_subset_idx)})  val: {len(val_idx)}  "
        f"test (reserved): {len(boot_test)}  | n_inputs varies by collapse  "
        f"precision=float{args.precision}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Per-(bin, collapse) split cache. Index sets above are binning-independent
    # because load_shd_binned's sample order is deterministic across binnings;
    # we assert that invariant once per combo (speaker mode).
    # ------------------------------------------------------------------
    _SPLIT_CACHE = {}

    def _slice_split(train_data, test_data):
        val_data = [train_data[i] for i in val_idx]
        train_pool = [train_data[i] for i in pool_idx]
        train_subset = [train_data[i] for i in train_subset_idx]
        n_inputs = train_pool[0][0].shape[1]
        return (train_subset, train_pool, val_data, test_data, n_inputs)

    def get_split_data(bin_size_ms, collapse_factor):
        key = (int(collapse_factor), float(bin_size_ms))
        if key not in _SPLIT_CACHE:
            train_data, test_data, spk_tr = _load_combo(bin_size_ms, collapse_factor)
            if use_speakers:
                assert np.array_equal(np.asarray(spk_tr), np.asarray(boot_spk)), (
                    "SHD sample order changed across binning "
                    f"(bin={bin_size_ms}, collapse={collapse_factor}); "
                    "split indices would be invalid."
                )
            _SPLIT_CACHE[key] = _slice_split(train_data, test_data)
        return _SPLIT_CACHE[key]

    # seed the bootstrap binning's split so it isn't re-binned
    _SPLIT_CACHE[(int(boot_collapse), float(boot_bin))] = _slice_split(boot_train, boot_test)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.prune:
        warmup = args.pruner_warmup if args.pruner_warmup is not None else max(args.epochs - 5, 5)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=args.pruner_startup,
            n_warmup_steps=warmup,
        )
        print(f"Pruner: MedianPruner(startup={args.pruner_startup}, warmup={warmup})", flush=True)
    else:
        pruner = optuna.pruners.NopPruner()
        print("Pruner: NopPruner (no early pruning — slow-but-better trials survive)", flush=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )

    # --- record the fixed config so the study/DB is self-documenting ---
    fixed_config = build_fixed_config(args)
    prev = study.user_attrs.get("fixed_config")
    if prev is not None and prev != fixed_config:
        diffs = {k: (prev.get(k), fixed_config.get(k))
                 for k in set(prev) | set(fixed_config)
                 if prev.get(k) != fixed_config.get(k)}
        print("=" * 70, flush=True)
        print("WARNING: resuming study with a DIFFERENT fixed config than the "
              "trials already in this DB.\n"
              "         Existing and new trials are NOT apples-to-apples:", flush=True)
        for k, (old, new) in sorted(diffs.items()):
            print(f"           {k}: {old} -> {new}", flush=True)
        print("=" * 70, flush=True)
    history = list(study.user_attrs.get("config_history", []))
    history.append({"time": datetime.datetime.now().isoformat(timespec="seconds"),
                    "config": fixed_config})
    study.set_user_attr("config_history", history)
    study.set_user_attr("fixed_config", fixed_config)
    print("Fixed config (recorded to study):")
    for k, v in fixed_config.items():
        print(f"  {k:24s}  {v}")
    print(flush=True)

    def objective(trial):
        return run_trial(trial, args, get_split_data, fixed_config)

    print(f"Starting Optuna study '{args.study_name}' "
          f"({args.n_trials} trials, up to {args.epochs} epochs each, "
          f"selecting on VAL)...\n",
          flush=True)

    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\nCompleted: {len(completed)}  Pruned: {len(pruned)}")

    if not completed:
        print("No completed trials; nothing to confirm.")
        return

    # ---- final confirmation: re-validate top-K on FULL pool, then test once ----
    top = sorted(completed, key=lambda t: t.value, reverse=True)[:max(1, args.final_topk)]
    print(f"\n=== Final confirmation: re-validating top {len(top)} trials on the "
          f"FULL train pool ({len(pool_idx)} samples) ===", flush=True)

    best = None  # (full_val_acc, params, trial_number)
    for t in top:
        params = t.user_attrs["params"]
        # Re-fetch THIS trial's own binning on the full pool (cache hit unless
        # a rare combo was never touched during the study).
        _sub, train_pool, val_data, _test, n_inputs = get_split_data(
            params["bin_size_ms"], params["collapse_factor"])
        print(f"\n-- Re-validating trial #{t.number} "
              f"(subset val_acc={t.value:.2f}%)  params={params}", flush=True)
        full_val_acc, _ = train_and_eval(
            params, args, train_pool, val_data, n_inputs,
            seed=args.seed, trial=None, eval_label="val",
        )
        print(f"   full-pool val_acc = {full_val_acc:.2f}%", flush=True)
        if best is None or full_val_acc > best[0]:
            best = (full_val_acc, params, t.number)

    full_val_acc, best_params, best_trial_no = best
    # Final test confirmation trains on ALL train speakers (pool + held-out val),
    # at the best trial's own binning, matching real deployment.
    _sub, train_pool, val_data, test_data, n_inputs = get_split_data(
        best_params["bin_size_ms"], best_params["collapse_factor"])
    full_train = train_pool + val_data
    print(f"\n=== Best config (trial #{best_trial_no}, held-out val_acc={full_val_acc:.2f}%) — "
          f"retraining on ALL {len(full_train)} train samples, confirming on TEST once ===",
          flush=True)
    test_acc, _ = train_and_eval(
        best_params, args, full_train, test_data, n_inputs,
        seed=args.seed, trial=None, eval_label="test",
    )

    print("\n=== Result ===")
    print(f"  Best trial          : #{best_trial_no}")
    print(f"  Full-pool VAL acc   : {full_val_acc:.2f}%")
    print(f"  TEST acc (one-shot) : {test_acc:.2f}%")
    print("  Parameters          :")
    for k, v in best_params.items():
        print(f"    {k:22s}  {v}")


if __name__ == "__main__":
    main()
