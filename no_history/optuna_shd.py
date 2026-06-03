#!/usr/bin/env python3
"""Optuna hyperparameter search for the no-history SHD model.

This search is built for the *augmented, long-horizon* regime: channel-shift
augmentation is ON by default, there is no early pruning, trials run for many
epochs, and configs are selected on a held-out VALIDATION split carved from the
training set (the test set is touched exactly once, in the final confirmation
step). See no_history/optuna_shd.py header notes below.

Why this differs from a vanilla Optuna setup:
  * The channel-shift regime converges slowly but generalizes better; a median
    pruner would kill those trials early, so pruning is OFF by default.
  * Selecting on test accuracy tunes hyperparameters against the test set, so we
    select on validation instead and confirm on test only once at the end.
  * `loss_count_bias` is a no-op (a constant added to every logit before a
    shift-invariant softmax) and is intentionally NOT tunable here.

Each tunable parameter accepts 1, 2, or 3 values:
  --lr 1e-3              -> static (fixed at 1e-3)
  --lr 1e-5 1e-1         -> tune, uniform in [1e-5, 1e-1]
  --lr 1e-5 1e-1 log     -> tune, log-uniform in [1e-5, 1e-1]

If a flag is omitted, the built-in default is used as a static value.
"""
import argparse
import os
import sys

import optuna


def _precision_from_argv(argv):
    default = "64"
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

    Mirrors no_history/run_shd.py:augment_sample, but takes an explicit
    `channel_shift_range` (resolved per trial) instead of reading it off args.
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

    `params` is a fully-resolved dict (no Optuna suggestions inside). The LR
    scheduler and best-acc tracking key off `eval_set` accuracy. When `trial`
    is given, per-epoch accuracy is reported (used only for logging/pruning;
    the default study uses NopPruner so reporting never kills a trial).
    """
    config = NeuronConfig(
        dt=args.bin_size_ms,
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
        key, n_inputs, args.n_hidden, args.n_outputs, config,
        optimizer=args.optimizer,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_eps=args.adam_eps,
        dropout_rate=params["dropout"],
        weight_decay=params["weight_decay"],
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

    # Channel-shift range: prefer the collapse-invariant raw parametrization.
    if args.channel_shift_range is not None:
        csr = int(round(suggest_or_static(trial, "channel_shift_range", args.channel_shift_range)))
    else:
        raw = suggest_or_static(trial, "channel_shift_raw", args.channel_shift_raw)
        csr = max(1, int(round(raw / args.collapse_factor)))
    params["channel_shift_range"] = csr
    return params


def run_trial(trial, args, train_set, val_set, n_inputs):
    params = resolve_params(trial, args)
    trial.set_user_attr("params", params)

    print(
        f"\n--- Trial {trial.number + 1}/{args.n_trials} ---"
        f"  lr={params['lr']:.4g}  temp={params['loss_temperature']:.4g}"
        f"  smooth={params['loss_label_smoothing']:.4g}"
        f"  beta_s={params['beta_s']:.4g}  beta_d={params['beta_d']:.4g}"
        f"  dropout={params['dropout']:.4g}  wd={params['weight_decay']:.4g}"
        f"  a={params['a_adapt']:.4g}  b={params['b_adapt']:.4g}"
        f"  mu_th={params['mu_th']:.4g}  gamma={params['gamma']:.4g}"
        f"  lr_factor={params['lr_factor']:.4g}  lr_patience={int(round(params['lr_patience']))}"
        f"  ch_shift={params['channel_shift_range']}",
        flush=True,
    )

    best_acc, _ = train_and_eval(
        params, args, train_set, val_set, n_inputs,
        seed=args.seed + trial.number, trial=trial, eval_label="val",
    )
    return best_acc


# Tunable parameter names whose value lists live on args under the same name.
_TUNABLE_NAMES = [
    "lr", "loss_temperature", "loss_label_smoothing", "beta_s", "beta_d",
    "tau_w", "a_adapt", "b_adapt", "dropout", "weight_decay",
    "mu_th", "gamma", "tau_plat_min", "tau_plat_max",
    "lr_factor", "lr_patience",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter search for no-history SHD model "
                    "(augmented, long-horizon, validation-selected).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- tunable parameters (1, 2, or 3 values each) ----
    tunable = p.add_argument_group(
        "tunable parameters",
        "Pass 1 value = static, 2 values = uniform range, 3 values (last='log') = log-uniform.",
    )
    tunable.add_argument("--lr", nargs="+", default=["1e-3"], metavar="VAL",
                         help="Learning rate (default static 1e-3).")
    tunable.add_argument("--loss_temperature", nargs="+", default=["2.7"], metavar="VAL")
    tunable.add_argument("--loss_label_smoothing", nargs="+", default=["0.13"], metavar="VAL")
    tunable.add_argument("--beta_s", nargs="+", default=["1.0"], metavar="VAL",
                         help="Somatic surrogate gradient scale.")
    tunable.add_argument("--beta_d", nargs="+", default=["1.5"], metavar="VAL",
                         help="Dendritic surrogate gradient scale.")
    tunable.add_argument("--tau_w", nargs="+", default=["100.0"], metavar="VAL",
                         help="Adaptation time constant (ms).")
    tunable.add_argument("--a_adapt", nargs="+", default=["0.0"], metavar="VAL",
                         help="Subthreshold adaptation coupling.")
    tunable.add_argument("--b_adapt", nargs="+", default=["0.0"], metavar="VAL",
                         help="Spike-triggered adaptation jump.")
    tunable.add_argument("--dropout", nargs="+", default=["0.0"], metavar="VAL",
                         help="Hidden->readout dropout rate (wired & functional).")
    tunable.add_argument("--weight_decay", nargs="+", default=["0.0"], metavar="VAL",
                         help="Decoupled weight decay (AdamW-style / L2-equivalent).")
    tunable.add_argument("--mu_th", nargs="+", default=["1.0"], metavar="VAL",
                         help="Dendritic plateau threshold (temporal-memory knob).")
    tunable.add_argument("--gamma", nargs="+", default=["0.5"], metavar="VAL",
                         help="Plateau-induced threshold reduction (temporal-memory knob).")
    tunable.add_argument("--tau_plat_min", nargs="+", default=["100.0"], metavar="VAL",
                         help="Plateau duration min (ms).")
    tunable.add_argument("--tau_plat_max", nargs="+", default=["350.0"], metavar="VAL",
                         help="Plateau duration max (ms).")
    tunable.add_argument("--channel_shift_raw", nargs="+", default=["25"], metavar="VAL",
                         help="Channel-shift magnitude in RAW (700-channel) units; "
                              "mapped per trial to the collapsed axis as "
                              "round(raw/collapse_factor). Recommended (collapse-invariant).")
    tunable.add_argument("--channel_shift_range", nargs="+", default=None, metavar="VAL",
                         help="Alternative: channel-shift range directly in COLLAPSED units. "
                              "If set, overrides --channel_shift_raw.")

    # ---- Optuna study settings ----
    study = p.add_argument_group("optuna study")
    study.add_argument("--n_trials", type=int, default=50)
    study.add_argument("--n_jobs", type=int, default=1,
                       help="Parallel Optuna workers (use 1 with JAX).")
    study.add_argument("--study_name", default="shd_optuna_aug")
    study.add_argument("--storage", default=None,
                       help="Optuna storage URL, e.g. sqlite:///shd_aug.db. "
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
    splitg.add_argument("--val_fraction", type=float, default=0.2,
                        help="Fraction of TRAIN carved off as the validation split (default 0.2).")
    splitg.add_argument("--val_seed", type=int, default=None,
                        help="Seed for the train/val split (default: --seed).")
    splitg.add_argument("--train_fraction", type=float, default=1.0,
                        help="Fraction of the train POOL used per trial (fixed seeded subset).")

    # ---- fixed model / data settings (mirrored from run_shd.py) ----
    p.add_argument("--bin_size_ms", type=float, default=4.0)
    p.add_argument("--collapse_factor", type=int, default=5)
    p.add_argument("--max_duration_ms", type=float, default=1400.0)
    p.add_argument("--binarize", action="store_true")
    p.add_argument("--input_scale", type=float, default=1.0)
    p.add_argument("--n_hidden", type=int, default=150)
    p.add_argument("--n_outputs", type=int, default=20)
    p.add_argument("--epochs", type=int, default=80)
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
    # LR-scheduler knobs are tunable (1/2/3 values, like the other tunables).
    p.add_argument("--lr_factor", nargs="+", default=["0.7"], metavar="VAL",
                   help="ReduceLROnPlateau multiplier (lr := lr * factor). 1.0 disables.")
    p.add_argument("--lr_patience", nargs="+", default=["5"], metavar="VAL",
                   help="Epochs without val improvement before an LR drop (rounded to int).")
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="0 (default, recommended for long runs) disables early stopping.")
    p.add_argument("--precision", choices=["32", "64"], default=_PRECISION)

    return p.parse_args()


def _channel_shift_spec(args):
    """Return (name, values) for whichever channel-shift parametrization is active."""
    if args.channel_shift_range is not None:
        return "channel_shift_range", args.channel_shift_range
    return "channel_shift_raw", args.channel_shift_raw


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

    # Validate all tunable param specs early so we fail fast.
    cs_name, cs_values = _channel_shift_spec(args)
    all_specs = [(n, getattr(args, n)) for n in _TUNABLE_NAMES] + [(cs_name, cs_values)]
    for name, vals in all_specs:
        parse_param(name, vals)  # raises on bad input

    print("Parameter configuration:")
    for name, vals in all_specs:
        print(f"  {name:25s}  {_describe_param(name, vals)}")
    print(f"  {'loss_count_bias':25s}  EXCLUDED (no-op: constant added to all logits before softmax)")
    print(f"  {'augment_channel_shift':25s}  {args.augment_channel_shift}")
    print(f"  {'collapse_factor':25s}  {args.collapse_factor}  (bin_size_ms={args.bin_size_ms}ms)")
    print(flush=True)

    np.random.seed(args.seed)

    dtype = np.float64 if args.precision == "64" else np.float32
    print("Loading SHD data...", flush=True)
    X_tr, y_tr, _, X_te, y_te, _ = load_shd_binned(
        bin_size_ms=args.bin_size_ms,
        collapse_factor=args.collapse_factor,
        max_duration_ms=args.max_duration_ms,
        binarize=args.binarize,
        dtype=dtype,
    )
    if args.input_scale != 1.0:
        X_tr = X_tr * args.input_scale
        X_te = X_te * args.input_scale

    train_data = [(X_tr[i], int(y_tr[i])) for i in range(len(y_tr))]
    test_data = [(X_te[i], int(y_te[i])) for i in range(len(y_te))]

    # --- carve a fixed validation split off TRAIN (test is reserved) ---
    val_seed = args.val_seed if args.val_seed is not None else args.seed
    split_rng = np.random.RandomState(val_seed)
    perm = split_rng.permutation(len(train_data))
    n_val = max(1, int(round(len(train_data) * args.val_fraction)))
    val_idx = perm[:n_val]
    pool_idx = perm[n_val:]
    val_data = [train_data[int(i)] for i in val_idx]
    train_pool = [train_data[int(i)] for i in pool_idx]

    # --- per-trial training subset of the pool (fixed seeded subset) ---
    if args.train_fraction < 1.0:
        n_sub = max(1, int(round(len(train_pool) * args.train_fraction)))
        sub_idx = np.random.RandomState(args.seed).permutation(len(train_pool))[:n_sub]
        train_subset = [train_pool[int(i)] for i in sub_idx]
    else:
        train_subset = train_pool

    n_inputs = train_pool[0][0].shape[1]
    print(
        f"Splits  ->  train_pool: {len(train_pool)}  "
        f"(trial subset: {len(train_subset)})  val: {len(val_data)}  "
        f"test (reserved): {len(test_data)}  | n_inputs: {n_inputs}  "
        f"precision=float{args.precision}",
        flush=True,
    )

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

    def objective(trial):
        return run_trial(trial, args, train_subset, val_data, n_inputs)

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
          f"FULL train pool ({len(train_pool)} samples) ===", flush=True)

    best = None  # (full_val_acc, params, trial_number)
    for t in top:
        params = t.user_attrs["params"]
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
    print(f"\n=== Best config (trial #{best_trial_no}, full-pool val_acc={full_val_acc:.2f}%) — "
          f"confirming on TEST once ===", flush=True)
    test_acc, _ = train_and_eval(
        best_params, args, train_pool, test_data, n_inputs,
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
