#!/usr/bin/env python3
"""Optuna hyperparameter search for the no-history SHD model.

Each of the 6 tunable parameters accepts 1, 2, or 3 values:
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
from data.shd_binned import load_shd_binned


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


def run_trial(trial, args, train_data, test_data, n_inputs):
    lr = suggest_or_static(trial, "lr", args.lr)
    loss_temperature = suggest_or_static(trial, "loss_temperature", args.loss_temperature)
    loss_count_bias = suggest_or_static(trial, "loss_count_bias", args.loss_count_bias)
    loss_label_smoothing = suggest_or_static(trial, "loss_label_smoothing", args.loss_label_smoothing)
    beta_s = suggest_or_static(trial, "beta_s", args.beta_s)
    beta_d = suggest_or_static(trial, "beta_d", args.beta_d)
    tau_w = suggest_or_static(trial, "tau_w", args.tau_w)
    a_adapt = suggest_or_static(trial, "a_adapt", args.a_adapt)
    b_adapt = suggest_or_static(trial, "b_adapt", args.b_adapt)

    print(
        f"\n--- Trial {trial.number + 1}/{args.n_trials} ---"
        f"  lr={lr:.4g}  temp={loss_temperature:.4g}  bias={loss_count_bias:.4g}"
        f"  smooth={loss_label_smoothing:.4g}  beta_s={beta_s:.4g}  beta_d={beta_d:.4g}"
        f"  tau_w={tau_w:.4g}  a_adapt={a_adapt:.4g}  b_adapt={b_adapt:.4g}",
        flush=True,
    )

    config = NeuronConfig(
        dt=args.bin_size_ms,
        tau_soma=args.tau_soma,
        tau_dend=args.tau_dend,
        tau_m=args.tau_m,
        tau_plat_min=args.tau_plat_min,
        tau_plat_max=args.tau_plat_max,
        tau_w=tau_w,
        a_adapt=a_adapt,
        b_adapt=b_adapt,
        mu_th=args.mu_th,
        v_th=args.v_th,
        gamma=args.gamma,
        beta_s=beta_s,
        beta_d=beta_d,
        weight_scale=args.weight_scale,
        loss_temperature=loss_temperature,
        loss_count_bias=loss_count_bias,
        loss_label_smoothing=loss_label_smoothing,
    )

    key = random.PRNGKey(args.seed + trial.number)
    B = args.batch_size
    net = Network(
        key, n_inputs, args.n_hidden, args.n_outputs, config,
        optimizer=args.optimizer,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_eps=args.adam_eps,
        dropout_rate=args.dropout,
        weight_decay=args.weight_decay,
    )

    n_train = len(train_data)
    n_batches = n_train // B
    samples_per_epoch = n_batches * B

    current_lr = lr
    best_test_acc = 0.0
    epochs_since_lr_drop = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        idx = np.random.permutation(n_train)

        for bi in range(n_batches):
            start = bi * B
            batch_idx = idx[start: start + B]

            if B == 1:
                x, y = train_data[int(batch_idx[0])]
                if args.augment_jitter:
                    x = apply_temporal_jitter(x, args.jitter_range)
                net.train_step(
                    jnp.array(x), int(y), lr=current_lr, clip_value=args.gradient_clip,
                )
            else:
                x_batch_np = [
                    apply_temporal_jitter(train_data[int(i)][0], args.jitter_range)
                    if args.augment_jitter
                    else train_data[int(i)][0]
                    for i in batch_idx
                ]
                x_batch = jnp.stack(x_batch_np)
                y_batch = jnp.array([int(train_data[int(i)][1]) for i in batch_idx])
                net.batch_train_step(
                    x_batch, y_batch, lr=current_lr, clip_value=args.gradient_clip,
                )

        test_acc = evaluate(net, test_data, B)

        improved = test_acc > best_test_acc
        if improved:
            best_test_acc = test_acc
            epochs_since_lr_drop = 0
            epochs_without_improvement = 0
        else:
            epochs_since_lr_drop += 1
            epochs_without_improvement += 1

        marker = "*" if improved else " "
        print(
            f"  Epoch {epoch:2d}/{args.epochs}  test_acc={test_acc:.2f}%{marker}"
            f"  lr={current_lr:.2e}",
            flush=True,
        )

        trial.report(test_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if (args.lr_factor < 1.0
                and args.lr_patience > 0
                and current_lr > args.lr_min
                and epochs_since_lr_drop >= args.lr_patience):
            new_lr = max(current_lr * args.lr_factor, args.lr_min)
            if new_lr < current_lr:
                current_lr = new_lr
                epochs_since_lr_drop = 0

        if (args.early_stop_patience > 0
                and epochs_without_improvement >= args.early_stop_patience):
            break

    return best_test_acc


def parse_args():
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter search for no-history SHD model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- tunable parameters (1, 2, or 3 values each) ----
    tunable = p.add_argument_group(
        "tunable parameters",
        "Pass 1 value = static, 2 values = uniform range, 3 values (last='log') = log-uniform.",
    )
    tunable.add_argument("--lr", nargs="+", default=["1e-3"],
                         metavar="VAL", help="Learning rate (default static 1e-3).")
    tunable.add_argument("--loss_temperature", nargs="+", default=["2.7"],
                         metavar="VAL")
    tunable.add_argument("--loss_count_bias", nargs="+", default=["0.18"],
                         metavar="VAL")
    tunable.add_argument("--loss_label_smoothing", nargs="+", default=["0.13"],
                         metavar="VAL")
    tunable.add_argument("--beta_s", nargs="+", default=["1.0"],
                         metavar="VAL", help="Somatic surrogate gradient scale.")
    tunable.add_argument("--beta_d", nargs="+", default=["1.5"],
                         metavar="VAL", help="Dendritic surrogate gradient scale.")
    tunable.add_argument("--tau_w", nargs="+", default=["100.0"],
                         metavar="VAL", help="Adaptation time constant (ms).")
    tunable.add_argument("--a_adapt", nargs="+", default=["0.0"],
                         metavar="VAL", help="Subthreshold adaptation coupling.")
    tunable.add_argument("--b_adapt", nargs="+", default=["0.0"],
                         metavar="VAL", help="Spike-triggered adaptation jump.")

    # ---- Optuna study settings ----
    study = p.add_argument_group("optuna study")
    study.add_argument("--n_trials", type=int, default=50)
    study.add_argument("--n_jobs", type=int, default=1,
                       help="Parallel Optuna workers (use 1 with JAX).")
    study.add_argument("--study_name", default="shd_optuna")
    study.add_argument("--storage", default=None,
                       help="Optuna storage URL, e.g. sqlite:///study.db. "
                            "Default: in-memory (not persistent).")
    study.add_argument("--pruner_startup", type=int, default=5,
                       help="MedianPruner n_startup_trials (default 5).")
    study.add_argument("--pruner_warmup", type=int, default=3,
                       help="MedianPruner n_warmup_steps per trial (default 3).")

    # ---- fixed model / data settings (mirrored from run_shd.py) ----
    p.add_argument("--bin_size_ms", type=float, default=4.0)
    p.add_argument("--collapse_factor", type=int, default=5)
    p.add_argument("--max_duration_ms", type=float, default=1400.0)
    p.add_argument("--binarize", action="store_true")
    p.add_argument("--input_scale", type=float, default=1.0)
    p.add_argument("--n_hidden", type=int, default=64)
    p.add_argument("--n_outputs", type=int, default=20)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_clip", type=float, default=5.0)
    p.add_argument("--weight_scale", type=float, default=0.25)
    p.add_argument("--tau_soma", type=float, default=15.0)
    p.add_argument("--tau_dend", type=float, default=15.0)
    p.add_argument("--tau_m", type=float, default=20.0)
    p.add_argument("--tau_plat_min", type=float, default=100.0)
    p.add_argument("--tau_plat_max", type=float, default=350.0)
    p.add_argument("--mu_th", type=float, default=1.0)
    p.add_argument("--v_th", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--train_fraction", type=float, default=1.0,
                   help="Fraction of training data to use per trial (default 1.0 = all).")
    p.add_argument("--test_fraction", type=float, default=1.0,
                   help="Fraction of test data to use for evaluation (default 1.0 = all).")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--augment_jitter", action="store_true")
    p.add_argument("--jitter_range", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--lr_patience", type=int, default=5)
    p.add_argument("--lr_factor", type=float, default=0.7)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--early_stop_patience", type=int, default=0)
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

    # Validate all tunable param specs early so we fail fast.
    tunable_names = ["lr", "loss_temperature", "loss_count_bias",
                     "loss_label_smoothing", "beta_s", "beta_d",
                     "tau_w", "a_adapt", "b_adapt"]
    tunable_values = [args.lr, args.loss_temperature, args.loss_count_bias,
                      args.loss_label_smoothing, args.beta_s, args.beta_d,
                      args.tau_w, args.a_adapt, args.b_adapt]
    for name, vals in zip(tunable_names, tunable_values):
        parse_param(name, vals)  # raises on bad input

    print("Parameter configuration:")
    for name, vals in zip(tunable_names, tunable_values):
        print(f"  {name:25s}  {_describe_param(name, vals)}")
    print(f"  {'weight_scale':25s}  static={args.weight_scale}  (bin_size_ms={args.bin_size_ms}ms)")
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

    if args.train_fraction < 1.0:
        n = max(1, int(len(train_data) * args.train_fraction))
        train_data = train_data[:n]
    if args.test_fraction < 1.0:
        n = max(1, int(len(test_data) * args.test_fraction))
        test_data = test_data[:n]

    n_inputs = train_data[0][0].shape[1]
    print(
        f"Train: {len(train_data)}  Test: {len(test_data)}  "
        f"n_inputs: {n_inputs}  precision=float{args.precision}",
        flush=True,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.pruner_startup,
        n_warmup_steps=args.pruner_warmup,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )

    def objective(trial):
        return run_trial(trial, args, train_data, test_data, n_inputs)

    print(f"Starting Optuna study '{args.study_name}' "
          f"({args.n_trials} trials, {args.epochs} epochs each)...\n",
          flush=True)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
    )

    print("\n=== Best trial ===")
    best = study.best_trial
    print(f"  Test accuracy : {best.value:.2f}%")
    print(f"  Trial number  : {best.number}")
    print("  Parameters    :")
    for name, vals in zip(tunable_names, tunable_values):
        static_val, *_ = parse_param(name, vals)
        if static_val is not None:
            print(f"    {name:25s}  {static_val}  (static)")
        else:
            print(f"    {name:25s}  {best.params[name]:.6g}")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\nCompleted: {len(completed)}  Pruned: {len(pruned)}")


if __name__ == "__main__":
    main()
