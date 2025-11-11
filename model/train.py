from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Local modules
from dataset import build_dataset
from architectures import TCNGRUModel


SEED = 42
np.random.seed(SEED)
try:
    tf.random.set_seed(SEED)
except Exception:
    pass


@dataclass
class TrainConfig:
    data_dir: Path
    sr: int = 2000
    window_size: int = 4000
    downsampling_factor: int = 16
    test_size: float = 0.25
    batch_size: int = 32
    base_channels: int = 256
    noise_std: float = 0.01
    epochs: int = 50
    learning_rate: float = 1e-3
    best_dir: Path = Path("model/best_models")
    make_readme_plots: bool = False
    plots_outdir: Path = Path("model/images")
    mixed_precision: bool = False


def _plot_training_window(
    X: np.ndarray, Y: np.ndarray, idx: int, downsampling_factor: int, outpath: Path
):
    ts = X[idx].squeeze()
    steps = ts.shape[0]
    t = np.linspace(0, 1, steps)
    labels_idx = np.argmax(Y[idx], axis=1)
    labels_up = np.repeat(labels_idx, downsampling_factor)[:steps]
    plt.figure(figsize=(10, 3))
    plt.plot(t, ts, label="waveform")
    plt.plot(t, labels_up * np.abs(ts).max() / 2, label="labels (upsampled)")
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


def train(
    cfg: TrainConfig,
) -> Tuple[tf.keras.Model, dict, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if cfg.mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("[info] Using mixed precision.")
        except Exception:
            print("[warn] Mixed precision not available; continuing in float32.")

    train_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"[train] Starting training run {train_id}")

    # Data
    X, Y = build_dataset(
        data_dir=cfg.data_dir,
        sr=cfg.sr,
        window_size=cfg.window_size,
        downsampling_factor=cfg.downsampling_factor,
        require_fully_labeled=True,
        apply_bandpass=False,
    )
    X = X.reshape(-1, cfg.window_size, 1)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=cfg.test_size, random_state=SEED, shuffle=True
    )

    # Model
    model = TCNGRUModel(
        time_steps=cfg.window_size,
        n_classes=3,
        base_channels=cfg.base_channels,
        noise_std=cfg.noise_std,
    )
    opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.summary()

    # IO
    cfg.best_dir.mkdir(parents=True, exist_ok=True)

    ckpt_best = callbacks.ModelCheckpoint(
        filepath=str(cfg.best_dir / f"{train_id}.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    early = callbacks.EarlyStopping(
        monitor="val_loss", patience=12, restore_best_weights=True
    )
    rlrop = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=6, verbose=1
    )

    # Optional README plot
    if cfg.make_readme_plots:
        _plot_training_window(
            X_train,
            Y_train,
            idx=min(2, len(X_train) - 1),
            downsampling_factor=cfg.downsampling_factor,
            outpath=cfg.plots_outdir / f"{train_id}_training_window.png",
        )

    # Train
    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=[ckpt_best, early, rlrop],
        verbose=1,
    )

    # Eval
    eval_res = model.evaluate(X_val, Y_val, verbose=0)
    print(f"[eval] {dict(zip(model.metrics_names, eval_res))}")

    # Final snapshot
    final_name = f"final-{datetime.now().strftime('%Y%m%d-%H%M%S')}.keras"
    final_path = cfg.checkpoints_dir / final_name
    model.save(final_path)
    print(f"[save] Final model -> {final_path}")

    return model, history.history, (X_train, X_val, Y_train, Y_val)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train heart-sound segmentation")
    p.add_argument(
        "--data_dir", type=Path, required=True, help="Folder with *.wav and *.tsv"
    )
    p.add_argument("--sr", type=int, default=2000)
    p.add_argument(
        "--window_size", type=int, default=4000, help="Must be divisible by 16"
    )
    p.add_argument(
        "--downsampling_factor",
        type=int,
        default=16,
        help="Match model temporal stride",
    )
    p.add_argument("--test_size", type=float, default=0.25)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--base_channels", type=int, default=256)
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--best_dir", type=Path, default=Path("model/best_models"))
    p.add_argument("--make_readme_plots", action="store_true")
    p.add_argument("--plots_outdir", type=Path, default=Path("model/images"))
    p.add_argument("--mixed_precision", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    if args.window_size % args.downsampling_factor != 0:
        raise ValueError("window_size must be divisible by downsampling_factor")
    if args.window_size % 16 != 0:
        raise ValueError("window_size must be divisible by 16 (two 4x4 poolings)")
    cfg = TrainConfig(
        data_dir=args.data_dir,
        sr=args.sr,
        window_size=args.window_size,
        downsampling_factor=args.downsampling_factor,
        test_size=args.test_size,
        batch_size=args.batch_size,
        base_channels=args.base_channels,
        noise_std=args.noise_std,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        best_dir=args.best_dir,
        make_readme_plots=args.make_readme_plots,
        plots_outdir=args.plots_outdir,
        mixed_precision=args.mixed_precision,
    )
    train(cfg)


if __name__ == "__main__":
    main()
