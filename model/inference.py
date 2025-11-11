#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List
import numpy as np
import librosa
import tensorflow as tf
from tcn import TCN
from dataset import (
    bandpass,
    load_intervals_tsv,
    build_frame_labels,
    map_labels_to_three_classes,
)


def load_model(model_path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(
        model_path, compile=False, custom_objects={"TCN": TCN}
    )


def _predict_probs_full(
    y: np.ndarray,
    model: tf.keras.Model,
    sr: int,
    window_size: int,
    downsampling_factor: int,
    stride: int,
) -> np.ndarray:
    """Return per-step class probabilities over full signal (average overlaps)."""
    n_classes = model.output_shape[-1]
    total_steps = math.ceil(len(y) / downsampling_factor)
    acc = np.zeros((total_steps, n_classes), dtype=np.float32)
    cnt = np.zeros((total_steps,), dtype=np.float32)

    for s in range(0, max(1, len(y) - window_size + 1), stride):
        seg = y[s : s + window_size]
        if len(seg) < window_size:
            seg = np.pad(seg, (0, window_size - len(seg)))
        seg = (seg - seg.mean()) / (seg.std() + 1e-8)
        probs = model.predict(seg.reshape(1, window_size, 1), verbose=0)[0]
        t0 = s // downsampling_factor
        t1 = t0 + probs.shape[0]
        acc[t0:t1] += probs
        cnt[t0:t1] += 1.0

    cnt[cnt == 0] = 1.0
    return acc / cnt[:, None]


def _steps_to_segments(
    pred_steps: np.ndarray, sr: int, downsampling_factor: int, class_names: List[str]
) -> List[Dict]:
    segs: List[Dict] = []
    if len(pred_steps) == 0:
        return segs
    cur = int(pred_steps[0])
    start = 0
    for i in range(1, len(pred_steps)):
        if int(pred_steps[i]) != cur:
            segs.append(
                {
                    "start": start * downsampling_factor / sr,
                    "end": i * downsampling_factor / sr,
                    "class_id": int(cur),
                    "class_name": class_names[int(cur)]
                    if 0 <= int(cur) < len(class_names)
                    else f"C{int(cur)}",
                }
            )
            start = i
            cur = int(pred_steps[i])
    segs.append(
        {
            "start": start * downsampling_factor / sr,
            "end": len(pred_steps) * downsampling_factor / sr,
            "class_id": int(cur),
            "class_name": class_names[int(cur)]
            if 0 <= int(cur) < len(class_names)
            else f"C{int(cur)}",
        }
    )
    return segs


def plot_prediction(
    wav_path: Path,
    model: tf.keras.Model,
    sr: int = 2000,
    window_size: int = 4000,
    downsampling_factor: int = 16,
    apply_bandpass: bool = False,
    overlay_tsv: Path | None = None,
    outpath: Path | None = None,
):
    """
    Plot waveform + predicted labels (step plot). If overlay_tsv is provided (or
    a sibling .tsv exists), overlays ground truth classes.
    """
    import matplotlib.pyplot as plt

    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)

    if len(y) < window_size:
        y = np.pad(y, (0, window_size - len(y)))
    y = y[:window_size]

    y_vis = bandpass(y, sr) if apply_bandpass else y.copy()
    t = np.linspace(0, 1, window_size)

    x = (y_vis - y_vis.mean()) / (y_vis.std() + 1e-8)
    xin = x.reshape(1, window_size, 1)

    pred = model.predict(xin, verbose=0)[0]  # (T_out, C)
    pred_idx = np.argmax(pred, axis=1)  # (T_out,)

    z = np.repeat(pred_idx, downsampling_factor) / 8.0
    z = z[:window_size]

    z[z == 0] = 0.1
    z[z == 1 / 8] = 0.0

    gt_curve = None
    if overlay_tsv is None:
        auto = wav_path.with_suffix(".tsv")
        overlay_tsv = auto if auto.exists() else None
    if overlay_tsv and overlay_tsv.exists():
        try:
            intervals = load_intervals_tsv(overlay_tsv, sr)
            frame_labels = build_frame_labels(window_size, sr, intervals)
            mapped = map_labels_to_three_classes(frame_labels)
            mapped = mapped[::downsampling_factor][: len(pred_idx)]

            gt_up = np.repeat(mapped, downsampling_factor) / 8.0
            gt_up = gt_up[:window_size]
            gt_up[gt_up == 0] = 0.1
            gt_up[gt_up == 1 / 8] = 0.0

            gt_curve = gt_up
        except Exception as e:
            print(f"[warn] could not load ground truth: {e}")

    plt.figure(figsize=(10, 3))
    plt.plot(t, y_vis, label="Waveform")
    plt.plot(t, z, label="Prediction (upsampled)")

    if gt_curve is not None:
        plt.plot(t, gt_curve, label="Ground Truth (upsampled)")

    plt.xlabel("Normalized time (0-1)")
    plt.title(f"Notebook-style prediction plot — {wav_path.name}")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if outpath:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"[plot] saved -> {outpath}")
    else:
        plt.show()


def run_on_wav(
    wav_path: Path,
    model: tf.keras.Model,
    sr: int,
    window_size: int,
    downsampling_factor: int,
    stride: int,
    class_names: List[str],
) -> Dict:
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    probs = _predict_probs_full(y, model, sr, window_size, downsampling_factor, stride)
    pred = probs.argmax(axis=1)
    segs = _steps_to_segments(pred, sr, downsampling_factor, class_names)
    return {
        "file": str(wav_path),
        "sr": sr,
        "window_size": window_size,
        "downsampling_factor": downsampling_factor,
        "stride": stride,
        "segments": segs,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference for heart-sound segmentation (with plotting)"
    )
    p.add_argument("--model_path", type=Path, required=True)
    p.add_argument("--wav", type=Path, help="Path to a .wav file")
    p.add_argument("--wav_dir", type=Path, help="Directory of .wav files")
    p.add_argument("--sr", type=int, default=2000)
    p.add_argument("--window_size", type=int, default=4000)
    p.add_argument("--downsampling_factor", type=int, default=16)
    p.add_argument(
        "--stride",
        type=int,
        default=4000,
        help="Sliding window stride in samples (default: window_size)",
    )
    p.add_argument("--class_names", nargs="*", default=["C0", "C1", "C2"])
    p.add_argument("--out_json", type=Path, help="Write predictions to JSON")

    # plotting flags
    p.add_argument(
        "--plot", action="store_true", help="Visualize waveform + predicted labels"
    )
    p.add_argument(
        "--plot_outdir",
        type=Path,
        help="Folder to save PNG plots (if omitted, will show interactively)",
    )
    p.add_argument(
        "--apply_bandpass",
        action="store_true",
        help="Bandpass filter for nicer visuals",
    )
    p.add_argument(
        "--overlay_tsv",
        action="store_true",
        help="Overlay ground-truth from sibling .tsv if available",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    if not args.wav and not args.wav_dir:
        raise SystemExit("Provide --wav or --wav_dir")

    model = load_model(args.model_path)
    if args.stride <= 0:
        args.stride = args.window_size

    results: List[Dict] = []
    targets = []
    if args.wav:
        targets.append(args.wav)
    if args.wav_dir:
        targets.extend(sorted(args.wav_dir.glob("*.wav")))

    for wav in targets:
        res = run_on_wav(
            wav,
            model,
            sr=args.sr,
            window_size=args.window_size,
            downsampling_factor=args.downsampling_factor,
            stride=args.stride,
            class_names=args.class_names,
        )
        results.append(res)

        if args.plot:
            outpath = None
            if args.plot_outdir:
                outpath = args.plot_outdir / (wav.stem + ".png")
            overlay = wav.with_suffix(".tsv") if args.overlay_tsv else None
            plot_prediction(
                wav_path=wav,
                model=model,
                sr=args.sr,
                window_size=args.window_size,
                downsampling_factor=args.downsampling_factor,
                apply_bandpass=args.apply_bandpass,
                overlay_tsv=overlay,
                outpath=outpath,
            )

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, indent=2))
        print(f"[save] {args.out_json}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
