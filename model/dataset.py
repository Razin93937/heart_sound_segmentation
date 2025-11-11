from pathlib import Path
from typing import List, Tuple
import librosa
import numpy as np
from scipy.signal import butter, lfilter
import tensorflow as tf


def round_to_sample_grid(x: np.ndarray, sr: int) -> np.ndarray:
    """Round times to the nearest sample index at a given sampling rate.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (N, 3): [start_time, end_time, label].
    sr : int
        Sampling rate.

    Returns
    -------
    np.ndarray
        Same shape as input, with start/end times snapped to exact sample steps.
    """
    if x.ndim != 2 or x.shape[1] < 3:
        return x
    base = 1.0 / sr
    out = x.copy()
    # Round start and end columns to sample grid
    out[:, 0] = np.around(base * np.around(out[:, 0] / base), 10)
    out[:, 1] = np.around(base * np.around(out[:, 1] / base), 10)
    return out


def load_intervals_tsv(tsv_path: Path, sr: int) -> np.ndarray:
    """Load TSV annotation file as [start, end, label].

    Handles header rows and snaps times to sample grid.
    """
    if not tsv_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {tsv_path}")

    try:
        # Try with delimiter=\t first (TSV). Some files may not include headers.
        d = np.genfromtxt(str(tsv_path), dtype=float, delimiter="\t")
    except Exception:
        d = np.genfromtxt(str(tsv_path), dtype=float)

    if d.ndim == 1:
        d = np.expand_dims(d, 0)

    # Drop rows that contain NaNs (e.g., header or malformed rows)
    d = d[~np.isnan(d).any(axis=1)]

    # Ensure at least 3 columns (start, end, label)
    if d.shape[1] < 3:
        raise ValueError(f"Annotation file has <3 columns: {tsv_path}")

    d = d[:, :3]
    d = round_to_sample_grid(d, sr=sr)
    return d


def build_frame_labels(y_len: int, sr: int, intervals: np.ndarray) -> np.ndarray:
    """Build a per-sample integer label track matching the audio length.

    Parameters
    ----------
    y_len : int
        Length of the waveform in samples.
    sr : int
        Sampling rate.
    intervals : np.ndarray
        Array of [start_time, end_time, label].

    Returns
    -------
    np.ndarray
        Integer array of shape (y_len,) with 0 for unlabeled, otherwise the label int.
    """
    labels = np.zeros(y_len, dtype=np.int32)
    for start_t, end_t, lab in intervals:
        s = int(start_t * sr)
        e = int(end_t * sr)
        s = max(0, min(s, y_len))
        e = max(0, min(e, y_len))
        if e > s:
            labels[s:e] = int(lab)
    return labels


def map_labels_to_three_classes(labels: np.ndarray) -> np.ndarray:
    """Map dataset labels to 3 classes, following the original notebook logic.

    Original behavior:
    - Labels come in as {0 (unlabeled), 1, 2, 3, 4?}
    - Convert to {0,1,2,3?} -> subtract 1 => {-1,0,1,2,3?}
    - Then map 3 -> 1 (merging a 4th class into class index 1)
    - Unlabeled windows are filtered out before one-hot (see preprocessing).

    This function assumes input labels are in {1,2,3,4} for labeled frames and 0 for unlabeled.
    Returns an array with values in {0,1,2} for labeled frames; unlabeled should be handled upstream.
    """
    x = labels.astype(np.int32) - 1  # labeled {1..} -> {0..}
    x[x == 3] = 1  # merge 3 -> 1 to get 3 classes total (0,1,2)
    return x


def standardize_signal(x: np.ndarray) -> np.ndarray:
    """Zero-mean unit-variance normalization per window (robust training)."""
    mu = np.mean(x)
    sigma = np.std(x) + 1e-8
    return (x - mu) / sigma


def bandpass(
    x: np.ndarray, sr: int, low: float = 25.0, high: float = 400.0, order: int = 4
) -> np.ndarray:
    """Optional band-pass filter for heart sounds; used only if SciPy is available.

    Defaults are conservative and intended for visualization. Disabled by default.
    """
    nyq = 0.5 * sr
    low_n = low / nyq
    high_n = high / nyq
    b, a = butter(order, [low_n, high_n], btype="band")
    return lfilter(b, a, x)


def list_record_ids(data_dir: Path) -> List[str]:
    """Return basenames (without extension) for all WAV files in data_dir."""
    ids = []
    for wav in sorted(Path(data_dir).glob("*.wav")):
        ids.append(wav.stem)
    if not ids:
        raise FileNotFoundError(f"No .wav files found in {data_dir}")
    return ids[:100]


def preprocess_record(
    record_id: str,
    data_dir: Path,
    sr: int = 2000,
    window_size: int = 4000,
    downsampling_factor: int = 16,
    require_fully_labeled: bool = True,
    apply_bandpass: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice one record into windows and one-hot labels.

    Parameters
    ----------
    record_id : str
        Base name (without extension) for the WAV/TSV pair.
    data_dir : Path
        Directory containing `record_id.wav` and `record_id.tsv`.
    sr : int
        Target sampling rate.
    window_size : int
        Number of samples per training window.
    downsampling_factor : int
        Factor to reduce label resolution (model outputs one label per `downsampling_factor` samples).
    require_fully_labeled : bool
        If True, drop windows that contain any unlabeled (0) frames.
    apply_bandpass : bool
        If True and SciPy is available, apply a mild bandpass (for visualization only).

    Returns
    -------
    X_windows : np.ndarray, shape (n_windows, window_size)
    Y_onehot : np.ndarray, shape (n_windows, window_size/downsampling_factor, 3)
    """
    wav_path = data_dir / f"{record_id}.wav"
    tsv_path = data_dir / f"{record_id}.tsv"

    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    intervals = load_intervals_tsv(tsv_path, sr=sr)
    frame_labels = build_frame_labels(len(y), sr=sr, intervals=intervals)

    # Windowing
    start = 0
    end = len(y)
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    while start + window_size < end:
        segment = y[start : start + window_size]
        labels_seg = frame_labels[start : start + window_size]

        if require_fully_labeled and np.any(labels_seg == 0):
            # Skip windows with any unlabeled frames (mirrors original notebook logic)
            start += window_size
            continue

        # Optional filter for visualization; NOT applied by default
        if apply_bandpass:
            segment = bandpass(segment, sr)

        # Normalize for stability
        segment = standardize_signal(segment)

        # Downsample labels to match model output resolution
        ds_labels = labels_seg[::downsampling_factor]
        ds_labels = map_labels_to_three_classes(ds_labels)

        # One-hot encode 3 classes
        onehot = tf.one_hot(ds_labels, depth=3)
        onehot = onehot.numpy().astype(np.float32)

        X_list.append(segment.astype(np.float32))
        Y_list.append(onehot)

        start += window_size  # non-overlapping windows; change to stride for overlap

    if not X_list:
        return np.empty((0, window_size), dtype=np.float32), np.empty(
            (0, window_size // downsampling_factor, 3), dtype=np.float32
        )

    X_windows = np.stack(X_list, axis=0)
    Y_onehot = np.stack(Y_list, axis=0)

    # Sanity check shapes
    assert X_windows.shape[1] == window_size
    assert Y_onehot.shape[1] == window_size // downsampling_factor
    assert Y_onehot.shape[2] == 3

    return X_windows, Y_onehot


def build_dataset(
    data_dir: Path,
    sr: int = 2000,
    window_size: int = 4000,
    downsampling_factor: int = 16,
    require_fully_labeled: bool = True,
    apply_bandpass: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct dataset arrays X (N, W) and Y (N, W/downsampling_factor, 3)."""
    record_ids = list_record_ids(data_dir)

    X_all: List[np.ndarray] = []
    Y_all: List[np.ndarray] = []

    for i, rid in enumerate(record_ids, 1):
        Xi, Yi = preprocess_record(
            rid,
            data_dir=data_dir,
            sr=sr,
            window_size=window_size,
            downsampling_factor=downsampling_factor,
            require_fully_labeled=require_fully_labeled,
            apply_bandpass=apply_bandpass,
        )
        if Xi.size and Yi.size:
            X_all.append(Xi)
            Y_all.append(Yi)
        if i % 100 == 0:
            print(
                f"Processed {i}/{len(record_ids)} records ({100 * i / len(record_ids):.1f}%)"
            )

    if not X_all:
        raise RuntimeError(
            "No usable windows found. Check labels or turn off `require_fully_labeled`."
        )

    X = np.concatenate(X_all, axis=0)
    Y = np.concatenate(Y_all, axis=0)
    print(f"Dataset windows: X={X.shape}, Y={Y.shape}")
    return X, Y
