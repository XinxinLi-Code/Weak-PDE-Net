"""Run Weak-PDE-Net on the analytic 3D wave equation data."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
for path in (str(PROJECT_ROOT), str(DATASETS_DIR), str(RUN_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from generate_wave3d_data import main as generate_wave3d_data_main
from PDE_Discover import PDE_Discover
from wave3d_params import Params


def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def safe_config_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(k): safe_config_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_config_value(item) for item in value]
    if hasattr(value, "__class__") and value.__class__.__module__ != "builtins":
        return value.__class__.__name__
    return value


def save_config(config: Any, save_dir: Path) -> Path:
    params_dict = {}
    for name in dir(config):
        if name.startswith("_"):
            continue
        value = getattr(config, name)
        if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isclass(value):
            continue
        params_dict[name] = safe_config_value(value)

    equation_type = getattr(config, "equation_type", "Wave3D")
    sample_count = getattr(config, "sample_count", None)
    sample_ratio = getattr(config, "sample_ratio", "unknown")
    sigma_nr = getattr(config, "sigma_NR", "unknown")
    run_tag = str(getattr(config, "run_tag", "") or "")
    run_suffix = f"_{run_tag}" if run_tag else ""
    if sample_count is not None:
        sample_tag = f"N{int(sample_count)}"
    else:
        sample_tag = str(sample_ratio)

    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / f"params_{equation_type}_{sample_tag}_{sigma_nr}{run_suffix}.json"
    with filepath.open("w", encoding="utf-8") as handle:
        json.dump(params_dict, handle, indent=4, ensure_ascii=False)
    print(f"[config] Saved config: {filepath}")
    return filepath


def ensure_dataset(data_path: Path) -> Path:
    if data_path.suffix.lower() == ".mat" and data_path.with_suffix(".npz").exists():
        return data_path.with_suffix(".npz")
    if data_path.exists():
        return data_path

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "generate_wave3d_data.py",
            "--output_dir",
            str(data_path.parent),
            "--stem",
            data_path.stem,
            "--force",
        ]
        print(f"[data] Generating dataset: {data_path}")
        generate_wave3d_data_main()
    finally:
        sys.argv = old_argv

    if not data_path.exists():
        npz_fallback = data_path.with_suffix(".npz")
        if data_path.suffix.lower() == ".mat" and npz_fallback.exists():
            return npz_fallback
        raise FileNotFoundError(f"Dataset was not generated: {data_path}")
    return data_path


def read_data_file(data_path: Path) -> dict[str, Any]:
    suffix = data_path.suffix.lower()
    if suffix == ".npz":
        with np.load(data_path, allow_pickle=True) as loaded:
            return {name: loaded[name] for name in loaded.files}

    if suffix == ".mat":
        try:
            import scipy.io
        except ImportError as exc:
            npz_fallback = data_path.with_suffix(".npz")
            if npz_fallback.exists():
                print(f"[data] scipy is unavailable; using fallback {npz_fallback}")
                return read_data_file(npz_fallback)
            raise ImportError("scipy is required to read .mat files.") from exc
        return scipy.io.loadmat(data_path)

    raise ValueError("Wave3D data_path must end with .npz or .mat.")


def load_wave3d_data(data_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    raw = read_data_file(data_path)
    t = np.asarray(raw["t"], dtype=np.float64).flatten()
    x = np.asarray(raw["x"], dtype=np.float64).flatten()
    y = np.asarray(raw["y"], dtype=np.float64).flatten()
    z = np.asarray(raw["z"], dtype=np.float64).flatten()
    usol = np.real(np.asarray(raw["usol"]))

    expected_shape = (t.size, x.size, y.size, z.size)
    if usol.shape != expected_shape:
        raise ValueError(f"usol must have shape {expected_shape}, got {usol.shape}.")

    t_grid, x_grid, y_grid, z_grid = np.meshgrid(t, x, y, z, indexing="ij")
    coords = np.column_stack(
        (
            t_grid.ravel(),
            x_grid.ravel(),
            y_grid.ravel(),
            z_grid.ravel(),
        )
    )
    values = usol.reshape(-1, 1)
    full_data = np.hstack((coords, values))

    metadata = {
        "input_bounds": [
            [float(t.min()), float(t.max())],
            [float(x.min()), float(x.max())],
            [float(y.min()), float(y.max())],
            [float(z.min()), float(z.max())],
        ],
        "grid_size": list(expected_shape),
        "total_points": int(values.shape[0]),
        "u_rms": float(np.sqrt(np.mean(values.astype(np.float64) ** 2))),
    }
    if "metadata" in raw:
        try:
            loaded_meta = json.loads(str(np.asarray(raw["metadata"]).item()))
            metadata.update(loaded_meta)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    return coords, values, full_data, metadata


def get_incremental_sampling_idx_by_count(
    total_size: int,
    target_count: int,
    save_dir: Path,
    seed: int,
) -> np.ndarray:
    target_count = int(target_count)
    if target_count <= 0:
        raise ValueError("sample_points must be positive.")
    if target_count > total_size:
        raise ValueError(
            f"sample_points={target_count} exceeds total points={total_size}."
        )

    save_dir.mkdir(parents=True, exist_ok=True)
    target_file = save_dir / f"sampling_idx_N{target_count}.npy"
    if target_file.exists():
        cached = np.load(target_file)
        cached_ok = len(cached) == target_count and (
            len(cached) == 0 or int(np.max(cached)) < total_size
        )
        if cached_ok:
            print(f"[sampling] Loading existing index: {target_file}")
            return cached.astype(np.int64, copy=False)
        print(f"[sampling] Ignoring incompatible index cache: {target_file}")

    existing: list[tuple[int, Path]] = []
    for path in save_dir.glob("sampling_idx_N*.npy"):
        try:
            count = int(path.stem.split("_N")[-1])
        except ValueError:
            continue
        if count < target_count:
            candidate = np.load(path)
            ok = len(candidate) == count and (
                len(candidate) == 0 or int(np.max(candidate)) < total_size
            )
            if ok:
                existing.append((count, path))
    existing.sort()

    rng = np.random.default_rng(seed)
    if existing:
        base_count, base_path = existing[-1]
        base_idx = np.load(base_path).astype(np.int64, copy=False)
        print(f"[sampling] Extending N={base_count} to N={target_count}")
    else:
        base_idx = np.array([], dtype=np.int64)
        print(f"[sampling] Creating N={target_count}")

    remaining = np.setdiff1d(np.arange(total_size, dtype=np.int64), base_idx, assume_unique=False)
    add_count = target_count - len(base_idx)
    new_idx = rng.choice(remaining, size=add_count, replace=False)
    idx = np.concatenate([base_idx, new_idx]).astype(np.int64, copy=False)
    np.save(target_file, idx)
    print(f"[sampling] Saved index: {target_file}")
    return idx


def choose_sample_count(args: argparse.Namespace, config: Any, total_points: int) -> int:
    if args.sample_points is not None:
        return int(args.sample_points)
    if args.sample_ratio is not None:
        return int(total_points * float(args.sample_ratio))

    default_counts = getattr(config, "default_sample_counts", None)
    if default_counts:
        return int(default_counts[1 if len(default_counts) > 1 else 0])
    return int(total_points * float(getattr(config, "sample_ratio", 0.006)))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PDE discovery on analytic 3D wave data.")
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=None,
        help="Ratio of data samples to use.",
    )
    parser.add_argument(
        "--sigma_NR",
        type=float,
        default=None,
        help="Noise ratio to add to the field values.",
    )
    parser.add_argument(
        "--sample_points",
        type=int,
        default=None,
        help="Number of samples to use; overrides --sample_ratio when provided.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    os.chdir(RUN_DIR)

    config = Params()
    seed = int(getattr(config, "seed", 42))
    set_all_seeds(seed)

    if args.sigma_NR is not None:
        config.sigma_NR = float(args.sigma_NR)

    data_path = ensure_dataset(PROJECT_ROOT / getattr(config, "data_file"))
    coords, values, full_data, metadata = load_wave3d_data(data_path)

    total_points = int(metadata["total_points"])
    if args.sample_ratio is not None and not 0 < args.sample_ratio <= 1:
        parser.error("--sample_ratio must be in the interval (0, 1].")
    sample_count = choose_sample_count(args, config, total_points)
    if not 0 < sample_count <= total_points:
        parser.error(f"--sample_points must be between 1 and {total_points}.")
    sample_ratio = sample_count / float(total_points)
    config.sample_count = sample_count
    config.sample_ratio = sample_ratio
    config.sigma_NR = float(getattr(config, "sigma_NR", 0.0))
    config.grid_size = list(metadata["grid_size"])
    config.grid_plot_size = list(metadata["grid_size"])
    config.input_bounds = metadata["input_bounds"]
    config.run_tag = f"N{sample_count}"

    print(f"data_path    = {data_path}")
    print(f"grid_size    = {config.grid_size}")
    print(f"sample_points = {sample_count}")
    print(f"sample_ratio = {sample_ratio:.8f}")
    print(f"sigma_NR     = {config.sigma_NR}")
    print("true PDE     = D_tt(u) = D_x^2(u) + D_y^2(u) + D_z^2(u)")

    sampling_idx = get_incremental_sampling_idx_by_count(
        total_size=total_points,
        target_count=sample_count,
        save_dir=RUN_DIR / "sampling_idx",
        seed=seed,
    )
    coords_train = torch.tensor(coords[sampling_idx, :], dtype=torch.float32, requires_grad=True)
    values_train = torch.tensor(values[sampling_idx, :], dtype=torch.float32)

    if config.sigma_NR > 0:
        y_rms = torch.sqrt(torch.mean(values_train**2))
        values_train = values_train + torch.randn_like(values_train) * (config.sigma_NR * y_rms)

    input_train = torch.cat([coords_train, values_train], dim=1)
    input_data_full = torch.tensor(full_data, dtype=torch.float32)

    save_config(config, RUN_DIR / "configs")

    solver_input = input_train if config.data_spase else input_data_full
    solver = PDE_Discover(config, input_data=solver_input, inputs_test=input_data_full)
    pde_list = solver.Solve_problem()
    print(pde_list)


if __name__ == "__main__":
    main()
