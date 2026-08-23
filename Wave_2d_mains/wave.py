import argparse
import inspect
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import scipy.io
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAVE_DIR = Path(__file__).resolve().parent
PARAMS_DIR = PROJECT_ROOT / "params"
for path in (PARAMS_DIR, PROJECT_ROOT, WAVE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from wave_params import Params
from PDE_Discover import PDE_Discover


def set_all_seeds(seed=42):
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


def get_incremental_sampling_idx(
    total_size,
    target_ratio,
    save_dir="./sampling_idx",
    seed=42,
    target_size=None,
):
    """
    Incrementally sample by ratio, extending the largest smaller sample when possible.
    Files are named sampling_idx_0.125.npy, sampling_idx_0.250.npy, and so on.
    """
    os.makedirs(save_dir, exist_ok=True)
    target_ratio = float(target_ratio)
    if target_size is None:
        target_size = int(total_size * target_ratio)
    target_file = os.path.join(save_dir, f"sampling_idx_{target_ratio:.3f}.npy")

    # Load an existing sample with the requested ratio.
    if os.path.exists(target_file):
        print(f"[INFO] Loading existing sampling file: {target_file}")
        return np.load(target_file)

    # Find previously generated samples with smaller ratios.
    existing_ratios = sorted([
        float(filename.split("_")[-1].replace(".npy", ""))
        for filename in os.listdir(save_dir)
        if filename.startswith("sampling_idx_")
    ])
    smaller_ratios = [ratio for ratio in existing_ratios if ratio < target_ratio]

    rng = np.random.default_rng(seed)
    if target_ratio > 0.125 and smaller_ratios:
        last_ratio = max(smaller_ratios)
        last_file = os.path.join(save_dir, f"sampling_idx_{last_ratio:.3f}.npy")
        base_idx = np.load(last_file)
        print(f"[INFO] Extending the {last_ratio:.3f} sample to {target_ratio:.3f}")
    else:
        base_idx = np.array([], dtype=int)
        print(f"[INFO] Creating a new sample with ratio {target_ratio:.3f}")

    # Add only the samples needed to reach the requested size.
    new_sample_size = target_size - len(base_idx)
    remaining_idx = np.setdiff1d(np.arange(total_size), base_idx)
    new_samples = rng.choice(remaining_idx, size=new_sample_size, replace=False)
    updated_idx = np.concatenate([base_idx, new_samples])
    np.save(target_file, updated_idx)

    print(f"[INFO] Saved sampling indices to {target_file} (samples={len(updated_idx)})")
    return updated_idx


def save_config(config, save_dir="configs"):
    """Save the class and instance attributes of config to a JSON file."""

    def safe_value(value):
        """Convert non-serializable values to readable representations."""
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, torch.device):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [safe_value(item) for item in value]
        if hasattr(value, "__class__") and value.__class__.__module__ != "builtins":
            return value.__class__.__name__
        return value

    params_dict = {}
    for key in dir(config):
        if key.startswith("_"):
            continue
        value = getattr(config, key)
        if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isclass(value):
            continue
        params_dict[key] = safe_value(value)

    equation_type = getattr(config, "equation_type", "unknown")
    sample_ratio = getattr(config, "sample_ratio", "unknown")
    sigma_nr = getattr(config, "sigma_NR", "unknown")
    run_tag = getattr(config, "run_tag", "")
    run_suffix = f"_{run_tag}" if run_tag else ""
    filename = f"params_{equation_type}_{sample_ratio}_{sigma_nr}{run_suffix}.json"

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(params_dict, file, indent=4, ensure_ascii=False)

    print(f"[INFO] Saved {len(params_dict)} configuration parameters to {filepath}")
    return filepath


def load_wave_data(data_path=None, time_steps=100):
    """Load the Wave dataset and return coordinates, values, and the grid shape."""
    if data_path is None:
        data_path = PROJECT_ROOT / "datasets" / "Wave_Sine_Exp_2D.mat"
    mat_data = scipy.io.loadmat(data_path)

    t = mat_data["t"].flatten()[:time_steps]
    x = mat_data["x"].flatten()
    y = mat_data["y"].flatten()
    wave_values = np.real(mat_data["usol"][:time_steps, :, :])

    # The coordinate order used by PDE_Discover is (t, x, y).
    t_grid, x_grid, y_grid = np.meshgrid(t, x, y, indexing="ij")
    coordinates = np.column_stack((t_grid.ravel(), x_grid.ravel(), y_grid.ravel()))
    values = wave_values.reshape(-1, 1)
    input_data = torch.tensor(
        np.column_stack((coordinates, values)), dtype=torch.float32
    )
    return coordinates, values, input_data, wave_values.shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discover the Wave equation with configurable sampling and noise."
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.15,
        help="Ratio of data samples to use (default: 0.20)",
    )
    parser.add_argument(
        "--sigma_NR",
        type=float,
        default=0.0,
        help="Noise ratio to add to the field values (default: 0.0)",
    )
    parser.add_argument(
        "--sample_points",
        type=int,
        default=None,
        help="Number of samples to use; overrides --sample_ratio when provided.",
    )
    args = parser.parse_args()

    sample_ratio = args.sample_ratio
    sigma_nr = args.sigma_NR
    sample_points = args.sample_points

    config = Params()
    seed = int(getattr(config, "seed", 42))
    set_all_seeds(seed)
    config.sigma_NR = sigma_nr

    coordinates, values, input_data, grid_shape = load_wave_data()
    print(f"Grid shape: {grid_shape}")

    total_points = coordinates.shape[0]
    if not 0 < sample_ratio <= 1:
        parser.error("--sample_ratio must be in the interval (0, 1].")
    if sample_points is None:
        sample_points = int(total_points * sample_ratio)
    else:
        if not 0 < sample_points <= total_points:
            parser.error(f"--sample_points must be between 1 and {total_points}.")
        sample_ratio = sample_points / total_points
    config.sample_ratio = sample_ratio

    print(f"sample_ratio  = {sample_ratio}")
    print(f"sigma_NR      = {sigma_nr}")
    print(f"sample_points = {sample_points}")

    sampling_idx = get_incremental_sampling_idx(
        total_points,
        sample_ratio,
        save_dir=str(WAVE_DIR / "sampling_idx"),
        seed=seed,
        target_size=sample_points,
    )
    coordinates_train = torch.tensor(
        coordinates[sampling_idx], dtype=torch.float32, requires_grad=True
    )
    values_train = torch.tensor(values[sampling_idx], dtype=torch.float32)

    if sigma_nr > 0:
        values_rms = torch.sqrt(torch.mean(values_train ** 2))
        noise_scale = sigma_nr * values_rms
        values_train = values_train + torch.randn_like(values_train) * noise_scale

    input_train = torch.cat((coordinates_train, values_train), dim=1)
    save_config(config, save_dir=str(WAVE_DIR / "configs"))

    os.chdir(WAVE_DIR)
    solver_input = input_train if config.data_spase else input_data
    solver = PDE_Discover(config, input_data=solver_input, inputs_test=input_data)
    pde_list = solver.Solve_problem()
    print(pde_list)
