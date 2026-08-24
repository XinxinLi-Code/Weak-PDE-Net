import argparse
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CGLE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
for path in (PROJECT_ROOT, CGLE_DIR, DATASETS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cgle_params import Params
from generate_cgle_data import generate_cgle_data
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
    save_dir="sampling_idx",
    seed=42,
    target_size=None,
):
    """Return a nested sample generated from a deterministic permutation."""
    os.makedirs(save_dir, exist_ok=True)
    target_ratio = float(target_ratio)
    if not 0.0 < target_ratio <= 1.0:
        raise ValueError("sample_ratio must be in (0, 1].")
    if target_size is None:
        target_size = int(total_size * target_ratio)
    if not 0 < target_size <= total_size:
        raise ValueError(f"sample_points must be between 1 and {total_size}.")

    permutation_file = os.path.join(
        save_dir, f"sampling_permutation_seed{int(seed)}_n{int(total_size)}.npy"
    )
    if os.path.exists(permutation_file):
        permutation = np.load(permutation_file)
        is_valid = (
            permutation.shape == (total_size,)
            and np.issubdtype(permutation.dtype, np.integer)
            and np.unique(permutation).size == total_size
            and int(permutation.min()) == 0
            and int(permutation.max()) == total_size - 1
        )
        if is_valid:
            print(f"[INFO] Loading nested sampling permutation: {permutation_file}")
        else:
            print(f"[WARN] Rebuilding invalid sampling permutation: {permutation_file}")
            rng = np.random.default_rng(seed)
            permutation = rng.permutation(total_size)
            np.save(permutation_file, permutation)
    else:
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(total_size)
        np.save(permutation_file, permutation)
        print(f"[INFO] Saved nested sampling permutation: {permutation_file}")

    # Each smaller sample is a prefix of every larger sample for the same seed and grid.
    sampling_idx = permutation[:target_size].astype(int, copy=True)
    target_file = os.path.join(save_dir, f"sampling_idx_{target_ratio:.3f}.npy")
    if os.path.exists(target_file):
        existing_idx = np.load(target_file)
        if np.array_equal(existing_idx, sampling_idx):
            print(
                f"[INFO] Loading nested sampling index: {target_file} "
                f"(samples={len(sampling_idx)})"
            )
            return sampling_idx
        print(f"[WARN] Overwriting incompatible sampling index: {target_file}")

    np.save(target_file, sampling_idx)
    print(
        f"[INFO] Saved nested sampling index: {target_file} "
        f"(samples={len(sampling_idx)}, ratio={target_ratio:.3f})"
    )
    return sampling_idx


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
    for name in dir(config):
        if name.startswith("_"):
            continue
        value = getattr(config, name)
        if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isclass(value):
            continue
        params_dict[name] = safe_value(value)

    equation_type = getattr(config, "equation_type", "unknown")
    sample_ratio = getattr(config, "sample_ratio", "unknown")
    sigma_nr = getattr(config, "sigma_NR", "unknown")
    num_gaussians = getattr(config, "num_gaussians", "unknown")
    run_tag = getattr(config, "run_tag", "")
    run_suffix = f"_{run_tag}" if run_tag else ""
    use_search = getattr(config, "use_search", True)

    if num_gaussians == 0:
        filename = (
            f"params_{equation_type}_{sample_ratio}_{sigma_nr}"
            f"_no_gauss{run_suffix}.json"
        )
    elif not use_search:
        filename = (
            f"params_{equation_type}_{sample_ratio}_{sigma_nr}"
            f"_no_nas{run_suffix}.json"
        )
    else:
        filename = (
            f"params_{equation_type}_{sample_ratio}_{sigma_nr}{run_suffix}.json"
        )

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(params_dict, handle, indent=4, ensure_ascii=False)

    print(f"[INFO] Saved configuration to {filepath}")
    return filepath


def ensure_data(data_path):
    """Return the configured dataset, generating it when necessary."""
    data_path = Path(data_path)
    npz_fallback = data_path.with_suffix(".npz")
    if data_path.exists():
        return data_path
    if data_path.suffix.lower() == ".mat" and npz_fallback.exists():
        return npz_fallback

    print(f"[INFO] Generating CGLE data at: {data_path}")
    generated_path, _, _ = generate_cgle_data(output_path=data_path)
    return generated_path


def read_data_file(data_path):
    """Read a CGLE dataset stored as a MATLAB or NumPy file."""
    data_path = Path(data_path)
    if data_path.suffix.lower() == ".mat":
        try:
            import scipy.io
        except ImportError as exc:
            fallback_path = data_path.with_suffix(".npz")
            if fallback_path.exists():
                print(f"[WARN] scipy is unavailable; using {fallback_path}")
                return read_data_file(fallback_path)
            raise ImportError("scipy is required to read .mat files.") from exc
        return scipy.io.loadmat(data_path)

    if data_path.suffix.lower() == ".npz":
        with np.load(data_path, allow_pickle=True) as loaded:
            return {name: loaded[name] for name in loaded.files}

    raise ValueError("CGLE data files must end with .mat or .npz.")


def load_cgle_data(data_path):
    """Load CGLE coordinates and split the complex field into real and imaginary parts."""
    raw_data = read_data_file(data_path)
    t = np.asarray(raw_data["t"]).flatten()
    x = np.asarray(raw_data["x"]).flatten()
    if "U_exact" in raw_data:
        solution = np.asarray(raw_data["U_exact"])
    else:
        solution = np.asarray(raw_data["u"]) + 1.0j * np.asarray(raw_data["v"])

    if solution.shape == (x.size, t.size):
        solution = solution.T
    if solution.shape != (t.size, x.size):
        raise ValueError(
            f"CGLE solution must have shape {(t.size, x.size)} or "
            f"{(x.size, t.size)}, got {solution.shape}."
        )

    t_grid, x_grid = np.meshgrid(t, x, indexing="ij")
    coordinates = np.column_stack((t_grid.ravel(), x_grid.ravel()))
    values = np.column_stack((solution.real.ravel(), solution.imag.ravel()))
    full_data = np.column_stack((coordinates, values))
    return coordinates, values, full_data, solution.shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discover the one-dimensional cubic CGLE system."
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.25,
        help="Ratio of data samples to use (default: 0.25)",
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
    seed = int(getattr(config, "seed", 24))
    set_all_seeds(seed)
    config.sigma_NR = sigma_nr

    data_path = ensure_data(PROJECT_ROOT / config.data_file)
    coordinates, values, full_data, grid_shape = load_cgle_data(data_path)
    print(f"Grid shape: {grid_shape}")

    total_points = math.prod(config.grid_plot_size)
    if total_points != full_data.shape[0]:
        raise ValueError(
            f"grid_plot_size implies {total_points} points, but the loaded data "
            f"contains {full_data.shape[0]} points."
        )
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
        save_dir=str(CGLE_DIR / "sampling_idx"),
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
    input_data = torch.tensor(full_data, dtype=torch.float32)
    save_config(config, save_dir=str(CGLE_DIR / "configs"))

    os.chdir(CGLE_DIR)
    solver_input = input_train if config.data_spase else input_data
    solver = PDE_Discover(config, input_data=solver_input, inputs_test=input_data)
    pde_list = solver.Solve_problem()
    print(pde_list)
