import os
import numpy as np
import math
import torch
import sys
import argparse
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARAMS_DIR = os.path.join(PROJECT_ROOT, "params")
for path in (PARAMS_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
from burgers_params import Params
import random
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

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def get_incremental_sampling_idx(
    total_size, target_ratio, save_dir="./sampling_idx", seed=42, target_size=None
):
    """
    Incrementally sample by ratio, extending the largest existing sample when possible.
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

    # Find previously generated sampling files.
    existing_ratios = sorted([
        float(f.split("_")[-1].replace(".npy", ""))
        for f in os.listdir(save_dir)
        if f.startswith("sampling_idx_")
    ])

    rng = np.random.default_rng(seed)

    if target_ratio>0.125 and existing_ratios:
        # Extend the sample with the largest existing ratio.
        last_ratio = existing_ratios[-1]
        last_file = os.path.join(save_dir, f"sampling_idx_{last_ratio:.3f}.npy")
        base_idx = np.load(last_file)
        print(f"[INFO] Extending the {last_ratio:.3f} sample to {target_ratio:.3f}")
    else:
        # Start a new sample.
        base_idx = np.array([], dtype=int)
        print(f"[INFO] Creating a new sample with ratio {target_ratio:.3f}")

    # Calculate the number of additional samples.
    new_sample_size = target_size - len(base_idx)
    remaining_idx = np.setdiff1d(np.arange(total_size), base_idx)
    new_samples = rng.choice(remaining_idx, size=new_sample_size, replace=False)
    updated_idx = np.concatenate([base_idx, new_samples])
    np.save(target_file, updated_idx)

    print(f"[INFO] Saved sampling indices to {target_file} (samples={len(updated_idx)})")
    return updated_idx

import json
import inspect

def save_config(config, save_dir="configs"):
    """
    Save the class and instance attributes of config to a JSON file.
    Convert custom objects such as Identity() and Square() to readable names.
    The file name follows this format:
        params_<equation_type>_<sample_ratio>_<sigma_nr>.json
    """

    def safe_value(v):
        """Convert non-serializable values to readable representations."""
        if isinstance(v, torch.Tensor):
            return v.tolist()
        elif isinstance(v, torch.device):
            return str(v)
        elif isinstance(v, (list, tuple)):
            return [safe_value(i) for i in v]
        elif hasattr(v, "__class__") and v.__class__.__module__ != "builtins":
            # Store the class name for custom objects.
            return v.__class__.__name__
        else:
            return v

    # Collect class and instance attributes.
    params_dict = {}
    for k in dir(config):
        if k.startswith("_"):
            continue
        v = getattr(config, k)
        if inspect.isfunction(v) or inspect.ismethod(v) or inspect.isclass(v):
            continue
        params_dict[k] = safe_value(v)

    # Build the output file name.
    equation_type = getattr(config, "equation_type", "unknown")
    sample_ratio = getattr(config, "sample_ratio", "unknown")
    sigma_nr = getattr(config, "sigma_NR", "unknown")
    run_tag = getattr(config, "run_tag", "")
    run_suffix = f"_{run_tag}" if run_tag else ""
    filename = f"params_{equation_type}_{sample_ratio}_{sigma_nr}{run_suffix}.json"

    # Ensure that the output directory exists.
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    # Save the configuration as JSON.
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(params_dict, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Saved {len(params_dict)} configuration parameters to {filepath}")
    return filepath

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discover the Burgers equation with configurable sampling and noise."
    )
    parser.add_argument("--sample_ratio", type=float, default=0.20,
                        help="Ratio of data samples to use (default: 0.20)")
    parser.add_argument("--sigma_NR", type=float, default=0.0,
                        help="Noise ratio to add to y (default: 0.0)")
    parser.add_argument("--sample_points", type=int, default=None,
                        help="Number of samples to use; overrides --sample_ratio when provided.")
    args = parser.parse_args()

    sample_ratio = args.sample_ratio
    sigma_NR = args.sigma_NR
    sample_points = args.sample_points

    # Set random seeds.
    seed = 42
    set_all_seeds(seed)
    config = Params()
    seed = int(getattr(config, "seed", seed))
    set_all_seeds(seed)
    config.sigma_NR = sigma_NR

    # Load the data provided by DeepMoD.
    data = np.load(os.path.join(PROJECT_ROOT, "datasets", "burgers.npy"),
                   allow_pickle=True).item()
    print('Shape of grid:', data['x'].shape)
    X = np.transpose((data['t'].flatten(), data['x'].flatten()))
    y = np.real(data['u']).reshape((data['u'].size, 1))
    data = np.hstack((X, y))
    input_data = torch.tensor(data, dtype=torch.float32)

    total_points = math.prod(config.grid_plot_size)
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
    print(f"sigma_NR      = {sigma_NR}")
    print(f"sample_points = {sample_points}")

    save_dir="./sampling_idx"
    # Generate or load incremental sampling indices.
    sampling_idx = get_incremental_sampling_idx(
        total_points, sample_ratio, save_dir, seed, target_size=sample_points
    )
    X_train = torch.tensor(X[sampling_idx, :], dtype=torch.float32, requires_grad=True)
    y_train = torch.tensor(y[sampling_idx, :], dtype=torch.float32)

    # Add noise when sigma_NR is positive.
    if sigma_NR > 0:
        y_rms = torch.sqrt(torch.mean(y_train**2))
        sigma = sigma_NR * y_rms
        noise = torch.randn_like(y_train) * sigma
        y_train = y_train + noise

    input_train = torch.cat([X_train, y_train], dim=1)
    save_config(config)
    # Use either the sampled training data or the full dataset.
    if config.data_spase:
        Solver = PDE_Discover(config, input_data=input_train, inputs_test=input_data)
    else:
        Solver = PDE_Discover(config, input_data=input_data, inputs_test=input_data)
    PDE_list = Solver.Solve_problem()
    print(PDE_list)
