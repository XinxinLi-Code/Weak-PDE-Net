"""Generate analytic 3D wave equation data.

The generated data follow the PDE-LEARN-style protocol used by the existing
2D wave benchmark: save a complete reference data set first, then train with
fixed random index subsets.  The default solution is a non-degenerate
multi-component 3D wave so that the zero-order term u is not collinear with the
second-derivative terms.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET_STEM = "Wave3D_Analytic"


def parse_counts(value: str | None) -> list[int]:
    if value is None or not str(value).strip():
        return []
    counts = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        counts.append(int(item))
    return sorted(set(counts))


def make_coordinate_vector(start: float, stop: float, n: int) -> np.ndarray:
    if n < 2:
        raise ValueError("Each coordinate axis must have at least two points.")
    return np.linspace(float(start), float(stop), int(n), dtype=np.float64)


def default_components(c: float) -> list[dict[str, float | str]]:
    """Return exact 3D wave components with several spatial frequencies."""
    return [
        {"kind": "sin", "amplitude": 1.0, "omega": c, "kx": 1.0, "ky": 0.0, "kz": 0.0, "phase": 0.0, "scale": 1.0},
        {"kind": "sin", "amplitude": 0.8, "omega": c, "kx": 0.0, "ky": 1.0, "kz": 0.0, "phase": 0.3, "scale": 1.0},
        {"kind": "sin", "amplitude": 0.7, "omega": c, "kx": 0.0, "ky": 0.0, "kz": 1.0, "phase": -0.2, "scale": 1.0},
        {
            "kind": "sin",
            "amplitude": 0.5,
            "omega": c * math.sqrt(2.0),
            "kx": 1.0,
            "ky": 1.0,
            "kz": 0.0,
            "phase": 0.1,
            "scale": 1.0,
        },
        {
            "kind": "sin",
            "amplitude": 0.4,
            "omega": c * math.sqrt(2.0),
            "kx": 1.0,
            "ky": 0.0,
            "kz": 1.0,
            "phase": -0.4,
            "scale": 1.0,
        },
        {
            "kind": "sin",
            "amplitude": 0.35,
            "omega": c * math.sqrt(3.0),
            "kx": 1.0,
            "ky": 1.0,
            "kz": 1.0,
            "phase": 0.2,
            "scale": 1.0,
        },
        {
            "kind": "exp",
            "amplitude": 0.25,
            "omega": c * math.sqrt(2.0),
            "kx": 1.0,
            "ky": 1.0,
            "kz": 0.0,
            "phase": 0.0,
            "scale": 0.05,
        },
    ]


def component_label(component: dict[str, float | str]) -> str:
    kind = str(component["kind"])
    amplitude = float(component["amplitude"])
    omega = float(component["omega"])
    kx = float(component["kx"])
    ky = float(component["ky"])
    kz = float(component["kz"])
    phase = float(component["phase"])
    scale = float(component["scale"])
    phase_expr = f"{omega:.12g}*t-{kx:.12g}*x-{ky:.12g}*y-{kz:.12g}*z"
    if phase:
        phase_expr += f"{phase:+.12g}"
    if kind == "exp":
        return f"{amplitude:.12g}*exp({scale:.12g}*({phase_expr}))"
    return f"{amplitude:.12g}*sin({phase_expr})"


def generate_solution(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    c: float,
    dtype: np.dtype,
) -> tuple[np.ndarray, dict[str, object]]:
    components = default_components(float(c))
    x_grid = x[:, None, None]
    y_grid = y[None, :, None]
    z_grid = z[None, None, :]

    usol = np.empty((len(t), len(x), len(y), len(z)), dtype=dtype)
    uxx_all = np.empty_like(usol, dtype=np.float64)
    uyy_all = np.empty_like(usol, dtype=np.float64)
    uzz_all = np.empty_like(usol, dtype=np.float64)

    max_abs_residual = 0.0
    for i, t_value in enumerate(t):
        u = np.zeros((len(x), len(y), len(z)), dtype=np.float64)
        u_tt = np.zeros_like(u)
        u_xx = np.zeros_like(u)
        u_yy = np.zeros_like(u)
        u_zz = np.zeros_like(u)

        for component in components:
            kind = str(component["kind"])
            amplitude = float(component["amplitude"])
            omega = float(component["omega"])
            kx = float(component["kx"])
            ky = float(component["ky"])
            kz = float(component["kz"])
            phase = float(component["phase"])
            scale = float(component["scale"])
            theta = omega * float(t_value) - kx * x_grid - ky * y_grid - kz * z_grid + phase

            if kind == "sin":
                value = amplitude * np.sin(theta)
                u += value
                u_tt += -(omega * omega) * value
                u_xx += -(kx * kx) * value
                u_yy += -(ky * ky) * value
                u_zz += -(kz * kz) * value
            elif kind == "exp":
                value = amplitude * np.exp(scale * theta)
                u += value
                u_tt += (scale * omega) ** 2 * value
                u_xx += (scale * kx) ** 2 * value
                u_yy += (scale * ky) ** 2 * value
                u_zz += (scale * kz) ** 2 * value
            else:
                raise ValueError(f"Unsupported component kind: {kind}")

        usol[i] = u.astype(dtype, copy=False)
        uxx_all[i] = u_xx
        uyy_all[i] = u_yy
        uzz_all[i] = u_zz
        residual = u_tt - (float(c) ** 2) * (u_xx + u_yy + u_zz)
        max_abs_residual = max(max_abs_residual, float(np.max(np.abs(residual))))

    feature_matrix = np.column_stack(
        (
            usol.astype(np.float64, copy=False).reshape(-1),
            uxx_all.reshape(-1),
            uyy_all.reshape(-1),
            uzz_all.reshape(-1),
        )
    )
    singular_values = np.linalg.svd(feature_matrix, compute_uv=False)
    corr = np.corrcoef(feature_matrix, rowvar=False)

    component_table = [
        [
            1.0 if str(component["kind"]) == "sin" else 2.0,
            float(component["amplitude"]),
            float(component["omega"]),
            float(component["kx"]),
            float(component["ky"]),
            float(component["kz"]),
            float(component["phase"]),
            float(component["scale"]),
        ]
        for component in components
    ]
    stats: dict[str, object] = {
        "components": components,
        "component_labels": [component_label(component) for component in components],
        "component_table_columns": ["kind_code", "amplitude", "omega", "kx", "ky", "kz", "phase", "scale"],
        "component_table": component_table,
        "residual_max_abs": max_abs_residual,
        "u_min": float(np.min(usol)),
        "u_max": float(np.max(usol)),
        "u_rms": float(np.sqrt(np.mean(usol.astype(np.float64) ** 2))),
        "feature_columns": ["u", "u_xx", "u_yy", "u_zz"],
        "feature_rank": int(np.linalg.matrix_rank(feature_matrix)),
        "feature_singular_values": [float(value) for value in singular_values],
        "feature_correlation": corr.tolist(),
    }
    return usol, stats


def save_npz(
    output_path: Path,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    usol: np.ndarray,
    metadata: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        t=t,
        x=x,
        y=y,
        z=z,
        usol=usol,
        metadata=json.dumps(metadata, indent=2),
    )


def save_mat(
    output_path: Path,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    usol: np.ndarray,
    metadata: dict,
    require: bool = False,
) -> bool:
    try:
        import scipy.io
    except ImportError as exc:
        message = "[data] scipy is not available; skipped MATLAB .mat copy."
        if require:
            raise RuntimeError("scipy is required to save the MATLAB .mat copy.") from exc
        print(message)
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(
        output_path,
        {
            "t": t.reshape(-1, 1),
            "x": x.reshape(-1, 1),
            "y": y.reshape(-1, 1),
            "z": z.reshape(-1, 1),
            "usol": usol,
            "c": np.array([[metadata["wave_speed"]]], dtype=np.float64),
            "component_table": np.array(metadata["component_table"], dtype=np.float64),
            "component_labels": np.array(metadata["component_labels"], dtype=object),
            "equation": metadata["equation"],
            "solution": metadata["solution"],
            "layout": metadata["layout"],
        },
        do_compression=True,
    )
    return True


def get_incremental_sample_indices(
    total_size: int,
    target_count: int,
    save_dir: Path,
    seed: int,
) -> np.ndarray:
    target_count = int(target_count)
    if target_count <= 0:
        raise ValueError("Sample counts must be positive.")
    if target_count > total_size:
        raise ValueError(f"Sample count {target_count} exceeds total size {total_size}.")

    save_dir.mkdir(parents=True, exist_ok=True)
    target_file = save_dir / f"sampling_idx_N{target_count}.npy"
    if target_file.exists():
        cached = np.load(target_file)
        if len(cached) == target_count and (len(cached) == 0 or int(np.max(cached)) < total_size):
            return cached

    existing: list[tuple[int, Path]] = []
    for path in save_dir.glob("sampling_idx_N*.npy"):
        try:
            count = int(path.stem.split("_N")[-1])
        except ValueError:
            continue
        if count < target_count:
            idx = np.load(path)
            if len(idx) == count and (len(idx) == 0 or int(np.max(idx)) < total_size):
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
    return idx


def write_summary(path: Path, metadata: dict, sample_counts: Iterable[int]) -> None:
    summary = dict(metadata)
    summary["sample_counts"] = list(sample_counts)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate analytic 3D wave equation data.")
    parser.add_argument("--nt", type=int, default=50)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--t_min", type=float, default=0.0)
    parser.add_argument("--t_max", type=float, default=10.0)
    parser.add_argument("--x_min", type=float, default=-5.0)
    parser.add_argument("--x_max", type=float, default=5.0)
    parser.add_argument("--y_min", type=float, default=-5.0)
    parser.add_argument("--y_max", type=float, default=5.0)
    parser.add_argument("--z_min", type=float, default=-5.0)
    parser.add_argument("--z_max", type=float, default=5.0)
    parser.add_argument("--c", type=float, default=1.0, help="Wave speed.")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--stem", type=str, default=DEFAULT_DATASET_STEM)
    parser.add_argument("--skip_mat", action="store_true", help="Do not write the optional .mat copy.")
    parser.add_argument(
        "--require_mat",
        action="store_true",
        help="Fail if scipy is unavailable and the .mat copy cannot be written.",
    )
    parser.add_argument("--sample_counts", type=str, default="5000,10000,20000")
    parser.add_argument("--sampling_dir", type=Path, default=SCRIPT_DIR / "sampling_idx")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset files.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    npz_path = output_dir / f"{args.stem}.npz"
    mat_path = output_dir / f"{args.stem}.mat"
    if (npz_path.exists() or mat_path.exists()) and not args.force:
        raise FileExistsError(
            f"{npz_path} or {mat_path} already exists. Use --force to regenerate."
        )

    dtype = np.dtype(args.dtype)
    t = make_coordinate_vector(args.t_min, args.t_max, args.nt)
    x = make_coordinate_vector(args.x_min, args.x_max, args.nx)
    y = make_coordinate_vector(args.y_min, args.y_max, args.ny)
    z = make_coordinate_vector(args.z_min, args.z_max, args.nz)
    usol, stats = generate_solution(t, x, y, z, float(args.c), dtype)
    metadata = {
        "equation": "u_tt = c^2 * (u_xx + u_yy + u_zz)",
        "solution": " + ".join(stats["component_labels"]),
        "wave_speed": float(args.c),
        "data_generation": "PDE-LEARN-style full reference grid with fixed random index subsets",
        "grid_size": [int(args.nt), int(args.nx), int(args.ny), int(args.nz)],
        "input_bounds": [
            [float(args.t_min), float(args.t_max)],
            [float(args.x_min), float(args.x_max)],
            [float(args.y_min), float(args.y_max)],
            [float(args.z_min), float(args.z_max)],
        ],
        "layout": "usol[t_index, x_index, y_index, z_index]",
        "flatten_order": "np.meshgrid(t, x, y, z, indexing='ij') followed by C-order flatten",
        "dtype": str(dtype),
        "total_points": int(usol.size),
        **stats,
    }

    save_npz(npz_path, t, x, y, z, usol, metadata)
    mat_saved = False
    if not args.skip_mat:
        mat_saved = save_mat(mat_path, t, x, y, z, usol, metadata, require=args.require_mat)

    counts = parse_counts(args.sample_counts)
    for count in counts:
        get_incremental_sample_indices(usol.size, count, args.sampling_dir, args.seed)

    summary_path = SCRIPT_DIR / "wave3d_dataset_summary.json"
    write_summary(summary_path, metadata, counts)

    print(f"[data] saved {npz_path}")
    if mat_saved:
        print(f"[data] saved {mat_path}")
    print(f"[data] shape usol={usol.shape}, total_points={usol.size}")
    print(f"[data] residual_max_abs={stats['residual_max_abs']:.3e}")
    print(f"[data] summary {summary_path}")


if __name__ == "__main__":
    main()
