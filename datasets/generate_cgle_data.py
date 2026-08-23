import argparse
import struct
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "CGLE.mat"
DEFAULT_PREVIEW = Path(__file__).resolve().parent / "cgle_data_preview.png"


def _nonlinear(a):
    return -(1.0 + 1.0j) * np.abs(a) ** 2 * a


def _etdrk4_coefficients(linear_operator, dt, contour_points=32):
    # Use the full unit-circle contour; the CGLE linear operator is complex.
    roots = np.exp(
        2.0j
        * np.pi
        * (np.arange(1, contour_points + 1, dtype=np.float64) - 0.5)
        / contour_points
    )
    lr = dt * linear_operator[:, None] + roots[None, :]

    e = np.exp(dt * linear_operator)
    e2 = np.exp(0.5 * dt * linear_operator)
    q = dt * np.mean((np.exp(0.5 * lr) - 1.0) / lr, axis=1)
    f1 = dt * np.mean(
        (-4.0 - lr + np.exp(lr) * (4.0 - 3.0 * lr + lr**2)) / lr**3,
        axis=1,
    )
    f2 = dt * np.mean(
        (2.0 + lr + np.exp(lr) * (-2.0 + lr)) / lr**3,
        axis=1,
    )
    f3 = dt * np.mean(
        (-4.0 - 3.0 * lr - lr**2 + np.exp(lr) * (4.0 - lr)) / lr**3,
        axis=1,
    )
    return e, e2, q, f1, f2, f3


def _initial_condition(x, x_min, x_max):
    domain_length = x_max - x_min
    phase = 2.0 * np.pi * (x - x_min) / domain_length
    modes = np.arange(1, 9, dtype=np.float64)
    alpha = np.array([0.75, -0.54, 1.12, 0.38, -0.86, 0.63, -0.41, 0.27])
    beta = np.array([-0.48, 0.91, 0.35, -0.72, 0.58, -0.29, 0.44, -0.33])
    coefficients = (alpha + 1.0j * beta) / np.sqrt(modes)
    spectral_field = np.sum(
        coefficients[:, None] * np.exp(1.0j * modes[:, None] * phase[None, :]),
        axis=0,
    )
    centered = spectral_field - np.mean(spectral_field)
    rms = np.sqrt(np.mean(np.abs(centered) ** 2))
    if rms == 0:
        raise FloatingPointError("Degenerate CGLE initial condition has zero RMS.")
    return 0.06 + 0.04j + 0.35 * centered / rms


def generate_cgle_data(
    output_path=DEFAULT_OUTPUT,
    preview_path=DEFAULT_PREVIEW,
    nx=256,
    nt=251,
    x_min=-10.0,
    x_max=10.0,
    t_max=5.0,
    substeps=20,
):
    output_path = Path(output_path)
    preview_path = Path(preview_path) if preview_path else None

    if nx <= 0 or nt <= 1 or substeps <= 0:
        raise ValueError("nx must be positive, nt must be greater than 1, and substeps must be positive.")

    dx = (x_max - x_min) / nx
    x = x_min + dx * np.arange(nx, dtype=np.float64)
    t = np.linspace(0.0, t_max, nt, dtype=np.float64)
    dt = (t[1] - t[0]) / substeps

    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    linear_operator = 1.0 - (1.0 + 1.0j) * wave_numbers**2
    e, e2, q, f1, f2, f3 = _etdrk4_coefficients(linear_operator, dt)

    solution = np.empty((nt, nx), dtype=np.complex128)
    a0 = _initial_condition(x, x_min, x_max)
    solution[0] = a0
    a_hat = np.fft.fft(a0)

    total_steps = (nt - 1) * substeps
    for step in range(1, total_steps + 1):
        n_v = np.fft.fft(_nonlinear(np.fft.ifft(a_hat)))
        a_stage = e2 * a_hat + q * n_v
        n_a = np.fft.fft(_nonlinear(np.fft.ifft(a_stage)))
        b_stage = e2 * a_hat + q * n_a
        n_b = np.fft.fft(_nonlinear(np.fft.ifft(b_stage)))
        c_stage = e2 * a_stage + q * (2.0 * n_b - n_v)
        n_c = np.fft.fft(_nonlinear(np.fft.ifft(c_stage)))
        a_hat = e * a_hat + f1 * n_v + 2.0 * f2 * (n_a + n_b) + f3 * n_c

        if step % substeps == 0:
            frame = np.fft.ifft(a_hat)
            if not np.all(np.isfinite(frame)):
                raise FloatingPointError(f"Non-finite values encountered at internal step {step}.")
            solution[step // substeps] = frame

    payload = {
        "t": t,
        "x": x,
        "U_exact": solution,
        "U_obs": solution,
        "u": solution.real,
        "v": solution.imag,
        "dims": np.array([nt, nx], dtype=np.int32),
        "c1": np.array([[1.0]]),
        "c3": np.array([[1.0]]),
        "equation": "A_t = A + (1+i)A_xx - (1+i)|A|^2A",
    }
    output_path = save_dataset(output_path, payload)

    if preview_path:
        try:
            save_preview(x, t, solution, preview_path)
        except ImportError as exc:
            print(f"[WARN] Could not save preview because a plotting dependency is missing: {exc}")
            preview_path = None

    return output_path, preview_path, solution


def save_dataset(output_path, payload):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".mat":
        try:
            import scipy.io
        except ImportError:
            save_mat_v5(output_path, payload)
            fallback_path = output_path.with_suffix(".npz")
            np.savez(fallback_path, **payload)
            print(f"[WARN] scipy is not available; saved MATLAB v5 data with a NumPy fallback: {fallback_path}")
            return output_path
        scipy.io.savemat(output_path, payload)
        return output_path

    if output_path.suffix.lower() == ".npz":
        np.savez(output_path, **payload)
        return output_path

    raise ValueError("output path must end with .mat or .npz")


def save_mat_v5(output_path, payload):
    mi_int8 = 1
    mi_int32 = 5
    mi_uint32 = 6
    mi_double = 9
    mi_matrix = 14
    mx_double_class = 6
    mx_complex = 0x0800

    def pad_bytes(data):
        return data + (b"\x00" * ((8 - len(data) % 8) % 8))

    def tag(data_type, byte_count):
        return struct.pack("<II", data_type, byte_count)

    def element(data_type, data):
        return tag(data_type, len(data)) + pad_bytes(data)

    def matrix_element(name, value):
        array = np.asarray(value)
        if array.dtype.kind not in {"b", "i", "u", "f", "c"}:
            return b""
        if array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.ndim == 1:
            array = array.reshape(1, -1)

        is_complex = array.dtype.kind == "c"
        flags = mx_double_class | (mx_complex if is_complex else 0)
        flags_element = element(mi_uint32, struct.pack("<II", flags, 0))
        dims_element = element(mi_int32, np.asarray(array.shape, dtype="<i4").tobytes())
        name_element = element(mi_int8, name.encode("latin1"))

        real_data = np.asarray(array.real, dtype="<f8").ravel(order="F").tobytes()
        body = flags_element + dims_element + name_element + element(mi_double, real_data)
        if is_complex:
            imag_data = np.asarray(array.imag, dtype="<f8").ravel(order="F").tobytes()
            body += element(mi_double, imag_data)
        return element(mi_matrix, body)

    description = b"MATLAB 5.0 MAT-file, Created by cgle_mains_final/generate_cgle_data.py"
    header = description[:116].ljust(116, b" ") + (b"\x00" * 8) + struct.pack("<H", 0x0100) + b"IM"

    with Path(output_path).open("wb") as handle:
        handle.write(header)
        for name, value in payload.items():
            handle.write(matrix_element(name, value))


def save_preview(x, t, solution, preview_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preview_path = Path(preview_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    extent = [x.min(), x.max(), t.min(), t.max()]
    fields = [
        ("u = Re(A)", solution.real),
        ("v = Im(A)", solution.imag),
        ("|A|", np.abs(solution)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for ax, (title, field) in zip(axes, fields):
        im = ax.imshow(field, origin="lower", aspect="auto", extent=extent, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(preview_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate data for the 1D cubic complex Ginzburg-Landau equation.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--preview", type=str, default=str(DEFAULT_PREVIEW))
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--nt", type=int, default=251)
    parser.add_argument("--x_min", type=float, default=-10.0)
    parser.add_argument("--x_max", type=float, default=10.0)
    parser.add_argument("--t_max", type=float, default=5.0)
    parser.add_argument("--substeps", type=int, default=20)
    args = parser.parse_args()

    output_path, preview_path, solution = generate_cgle_data(
        output_path=args.output,
        preview_path=args.preview,
        nx=args.nx,
        nt=args.nt,
        x_min=args.x_min,
        x_max=args.x_max,
        t_max=args.t_max,
        substeps=args.substeps,
    )

    print(f"Saved data: {output_path}")
    if preview_path:
        print(f"Saved preview: {preview_path}")
    print(f"shape={solution.shape}, |A|_min={np.abs(solution).min():.6f}, |A|_max={np.abs(solution).max():.6f}")


if __name__ == "__main__":
    main()
