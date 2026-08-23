"""Configuration for the analytic 3D wave dimensionality experiment."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from functions import Identity, Product, Square


class Params:
    # Problem setup
    problem_dim = 3
    equation_type = "Wave3D"
    lhs_type = 2
    data_spase = True
    data_file = "datasets/Wave3D_Analytic.npz"
    sample_ratio = 0.006
    default_sample_counts = [5000, 10000, 20000]
    sigma_NR = 0.0
    seed = 42

    # Analytic data definition
    wave_speed = 1.0
    wave_solution = (
        "multi-component PDE-LEARN-style analytic 3D wave; "
        "see datasets/generate_wave3d_data.py"
    )
    input_bounds = [
        [0.0, 10.0],
        [-5.0, 5.0],
        [-5.0, 5.0],
        [-5.0, 5.0],
    ]
    grid_size = [50, 32, 32, 32]
    grid_plot_size = [50, 32, 32, 32]
    time_step = 5
    expected_coefficients = {
        "D_x^2 (u)": 1.0,
        "D_y^2 (u)": 1.0,
        "D_z^2 (u)": 1.0,
    }

    # Symbolic library and architecture search
    funcs = [Identity(), Square(), Product()]
    depth_candidates = [1, 2, 3]
    repeats_candidates = [[1, 2, 3], [0, 1, 2], [0, 1, 2]]
    max_order = 2
    init_uniform = 1
    add_bias = False

    # PIG frontend
    full_cov = False
    num_layers = 2
    mlp_dim = 4
    num_gaussians = 100
    sigma_init = 0.1
    hidden_dim = 32
    activation = "tanh"
    f_scale = 0.1
    norm_flag = True
    clip_grad = True
    max_norm = 1

    # Optimization
    learning_rate = 1e-2
    learning_rate_nas = 1e-2
    learning_rate_tune = 1e-2
    reg_weight = 1e-5
    n_epochs_nas = 1500
    n_epochs_adam = 1000
    n_epochs_lbfgs = 0
    n_epochs_tune = 1000
    tune_repeats = 3
    threshold = 0.05
    final_threshold = 0.25
    summary_step = 1000
    tune_type = "adam"
    decay_rate = 0.99

    # Weak-form test functions
    weight_function_type = "exp_bump"
    p_order = 4
    Num_Weight_Functions = 200
    update_weight_epochs = 25

    # Device is initialized in __init__ to avoid importing torch during data-only runs.
    Device = None

    def __init__(self):
        import torch

        self.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
