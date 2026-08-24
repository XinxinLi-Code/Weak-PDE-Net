import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from functions import Identity, Product, Square


class Params:
    # Problem setup.
    problem_dim = 1
    equation_type = "CGLE"
    data_spase = True
    data_file = "datasets/CGLE.mat"
    sample_ratio = 0.25
    sigma_NR = 0.0
    seed = 24

    # Symbolic library and architecture search.
    funcs = [Identity(), Square(), Product()]
    depth_candidates = [1, 2, 3]
    repeats_candidates = [[2, 3, 4], [0, 1, 2], [0, 1, 2]]
    max_order = 2
    init_uniform = 1
    add_bias = False

    # Physics-informed Gaussian network.
    grid_size = [400, 400]
    grid_plot_size = [251, 256]
    input_bounds = torch.tensor(
        [[0.0, 5.0], [-10.0, 9.921875]],
        dtype=torch.float32,
    )
    full_cov = False
    num_layers = 2
    mlp_dim = 6
    num_gaussians = 1200
    sigma_init = 0.1
    hidden_dim = 32
    activation = "tanh"
    f_scale = 1

    # Network training.
    learning_rate = 1e-2
    learning_rate_nas = 1e-2
    reg_weight = 1e-6
    learning_rate_tune = 1e-2
    n_epochs_tune = 2000
    tune_print_every = 200
    threshold = 0.1
    final_threshold = [0.25, 0.25]
    n_epochs_nas = 1000
    n_epochs_adam = 1000
    n_epochs_lbfgs = 0
    summary_step = 1000
    tune_type = "adam"
    decay_rate = 0.9

    # Complete the U(1)-equivariant coupled library before each tuning repeat.
    skew_symmetic = True
    tune_repeats = 10
    skew_completion_repeats = 10

    # Weak formulation.
    weight_function_type = "exp_bump"
    p_order = 6
    Num_Weight_Functions = 200
    update_weight_epochs = 25
    weight_radius_min = 0.1
    weight_radius_max = 0.3

    Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
