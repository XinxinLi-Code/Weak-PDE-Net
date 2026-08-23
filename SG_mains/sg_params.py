import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from functions import Identity, Product, Sin


class Params:
    # Problem setup.
    problem_dim = 2
    equation_type = "SG"
    lhs_type = 2
    data_spase = True
    sample_ratio = 0.25
    sigma_NR = 0.0
    seed = 42

    # Symbolic library and architecture search.
    funcs = [Identity(), Sin(), Product()]
    depth_candidates = [1, 2, 3]
    repeats_candidates = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    max_order = 2
    init_uniform = 1
    add_bias = False

    # Physics-informed Gaussian network.
    grid_size = [70, 403, 129]
    time_step = 5
    grid_plot_size = [70, 403, 129]
    full_cov = False
    num_layers = 2
    mlp_dim = 2
    num_gaussians = 120
    sigma_init = 0.1
    hidden_dim = 32
    activation = "tanh"
    f_scale = 0.1
    clip_grad = True
    max_norm = 1

    # Network training.
    learning_rate = 5e-2
    learning_rate_nas = 5e-2
    reg_weight = 1e-5
    learning_rate_tune = 5e-2
    n_epochs_tune = 600
    threshold = 0.01
    final_threshold = 0.1
    n_epochs_nas = 300
    n_epochs_adam = 2000
    n_epochs_lbfgs = 0
    summary_step = 1000
    tune_type = "adam"
    decay_rate = 0.99

    # Weak formulation.
    weight_function_type = "exp_bump"
    p_order = 6
    Num_Weight_Functions = 200
    update_weight_epochs = 25

    Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
