import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from functions import *


class Params:
    # Symbolic network.
    problem_dim = 2
    funcs = [Identity(), Square(), Product()]
    depth_candidates = [1, 2, 3]
    repeats_candidates = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    lhs_type = 2
    clip_grad = True
    max_norm = 1

    # Physics-informed Gaussian network.
    data_spase = True
    grid_size = [100, 64, 64]
    time_step = 10
    grid_plot_size = [100, 64, 64]
    full_cov = False
    num_layers = 2
    mlp_dim = 2
    num_gaussians = 800
    sigma_init = 0.1
    hidden_dim = 32
    activation = "tanh"
    f_scale = 0.1

    # Network training.
    learning_rate = 1e-1
    learning_rate_nas = 1e-2
    reg_weight = 1e-3
    learning_rate_tune = 1e-2
    n_epochs_tune = 0
    threshold = 0.05
    final_threshold = 0.5
    n_epochs_nas = 800
    n_epochs_adam = 1600
    n_epochs_lbfgs = 0
    summary_step = 1000
    add_bias = False

    # Weak formulation.
    weight_function_type = "exp_bump"
    p_order = 4
    tune_type = "adam"
    max_order = 2
    init_uniform = 1
    Num_Weight_Functions = 200
    update_weight_epochs = 25
    Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decay_rate = 0.99

    equation_type = "Wave"
    seed = 42
