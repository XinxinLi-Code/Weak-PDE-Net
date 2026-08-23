import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from functions import *
# Burgers Equation Params without noisy
class Params:
    # Total Parameters
    problem_dim = 1
    # Optional operators during sampling
    funcs = [Identity(), Square(), Product()]
    depth_candidates = [1, 2, 3]  # optional number of layers
    repeats_candidates = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

    # PIG
    data_spase = True
    grid_size = [400,400]
    grid_plot_size = [256,101]
    full_cov = False
    num_layers = 2
    mlp_dim = 4
    num_gaussians =800
    sigma_init = 0.1
    hidden_dim = 32
    activation = 'tanh'
    f_scale = 0.01
    input_bounds = torch.tensor([[0,10],[-8,7.9375]], dtype=torch.float32)

    # network training parameters
    learning_rate = 1e-2
    learning_rate_nas = 1e-2
    reg_weight = 1e-3
    learning_rate_tune = 1e-2
    n_epochs_tune = 200
    threshold = 0.05
    final_threshold = 0.07
    n_epochs_nas = 300
    n_epochs_adam = 400
    n_epochs_lbfgs = 100
    summary_step = 1000
    add_bias = False
    weight_function_type = 'exp_bump'


    tune_type = 'adam'

    max_order = 2
    init_uniform = 1
    Num_Weight_Functions = 200
    update_weight_epochs = 25
    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decay_rate = 0.99


    equation_type = 'Burgers'
    seed = 42
