from torch.utils.tensorboard import SummaryWriter
import copy
from Weight_Function import Weight_Function, Build_From_Other
from train import Model
import numpy as np
import torch
import random
from train import Training
from pretty_print import print_model
import sympy as sp
from utils import parse_derivatives, replace_str_with_fun, decompose_expression, simplify_trig_coefficients, skew_symmetric_completion, extract_coeff, is_zero_coeff, replace_coeff, complete_symmetric_term_pair
import Derivative
import Regression
from pig_network import *
from Tune_Net import *
import Poly_Weight_Function as poly_w
import os
import functions as functions
from functions import *


def config_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        return default
    return bool(value)


def find_close_coeff_groups(coff_list, threshold=1e-3):
    """
    Group entries in coff_list whose values differ by less than threshold.
    For example, coff_list = [0.5, 0.501, 0.8, 0.7995]
    returns [[0, 1], [2, 3]].

    Arguments:
        coff_list: A list or NumPy array.
        threshold: The maximum difference between nearby values.
    Returns:
        groups: A list of index lists, with each inner list identifying a group
            of nearby entries.
    """
    import numpy as np
    
    coff_list = np.array(coff_list)
    M = len(coff_list)
    used = np.zeros(M, dtype=bool)
    groups = []
    
    for i in range(M):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i+1, M):
            if not used[j] and abs(coff_list[i] - coff_list[j]) < threshold:
                group.append(j)
                used[j] = True
        if len(group) > 1:  # A group must contain at least two entries.
            groups.append(group)
    
    return groups


class PDE_Discover():
    def __init__(self, config, input_data, inputs_test=None, other_data=None):
        """
        config: Default configuration parameters.
        input_data: Data arranged as [x, t, u, v, w].
        inputs_test: Ground-truth data on a dense grid.
        We assume that u, v, and w share the same grid.

        """
        self.data_spase = config.data_spase
        self.grid_size = config.grid_size
        self.Train_Weight_Functions_Lists = []
        self.Targeted_Weight_Functions_Lists = []
        self.Random_Weight_Functions_Lists = []
        self.config = config
        self.grid_plot_size = config.grid_plot_size
        self.problem_dim = config.problem_dim
        self.decay_rate = config.decay_rate
        self.weight_function_type = config.weight_function_type
        self.weight_radius_min = getattr(config, "weight_radius_min", None)
        self.weight_radius_max = getattr(config, "weight_radius_max", None)
        self.weight_center_strategy = getattr(config, "weight_center_strategy", "uniform")
        self.equation_type = config.equation_type
        self.sample_ratio = config.sample_ratio
        self.sigma_NR = config.sigma_NR
        self.run_tag = getattr(config, "run_tag", "")
        self.run_suffix = f"_{self.run_tag}" if self.run_tag else ""
        self.Device = config.Device
        self.other_data = other_data
        self.bounds_flag = config_bool(getattr(config, "bounds_flag", None), False)
        try:
            self.enforce_galilean_invariance = config.enforce_galilean_invariance
            self.init_to_ones= config.init_to_ones
        except:
            self.enforce_galilean_invariance = False
            self.init_to_ones = False
        try:
            self.weights_functions_path = config.weights_functions_path
        except:
            self.weights_functions_path = None
        if other_data is not None:
            self.other_data = self.other_data.to(self.Device)
        try:
            self.use_search = config.use_search
            self.depth = config.depth
            self.repeats_list = config.repeats_list
        except:
            self.use_search = True
        try:
            self.kill_small_items = config.kill_small_items
        except:
            self.kill_small_items = False
        # print("0:",self.kill_small_items)
        try:
            self.joint_tune = config.joint_tune
        except:
            self.joint_tune = False
        try:
            self.lhs_type = config.lhs_type
        except:
            self.lhs_type = 1
        try:
            self.clip_grad = config.clip_grad
        except:
            self.clip_grad = False
        try:
            self.skew_symmetic = config.skew_symmetic
        except:
            self.skew_symmetic = False
        try:
            self.max_norm = config.max_norm
            self.norm_flag = config.norm_flag
        except:
            self.max_norm = 1
            self.norm_flag = False
        try:
            self.res_net = config.res_net
        except:
            self.res_net = False
        try:
            self.grad_all = config.grad_all
        except:
            self.grad_all = False
        try:
            self.int_type = config.int_type
        except:
            self.int_type = 'Riemann'
        try:
            self.tune_repeats = config.tune_repeats
        except:
            self.tune_repeats = 1
        self.tune_print_every = getattr(config, "tune_print_every", 200)
        try:
            self.skew_completion_repeats = int(config.skew_completion_repeats)
        except:
            self.skew_completion_repeats = 1 if self.skew_symmetic else 0
        try:
            self.fourier_encoding=config.fourier_encoding
            self.num_freqs=config.num_freqs
        except:
            self.fourier_encoding=False
            self.num_freqs=10
        if self.problem_dim!=1:
            self.time_step = config.time_step
        else:
            self.time_step = 0
        print("self.config.num_gaussians:",self.config.num_gaussians)
        if self.config.num_gaussians == 0:
            self.results_folder_name = f"results_{self.equation_type}_{self.sample_ratio}_{self.sigma_NR}_no_gauss{self.run_suffix}"
        elif not self.use_search:
            self.results_folder_name = f"results_{self.equation_type}_{self.sample_ratio}_{self.sigma_NR}_no_nas{self.run_suffix}"
        else:
            self.results_folder_name = f"results_{self.equation_type}_{self.sample_ratio}_{self.sigma_NR}{self.run_suffix}"
        os.makedirs(self.results_folder_name, exist_ok=True)
        print(f"Results will be saved to: {self.results_folder_name}")
        subfolders = ["search", "train", "tune"]

        self.subfolder_paths = {}

        for folder in subfolders:
            path = os.path.join(self.results_folder_name, folder)
            os.makedirs(path, exist_ok=True)
            self.subfolder_paths[folder] = path
            print(f"Created subdirectory: {path}")
        self.num_equation = input_data.shape[-1] - (self.problem_dim + 1)
        print("self.num_equation:",self.num_equation)
        if self.other_data is not None:
            self.num_var = input_data.shape[-1] - (self.problem_dim + 1) + self.other_data.shape[-1]
        else:
            self.num_var = self.num_equation
        if self.num_var == 1:
            self.var_names = ['u']
        elif self.num_var == 2:
            self.var_names = ['u', 'v']
        elif self.num_var == 3:
            self.var_names = ['w', 'u', 'v']
        else:
            print("Only problems with at most three datasets are supported.")
        self.input_data = input_data
        
        # self.Device = config.Device
        # self.other_data = other_data.to(self.Device)
        self.inputs_test = inputs_test
        # if inputs_test is None:
        bound_list = []
        center_list = []
        self.input_train = input_data[:,0:self.problem_dim+1]
        configured_bounds = getattr(config, "input_bounds", None)
        if configured_bounds is not None:
            if isinstance(configured_bounds, torch.Tensor):
                self.Input_Bounds = configured_bounds.detach().cpu().float()
            else:
                self.Input_Bounds = torch.tensor(configured_bounds, dtype=torch.float32)
            if self.Input_Bounds.shape != (self.problem_dim + 1, 2):
                raise ValueError(
                    f"input_bounds must have shape {(self.problem_dim + 1, 2)}, "
                    f"got {tuple(self.Input_Bounds.shape)}"
                )
            for dim in range(self.problem_dim + 1):
                center_list.append((self.Input_Bounds[dim, 0] + self.Input_Bounds[dim, 1]) / 2.0)
        else:
            for dim in range(self.problem_dim + 1):
                max_value = max(self.input_train[:, dim])
                min_value = min(self.input_train[:, dim])
                center_dim = (max_value + min_value) / 2.0
                bound_dim = np.array([min_value.item(), max_value.item()])
                bound_list.append(bound_dim)
                center_list.append(center_dim)
            bound_array = np.array(bound_list)
            self.Input_Bounds = torch.tensor(bound_array, dtype=torch.float32)
        print(self.Input_Bounds)
        self.input_test = self.get_full_grid_coords()
        print(self.input_test.shape)
        self.input_test = torch.tensor(self.input_test, dtype=torch.float32)
        self.u_test = None
        
        
        if self.data_spase:
            self.grid_data = self.input_test.clone().detach()
        else:
            self.grid_data = self.input_train.clone().detach()
        self.grid_data = self.grid_data.to(self.Device)
        self.input_test = self.input_test.to(self.Device)
        

        if isinstance(center_list[0], torch.Tensor):
            self.Input_Center = torch.stack([c.detach() for c in center_list]).float()
        else:
            self.Input_Center = torch.tensor(center_list, dtype=torch.float32)

        self.update_weight_epochs = config.update_weight_epochs
        self.funcs = config.funcs
        self.n_double = functions.count_double(self.funcs) 
        self.depth_candidates = config.depth_candidates
        self.repeats_candidates = config.repeats_candidates
        self.learning_rate = config.learning_rate
        self.learning_rate_nas = config.learning_rate_nas
        self.reg_weight = config.reg_weight
        self.threshold = config.threshold
        self.n_epochs_nas = config.n_epochs_nas
        self.n_epochs_adam = config.n_epochs_adam
        self.n_epochs_lbfgs = config.n_epochs_lbfgs
        self.summary_step = config.summary_step
        self.add_bias = config.add_bias
        self.max_order = config.max_order
        self.init_uniform = config.init_uniform
        self.Num_Weight_Functions = config.Num_Weight_Functions
        self.input_data.to(self.Device)
        if self.data_spase:
            self.full_cov = config.full_cov
            self.num_layers = config.num_layers
            self.mlp_dim = config.mlp_dim
            self.num_gaussians = config.num_gaussians
            self.sigma_init = config.sigma_init
            self.hidden_dim = config.hidden_dim
            self.in_dim = self.problem_dim+1
            self.out_dim = self.num_equation
            self.activation = config.activation
            self.f_scale = config.f_scale
            self.Pig_Net = Base(self.full_cov, self.num_layers, self.mlp_dim, self.num_gaussians, self.sigma_init, self.hidden_dim, self.in_dim, self.out_dim, self.activation, self.fourier_encoding, self.num_freqs, self.res_net)
            self.Pig_Net = self.Pig_Net.to(self.Device)
            if self.full_cov and self.problem_dim == 1:
                self.Pig_Net.additional_params()
        else:
            self.f_scale = 1
            self.Pig_Net = None
            self.grid_size = []
            self.out_dim = self.num_equation
        self.model_list = []
        print("self.out_dim:",self.out_dim)
        if self.use_search:
            for i in range(self.out_dim):
                model = Model(problem_dim=self.problem_dim, x_dim=self.num_var, depth=None, repeats_list=[],
                            funcs=self.funcs, max_order=self.max_order, device=self.Device, grad_all=False, 
                            initial_weights_sym=None, initial_weights_int=None, init_uniform=self.init_uniform,
                            add_bias=self.add_bias, depth_candidates=self.depth_candidates,
                            repeats_candidates=self.repeats_candidates, use_search=self.use_search,
                            enforce_galilean_invariance=self.enforce_galilean_invariance)
                model.to(self.Device)
                self.model_list.append(model)
        else:
            for i in range(self.out_dim):
                model = Model(problem_dim=self.problem_dim, x_dim=self.num_var, depth=self.depth, repeats_list=self.repeats_list,
                            funcs=self.funcs, max_order=self.max_order, device=self.Device, grad_all=False, 
                            initial_weights_sym=None, initial_weights_int=None, init_uniform=self.init_uniform,
                            add_bias=self.add_bias, depth_candidates=[],
                            repeats_candidates=[], use_search=self.use_search,
                            enforce_galilean_invariance=self.enforce_galilean_invariance)
                model.to(self.Device)
                self.model_list.append(model)

        Min_Side_Length = self.Input_Bounds[0, 1] - self.Input_Bounds[0, 0]
        for i in range(1, self.problem_dim + 1):
            if self.Input_Bounds[i, 1] - self.Input_Bounds[i, 0] < Min_Side_Length:
                Min_Side_Length = self.Input_Bounds[i, 1] - self.Input_Bounds[i, 0]

        V = 1
        for j in range(self.problem_dim + 1):
            V *= (self.Input_Bounds[j, 1] - self.Input_Bounds[j, 0])
        self.V = V / float(self.grid_data.shape[0])

        self.radius = .5 * Min_Side_Length
        if self.weight_function_type == 'exp_bump':
            self.master_weight_function = Weight_Function(
                X_0=self.Input_Center.to(self.Device),
                r=self.radius.item(),
                Coords=self.grid_data,
                V=self.V)
        elif self.weight_function_type == 'poly_bump':
            self.master_weight_function = poly_w.Poly_Weight_Function(
                X_0=self.Input_Center.to(self.Device),
                r=self.radius.item(),
                p=self.config.p_order,
                Coords=self.grid_data,
                V=self.V)
        elif self.weight_function_type == 'mix_bump':
            self.master_weight_function = poly_w.Poly_Weight_Function(
                X_0=self.Input_Center.to(self.Device),
                r=self.radius.item(),
                p=self.config.p_order,
                Coords=self.grid_data,
                V=self.V)
            self.master_weight_function_1 = Weight_Function(
                X_0=self.Input_Center.to(self.Device),
                r=self.radius.item(),
                Coords=self.grid_data,
                V=self.V)

    def threshold_for_equation(self, equation_idx):
        threshold = getattr(self.config, "final_threshold_final", None)
        if threshold is None:
            threshold = self.config.final_threshold
        if isinstance(threshold, (list, tuple, np.ndarray)):
            return threshold[equation_idx]
        return threshold

    def expression_from_terms(self, Fu_list, coff_list, derivative_list):
        parts = []
        for Fu, coeff, derivative in zip(Fu_list, coff_list, derivative_list):
            coeff = float(coeff)
            if abs(coeff) < 1e-14:
                continue
            term = str(Fu).replace(" ", "")
            derivative_text = str(derivative).strip()
            if derivative_text:
                term = f"{derivative_text}({term})"
            parts.append(f"{coeff:+.16g}*{term}")
        expr = "".join(parts)
        return expr[1:] if expr.startswith("+") else expr

    def compact_expression_for_completion(self, rhs_expr, equation_idx):
        threshold = self.threshold_for_equation(equation_idx)
        Fu_list, coff_list, derivative_list = self.process_expression(rhs_expr, threshold)
        compact_expr = self.expression_from_terms(Fu_list, coff_list, derivative_list)
        print(
            f"[U(1) completion] component {equation_idx}: "
            f"{len(Fu_list)} thresholded terms"
        )
        return compact_expr

    def complete_u1_library_if_needed(self, rhs_expr_list, repeat_idx):
        if not self.skew_symmetic:
            return rhs_expr_list
        if repeat_idx >= self.skew_completion_repeats:
            return rhs_expr_list
        if len(rhs_expr_list) != 2:
            print(
                f"[U(1) completion] skipped at repeat {repeat_idx}: "
                f"expected 2 components, got {len(rhs_expr_list)}"
            )
            return rhs_expr_list

        print(f"[U(1) completion] repeat {repeat_idx}: completing candidate library")
        compact_rhs_expr_list = [
            self.compact_expression_for_completion(rhs_expr, i)
            for i, rhs_expr in enumerate(rhs_expr_list)
        ]
        u_new, v_new = skew_symmetric_completion(
            compact_rhs_expr_list[0],
            compact_rhs_expr_list[1],
        )
        print(f"[U(1) completion] u: {u_new}")
        print(f"[U(1) completion] v: {v_new}")
        return [u_new, v_new]

    def get_full_grid_coords(self):
        grid_dim = len(self.grid_size)
        axis_list = []

        for i in range(grid_dim):
            # Extract two floating-point values from the tensor.
            x_min = self.Input_Bounds[i, 0].item()
            x_max = self.Input_Bounds[i, 1].item()

            num_points = int(self.grid_size[i])  # Ensure that the value is an integer.

            grid_1d = np.linspace(x_min, x_max, num_points)
            axis_list.append(grid_1d)

        mesh = np.meshgrid(*axis_list, indexing='ij')  # Preserve row-major ordering.
        full_coords = np.stack(mesh, axis=-1)
        full_coords = full_coords.reshape(-1, grid_dim)
        return full_coords
    
    def Make_Random_Weight_Functions(self):
        Bounds = self.Input_Bounds
        W_Master = self.master_weight_function
        Num_Weight_Functions = self.Num_Weight_Functions

        # Get the number of dimensions.
        Num_Dimensions = Bounds.shape[0]

        # Determine the shortest side length of the i-th problem domain.
        ith_Min_Side_Length = Bounds[0, 1] - Bounds[0, 0]
        for i in range(1, Num_Dimensions):
            if Bounds[i, 1] - Bounds[i, 0] < ith_Min_Side_Length:
                ith_Min_Side_Length = Bounds[i, 1] - Bounds[i, 0]

        # Set up the random weight functions for the i-th problem domain.
        # If the problem domain is [a_1, b_1] x ... x [a_n, b_n], then we place the
        # centers in [a_1 + r + e, b_1 - r - e] x ... x [a_n - r + e, b_n - r - e],
        # where e = Epsilon is some small positive number (to ensure the weight
        # function support is in the domain).
        Random_Weight_Functions = []
        Epsilon: float = 0.0005
        if self.weight_radius_min is None and self.weight_radius_max is None:
            Radius_Ratio_Min, Radius_Ratio_Max = .2, .4
        elif self.weight_radius_min is not None and self.weight_radius_max is not None:
            Radius_Ratio_Min = float(self.weight_radius_min)
            Radius_Ratio_Max = float(self.weight_radius_max)
        else:
            raise ValueError("weight_radius_min and weight_radius_max must be set together.")
        if Radius_Ratio_Min <= 0 or Radius_Ratio_Max <= Radius_Ratio_Min:
            raise ValueError("Invalid weight-function radius range.")

        Center_Strategy = str(self.weight_center_strategy or "uniform").lower()
        if Center_Strategy == "latin_hypercube":
            Center_Strategy = "lhs"
        if Center_Strategy not in {"uniform", "lhs"}:
            raise ValueError(f"Unknown weight center strategy: {self.weight_center_strategy}")

        Lhs_Points = None
        if Center_Strategy == "lhs":
            Lhs_Points = np.empty((Num_Weight_Functions, Num_Dimensions), dtype=np.float32)
            for k in range(Num_Dimensions):
                Strata = [(j + random.random()) / Num_Weight_Functions for j in range(Num_Weight_Functions)]
                random.shuffle(Strata)
                Lhs_Points[:, k] = Strata

        for j in range(Num_Weight_Functions):
            # Set the radius of the j-th weight function.
            jth_Rand = random.uniform(Radius_Ratio_Min, Radius_Ratio_Max)
            jth_Radius = float(jth_Rand * ith_Min_Side_Length)

            # Set the center of the j-th weight function.
            jth_Center = torch.empty(Num_Dimensions, dtype=torch.float32).to(self.Device)
            for k in range(Num_Dimensions):
                lower_bound = float(Bounds[k, 0] + jth_Radius + Epsilon)
                upper_bound = float(Bounds[k, 1] - jth_Radius - Epsilon)
                if Center_Strategy == "lhs":
                    jth_Center[k] = lower_bound + float(Lhs_Points[j, k]) * (upper_bound - lower_bound)
                else:
                    jth_Center[k] = random.uniform(a=lower_bound, b=upper_bound)
            if self.weight_function_type == 'exp_bump':
                Random_Weight_Functions.append(Weight_Function(X_0=jth_Center.to(self.Device), r=jth_Radius, Coords=self.grid_data,V=self.V))
            elif self.weight_function_type == 'poly_bump':
                Random_Weight_Functions.append(poly_w.Poly_Weight_Function(X_0=jth_Center.to(self.Device), p=self.config.p_order, r=jth_Radius, Coords=self.grid_data,V=self.V))
            elif self.weight_function_type == 'mix_bump':
                Random_Weight_Functions.append(Weight_Function(X_0=jth_Center.to(self.Device), r=jth_Radius, Coords=self.grid_data,V=self.V))
                Random_Weight_Functions.append(poly_w.Poly_Weight_Function(X_0=jth_Center.to(self.Device), p=self.config.p_order, r=jth_Radius, Coords=self.grid_data,V=self.V))
        # Add the weight functions to the list of lists.
        return Random_Weight_Functions

    def Solve_problem(self):
        print("Strat Train")
        print("0:",self.num_equation)
        if self.use_search:
            beta_list, alphas_list, weights_sym_list, weights_int_list, max_repeats_list = self.NAS_Sym_Net()
            for i, model in enumerate(self.model_list):
                beta = beta_list[i]
                alphas = alphas_list[i]
                weights_sym = weights_sym_list[i]
                weights_int = weights_int_list[i]
                depth, repeats_list, new_weights_sym_list, new_weights_int_list = self.fix_na(beta, alphas, weights_sym,
                                                                                            weights_int, max_repeats_list)
                print(repeats_list)
                if not self.init_to_ones:
                    model.update_params(repeats_list=repeats_list, depth=depth, initial_weights_symnet=new_weights_sym_list,
                                        initial_weights_int=new_weights_int_list, grad_all=self.grad_all,kill_small_items=self.kill_small_items,
                                        threshold=self.threshold,
                                        enforce_galilean_invariance=self.enforce_galilean_invariance
                                        )
                elif self.init_to_ones:
                    model.update_params(repeats_list=repeats_list, depth=depth, initial_weights_symnet=None,
                                        initial_weights_int=None, grad_all=self.grad_all,kill_small_items=self.kill_small_items,
                                        threshold=self.threshold,
                                        enforce_galilean_invariance=self.enforce_galilean_invariance,
                                        init_to_ones=self.init_to_ones
                                        )
        best_pdes = self.final_train()
        print("Optimal PDEs Summary (Lowest Total Loss per Model):")
        PDE_list = []
        self.Tune_Net_list = []
        for i, data in best_pdes.items():
            expr = data['pde']
            print(f"\nModel #{i + 1}:")
            print(f"  Best PDE: {expr[0, 0]}")
            print(f"  Achieved Loss: {data['total_loss']:.6e}")
            threshold = self.threshold_for_equation(i)
            if not self.data_spase:
                # if self.other_data is not None:
                u_data = self.input_data[:, self.problem_dim + 1 + i]
                print("i",i)
                print("self.lhs_type:",self.lhs_type)
                print(u_data)
                LHS = Regression.compute_lhs(self.Random_Weight_Functions_Lists[i], self.problem_dim, u_data, self.lhs_type)
                if self.equation_type != 'NS':
                    Fu_list, coff_list, derivative_list = self.process_expression(str(expr[0, 0]), threshold)
                    RHS = Regression.compute_rhs(self.Random_Weight_Functions_Lists[i], Fu_list, derivative_list,
                                                self.input_data, self.var_names)
                    result, coff_list, rhs_expr = Regression.regression(RHS, LHS, coff_list, Fu_list, derivative_list)
                    print("coff_list:",coff_list)
                    print("rhs_expr:",rhs_expr)
                elif self.equation_type == 'NS':
                    Fu_list, coff_list, derivative_list = self.process_expression(str(expr[0, 0]), threshold)
                    if self.other_data is not None:
                        # print(self.other_data.device)
                        # print(self.input_data.device)
                        self.other_data=self.other_data.to(self.input_data.device)
                        RHS = Regression.compute_rhs(self.Random_Weight_Functions_Lists[i], Fu_list, derivative_list,
                                                    torch.cat([self.input_data,self.other_data],dim=1), self.var_names)
                    else:
                        RHS = Regression.compute_rhs(self.Random_Weight_Functions_Lists[i], Fu_list, derivative_list,
                                                    self.input_data, self.var_names)
                                                
                    min_loss, coff_list, rhs_expr = Regression.mstls_regression(
                                                            RHS, LHS,
                                                            Fu_list, derivative_list,
                                                            print_result_flag=True,
                                                            lambda_list=None,
                                                        )
                    print("coff_list:",coff_list)
                    print("rhs_expr:",rhs_expr)
                    Fu_list, coff_list, derivative_list = self.process_expression(str(rhs_expr), threshold)
                    if self.other_data is not None:
                        # print(self.other_data.device)
                        # print(self.input_data.device)
                        self.other_data=self.other_data.to(self.input_data.device)
                        RHS = Regression.compute_rhs(self.Random_Weight_Functions_Lists[i], Fu_list, derivative_list,
                                                    torch.cat([self.input_data,self.other_data],dim=1), self.var_names)
                    else:
                        RHS = Regression.compute_rhs(self.Random_Weight_Functions_Lists[i], Fu_list, derivative_list,
                                                    self.input_data, self.var_names)
                    result, coff_list, rhs_expr = Regression.regression(RHS, LHS, coff_list, Fu_list, derivative_list,print=True,bounds=True)
                   
                if self.lhs_type==1:
                    lhs_expr = 'D_t(' + self.var_names[i] + ')' + '='
                elif self.lhs_type == 2:
                    lhs_expr = 'D_tt(' + self.var_names[i] + ')' + '='
                PDE = lhs_expr + rhs_expr
                PDE_list.append(PDE)
            else:
                if not self.use_search:
                    if self.weights_functions_path is not None:
                        self.Random_Weight_Functions_Lists[i] = torch.load(self.weights_functions_path,weights_only=False)
                        print("weights functions have been loaded!")
                    else:
                        self.Random_Weight_Functions_Lists[i] = self.Make_Random_Weight_Functions()
                Fu_list, coff_list, derivative_list = self.process_expression(str(expr[0, 0]), threshold)
                if self.equation_type == 'NS':
                    with torch.no_grad():
                        input_data = self.Pig_Net(self.input_test, self.norm_flag)
                    u_data = input_data[:, i]
                    LHS = Regression.compute_lhs(self.Random_Weight_Functions_Lists[i], self.problem_dim, u_data, self.lhs_type)
                    if self.other_data is not None:
                        self.other_data=self.other_data.to(input_data.device)
                        RHS = Regression.compute_rhs(self.Random_Weight_Functions_Lists[i], Fu_list, derivative_list,
                                                    torch.cat([input_data,self.other_data],dim=1), self.var_names)
                    else:
                        RHS = Regression.compute_rhs(self.Random_Weight_Functions_Lists[i], Fu_list, derivative_list,
                                                    input_data, self.var_names)
                                                
                    min_loss, coff_list, rhs_expr = Regression.mstls_regression(
                                                            RHS, LHS,
                                                            Fu_list, derivative_list,
                                                            print_result_flag=True,
                                                            lambda_list=None,
                                                        )
                    print("coff_list:",coff_list)
                    print("rhs_expr:",rhs_expr)
                    Fu_list, coff_list, derivative_list = self.process_expression(str(rhs_expr), threshold)
                if self.joint_tune:
                    Tune_net = Tune_Net(self.problem_dim, Fu_list, derivative_list, self.var_names,
                                    coff_list, self.Device,self.joint_tune,coff_list,self.reg_weight,
                                    bounds_flag=self.bounds_flag,lhs_type=self.lhs_type)
                else:
                    Tune_net = Tune_Net(self.problem_dim, Fu_list, derivative_list, self.var_names,
                                        coff_list, self.Device,joint_optim=self.joint_tune,int_xi=coff_list,
                                        reg_lambda=1e-5,bounds_flag=self.bounds_flag,lhs_type=self.lhs_type)
                self.Tune_Net_list.append(Tune_net)
        
        if self.data_spase:
            if self.equation_type == 'SG':
                self.config.f_scale = 1e-3
            elif self.equation_type == 'NS':
                self.config.f_scale = 100
            if self.config.n_epochs_tune > 0:
                if not self.use_search:
                    self.input_test = self.input_test.to(self.Device)
                rhs_expr_list = train_tune_net(Tune_net_list=self.Tune_Net_list, Num_datasets=self.num_equation,
                                        input_train=self.input_train,
                                        u_train=self.input_data[:, self.problem_dim + 1:].to(self.Device), input_test=self.input_test,
                                        Random_Weight_Functions_Lists=self.Random_Weight_Functions_Lists,
                                        f_scale=self.config.f_scale, problem_dim=self.problem_dim, Device=self.Device,
                                        output_path=self.subfolder_paths['tune'],
                                        Pig_Net=self.Pig_Net,other_data=self.other_data, norm_flag=self.norm_flag, u_test=self.inputs_test, grid_plot_size=self.grid_plot_size,
                                        grid_size=self.config.grid_size,
                                        optimizer_type=self.config.tune_type, lr=self.config.learning_rate_tune, max_iter=self.config.n_epochs_tune,
                                        print_every=self.tune_print_every,step=self.time_step)
                if self.tune_repeats!=1:
                    for k in range(self.tune_repeats):
                        rhs_expr_list = self.complete_u1_library_if_needed(rhs_expr_list, k)
                        self.Tune_Net_list = []
                        for i in range(self.num_equation):
                            threshold = self.threshold_for_equation(i)
                            Fu_list, coff_list, derivative_list = self.process_expression(rhs_expr_list[i], threshold)
                            if self.equation_type == 'NS':
                                with torch.no_grad():
                                    input_data = self.Pig_Net(self.input_test, self.norm_flag)
                                u_data = input_data[:, i]
                                LHS = Regression.compute_lhs(self.Random_Weight_Functions_Lists[i], self.problem_dim, u_data, self.lhs_type)
                                if self.other_data is not None:
                                    self.other_data=self.other_data.to(input_data.device)
                                    RHS = Regression.compute_rhs(self.Random_Weight_Functions_Lists[i], Fu_list, derivative_list,
                                                                torch.cat([input_data,self.other_data],dim=1), self.var_names)
                                else:
                                    RHS = Regression.compute_rhs(self.Random_Weight_Functions_Lists[i], Fu_list, derivative_list,
                                                                input_data, self.var_names)
                                if len(Fu_list)<=4:
                                    result, coff_list, rhs_expr = Regression.regression(
                                                                                    RHS, LHS,
                                                                                    coff_list, Fu_list, derivative_list,
                                                                                    print=True,bounds=True)
                                else:
                                    min_loss,coff_list, rhs_expr = Regression.mstls_regression(
                                                                                RHS, LHS,
                                                                                Fu_list, derivative_list,
                                                                                print_result_flag=True,
                                                                                lambda_list=None,
                                                                            )
                                print("coff_list:",coff_list)
                                print("rhs_expr:",rhs_expr)
                                rhs_expr = complete_symmetric_term_pair(rhs_expr, 'D_x^2 (w)', 'D_y^2 (w)')
                                rhs_expr = complete_symmetric_term_pair(rhs_expr, 'D_x (u*w)', 'D_y (v*w)', zero_fallback=1.0)
                                Fu_list, coff_list, derivative_list = self.process_expression(str(rhs_expr), threshold)
                            if self.equation_type=="NS" and len(Fu_list)<=4:
                                self.config.n_epochs_tune = 2000
                                self.bounds_flag = True
                                self.reg_weight = 0
                                self.config.f_scale = 1e-1
                            if self.joint_tune:
                                Tune_net = Tune_Net(self.problem_dim, Fu_list, derivative_list, self.var_names,
                                                coff_list, self.Device,self.joint_tune,coff_list,self.reg_weight,
                                                bounds_flag=self.bounds_flag,lhs_type=self.lhs_type)
                            else:
                                Tune_net = Tune_Net(self.problem_dim, Fu_list, derivative_list, self.var_names,
                                                    coff_list, self.Device,joint_optim=self.joint_tune,int_xi=coff_list,
                                                    reg_lambda=1e-5,bounds_flag=self.bounds_flag,lhs_type=self.lhs_type)
                            self.Tune_Net_list.append(Tune_net)
                        if self.config.n_epochs_tune > 0:
                            rhs_expr_list = train_tune_net(Tune_net_list=self.Tune_Net_list, Num_datasets=self.num_equation,
                                                    input_train=self.input_train,
                                                    u_train=self.input_data[:, self.problem_dim + 1:].to(self.Device), input_test=self.input_test,
                                                    Random_Weight_Functions_Lists=self.Random_Weight_Functions_Lists,
                                                    f_scale=self.config.f_scale, problem_dim=self.problem_dim, Device=self.Device,
                                                    output_path=self.subfolder_paths['tune'],
                                                    Pig_Net=self.Pig_Net,other_data=self.other_data,norm_flag=self.norm_flag, u_test=self.inputs_test, grid_plot_size=self.grid_plot_size,
                                                    grid_size=self.config.grid_size,
                                                    optimizer_type=self.config.tune_type, lr=self.config.learning_rate_tune, max_iter=self.config.n_epochs_tune,
                                                    print_every=self.tune_print_every,step=self.time_step)
            if self.lhs_type == 1:
                if len(self.var_names)==1:
                    lhs_expr = ['D_t(' + self.var_names[0] + ')' + '=']
                elif len(self.var_names)==2:
                    lhs_expr = ['D_t(' + self.var_names[0] + ')' + '=', 'D_t(' + self.var_names[1] + ')' + '=']
                elif len(self.var_names)==3:
                    lhs_expr = ['D_t(' + self.var_names[0] + ')' + '=', 'D_t(' + self.var_names[1] + ')' + '=', 'D_t(' + self.var_names[2] + ')' + '=']
                
                
            elif self.lhs_type == 2:
                if len(self.var_names)==1:
                    lhs_expr = ['D_tt(' + self.var_names[0] + ')' + '=']
                elif len(self.var_names)==2:
                    lhs_expr = ['D_tt(' + self.var_names[0] + ')' + '=', 'D_tt(' + self.var_names[1] + ')' + '=']
                elif len(self.var_names)==3:
                    lhs_expr = ['D_tt(' + self.var_names[0] + ')' + '=', 'D_tt(' + self.var_names[1] + ')' + '=', 'D_tt(' + self.var_names[2] + ')' + '=']
                
            for i in range(self.num_equation):
                if self.config.n_epochs_tune > 0:
                    rhs_expr = rhs_expr_list[i]
                else:
                    rhs_expr = Regression.print_result(coff_list,Fu_list,derivative_list)
                PDE = lhs_expr[i] + rhs_expr
                PDE_list.append(PDE)
        return PDE_list

    def NAS_Sym_Net(self):
        if self.data_spase:
            if self.num_gaussians == 0:
                log_dir = f"runs_{self.equation_type}_{self.sample_ratio}_{self.sigma_NR}_no_gauss{self.run_suffix}/search"
            elif not self.use_search:
                log_dir = f"runs_{self.equation_type}_{self.sample_ratio}_{self.sigma_NR}_no_nas{self.run_suffix}/search"
            else:
                log_dir = f"runs_{self.equation_type}_{self.sample_ratio}_{self.sigma_NR}{self.run_suffix}/search"
        else:
            log_dir = f"runs_{self.equation_type}_{self.sample_ratio}_{self.sigma_NR}{self.run_suffix}/search"
        reg_weight = [0.0] * self.num_equation
        writer = SummaryWriter(log_dir=log_dir)

        # Set up weight function lists. We initialize each list to be empty.
        self.Random_Weight_Functions_Lists = []
        self.Targeted_Weight_Functions_Lists = []
        self.Train_Weight_Functions_Lists = []
        if self.data_spase:
            trainable_params = list(self.Pig_Net.parameters())
        else:
            trainable_params = []
        for i in range(self.num_equation):
            self.Random_Weight_Functions_Lists.append([])
            self.Targeted_Weight_Functions_Lists.append([])
            self.Train_Weight_Functions_Lists.append([])
            # print("i",i)
            trainable_params = trainable_params + list(self.model_list[i].parameters())
        Optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate_nas)
        print("\nRunning %d epochs..." % self.n_epochs_nas)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(Optimizer, gamma=self.decay_rate)
        best_params = {}
        best_pig_net = {
            "state_dict": None,
            "loss": float('inf')
        }

        for t in range(0, self.n_epochs_nas):
            # Train
            if t % self.update_weight_epochs == 0:
                for i in range(self.num_equation):
                    if self.weights_functions_path is not None:
                        self.Random_Weight_Functions_Lists[i] = torch.load(self.weights_functions_path,weights_only=False)
                        print("weights functions have been loaded!")
                    else:
                        self.Random_Weight_Functions_Lists[i] = self.Make_Random_Weight_Functions()
            for i in range(self.num_equation):
                if self.weights_functions_path is None:
                    self.Train_Weight_Functions_Lists[i] = (
                            self.Random_Weight_Functions_Lists[i] + self.Targeted_Weight_Functions_Lists[i])
                else:
                    self.Train_Weight_Functions_Lists[i] = self.Random_Weight_Functions_Lists[i]

            # Run one epoch of training.
            Data_Loss, Residual_List, Weak_Loss_List, L1_Loss_List, Total_Loss_List \
                = Training(Model_list=self.model_list, problem_dim=self.problem_dim,
                           Inputs_train=self.input_data, Weight_Functions_List=self.Train_Weight_Functions_Lists,
                           reg_weight=reg_weight, Optimizer=Optimizer, Device=self.Device, 
                           input_test=self.input_test,lhs_type=self.lhs_type, max_iter=self.n_epochs_nas, grid_plot_size=self.grid_plot_size,
                           grid_size=self.config.grid_size,u_test=self.inputs_test,
                           f_scale= self.f_scale, Pig_Net=self.Pig_Net, norm_flag=self.norm_flag, data_spase=self.data_spase,
                           Writer=writer, Epoch=t,output_path=self.subfolder_paths['search'],
                           train_pig=True,step=self.time_step,clip_grad=self.clip_grad,max_norm=self.max_norm,int_type=self.int_type,other_data=self.other_data)

            # Set up targeted weight functions for the next epoch.
            Residual_Cutoffs = []
            for i in range(self.num_equation):
                # Determine the cutoff. We will target any weight function with a
                # residual above this level.
                Abs_Residual: torch.Tensor = torch.abs(Residual_List[i])

                ith_Mean: float = torch.mean(Abs_Residual).item()
                ith_SD: float = torch.std(Abs_Residual).item()

                Residual_Cutoffs.append(ith_Mean + 3 * ith_SD)

                # Determine which weight functions have a large residual.
                self.Targeted_Weight_Functions_Lists[i] = []
                for j in range(len(self.Train_Weight_Functions_Lists[i])):
                    if Abs_Residual[j] >= Residual_Cutoffs[i]:
                        self.Targeted_Weight_Functions_Lists[i].append(self.Train_Weight_Functions_Lists[i][j])

            with torch.no_grad():
                if self.data_spase and Data_Loss.item() < best_pig_net["loss"]:
                    best_pig_net["state_dict"] = copy.deepcopy(self.Pig_Net.state_dict())
                    best_pig_net["loss"] = Data_Loss.item()

                for i, model in enumerate(self.model_list):
                    SymNet = model.Sym_Net
                    IntNet = model.Int_Net
                    beta = SymNet.get_beta()
                    layers = SymNet.hidden_layers
                    weights_sym = []
                    weights_int = [IntNet.W.detach().clone(), IntNet.Xi.detach().clone()]
                    alphas = []
                    sp_funcs = []
                    max_repeats_list = SymNet.repeats_list
                    for layer in layers:
                        weights_sym.append(layer.get_weight())
                        alphas.append(layer.get_alpha())
                        alphas = [alpha for alpha in alphas if alpha is not None]
                        # print("alphas:", alphas)
                        sp_funcs.append(layer.funcs_sp)
                    
                    current_total_loss = Total_Loss_List[i]
                    current_weak_loss = Weak_Loss_List[i]
                    current_reg_loss = L1_Loss_List[i]
                    print(f"\n{'=' * 50}")
                    print(f"Model #{i + 1} Summary:")
                    print(f"  Total Loss: {current_total_loss:.6e}")
                    print(f"  Data Loss: {Data_Loss.item():.6e}")
                    print(f"  Weak Loss: {current_weak_loss:.6e}")
                    print(f"  Regularization Loss: {current_reg_loss:.6e}")
                    print('=' * 50)
                    if i not in best_params or current_total_loss < best_params[i]["total_loss"]:
                        best_params[i] = {
                            "beta": copy.deepcopy(beta),
                            "alphas": copy.deepcopy(alphas),
                            "weights_sym": copy.deepcopy(weights_sym),
                            "weights_int": weights_int,
                            "total_loss": current_total_loss
                        }
        if self.data_spase and best_pig_net["state_dict"] is not None:
            self.Pig_Net.load_state_dict(best_pig_net["state_dict"])
        beta_list = []
        alphas_list = []
        weights_sym_list = []
        weights_int_list = []
        for i, data in best_params.items():
            beta_list.append(data["beta"])
            alphas_list.append(data["alphas"])
            weights_sym_list.append(data["weights_sym"])
            weights_int_list.append(data["weights_int"])
        return beta_list, alphas_list, weights_sym_list, weights_int_list, max_repeats_list

    def fix_na(self, beta, alphas_list, weights_sym, weights_int, max_repeats_list):
        """
        beta: (b_1, b_2, b_3, ..., b_depth)
        alphas_list: [[(a_1, a_2, ...), (), ()], [], [], ...]
        alpha_list: A nested list such as [[a_1, a_2, ...], [], []], where each
            inner list contains the alpha values for the operators in one layer.
        weights: The weights for each layer.
        repeats_list: A nested list such as [[0, 1, 0, 0]], where
            repeats_list[i] represents layer i.
        max_repeats_list: A list such as [2, 2, 2, 2].
        """
        beta_index = np.argmax(beta)
        depth = self.depth_candidates[beta_index]
        repeats_list = []
        for alphas in alphas_list:
            repeats = []
            for (alpha, repeats_candidate) in zip(alphas, self.repeats_candidates):
                alpha_index = np.argmax(alpha)
                repeat = repeats_candidate[alpha_index]
                repeats.append(repeat)
            repeats_list.append(repeats)
        repeats_list = repeats_list[0:depth]
        weights_sym = weights_sym[0:depth]

        new_weights_sym_list = []
        new_weights_int_list = []
        for i in range(len(weights_sym)):
            weight = weights_sym[i]
            # print("weight.shape:", weight.shape)
            repeats = repeats_list[i]
            max_repeats = max_repeats_list[i]
            if i == 0:
                partial_index = 0
                selected_weight_list = []
                selected_indices = []  # Record the indices of retained columns.
                for j in range(len(max_repeats)):
                    max_repeat = max_repeats[j]
                    repeat = repeats[j]
                    if j < len(max_repeats) - self.n_double:
                        start_idx = partial_index
                        end_idx = partial_index + max_repeat
                        partial_weight = weight[:, start_idx:end_idx]
                        selected_weight = partial_weight[:, 0:repeat]
                        selected_weight_list.append(selected_weight)
                        current_indices = list(range(start_idx, start_idx + repeat))
                        selected_indices.extend(current_indices)
                        partial_index += max_repeat
                    else:
                        start_idx = partial_index
                        end_idx = partial_index + max_repeat*2
                        partial_weight = weight[:, start_idx:end_idx]
                        selected_weight = partial_weight[:, 0:2*repeat]
                        selected_weight_list.append(selected_weight)
                        current_indices = list(range(start_idx, start_idx + repeat))
                        selected_indices.extend(current_indices)
                        partial_index += 2*max_repeat

                new_weight = np.concatenate(selected_weight_list, axis=1)
                print(f"Retained column indices: {selected_indices}")
                # Alternatively, save them in a variable: retained_indices = selected_indices.
            else:
                partial_index = 0
                selected_weight_list = []
                weight = weight[selected_indices]
                selected_indices = []
                for j in range(len(max_repeats)):
                    max_repeat = max_repeats[j]
                    repeat = repeats[j]
                    if j < len(max_repeats) - 1:
                        start_idx = partial_index
                        end_idx = partial_index + max_repeat
                        partial_weight = weight[:, start_idx:end_idx]
                        selected_weight = partial_weight[:, 0:repeat]
                        selected_weight_list.append(selected_weight)
                        current_indices = list(range(start_idx, start_idx + repeat))
                        selected_indices.extend(current_indices)
                        partial_index += max_repeat
                    else:
                        start_idx = partial_index
                        end_idx = partial_index + max_repeat * 2
                        partial_weight = weight[:, start_idx:end_idx]
                        selected_weight = partial_weight[:, 0:2 * repeat]
                        selected_weight_list.append(selected_weight)
                        current_indices = list(range(start_idx, start_idx + repeat))
                        selected_indices.extend(current_indices)
                        partial_index += 2 * max_repeat
                new_weight = np.concatenate(selected_weight_list, axis=1)
                print(f"Retained column indices: {selected_indices}")
            new_weights_sym_list.append(new_weight)
        for i in range(len(weights_int)):
            weight = weights_int[i]
            if i == 0:
                new_weight = weight[selected_indices]
            else:
                new_weight = weight
            new_weights_int_list.append(new_weight)
            # print(repeats_list)
        return depth, repeats_list, new_weights_sym_list, new_weights_int_list

    def collect_symbolic_biases(self, layers):
        if not self.add_bias:
            return None
        biases = []
        for layer in layers:
            if not hasattr(layer, "get_bias"):
                return None
            bias = layer.get_bias()
            if bias is None:
                return None
            biases.append(bias)
        return biases

    def collect_integrate_biases(self, int_net):
        if not self.add_bias or not hasattr(int_net, "b"):
            return None
        first_bias = int_net.b.cpu().detach().numpy()
        if first_bias.shape[1] == 1 and int_net.diff_len != 1:
            first_bias = np.repeat(first_bias, int_net.diff_len, axis=1)
        output_bias = np.zeros((1, int_net.output_dim))
        return [first_bias, output_bias]

    def final_train(self):
        if self.data_spase:
            if self.num_gaussians == 0:
                log_dir = f"runs_{self.equation_type}_{self.sample_ratio}_{self.sigma_NR}_no_gauss{self.run_suffix}/train"
            elif not self.use_search:
                log_dir = f"runs_{self.equation_type}_{self.sample_ratio}_{self.sigma_NR}_no_nas{self.run_suffix}/train"
            else:
                log_dir = f"runs_{self.equation_type}_{self.sample_ratio}_{self.sigma_NR}{self.run_suffix}/train"
        else:
            log_dir = f"runs_{self.equation_type}_{self.sample_ratio}_{self.sigma_NR}{self.run_suffix}/train"
        if not self.use_search:
            # Set up weight function lists. We initialize each list to be empty.
            self.Random_Weight_Functions_Lists = []
            self.Targeted_Weight_Functions_Lists = []
            self.Train_Weight_Functions_Lists = []
            for i in range(self.num_equation):
                self.Random_Weight_Functions_Lists.append([])
                self.Targeted_Weight_Functions_Lists.append([])
                self.Train_Weight_Functions_Lists.append([])
        writer = SummaryWriter(log_dir=log_dir)
        reg_weight = [self.reg_weight] * self.num_equation
        if self.data_spase:
            trainable_params = list(self.Pig_Net.parameters())
        else:
            trainable_params = []
        for i in range(self.num_equation):
            trainable_params = trainable_params + list(self.model_list[i].parameters())

        adam_optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate)
        lbfgs_optimizer = torch.optim.LBFGS(
            list(self.model_list[i].parameters()),
            lr=self.learning_rate_nas)
        n_epochs = self.n_epochs_adam + self.n_epochs_lbfgs
        print("\nRunning %d epochs..." % n_epochs)
        if n_epochs!=0:
            for t in range(0, n_epochs):
                if t < self.n_epochs_adam:
                    Optimizer = adam_optimizer
                    # reg_weight = [self.reg_weight] * self.num_equation
                    train_pig = True
                else:
                    Optimizer = lbfgs_optimizer
                    try:
                        reg_weight = [self.config.reg_weight_lbfgs] * self.num_equation
                    except:
                        reg_weight = [0.0] * self.num_equation
                    train_pig = False
                    self.f_scale = 1
                # Train
                if t % self.update_weight_epochs == 0:
                    for i in range(self.num_equation):
                        # if self.
                        self.Random_Weight_Functions_Lists[i] = self.Make_Random_Weight_Functions()
                        self.Random_Weight_Functions_Lists[i].append(self.master_weight_function)
                # Next, combine the random and targeted weight functions.
                for i in range(self.num_equation):
                    if self.weights_functions_path is None:
                        self.Train_Weight_Functions_Lists[i] = (
                                self.Random_Weight_Functions_Lists[i] + self.Targeted_Weight_Functions_Lists[i])
                    else:
                        self.Train_Weight_Functions_Lists[i] = self.Random_Weight_Functions_Lists[i] 
                # self.f_scale = 0.1
                # Run one epoch of training.
                Data_Loss, Residual_List, Weak_Loss_List, L1_Loss_List, Total_Loss_List \
                    = Training(Model_list=self.model_list, problem_dim=self.problem_dim,
                            Inputs_train=self.input_data, Weight_Functions_List=self.Train_Weight_Functions_Lists,
                            reg_weight=reg_weight, Optimizer=Optimizer, Device=self.Device, 
                            input_test=self.input_test, lhs_type=self.lhs_type, max_iter=self.n_epochs_adam, grid_plot_size=self.grid_plot_size,
                            grid_size=self.config.grid_size, u_test=self.inputs_test,
                            f_scale= self.f_scale, Pig_Net=self.Pig_Net,norm_flag=self.norm_flag,data_spase=self.data_spase,
                            Writer=writer, Epoch=t,output_path=self.subfolder_paths['train'],
                            train_pig=train_pig, step=self.time_step,clip_grad=self.clip_grad,max_norm=self.max_norm,other_data=self.other_data)
                # Set up targeted weight functions for the next epoch.
                Residual_Cutoffs = []
                for i in range(self.num_equation):
                    # Determine the cutoff. We will target any weight function with a
                    # residual above this level.
                    Abs_Residual: torch.Tensor = torch.abs(Residual_List[i])

                    ith_Mean: float = torch.mean(Abs_Residual).item()
                    ith_SD: float = torch.std(Abs_Residual).item()

                    Residual_Cutoffs.append(ith_Mean + 3 * ith_SD)

                    # Determine which weight functions have a large residual.
                    self.Targeted_Weight_Functions_Lists[i] = []
                    for j in range(len(self.Train_Weight_Functions_Lists[i])):
                        if Abs_Residual[j] >= Residual_Cutoffs[i]:
                            self.Targeted_Weight_Functions_Lists[i].append(self.Train_Weight_Functions_Lists[i][j])

                best_pdes = {}
                with torch.no_grad():
                    for i, model in enumerate(self.model_list):
                        SymNet = model.Sym_Net
                        IntNet = model.Int_Net
                        diff_list = IntNet.diff_list
                        layers = SymNet.hidden_layers
                        weights_sym = []
                        # print("IntNet.W:", IntNet.W)
                        # print(self.grad_all)
                        if self.grad_all:
                            IntNet.Xi[0] = 0
                        # print(IntNet.Xi)
                        weights_int = [IntNet.W.cpu().detach().numpy(), IntNet.Xi.cpu().detach().numpy()]
                        # if self.grad_all:

                        funcs_list = []
                        for layer in layers:
                            weights_sym.append(layer.get_weight())
                            funcs_list.append(layer.func_list)
                        depth = len(weights_sym)
                        biases_sym = self.collect_symbolic_biases(layers)
                        biases_int = self.collect_integrate_biases(IntNet)

                        current_pde = print_model(weights_sym_list=weights_sym, funcs_list=funcs_list,
                                                var_names=self.var_names,
                                                weights_int_list=weights_int, diff_list=diff_list,
                                                threshold=self.threshold,
                                                add_bias=self.add_bias,
                                                biases=biases_sym,
                                                int_biases=biases_int)
                        current_total_loss = Total_Loss_List[i]
                        current_weak_loss = Weak_Loss_List[i]
                        current_reg_loss = L1_Loss_List[i]
                        # if reg_weight[i]!= 0.0:
                        #     scale_weight = current_weak_loss/current_reg_loss
                        #     reg_weight[i] *= scale_weight

                        print(f"\n{'=' * 50}")
                        print(f"Model #{i + 1} Summary:")
                        print(f"  PDE Expression: {current_pde}")
                        print(f"  Total Loss: {current_total_loss:.6e}")
                        print(f"  Data Loss: {Data_Loss.item():.6e}")
                        print(f"  Weak Loss: {current_weak_loss:.6e}")
                        print(f"  Regularization Loss: {current_reg_loss:.6e}")
                        print('=' * 50)

                        if i not in best_pdes:
                            best_pdes[i] = {
                                "pde": current_pde,
                                "total_loss": float('inf')
                            }
                        if current_total_loss < best_pdes[i]["total_loss"]:
                            best_pdes[i]["pde"] = current_pde
                            best_pdes[i]["total_loss"] = current_total_loss
                    if self.data_spase:
                        input_plot_test = self.inputs_test[:,0:self.problem_dim+1].to(self.Device)
                        l2_loss = torch.mean((self.inputs_test[:,self.problem_dim+1:].to(self.Device) - self.Pig_Net(input_plot_test, self.norm_flag)) ** 2)
                        print('[Loss: %.5e]'%(l2_loss))
        else:
            best_pdes = {}
            with torch.no_grad():
                for i, model in enumerate(self.model_list):
                    SymNet = model.Sym_Net
                    IntNet = model.Int_Net
                    diff_list = IntNet.diff_list
                    layers = SymNet.hidden_layers
                    weights_sym = []
                    weights_int = [IntNet.W.cpu().detach().numpy(), IntNet.Xi.cpu().detach().numpy()]
                    # if self.grad_all:

                    funcs_list = []
                    for layer in layers:
                        weights_sym.append(layer.get_weight())
                        funcs_list.append(layer.func_list)
                    depth = len(weights_sym)
                    biases_sym = self.collect_symbolic_biases(layers)
                    biases_int = self.collect_integrate_biases(IntNet)

                    current_pde = print_model(weights_sym_list=weights_sym, funcs_list=funcs_list,
                                            var_names=self.var_names,
                                            weights_int_list=weights_int, diff_list=diff_list,
                                            threshold=self.threshold,
                                            add_bias=self.add_bias,
                                            biases=biases_sym,
                                            int_biases=biases_int)
                    best_pdes[i] = {
                        "pde": current_pde,
                        "total_loss": float('inf')
                    }
                    
        return best_pdes
    
    def process_expression(self,expr_str,threshold):
        x = sp.Symbol('x')
        y =  sp.Symbol('y')
        z =  sp.Symbol('z')
        u_func = sp.Function('u')
        v_func = sp.Function('v')
        w_func = sp.Function('w')
        # threshold = self.config.final_threshold
        dim = self.problem_dim
        expr_str = simplify_trig_coefficients(expr_str)
        expr_str = parse_derivatives(dim, expr_str)
        expr_str = replace_str_with_fun(expr_str)
        expr = sp.sympify(expr_str, locals={'Derivative': sp.Derivative, 'u': u_func, 'v': v_func, 'w': w_func,'x': x,'y': y,'z': z})
        result = decompose_expression(expr,dim)
        print(result)
        Fu_list = []
        coff_list = []
        derivative_list = []
        for order in sorted(result.keys()):
            new_order = np.array((0,) + order)
            print(new_order)
            derivative = Derivative.Derivative(new_order)
            expr_str = result[order]
            filtered_expr = sum(
                term for term in expr_str.expand().as_ordered_terms()
                if abs(term.coeff(x)) >= threshold or abs(term.as_coeff_Mul()[0]) >= threshold
            )

            # Simplify the output.
            simplified_expr = sp.simplify(filtered_expr)
            coefficient, Fu_part = simplified_expr.as_coeff_Mul()
            if coefficient != 0:
                Fu_part = Fu_part.subs(u_func(x), sp.Symbol('u'))
                Fu_part = Fu_part.subs(v_func(x), sp.Symbol('v'))
                Fu_part = Fu_part.subs(w_func(x), sp.Symbol('w'))

                # Expand terms, for example: u*(-0.88*u - 0.27) -> [-0.88*u**2, -0.27*u].
                expanded_terms = (coefficient * Fu_part).expand().as_ordered_terms()
                for term in expanded_terms:
                    c, f = term.as_coeff_Mul()
                    Fu_list.append(sp.simplify(f))  # Keep the pure function term.
                    coff_list.append(c)             # Store the coefficient.
                    derivative_list.append(derivative)  # Keep the corresponding derivative unchanged.

        print("Fu_list: ",Fu_list)
        print("coff_list: ",coff_list)
        print("derivative_list: ",derivative_list)
        return Fu_list, coff_list, derivative_list
