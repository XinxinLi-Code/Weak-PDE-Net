"""Contains the symbolic regression neural network architecture."""
import numpy as np
import torch.nn as nn
import functions as functions
from functions import *
import torch.nn.init as init
import collections
import torch.nn.functional as F
import random
class SymbolicLayer(nn.Module):
    def __init__(self, layer_index, repeats, funcs, initial_weight=None, init_uniform=1, in_dim=None, add_bias=False,
                 repeats_candidates=[], use_search=False, kill_small_items=False, threshold=0,
                 prev_layer_funcs_list=None, enforce_galilean_invariance=False, init_to_ones=False):  
        """
        funcs: Activation-function objects, for example,
            funcs = [Identity(), Square(), Sin(), Product()].
        repeats: Number of copies of each activation function, for example,
            repeats = [1, 0, 0, 1].
        repeats_candidates: Candidate repeat counts, for example,
            [[1, 2, 3], [0, 1, 2], [0, 1], [0, 1]].
        initial_weight: Optional initial value for the weight matrix.
        variable: Whether initial_weight is a variable.
        init_uniform: Uniform-initialization scale used when initial_weight is not provided.
        """
        super().__init__()
        self.layer_index = layer_index
        self.prev_layer_funcs_list = prev_layer_funcs_list
        self.enforce_galilean_invariance = enforce_galilean_invariance
        self.init_to_ones = init_to_ones
        self.init_uniform = init_uniform
        self.use_search = use_search
        if repeats_candidates:
            assert len(repeats_candidates) == len(
                funcs), "Length mismatch: repeats_candidates and funcs must have the same length."
            self.repeats_candidates = repeats_candidates
        self.funcs = funcs
        # sym-parameters
        self.initial_weight = initial_weight
        self.W = None  # Weight matrix.
        self.add_bias = add_bias
        self.output = None  # Tensor for the layer output.
        self.kill_small_items = kill_small_items
        self.threshold = threshold
        if self.use_search and self.repeats_candidates:
            self.repeats = []
            self.func_list = []
            for i in range(len(self.repeats_candidates)):
                repeat = max(repeats_candidates[i])
                self.repeats.append(repeat)
                self.func_list.extend([self.funcs[i]] * repeat)
        else:
            assert len(repeats) == len(
                funcs), "Length mismatch: repeats_candidates and funcs must have the same length."
            self.repeats = repeats  # Number of activation functions (and number of layer outputs)
            self.func_list = []
            for i in range(len(self.repeats)):
                self.func_list.extend([self.funcs[i]] * self.repeats[i])

        self.funcs_sp = [func.sp for func in self.func_list]
        self.funcs = [func.torch for func in self.func_list]
        # print(self.funcs_sp)
        self.n_funcs = len(self.funcs)
        self.n_double = functions.count_double(self.func_list)  # Number of activation functions that take two inputs.
        self.n_single = self.n_funcs - self.n_double  # Number of activation functions that take one input.
        self.out_dim = self.n_funcs + self.n_double
        self.W = None  # Weight matrix.

        if self.initial_weight is not None:  # Use the provided initial weight.
            if isinstance(initial_weight, np.ndarray):
                self.initial_weight = torch.from_numpy(initial_weight).float()
            else:
                self.initial_weight = initial_weight.clone().detach()
            self.W = nn.Parameter(self.initial_weight)
        else:
            self.W = nn.Parameter(torch.empty(in_dim, self.out_dim))
            if self.init_to_ones:
                nn.init.constant_(self.W, 1.0)
            else:
                nn.init.uniform_(self.W, a=-self.init_uniform, b=self.init_uniform)
            sin_indices = [i for i, func in enumerate(self.func_list) if func.name == 'sin']
            with torch.no_grad():
                for i in sin_indices:
                    self.W[:, i] = 1.0
                    self.W[:, i].requires_grad = False
                
            if add_bias:
                self.b = nn.Parameter(torch.empty(1, self.out_dim))
                nn.init.uniform_(self.b, a=-1, b=1)

        if add_bias and not hasattr(self, "b"):
            self.b = nn.Parameter(torch.empty(1, self.out_dim))
            nn.init.uniform_(self.b, a=-1, b=1)
        if self.enforce_galilean_invariance:
            with torch.no_grad():
                if self.layer_index == 0:
                    self.W[1:, :self.n_single] = 0.0

                    current_binary_col = self.n_single
                    for _ in range(self.n_double):
                        self.W[1:, current_binary_col] = 0.0
                        if self.W[0, current_binary_col] == 0:
                            self.W[0, current_binary_col] = 1.0
                        current_binary_col += 2

        if self.use_search and self.repeats_candidates:
            self.alpha_list = nn.ParameterList()
            for i in range(len(self.repeats_candidates)):
                alpha = nn.Parameter(torch.randn(len(self.repeats_candidates[i])))
                self.alpha_list.append(alpha)

    def forward(self, x):
        with torch.no_grad():
            sin_indices = [i for i, func in enumerate(self.func_list) if func.name == 'sin']
            self.W[:, sin_indices] = 1.0
            # print(self.kill_small_items)
            if self.kill_small_items:
                self.W[torch.abs(self.W) < self.threshold] = 0.0
        if self.enforce_galilean_invariance:
            with torch.no_grad():
                if self.layer_index == 0:
                    self.W[1:, :self.n_single] = 0.0
                    current_binary_col = self.n_single
                    for _ in range(self.n_double):
                        self.W[1:, current_binary_col] = 0.0
                        current_binary_col += 2

                elif self.layer_index > 0 and self.prev_layer_funcs_list is not None:
                    prev_prod_indices = [i for i, f in enumerate(self.prev_layer_funcs_list) if isinstance(f, Product)]
                    if prev_prod_indices:
                        current_binary_col = self.n_single
                        for _ in range(self.n_double):
                            for prev_idx in prev_prod_indices:
                                self.W[prev_idx, current_binary_col] = 0.0
                                self.W[prev_idx, current_binary_col + 1] = 0.0
                            current_binary_col += 2

        x = x.to(self.W.device)
        if self.add_bias:
            g = torch.matmul(x, self.W) + self.b
        else:
            g = torch.matmul(x, self.W)
        # print("x:", x.device)
        # print("w:", self.W.device)
        # print("g:", g)
        base_ouput = []
        in_i = 0  # Input index.
        out_i = 0  # Output index.
        # Apply unary functions first; binary operators must follow them.
        while out_i < self.n_single:
            base_ouput.append(self.funcs[out_i](g[:, in_i]))  # g[:, in_i] is the activation-function input.
            in_i += 1
            out_i += 1
        while out_i < self.n_funcs:
            base_ouput.append(self.funcs[out_i](g[:, in_i], g[:, in_i + 1]))
            in_i += 2
            out_i += 1
        base_ouput = torch.stack(base_ouput, dim=1)  # [n_points, n_funcs]
        if self.use_search and self.repeats_candidates:
            start_index = 0
            self.output = []
            for j in range(len(self.repeats_candidates)):
                alpha_weight = F.softmax(self.alpha_list[j], dim=0)
                weighted_output = 0
                for i, neuron_count in enumerate(self.repeats_candidates[j]):
                    partial_output = base_ouput[:, start_index:start_index + neuron_count]
                    weighted_partial = alpha_weight[i] * partial_output
                    padded_output = F.pad(
                        weighted_partial,
                        (0, self.repeats[j] - neuron_count),  
                        mode='constant',
                        value=0
                    )
                    weighted_output += padded_output
                self.output.append(weighted_output)
                # print(self.output)
                start_index += self.repeats[j]
            self.output = torch.cat(self.output, dim=1)
        else:
            self.output = base_ouput

        return self.output

    def get_weight(self):
        """Get masked weights for inspection."""
        return self.W.cpu().detach().numpy()

    def get_bias(self):
        if self.add_bias:
            return self.b.cpu().detach().numpy()
        return None

    def get_weight_tensor(self):
        return self.W.clone()

    def get_alpha(self):
        if self.use_search and self.repeats_candidates:
            alpha_list = []
            for alpha in self.alpha_list:
                alpha_list.append(alpha.cpu().detach().numpy())
            return alpha_list
        else:
            return None




class SymbolicNet(nn.Module):
    def __init__(self, problem_dim, x_dim, depth, repeats_list, funcs, device, initial_weights=None, init_uniform=1,
                 add_bias=False,
                 depth_candidates=[], repeats_candidates=[], use_search=False, kill_small_items=False, threshold=0,
                 enforce_galilean_invariance=False,init_to_ones=False): 
        super(SymbolicNet, self).__init__()
        self.enforce_galilean_invariance = enforce_galilean_invariance
        self.init_uniform = init_uniform
        self.x_dim = x_dim
        self.problem_dim = problem_dim
        self.use_search = use_search
        self.device = device
        self.kill_small_items = kill_small_items
        self.threshold = threshold
        self.funcs = funcs
        self.add_bias = add_bias
        self.init_to_ones = init_to_ones
        if use_search and depth_candidates:
            self.depth = max(depth_candidates)
            self.depth_candidates = depth_candidates
        else:
            self.depth = depth

        if use_search and repeats_candidates:
            repeats_list = []
            for sub_repeats_list in repeats_candidates:
                sub_repeats = max(sub_repeats_list)
                repeats_list.append(sub_repeats)
            max_repeats = sum(repeats_list)
            # print(max_repeats)
            self.repeats_candidates = repeats_candidates
            layer_in_dim = [x_dim] 
            layer_in_dim.extend([max_repeats] * self.depth)  # Extend the list in place again.
            self.add_bias = add_bias  
            self.funcs = funcs  
            self.repeats_list = [repeats_list] * self.depth
            # print("x_dim:", x_dim)
            # print("funcs_per_layer:", funcs)
            # print("layer_in_dim:", layer_in_dim)

        else:
            layer_in_dim = [x_dim]
            for repeats in repeats_list:
                in_dim = sum(repeats)
                layer_in_dim.append(in_dim)
            print(layer_in_dim)
            self.add_bias = add_bias  
            self.funcs = funcs  
            self.repeats_list = repeats_list
            # print("x_dim:", x_dim)
            # print("funcs_per_layer:", funcs)
        print(len(repeats_list))
        assert self.depth == len(
            self.repeats_list), "Length mismatch: repeats_list and depth must have the same length."

        layers = []
        prev_func_list = None

        for i in range(self.depth):
            curr_repeats = self.repeats_list[i] if not (use_search and repeats_candidates) else None
            curr_initial = initial_weights[i] if initial_weights is not None else None
            curr_uniform = init_uniform[i] if isinstance(init_uniform, list) else init_uniform

            layer = SymbolicLayer(
                layer_index=i,
                repeats=curr_repeats,
                funcs=funcs,
                initial_weight=curr_initial,
                init_uniform=curr_uniform,
                in_dim=layer_in_dim[i],
                add_bias=self.add_bias,
                repeats_candidates=repeats_candidates,
                use_search=use_search,
                kill_small_items=self.kill_small_items,
                threshold=self.threshold,
                prev_layer_funcs_list=prev_func_list,
                enforce_galilean_invariance=self.enforce_galilean_invariance,
                init_to_ones=self.init_to_ones
            )

            layers.append(layer)
            prev_func_list = layer.func_list

        self.hidden_layers = nn.Sequential(*layers)

        if use_search and depth_candidates:
            self.beta = nn.Parameter(torch.randn(len(depth_candidates)))

    def forward(self, x):
        device = self.device
        x = x.to(device)
        grid_data = x[:, :self.problem_dim + 1]
        input_data = x[:, self.problem_dim + 1:]

        if self.use_search and self.depth_candidates:
            weighted_beta = F.softmax(self.beta, dim=0)
            output_list = []
            hidden = input_data
            for layer in self.hidden_layers:
                hidden = layer(hidden)
                output_list.append(hidden)

            output = 0.0
            for i in range(len(self.depth_candidates)):
                idx = self.depth_candidates[i] - 1
                output += weighted_beta[i] * output_list[idx]
            output = torch.cat((grid_data, output), dim=1)
        else:
            output = input_data
            for layer in self.hidden_layers:
                output = layer(output)
            output = torch.cat((grid_data, output), dim=1)

        return output

    def get_weights(self):
        """Return the list of weight matrices."""
        # Iterate over the hidden weights, then append the output weight.
        return [self.hidden_layers[i].get_weight() for i in range(self.depth)]

    def get_alphas(self):
        """Return the list of alpha matrices."""
        return [self.hidden_layers[i].get_alpha() for i in range(self.depth)]

    def get_biases(self):
        return [self.hidden_layers[i].get_bias() for i in range(self.depth)]

    def get_weights_tensor(self):
        """Return the list of weight matrices as tensors."""
        return [self.hidden_layers[i].get_weight_tensor() for i in range(self.depth)]

    def get_beta(self):
        """Return the list of beta matrices."""
        if self.use_search and self.depth_candidates:
            return self.beta.cpu().detach().numpy()
        else:
            return None
