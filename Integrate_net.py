import Derivative
import torch.nn as nn
import numpy as np
import torch

def Integrate(w, F_u, grid_size, grid_data, derivative, type='Riemann'):
    if type == 'Riemann':
        result = Riemman_Integrate(w, F_u, grid_data, derivative)
    return result


def Riemman_Integrate(w, F_u, grid_data, derivative):
    """
    ``w`` is the weight function. ``F_u`` contains the constructed function
    values and has shape ``(batch_size,)``. ``grid_data`` contains grid
    coordinates aligned with ``F_u``. ``derivative`` is the derivative
    operator, denoted by alpha(m).
    """

    # First, fetch the coordinates in the support of w.
    Supported_Coords = w.Supported_Coords
    matched_indices = w.Supported_Indices
    # print("Supported_Coords:",Supported_Coords.device)
    # print("grid_data:",grid_data.device)
    grid_data = grid_data.to(Supported_Coords.device)
    F_u = F_u.to(Supported_Coords.device)
    # Next, evaluate U on these coordinates.
    # print("F_u:",F_u.shape)
    F_u_supported = F_u[matched_indices]
   
    ############################################################################
    # Second, compute the right-hand-side term integrals.
    if not derivative.__str__():
        D_w_Coords = w(Supported_Coords)
    else:
        w.Add_Derivative(derivative)
        D_w_Coords = w.Derivatives[tuple(derivative.Encoding)]
    
    # Sum the element-wise product of F_k_U_Coords and D_k(w). This yields
    # the summation \sum_{i = 1}^{N} D_k w(X_i) F_k(U(X_i)).
    Sum_k = torch.sum(torch.multiply(F_u_supported, D_w_Coords))
    # print("Sum_k:",Sum_k)
    # Finally, multiply the sum by (-1)^{|D_k|}V to obtain the integral approximation.
    Integrals = w.V * ((-1.) ** (derivative.Order)) * Sum_k

    return Integrals


class IntegrateNet(nn.Module):

    def __init__(self, problem_dim, max_order, input_dim, init_uniform, grad_all=False, initial_weights=None, add_bias=False,init_to_ones=False):
        super().__init__()
        """
        problem_dim: The dimension of the partial differential equation (PDE).
        max_order: The maximum order of the differential operator. For example, alpha = [1, 0, t] represents a total order of 1.
        """
        self.problem_dim = problem_dim
        self.max_order = max_order
        self.input_dim = input_dim
        self.diff_list = self.init_diff()
        self.diff_len = len(self.diff_list)
        self.output_dim = 1
        self.initial_weights = initial_weights
        self.init_uniform = init_uniform
        self.W = None  # Weight matrix.
        self.add_bias = add_bias
        self.init_to_ones = init_to_ones
        if self.initial_weights is not None:  # Use the provided initial weight.
            initial_weight = initial_weights[0]
            if isinstance(initial_weight, np.ndarray):
                self.initial_weight = torch.from_numpy(initial_weight).float()
            else:
                self.initial_weight = initial_weight.clone().detach()
            self.W = nn.Parameter(self.initial_weight)
        else:
            self.W = nn.Parameter(torch.empty(input_dim, self.diff_len))
            if self.init_to_ones:
                nn.init.constant_(self.W, 1.0)
            else:
                nn.init.uniform_(self.W, a=-self.init_uniform, b=self.init_uniform)

            if add_bias:
                self.b = nn.Parameter(torch.empty(1, self.diff_len))
                nn.init.uniform_(self.b, a=-1, b=1)

        self.Xi = None  # Weight matrix.
        if self.initial_weights is not None:  # Use the provided initial weight.
            initial_weight = self.initial_weights[1]
            if isinstance(initial_weight, np.ndarray):
                self.initial_weight = torch.from_numpy(initial_weight).float()
            else:
                self.initial_weight = initial_weight.clone().detach()
            self.Xi = nn.Parameter(self.initial_weight)
        else:
            self.Xi = nn.Parameter(torch.empty(self.diff_len, self.output_dim))
            if self.init_to_ones:
                nn.init.constant_(self.Xi, 1.0)
            else:
                nn.init.uniform_(self.Xi, a=-self.init_uniform, b=self.init_uniform)
            if grad_all:
                with torch.no_grad():
                    self.Xi[0] = 0.0
                self.Xi.register_hook(lambda grad: grad.clone().index_fill_(
                    0, torch.tensor(0, device=grad.device), 0))

            if add_bias and not hasattr(self, "b"):
                self.b = nn.Parameter(torch.empty(1, self.diff_len))
                nn.init.uniform_(self.b, a=-1, b=1)
        if add_bias and not hasattr(self, "b"):
            self.b = nn.Parameter(torch.empty(1, self.diff_len))
            nn.init.uniform_(self.b, a=-1, b=1)
        # print(self.Xi)
    def forward(self, x, weighted_function, grid_size, int_type):
        device = self.W.device
        x = x.to(device)
        weighted_function = weighted_function.to(device)
        grid_data = x[:, :self.problem_dim+1]
        # print("grid_data!:",grid_data.device)
        input_data = x[:, self.problem_dim+1:]
        if self.add_bias:
            g = torch.matmul(input_data, self.W) + self.b
        else:
            g = torch.matmul(input_data, self.W)
        output = 0.0
        for k, derivative in enumerate(self.diff_list):
            F_u = g[:, k]
            Integrate_k = Integrate(weighted_function, F_u, grid_size, grid_data, derivative, int_type)
            # print("Integrate_k", Integrate_k)
            # print("output", output)
            # print("self.Xi[k]:", self.Xi[k])
            output += Integrate_k * self.Xi[k]
        return output

    def init_diff(self):
        if self.problem_dim == 1:
            diff_list = []
            for order in range(self.max_order + 1):
                array = np.array([0, order])
                diff = Derivative.Derivative(array)
                diff_list.append(diff)
        elif self.problem_dim == 2:
            diff_list = []
            for a in range(self.max_order + 1):  
                for b in range(self.max_order + 1 - a):  
                    array = np.array([0, b, a])
                    diff = Derivative.Derivative(array)
                    diff_list.append(diff)
        else:
            diff_list = []
            for a in range(self.max_order + 1):  
                for b in range(self.max_order + 1 - a):  
                    for c in range(self.max_order + 1 - a - b):
                        array = np.array([0, c, b, a])
                        diff = Derivative.Derivative(array)
                        diff_list.append(diff)

        return diff_list


if __name__ == "__main__":
    integrate_net = IntegrateNet(problem_dim=1, max_order=3, input_dim=1, init_uniform=2, initial_weights=None)
    print(integrate_net.diff_list)
    for f in integrate_net.diff_list:
        print(f.__str__())
    from Weight_Function import Weight_Function

    X_0 = torch.tensor([0.5, 0.5])

    r = 0.3

    x_vals = torch.linspace(0, 1, 10)
    y_vals = torch.linspace(0, 1, 10)

    grid_x, grid_y = torch.meshgrid(x_vals, y_vals, indexing="ij")
    F_u = grid_x * grid_y
    print(F_u.shape)
    x = torch.stack([grid_x.flatten(), grid_y.flatten(), F_u.flatten()], dim=1)
    Coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # shape: [10000, 2]
    print(Coords.shape)
    V = (x_vals[1] - x_vals[0]) * (y_vals[1] - y_vals[0])

    w_0 = Weight_Function(X_0=X_0, r=r, Coords=Coords, V=V.item())
    y = integrate_net(x, w_0)

    print(y)
