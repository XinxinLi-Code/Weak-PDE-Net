import Derivative
import numpy as np
import torch
import sympy as sp
from scipy.optimize import least_squares

_LAMBDA_CACHE = {}
_SYMBOL_CACHE = {}


def _var_symbols(var_names):
    key = tuple(var_names)
    if key not in _SYMBOL_CACHE:
        if len(key) == 1:
            _SYMBOL_CACHE[key] = (sp.symbols(key[0]),)
        else:
            _SYMBOL_CACHE[key] = tuple(sp.symbols(" ".join(key)))
    return _SYMBOL_CACHE[key]


def _lambdified_fu(Fu, var_names):
    key = (tuple(var_names), sp.srepr(Fu))
    func = _LAMBDA_CACHE.get(key)
    if func is None:
        symbols = _var_symbols(var_names)
        args = symbols[0] if len(symbols) == 1 else symbols
        func = sp.lambdify(args, Fu, "numpy")
        _LAMBDA_CACHE[key] = func
    return func


def _input_arrays(input_data, var_names):
    if isinstance(input_data, torch.Tensor):
        input_array = input_data.detach().cpu().numpy()
        dtype = input_data.dtype
    else:
        input_array = np.asarray(input_data)
        if np.issubdtype(input_array.dtype, np.floating):
            dtype = torch.from_numpy(np.empty((), dtype=input_array.dtype)).dtype
        else:
            dtype = torch.float32
    num_vars = len(var_names)
    values = tuple(input_array[:, -num_vars + i] for i in range(num_vars))
    return input_array.shape[0], values, dtype


def _weight_derivative_coords(w, D):
    key = tuple(D.Encoding)
    cache = getattr(w, "_weak_derivative_cache", None)
    if cache is None:
        cache = {}
        setattr(w, "_weak_derivative_cache", cache)

    cached = cache.get(key)
    if cached is not None and cached.device == w.Supported_Coords.device:
        return cached

    if not D.__str__():
        coords = w(w.Supported_Coords)
    else:
        w.Add_Derivative(D)
        coords = w.Derivatives[key]
    coords = coords.detach()
    cache[key] = coords
    return coords


def compute_lhs(Random_Weight_Functions_Lists, problem_dim, u_data, lhs_type=1):
    LHS_list = []
    arr = np.zeros(problem_dim + 1)
    arr[0] = lhs_type
    D_0 = Derivative.Derivative(arr)
    if len(Random_Weight_Functions_Lists) == 0:
        return torch.empty(0, dtype=u_data.dtype if isinstance(u_data, torch.Tensor) else torch.float32)
    device = Random_Weight_Functions_Lists[0].Supported_Indices.device
    u_data = u_data.to(device)
    for i in range(len(Random_Weight_Functions_Lists)):
        w = Random_Weight_Functions_Lists[i]
        Supported_Indices = w.Supported_Indices
        D_0_Coords = _weight_derivative_coords(w, D_0)
        F_0_supported = u_data[Supported_Indices]
       
        Sum_k_0 = torch.sum(torch.multiply(F_0_supported, D_0_Coords))
        Integrals_0 = w.V * ((-1.) ** (D_0.Order)) * Sum_k_0
        LHS = Integrals_0
        LHS_list.append(LHS)
    LHS = torch.stack(LHS_list).detach().to(device)
    return LHS



def compute_rhs(Random_Weight_Functions_Lists, Fu_list, derivative_list, input_data, var_names):
    RHS_list = []
    if len(Random_Weight_Functions_Lists) == 0:
        return torch.empty((0, len(Fu_list)), dtype=torch.float32)
    device = Random_Weight_Functions_Lists[0].Supported_Indices.device
    num_points, var_values, input_dtype = _input_arrays(input_data, var_names)
    fu_value_cache = {}

    def evaluate_fu(j):
        if j not in fu_value_cache:
            f = _lambdified_fu(Fu_list[j], var_names)
            if len(var_values) == 1:
                values = f(var_values[0])
            else:
                values = f(*var_values)
            values = torch.as_tensor(values, dtype=input_dtype)
            if values.ndim == 0:
                values = values.expand(num_points)
            fu_value_cache[j] = values.to(device)
        return fu_value_cache[j]

    for i in range(len(Random_Weight_Functions_Lists)):
        w = Random_Weight_Functions_Lists[i]
        Supported_Indices = w.Supported_Indices
        RHS_sublist = []
        for j in range(len(Fu_list)):
            D = derivative_list[j]
            D_Coords = _weight_derivative_coords(w, D)
            F = evaluate_fu(j)
            F_supported = F[Supported_Indices]
            Sum_k = torch.sum(torch.multiply(F_supported, D_Coords))
            Integrals = w.V * ((-1.) ** (D.Order)) * Sum_k
            RHS_sublist.append(Integrals)
        RHS_list.append(RHS_sublist)
    RHS = torch.stack([torch.stack(row) for row in RHS_list]).detach().to(device)

    return RHS





def print_result(coff_list, Fu_list, derivative_list):
    expr = ""
    for i in range(len(coff_list)):
        derivative = derivative_list[i]
        sub_expr = str(coff_list[i]) + "*" + str(derivative) + '(' + str(Fu_list[i]) + ')'
        if i != 0 and coff_list[i] >= 0:
            expr += '+' + sub_expr
        else:
            expr += sub_expr
    return expr


def regression(A, b, coff_list, Fu_list, derivative_list,print=True,bounds=False):
    if isinstance(A, torch.Tensor):
        A = A.cpu().detach().numpy()
    if isinstance(b, torch.Tensor):
        b = b.cpu().detach().numpy()
    b = np.asarray(b).reshape(-1)

    def residuals(params, A, b):
        return A @ params - b

    x0 = np.asarray([float(c) for c in coff_list], dtype=np.float64).reshape(-1)
    x0[x0 == 0] = 1  
    if bounds:
        x0 = np.asarray([float(c) for c in coff_list], dtype=np.float64).reshape(-1)
        if x0.size != A.shape[1]:
            x0 = np.ones(A.shape[1], dtype=np.float64)
        eps = 1e-8
        x0 = np.nan_to_num(x0, nan=0.0, posinf=1.0 - eps, neginf=-1.0 + eps)
        x0 = np.clip(x0, -1.0 + eps, 1.0 - eps)
        result = least_squares(residuals, x0, args=(A, b), bounds=(-1, 1))
    else:
        x0 = np.asarray([float(c) for c in coff_list], dtype=np.float64).reshape(-1)
        if x0.size != A.shape[1]:
            x0 = np.ones(A.shape[1], dtype=np.float64)
        result = least_squares(residuals, x0, args=(A, b))

    coff_list = result.x
    if print:
        expr = print_result(coff_list, Fu_list,derivative_list)
    else:
        expr = None
    return result, coff_list, expr




def mstls_regression(
    A, b,
    Fu_list, derivative_list,
    print_result_flag=True,
    lambda_list=None,
):
    """
    Strict Python equivalent of:
    wsindy_pde_RGLS_seq2 (Messenger & Bortz)
    
    Returns: min_loss, final_coeffs, expr
    """

    # ------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------
    if isinstance(A, torch.Tensor):
        A = A.cpu().detach().numpy()
    if isinstance(b, torch.Tensor):
        b = b.cpu().detach().numpy()

    if b.ndim == 1:
        b = b.reshape(-1, 1)

    N, m = A.shape

    # ------------------------------------------------------------
    # 2. M-scale (Section 4.3)
    #   MATLAB: G = Theta ./ M_scale
    # ------------------------------------------------------------
    M_scale = np.linalg.norm(A, axis=0)
    M_scale[M_scale == 0] = 1.0

    G = A / M_scale

    # ------------------------------------------------------------
    # 3. Least-squares reference
    #   MATLAB: W_ls = G \ b
    # ------------------------------------------------------------
    W_ls = np.linalg.lstsq(G, b, rcond=None)[0]
    GW_ls = np.linalg.norm(G @ W_ls)

    # ------------------------------------------------------------
    # 4. Adaptive λ range
    # ------------------------------------------------------------
    if lambda_list is None:
        lam_max = min(
            np.max(np.abs(G.T @ b).flatten() / (np.linalg.norm(G, axis=0) ** 2)),
            1.0
        )

        lam_min = (
            np.linalg.norm(G @ W_ls)
            / m
            / np.max(np.linalg.norm(G, axis=0))
        )

        lambda_list = np.logspace(
            np.log10(lam_min),
            np.log10(lam_max),
            50
        )

    # ------------------------------------------------------------
    # 5. STLS
    # ------------------------------------------------------------
    def sparsify_dynamics(G, b, lam, max_iter=25):
        w = np.linalg.lstsq(G, b, rcond=None)[0].flatten()

        for _ in range(max_iter):
            small = np.abs(w) < lam

            if not np.any(~small):
                w[:] = 0.0
                break

            G_active = G[:, ~small]
            w_active = np.linalg.lstsq(G_active, b, rcond=None)[0].flatten()

            w_new = np.zeros_like(w)
            w_new[~small] = w_active

            if np.allclose(w, w_new):
                break

            w = w_new

        return w

    # ------------------------------------------------------------
    # 6. Weak SINDy loss
    # ------------------------------------------------------------
    alpha = 0.5
    best_w = W_ls.flatten()
    best_lambda = lambda_list[0]
    min_loss = np.inf

    for lam in lambda_list:
        w = sparsify_dynamics(G, b, lam)

        # Compute the projection cost
        proj_cost = (
            2 * alpha
            * np.linalg.norm(G @ (w.reshape(-1, 1) - W_ls))
            / GW_ls
        )

        # Compute the sparsity cost
        sparsity_cost = (
            2 * (1 - alpha)
            * np.count_nonzero(w)
            / m
        )

        loss = proj_cost + sparsity_cost

        # Update the best model
        if loss < min_loss:
            min_loss = loss
            best_w = w.copy()
            best_lambda = lam

    # ------------------------------------------------------------
    # 7. Undo M-scaling (MATLAB: W .* M)
    # ------------------------------------------------------------
    final_coeffs = best_w / M_scale

    # ------------------------------------------------------------
    # 8. Output
    # ------------------------------------------------------------
    expr = None
    if print_result_flag:
        try:
            expr = print_result(final_coeffs, Fu_list, derivative_list)
        except NameError:
            print(f"Optimal lambda: {best_lambda:.4e}")
            print("Identified coefficients:")
            print(final_coeffs)
            print(f"Final Loss: {min_loss:.6e}")  # Print the final loss as well

    # Return min_loss
    return min_loss, final_coeffs, expr
