"""
Generate a mathematical expression of the symbolic regression network (AKA EQL network) using SymPy. This expression
can be used to pretty-print the expression. SymPy also allows algebraic manipulation of the expression.
The main function is network(...)
There are several filtering functions to simplify expressions, although these are not always needed if the weight matrix
is already pruned.
"""
import pdb

import sympy as sp
import functions
import Derivative


def apply_activation(W, funcs, n_double=0):
    """Given an (n, m) matrix W and a length-m vector of functions, apply the functions to W.

    Arguments:
        W:  (n, m) matrix
        funcs: list of activation functions (SymPy functions)
        n_double: Number of activation functions that take two inputs.

    Returns:
        A one-column SymPy matrix representing the output of the activation functions.
    """
    W = sp.Matrix(W)
    # print(funcs)
    # print(W.shape)
    if n_double == 0:
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i, j] = funcs[j](W[i, j])
    else:
        W_new = W.copy()
        out_size = len(funcs)
        for i in range(W.shape[0]):
            in_j = 0
            out_j = 0
            while out_j < out_size - n_double:
                W_new[i, out_j] = funcs[out_j](W[i, in_j])
                in_j += 1
                out_j += 1
            while out_j < out_size:
                W_new[i, out_j] = funcs[out_j](W[i, in_j], W[i, in_j + 1])
                in_j += 2
                out_j += 1
        for i in range(n_double):
            W_new.col_del(-1)
        W = W_new
    return W


def sym_pp(W_list, funcs, var_names, threshold=0.01, n_double=None, add_bias=False, biases=None):
    """Pretty-print the hidden layers, excluding the final layer, of the symbolic regression network.

    Arguments:
        W_list: list of weight matrices for the hidden layers
        funcs: Dictionary of lambda functions using SymPy. It has the same size as W_list[i][j, :].
        var_names: List of variable names.
        threshold: Threshold for filtering expressions. Set it to 0 to disable filtering.
        n_double: Number of activation functions that take two inputs.

    Returns:
        Simplified SymPy expression.
    """
    vars = []
    for var in var_names:
        if isinstance(var, str):
            vars.append(sp.Symbol(var))
        else:
            vars.append(var)
    if True:
        expr = sp.Matrix(vars).T
        # print(funcs)
        if add_bias and biases is not None:
            assert len(W_list) == len(biases), "The number of biases must be equal to the number of weights."
            for i, (W, b) in enumerate(zip(W_list, biases)):
                W = filter_mat(sp.Matrix(W), threshold=threshold)
                b = filter_mat(sp.Matrix(b), threshold=threshold)
                expr = expr * W + b
                expr = apply_activation(expr, funcs[i], n_double=n_double[i])

        else:
            for i, W in enumerate(W_list):
                W = filter_mat(sp.Matrix(W), threshold=threshold)  # Pruning
                # print(W.shape)
                expr = expr * W
                expr = apply_activation(expr, funcs[i], n_double=n_double[i])
    # except:
    #     pdb.set_trace()
    # expr = expr * W_list[-1]
    return expr


def network(weights, funcs_list, var_names, threshold=0.01, add_bias=False, biases=None):
    """Pretty-print the entire symbolic regression network.

    Arguments:
        weights: list of weight matrices for the entire network
        funcs_list: List of lambda functions using SymPy.
        var_names: List of variable names.
        threshold: Threshold for filtering expressions. Set it to 0 to disable filtering.

    Returns:
        Simplified SymPy expression."""
    n_double = [functions.count_double(funcs_per_layer) for funcs_per_layer in funcs_list]
    # Translate operators to SymPy operators.
    sp_funcs = []
    for funcs in funcs_list:
        sp_value_per_layer = [func.sp for func in funcs]
        sp_funcs.append(sp_value_per_layer)

    if add_bias and biases is not None:
        assert len(weights) == len(biases), "The number of biases must be equal to the number of weights - 1."
        expr = sym_pp(weights, sp_funcs, var_names, threshold=threshold, n_double=n_double, add_bias=add_bias,
                      biases=biases)
    else:
        expr = sym_pp(weights, sp_funcs, var_names, threshold=threshold, n_double=n_double, add_bias=add_bias)
    return expr


def Integrate_net(weights, diff_list, expr, threshold=0.01, add_bias=False, biases=None):
    if add_bias and biases is not None:
        assert len(weights) == len(biases), "The number of biases must be equal to the number of weights."
        for i, (W, b) in enumerate(zip(weights, biases)):
            W = filter_mat(sp.Matrix(W), threshold=threshold)
            b = filter_mat(sp.Matrix(b), threshold=threshold)
            expr = expr * W + b
            if i == 0:
                for j in range(len(diff_list)):
                    diff = diff_list[j].__str__()
                    expr[j] = sp.Function(diff)(expr[j])

    else:
        for i, W in enumerate(weights):
            W = filter_mat(sp.Matrix(W), threshold=threshold)  # Pruning
            expr = expr * W
            if i == 0:
                for j in range(len(diff_list)):
                    diff = diff_list[j].__str__()
                    expr[j] = sp.Function(diff)(expr[j])
    return expr


def filter_mat(mat, threshold=0.01):
    """Remove elements of a matrix below a threshold."""
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if abs(mat[i, j]) < threshold:
                mat[i, j] = 0
    return mat


def filter_expr(expr, threshold=0.01):
    """Remove additive terms with coefficients below the threshold.
    TODO: Make more robust. This does not work in all cases."""
    expr_new = sp.Integer(0)
    for arg in expr.args:
        if arg.is_constant() and abs(arg) > threshold:  # Use a simple check for numeric values.
            expr_new = expr_new + arg
        elif not arg.is_constant() and abs(arg.args[0]) > threshold:
            expr_new = expr_new + arg
    return expr_new


def filter_expr2(expr, threshold=0.01):
    """Set all constants below the threshold to zero.
    TODO: Test"""
    for a in sp.preorder_traversal(expr):
        if isinstance(a, sp.Float) and a < threshold:
            expr = expr.subs(a, 0)
    return expr


def print_model(weights_sym_list, funcs_list, var_names, weights_int_list, diff_list, threshold=0.01, add_bias=False,
                biases=None, int_biases=None):
    expr = network(weights_sym_list, funcs_list, var_names, threshold=threshold, add_bias=add_bias, biases=biases)
    model = Integrate_net(weights_int_list, diff_list, expr, add_bias=add_bias, biases=int_biases)
    return model
