import sympy as sp
import re




def is_zero_coeff(coeff_str):
    return float(coeff_str) == 0.0

def replace_coeff(expr, operator, new_coeff):
    """
    Replace the coefficient before the operator in expr with new_coeff.
    """
    pattern = rf'([+-]?\s*\d*\.?\d+)\s*\*\s*{re.escape(operator)}'
    return re.sub(pattern, f'{new_coeff}*{operator}', expr, count=1)

def parse_derivatives(problem_dim, expr_str):
    # Define the allowed variables based on the dimension.
    if problem_dim == 1:
        allowed_vars = ['x']
    elif problem_dim == 2:
        allowed_vars = ['x', 'y']
    elif problem_dim == 3:
        allowed_vars = ['x', 'y', 'z']
    else:
        raise ValueError("problem_dim must be 1, 2, or 3")
    
    # Helper function for finding the matching closing parenthesis.
    def find_matching_paren(s, start):
        """Find the position of the matching closing parenthesis."""
        count = 1
        pos = start
        while pos < len(s) and count > 0:
            pos += 1
            if pos >= len(s):
                break
            if s[pos] == '(':
                count += 1
            elif s[pos] == ')':
                count -= 1
        return pos if count == 0 else -1
    
    # Step 0: Preprocessing - combine consecutive derivative notations
    expr_str = re.sub(r'D_([xyz])\s+D_([xyz])', r'D_\1\2', expr_str)
    
    # Step 1: Convert exponent notation to repeated variables (e.g., D_x^3(u) -> D_xxx(u))
    def replace_power(match):
        var = match.group(1)
        order = int(match.group(2))
        expr_start = match.end()
        
        # Find the matching parenthesis for the expression
        if expr_start < len(expr_str) and expr_str[expr_start] == '(':
            expr_end = find_matching_paren(expr_str, expr_start)
            if expr_end == -1:
                return match.group(0)  # Unmatched parentheses, return original
            
            inner_expr = expr_str[expr_start+1:expr_end]
            if var in allowed_vars:
                return f'D_{var * order}({inner_expr})'
        
        return match.group(0)  # Return original if issues found
    
    # Process exponent notation derivatives
    power_pattern = re.compile(r'D_([xyz])\^(\d+)\s*\(')
    new_expr = []
    pos = 0
    while pos < len(expr_str):
        match = power_pattern.search(expr_str, pos)
        if not match:
            new_expr.append(expr_str[pos:])
            break
        
        # Get the matched part and the expression inside parentheses
        start_idx = match.start()
        expr_start = match.end() - 1  # Position of '('
        expr_end = find_matching_paren(expr_str, expr_start)
        
        if expr_end == -1:
            new_expr.append(expr_str[pos:start_idx + len(match.group(0))])
            pos = start_idx + len(match.group(0))
            continue
        
        # Extract the inner expression
        inner_expr = expr_str[expr_start+1:expr_end]
        
        # Construct replacement
        var = match.group(1)
        order = int(match.group(2))
        if var in allowed_vars:
            replacement = f'D_{var * order}({inner_expr})'
        else:
            replacement = expr_str[start_idx:expr_end+1]
        
        new_expr.append(expr_str[pos:start_idx])
        new_expr.append(replacement)
        pos = expr_end + 1
    
    expr_str = ''.join(new_expr)
    
    # Step 2: Convert derivative notations to Derivative(...) format
    new_expr = []
    pos = 0
    deriv_pattern = re.compile(r'D_([xyz]+)\s*\(')
    
    while pos < len(expr_str):
        match = deriv_pattern.search(expr_str, pos)
        if not match:
            new_expr.append(expr_str[pos:])
            break
        
        # Get derivative variables and expression
        deriv_vars = match.group(1)
        start_idx = match.start()
        expr_start = match.end() - 1  # Position of '('
        expr_end = find_matching_paren(expr_str, expr_start)
        
        if expr_end == -1:
            new_expr.append(expr_str[pos:start_idx + len(match.group(0))])
            pos = start_idx + len(match.group(0))
            continue
        
        # Extract inner expression
        inner_expr = expr_str[expr_start+1:expr_end]
        
        # Validate variables and construct replacement
        if all(v in allowed_vars for v in deriv_vars):
            # Create derivative arguments string
            args = ', '.join(deriv_vars)
            replacement = f'Derivative({inner_expr}, {args})'
        else:
            replacement = expr_str[start_idx:expr_end+1]
        
        new_expr.append(expr_str[pos:start_idx])
        new_expr.append(replacement)
        pos = expr_end + 1
    
    expr_str = ''.join(new_expr)
    
    # Step 3: Handle basic first-order derivatives (e.g., D_x(u))
    new_expr = []
    pos = 0
    first_order_pattern = re.compile(r'D_([xyz])\s*\(')
    
    while pos < len(expr_str):
        match = first_order_pattern.search(expr_str, pos)
        if not match:
            new_expr.append(expr_str[pos:])
            break
        
        # Get derivative variable and expression
        var = match.group(1)
        start_idx = match.start()
        expr_start = match.end() - 1  # Position of '('
        expr_end = find_matching_paren(expr_str, expr_start)
        
        if expr_end == -1:
            new_expr.append(expr_str[pos:start_idx + len(match.group(0))])
            pos = start_idx + len(match.group(0))
            continue
        
        # Extract inner expression
        inner_expr = expr_str[expr_start+1:expr_end]
        
        # Validate variable and construct replacement
        if var in allowed_vars:
            replacement = f'Derivative({inner_expr}, {var})'
        else:
            replacement = expr_str[start_idx:expr_end+1]
        
        new_expr.append(expr_str[pos:start_idx])
        new_expr.append(replacement)
        pos = expr_end + 1
    
    return ''.join(new_expr)

def simplify_trig_coefficients(expr_str):
    """
    Simplify coefficients in trigonometric functions:
    sin(k*u) -> sin(±u), cos(k*u) -> cos(±u).
    """
    # Match the coefficient multiplying u inside sin or cos.
    pattern = r'(sin|cos)\(\s*([+-]?\s*\d*\.?\d+)\s*\*\s*u\s*\)'
    
    def replace_trig(match):
        trig_func = match.group(1)  # Either sin or cos.
        coeff_str = match.group(2).replace(' ', '')  # Remove spaces.
        sign = '-' if coeff_str.startswith('-') else ''
        return f'{trig_func}({sign}u)'
        
    return re.sub(pattern, replace_trig, expr_str)

def replace_str_with_fun(expr_str):
    expr_str = re.sub(r'\bu\b(?!\s*\()', 'u(x)', expr_str)
    expr_str = re.sub(r'\bv\b(?!\s*\()', 'v(x)', expr_str)
    expr_str = re.sub(r'\bw\b(?!\s*\()', 'w(x)', expr_str)
    return expr_str

def extract_derivative_info(term, dim):
    """
    Extract derivative information from a term.
    Returns a pair containing the coefficient and a derivative-information object.
    """
    # Create a derivative-information object.
    class DerivInfo:
        def __init__(self, order_type, expr):
            self.order_type = order_type  # Tuple describing the derivative type.
            self.expr = expr             # Expression being differentiated.
    
    # Handle the zeroth-order case.
    if not any(isinstance(arg, sp.Derivative) for arg in term.args if isinstance(term, sp.Mul)):
        return term, None
    
    # Extract the derivative object.
    deriv_obj = None
    coeff = 1
    for arg in term.args:
        if isinstance(arg, sp.Derivative):
            deriv_obj = arg
        elif arg.is_number:  # Numerical coefficient.
            coeff *= arg
    
    if deriv_obj is None:
        return term, None
    
    # Determine the derivative type.
    order_list = [0] * dim
    # Define the variable order as [x, y, z].
    variables = [sp.Symbol('x'), sp.Symbol('y'), sp.Symbol('z')][:dim]
    
    # Count the derivative order for each variable.
    for var in deriv_obj.variables:
        try:
            idx = variables.index(var)
            order_list[idx] += 1
        except ValueError:
            # Ignore derivatives with respect to variables outside the list.
            continue
    
    return coeff, DerivInfo(tuple(order_list), deriv_obj.expr)

def simplify_constants(expr):
    if expr.is_Number:
        return expr
    elif expr.is_Add:
        return sp.Add(*(simplify_constants(arg) for arg in expr.args))
    elif expr.is_Mul:
        args = [simplify_constants(arg) for arg in expr.args]
        numbers = [arg for arg in args if arg.is_Number]
        non_numbers = [arg for arg in args if not arg.is_Number]
        if numbers:
            num_val = sp.prod(numbers)
            if non_numbers:
                return num_val * sp.Mul(*non_numbers)
            return num_val
        return sp.Mul(*args)
    elif expr.is_Pow:
        base = simplify_constants(expr.base)
        exp = simplify_constants(expr.exp)
        return sp.Pow(base, exp)
    return expr


def decompose_expression(expr, dim=1):
    
    # Define derivative types according to the dimension.
    if dim == 1:
        # 1D: (x_order,)
        order_types = {(i,) for i in range(0, 4)}  # Orders zero through three.
    elif dim == 2:
        # 2D: (x_order, y_order)
        # Include all combinations up to second-order derivatives.
        order_types = {(i, j) for i in range(0, 3) for j in range(0, 3)}
    elif dim == 3:
        # 3D: (x_order, y_order, z_order)
        # Include all combinations up to second-order derivatives.
        order_types = {(i, j, k) for i in range(0, 3) for j in range(0, 3) for k in range(0, 3)}
    else:
        raise ValueError("The dimension must be 1, 2, or 3.")

    # Initialize the result dictionary.
    terms_by_type = {order_type: [] for order_type in order_types}
    terms_by_type[tuple([0]*dim)] = []  # Ensure that the zeroth-order entry exists.

    # Decompose the expression into terms.
    if expr.is_Add:
        terms = expr.args
    else:
        terms = [expr]

    # Process each term.
    for term in terms:
        # Extract derivative information.
        coeff, deriv_info = extract_derivative_info(term,dim)
        
        # Obtain the derivative-type tuple.
        if deriv_info is None:
            order_type = tuple([0]*dim)  # No derivatives.
            expr_part = coeff
        else:
            order_type = deriv_info.order_type  # For example, (1, 0) or (0, 1, 0).
            expr_part = coeff * deriv_info.expr
        
        # Store the term under the corresponding type.
        if order_type in terms_by_type:
            terms_by_type[order_type].append(expr_part)
        else:
            # Add unlisted higher-order derivatives dynamically.
            terms_by_type[order_type] = [expr_part]

    # Combine and simplify the terms.
    for order_type in list(terms_by_type.keys()):
        if terms_by_type[order_type]:  # Process only nonempty categories.
            total_expr = sp.Add(*terms_by_type[order_type])
            
            # Handle zeroth-order terms separately.
            if all(o == 0 for o in order_type):
                total_expr = sp.expand(total_expr)
            else:
                total_expr = simplify_constants(total_expr)
            
            terms_by_type[order_type] = total_expr
        else:
            # Remove empty categories.
            del terms_by_type[order_type]

    return terms_by_type

def split_terms(expr):
    """
    Input:
      0.1*(u**2)-0.3*(u*v)+0.5*D_x^2 (v)
    Output:
      [(0.1, 'u**2'), (-0.3, 'u*v'), (0.5, 'D_x^2 (v)')]
    """
    expr = expr.replace(' ', '')
    if expr[0] not in '+-':
        expr = '+' + expr

    pattern = r'([+-]\d*\.?\d+(?:e[+-]?\d+)?)(?:\*\(([^)]+)\)|\*([^+-]+))'
    matches = re.findall(pattern, expr)

    terms = []
    for coeff, poly, other in matches:
        term = poly if poly else other
        terms.append((float(coeff), term))

    return terms



_U_SYMBOL = sp.Symbol('u')
_V_SYMBOL = sp.Symbol('v')
_W_SYMBOL = sp.Symbol('w')
_SYMPY_LOCALS = {'u': _U_SYMBOL, 'v': _V_SYMBOL, 'w': _W_SYMBOL}


def _parse_derivative_term(term):
    compact = term.replace(' ', '')
    if not compact.startswith('D_'):
        return None
    open_idx = compact.find('(')
    if open_idx < 0 or not compact.endswith(')'):
        return None
    return compact[:open_idx], compact[open_idx + 1:-1]


def _sympify_uv_expr(expr):
    return sp.sympify(expr.replace('^', '**'), locals=_SYMPY_LOCALS)


def _format_sympy_expr(expr):
    return sp.sstr(sp.expand(expr)).replace(' ', '')


def _split_signed_expr(expr):
    expr = sp.expand(expr)
    coeff, body = expr.as_coeff_Mul()
    return _format_sympy_expr(body), float(coeff)


def canonicalize_uv_term(term):
    parsed = _parse_derivative_term(term)
    if parsed is not None:
        operator, inner = parsed
        try:
            return f"{operator}({_format_sympy_expr(_sympify_uv_expr(inner))})"
        except Exception:
            return term.replace(' ', '')

    try:
        return _format_sympy_expr(_sympify_uv_expr(term))
    except Exception:
        return term.replace(' ', '')


def u1_equivariant_partner(term, negate=True):
    """
    Generate the paired candidate under the U(1)-equivariant coupling rule.

    For a real-component candidate T(u, v), the imaginary-component partner is
    -T(-v, u). Set negate=False for the inverse map from the imaginary
    component back to the real component, T(-v, u).
    """
    sign = -1 if negate else 1
    parsed = _parse_derivative_term(term)

    if parsed is not None:
        operator, inner = parsed
        rotated_inner = _sympify_uv_expr(inner).subs(
            {_U_SYMBOL: -_V_SYMBOL, _V_SYMBOL: _U_SYMBOL},
            simultaneous=True,
        )
        body, coeff = _split_signed_expr(sign * rotated_inner)
        return f"{operator}({body})", coeff

    rotated = _sympify_uv_expr(term).subs(
        {_U_SYMBOL: -_V_SYMBOL, _V_SYMBOL: _U_SYMBOL},
        simultaneous=True,
    )
    return _split_signed_expr(sign * rotated)


def terms_to_dict(terms):
    d = {}
    for c, t in terms:
        t = canonicalize_uv_term(t)
        d[t] = d.get(t, 0.0) + c
    return d



def dict_to_expr(d):
    parts = []
    for term, coeff in sorted(d.items()):
        if abs(coeff) < 1e-14:
            continue
        parts.append(f"{coeff:+.16g}*{term}")
    expr = ''.join(parts)
    return expr[1:] if expr.startswith('+') else expr


def skew_symmetric_completion(expr_u, expr_v, tol=1e-12):
    """
    Complete a two-component complex system using the U(1)-equivariant rule.

    If the real component contains c*T(u, v), the imaginary component should
    contain c*(-T(-v, u)). This is the antisymmetric candidate coupling used
    in the paper and is broader than the old NLS-only u/v swap rule.
    """
    du = terms_to_dict(split_terms(expr_u))
    dv = terms_to_dict(split_terms(expr_v))

    all_real_terms = set(du)
    for imag_term in dv:
        real_term, _ = u1_equivariant_partner(imag_term, negate=False)
        all_real_terms.add(real_term)

    new_du = {}
    new_dv = {}

    for real_term in sorted(all_real_terms):
        imag_term, sign = u1_equivariant_partner(real_term)
        cu = du.get(real_term, 0.0)
        cv = dv.get(imag_term, 0.0)

        if abs(cu) < tol and abs(cv) < tol:
            continue

        if abs(cu) < tol:
            cu = sign * cv
        elif abs(cv) < tol:
            cv = sign * cu
        else:
            avg = 0.5 * (cu + sign * cv)
            cu = avg
            cv = sign * avg

        new_du[real_term] = new_du.get(real_term, 0.0) + cu
        new_dv[imag_term] = new_dv.get(imag_term, 0.0) + cv

    return dict_to_expr(new_du), dict_to_expr(new_dv)


def extract_coeff(expr, operator):
    """
    Extract the coefficient before the operator in expr.
    Return the coefficient as a signed string, such as '+0.5' or '-2.0'.
    """
    pattern = rf'([+-]?\s*\d*\.?\d+)\s*\*\s*{re.escape(operator)}'
    match = re.search(pattern, expr)
    if match:
        return match.group(1).replace(' ', '')
    return None


def complete_symmetric_term_pair(expr, first_term, second_term, zero_fallback=None):
    """Complete a symmetric term pair and synchronize zero coefficients."""
    first_present = first_term in expr
    second_present = second_term in expr

    if not first_present and not second_present:
        return expr

    first_coeff = extract_coeff(expr, first_term) if first_present else None
    second_coeff = extract_coeff(expr, second_term) if second_present else None

    if not first_present:
        if second_coeff is None:
            return expr
        if zero_fallback is not None and is_zero_coeff(second_coeff):
            return expr + f'+{zero_fallback}*{first_term}+{zero_fallback}*{second_term}'
        return expr + f'+{second_coeff}*{first_term}'

    if not second_present:
        if first_coeff is None:
            return expr
        if zero_fallback is not None and is_zero_coeff(first_coeff):
            return expr + f'+{zero_fallback}*{first_term}+{zero_fallback}*{second_term}'
        return expr + f'+{first_coeff}*{second_term}'

    if first_coeff is None or second_coeff is None:
        return expr
    if is_zero_coeff(first_coeff) and not is_zero_coeff(second_coeff):
        return replace_coeff(expr, first_term, second_coeff)
    if is_zero_coeff(second_coeff) and not is_zero_coeff(first_coeff):
        return replace_coeff(expr, second_term, first_coeff)
    if zero_fallback is not None and is_zero_coeff(first_coeff) and is_zero_coeff(second_coeff):
        return expr + f'+{zero_fallback}*{first_term}+{zero_fallback}*{second_term}'
    return expr
