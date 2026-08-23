import torch;

from Derivative import Derivative;


class Poly_Weight_Function(torch.nn.Module):

    def __init__(self,
                 X_0: torch.Tensor,
                 r: float,
                 p: int,
                 Coords: torch.Tensor,
                 V: float) -> None:

        # First, call the module initializer.
        super(Poly_Weight_Function, self).__init__();

        ########################################################################
        # Checks

        assert (len(X_0.shape) == 1);  # X_0 should be a 1D tensor.
        assert (len(Coords.shape) == 2);  # Coords should be a 2D tensor.
        assert (Coords.shape[1] == X_0.shape[0]);  # X_0 should be in the same space as Coords.
        assert (r > 0);  # The radius must be positive.

        ########################################################################
        # Assuming we passed the checks, assign X_0 and r members. We can also
        # determine the input dimension from the length of X_0.

        self.X_0: torch.Tensor = X_0;
        self.Input_Dim: int = X_0.numel();
        self.r: float = r;
        self.p: int = p;
        self.V: float = V;

        # Get Num_Coords.
        self.Num_Coords: int = Coords.shape[0];

        # Initialize this object's derivative dictionary.
        self.Derivatives: dict = {};

        ########################################################################
        # Now, determine which coordinates are in B_r(X_0).

        # First, calculate || X - X_0 ||_{infinity}.
        XmX0: torch.Tensor = torch.subtract(Coords, X_0);
        Max_XmX0: torch.Tensor = torch.linalg.vector_norm(XmX0, ord=float('inf'), dim=1);

        # Now, determine which coordinates are in B_r(X_0).
        Supported_Indices: torch.Tensor = torch.less(Max_XmX0, self.r);
        self.Supported_Indices = Supported_Indices
        # Record the coordinates that are.
        self.Supported_Coords: torch.Tensor = Coords[self.Supported_Indices, :];

    def to(self, device: torch.device):
        """
        Move the Poly_Weight_Function object to the specified device.
        This includes:
            - X_0
            - Supported_Coords
            - Supported_Indices
            - Values in Derivatives

        Note: r, V, Input_Dim, Num_Coords, and similar attributes are floats or integers and do not need to be moved.
        """

        self.X_0 = self.X_0.to(device)
        self.Supported_Coords = self.Supported_Coords.to(device)
        self.Supported_Indices = self.Supported_Indices.to(device)

        for key in self.Derivatives:
            self.Derivatives[key] = self.Derivatives[key].to(device)

        return self

    # This is a product of 1D bump functions. The initializer uses the
    # infinity norm to determine the supported points.
    def forward(self, X: torch.Tensor) -> torch.Tensor:

        # First, calculate || X - X_0 ||_{infinity}.
        XmX0: torch.Tensor = torch.subtract(X, self.X_0);
        Max_XmX0: torch.Tensor = torch.linalg.vector_norm(XmX0, ord=float('inf'), dim=1);

        # Determine which coordinates are in B_r(X_0).
        Supported_Indices: torch.Tensor = torch.less(Max_XmX0, self.r);
        XmX0_Supported = XmX0[Supported_Indices, :]
        v = XmX0_Supported / self.r
        base = 1 - v ** 2
        w_X_Supported: torch.Tensor = torch.prod(base ** self.p, dim=1);
        device = Supported_Indices.device
        # Set w to zero at the remaining coordinates.
        w_X: torch.Tensor = torch.zeros(X.shape[0], dtype=X.dtype, device=device);
        # print("Supported_Indices:",Supported_Indices.device)
        # print("w_X_Supported:", w_X_Supported.device)
        # print("w_X:",w_X.device)
        w_X[Supported_Indices] = w_X_Supported;

        # Return the evaluated values.
        return w_X;

    def Add_Derivative(self,
                       D: Derivative) -> None:

        # First, check if we've already evaluated D(w).
        if (tuple(D.Encoding) in self.Derivatives):
            return;

        # If not, then let's calculate it.
        Dw: torch.Tensor = Evaluate_Derivative(w=self,
                                               D=D,
                                               Coords=self.Supported_Coords);
        self.Derivatives[tuple(D.Encoding)] = Dw.detach();

        # The derivative is now cached.
        return;

    def Get_Derivative(self, D: Derivative) -> torch.Tensor:

        # First, get D(w) at self.Supported_Coords.
        Dw_Supported_Coords: torch.Tensor = self.Derivatives[tuple(D.Encoding)];

        # Next, extrapolate this derivative to the original set of Coordinates
        # we used to initialize w.
        Dw: torch.Tensor = torch.zeros(self.Num_Coords, dtype=torch.float32);
        Dw[self.Supported_Indices] = Dw_Supported_Coords;

        # Return the reconstructed derivative.
        return Dw;


def Build_From_Stand(
        X_1: torch.Tensor,
        r_1: float,
        p_0: int,
        points_per_dim) -> Poly_Weight_Function:
    # First, fetch the relevant parameters from W_0.
    r_0 = 1
    X_0 = torch.zeros_like(X_1)
    input_dim = X_1.shape[-1]
    x_ranges = [(-1, 1)] * input_dim
    grids = [torch.linspace(start, end, points_per_dim) for (start, end) in x_ranges]
    mesh = torch.meshgrid(*grids, indexing="ij")  
    W_0_Coords = torch.stack([m.flatten() for m in mesh], dim=-1)
    # Build new variables. 
    W_1_Coords: torch.Tensor = (W_0_Coords - X_0) * (r_1 / r_0) + X_1;
    deltas = [g[1] - g[0] for g in grids]
    V_0 = torch.prod(torch.stack(deltas))
    W_0 = Poly_Weight_Function(X_0, r_0, p_0, W_0_Coords, V_0)

    V_1: float = V_0 * ((r_1 / r_0) ** input_dim);
    # Initialize W_1.
    W_1 = Poly_Weight_Function(X_0=X_1, r=r_1, p=p_0, Coords=W_1_Coords, V=V_1);

    # Now, build the derivatives of W_1 from those of W_0. 
    W_0_Derivatives_Dict = W_0.Derivatives;
    for Encoding, W_0_Derivative in W_0_Derivatives_Dict.items():
        # First, determine the order of this derivative.
        Order: int = 0;
        for i in range(len(Encoding)):
            Order += Encoding[i];

        # Next, scale W_0_Derivative by (r_0/r_1)^Order to get the  
        # corresponding derivatives for W_1. This works because of how we 
        # define the coordinates of W_1. 
        W_1_Derivative: torch.Tensor = W_0_Derivative * ((r_0 / r_1) ** Order);
        W_1.Derivatives[Encoding] = W_1_Derivative;

    # Return the new weight function.
    return W_1;


def Build_From_Other(
        X_1: torch.Tensor,
        r_1: float,
        W_0):

    # First, fetch the relevant parameters from W_0.
    X_0: torch.Tensor = W_0.X_0;
    r_0: float = W_0.r;
    W_0_Coords: torch.Tensor = W_0.Supported_Coords;
    V_0: float = W_0.V;
    p_0 = W_0.p

    # Fetch the number of dimensions. We need this to adjust V.
    n: int = W_0_Coords.shape[1];

    # Build new variables.
    W_1_Coords: torch.Tensor = (W_0_Coords - X_0) * (r_1 / r_0) + X_1;
    V_1: float = V_0 * ((r_1 / r_0) ** n);

    # Initialize W_1.
    W_1 = Poly_Weight_Function(X_0=X_1, r=r_1, p=p_0, Coords=W_1_Coords, V=V_1);

    # Now, build the derivatives of W_1 from those of W_0.
    W_0_Derivatives_Dict = W_0.Derivatives;
    for Encoding, W_0_Derivative in W_0_Derivatives_Dict.items():
        # First, determine the order of this derivative.
        Order: int = 0;
        for i in range(len(Encoding)):
            Order += Encoding[i];

        # Next, scale W_0_Derivative by (r_0/r_1)^Order to get the
        # corresponding derivatives for W_1. This works because of how we
        # define the coordinates of W_1.
        W_1_Derivative: torch.Tensor = W_0_Derivative * ((r_0 / r_1) ** Order);
        W_1.Derivatives[Encoding] = W_1_Derivative;

    # Return the new weight function.
    return W_1;


def Evaluate_Derivative(
        w,
        D: Derivative,
        Coords: torch.Tensor) -> torch.Tensor:
    # Weight derivatives must be built with autograd even when the caller is
    # caching weak-form matrices under torch.no_grad().
    if not torch.is_grad_enabled():
        with torch.enable_grad():
            return Evaluate_Derivative(w=w, D=D, Coords=Coords)

    # First, we need to convert Coords to double precision. We need this to
    # prevent NaNs from appearing when computing high-order derivatives.
    # We will convert back to single precision when finished.
    Coords: torch.Tensor = Coords.detach().to(dtype=torch.float64);

    # Next, enable gradient tracking so that derivatives can be evaluated.
    Coords.requires_grad_(True);

    # Make sure we can actually compute the derivatives. For this, we need
    # the input dimension of f to be <= the size of D's encoding vector.
    assert (D.Encoding.size <= w.Input_Dim);

    # Now, let's get to work. The plan is the following: Suppose we want to find
    # D_t^{m(t)} D_x^{m(x)} D_y^{m(y)} D_z^{m(z)} w. First, we compute
    # D_t^{m(t)} w. From this, we calculate D_x^{m(x)} D_t^{m(t)} w, and so on.
    # We then use equality of mixed partials (remember, w is infinitely
    # differentiable) to rewrite this as the derivative we want.
    w_Coords: torch.Tensor = w(Coords).view(-1);

    ############################################################################
    # t derivatives.

    # Initialize Dt_w. If there are no t derivatives, then Dt_w = w_Coords.
    Dt_w: torch.Tensor = w_Coords;

    Dt_Order: int = D.Encoding[0];
    if (Dt_Order > 0):
        # Suppose Dt_Order = m. Compute D_t^k w from D_t^{k - 1} w for each k
        # in {1, 2, ... , m}.
        for k in range(1, Dt_Order + 1):
            # Compute the gradient.
            Grad_Dt_w: torch.Tensor = torch.autograd.grad(
                outputs=Dt_w,
                inputs=Coords,
                grad_outputs=torch.ones_like(Dt_w),
                retain_graph=True,
                create_graph=True)[0];

            # Update Dt_w; this replaces D_t^{k - 1} w with D_t^k w.
            Dt_w = Grad_Dt_w[:, 0].view(-1);

    ############################################################################
    # x derivatives.

    # Initialize Dx_Dt_w. If there are no x derivatives, then Dx_Dt_w = Dt_w.
    Dx_Dt_w: torch.Tensor = Dt_w;

    Dx_Order: int = D.Encoding[1];
    if (Dx_Order > 0):
        # Suppose Dx_Order = m. We compute D_x^k Dt_w from D_x^{k - 1} Dt_w for
        # each k in {1, 2, ... , m}.
        for k in range(1, Dx_Order + 1):
            # Compute the gradient.
            Grad_Dx_Dt_w: torch.Tensor = torch.autograd.grad(
                outputs=Dx_Dt_w,
                inputs=Coords,
                grad_outputs=torch.ones_like(Dx_Dt_w),
                retain_graph=True,
                create_graph=True)[0];

            # Update Dx_Dt_w; this replaces D_x^{k - 1} Dt_w with D_x^k Dt_w.
            Dx_Dt_w = Grad_Dx_Dt_w[:, 1].view(-1);

    ############################################################################
    # y derivatives.

    # First, check if there are any y derivatives (if Derivative.Encoding has a
    # 3rd element). If not, then we're done.
    if (D.Encoding.size < 3):
        return Dx_Dt_w.to(dtype=torch.float32);

    # Assuming we need y derivatives, initialize Dy_Dx_Dt_w. If there are no y
    # derivatives, then Dy_Dx_Dt_w = Dx_Dt_w.
    Dy_Dx_Dt_w: torch.Tensor = Dx_Dt_w;

    Dy_Order: int = D.Encoding[2];
    if (Dy_Order > 0):
        # Suppose Dy_Order = m. We compute D_y^k Dx_Dt_w from
        # D_y^{k - 1} Dx_Dt_w for each k in {1, 2, ... , m}.
        for k in range(1, Dy_Order + 1):
            # Compute the gradient.
            Grad_Dy_Dx_Dt_w: torch.Tensor = torch.autograd.grad(
                outputs=Dy_Dx_Dt_w,
                inputs=Coords,
                grad_outputs=torch.ones_like(Dy_Dx_Dt_w),
                retain_graph=True,
                create_graph=True)[0];

            # Update Dy_Dx_Dt_w; this replaces D_y^{k - 1} Dx_Dt_w with
            # D_y^k Dx_Dt_w.
            Dy_Dx_Dt_w = Grad_Dy_Dx_Dt_w[:, 2].view(-1);

    ############################################################################
    # z derivatives.

    # First, check if there are any z derivatives (if Derivative.Encoding has a
    # 4th element). If not, then we're done.
    if (D.Encoding.size < 4):
        return Dy_Dx_Dt_w.to(dtype=torch.float32);

    # Assuming we need z derivatives, initialize Dz_Dy_Dx_Dt_w. If there are no
    # z derivatives, then Dz_Dy_Dx_Dt_w = Dy_Dx_Dt_w.
    Dz_Dy_Dx_Dt_w: torch.Tensor = Dy_Dx_Dt_w;

    Dz_Order: int = D.Encoding[3];
    if (Dz_Order > 0):
        # Suppose Dz_Order = m. We compute D_z^k Dy_Dx_Dt_w from
        # D_z^{k - 1} Dy_Dx_Dt_w for each k in {1, 2, ... , m}.
        for k in range(1, Dz_Order + 1):
            # Compute the gradient.
            Grad_Dz_Dy_Dx_Dt_w: torch.Tensor = torch.autograd.grad(
                outputs=Dz_Dy_Dx_Dt_w,
                inputs=Coords,
                grad_outputs=torch.ones_like(Dz_Dy_Dx_Dt_w),
                retain_graph=True,
                create_graph=True)[0];

            # Update Dz_Dy_Dx_Dt_w; this replaces D_z^{k - 1} Dy_Dx_Dt_w with
            # D_z^k Dy_Dx_Dt_w.
            Dz_Dy_Dx_Dt_w = Grad_Dz_Dy_Dx_Dt_w[:, 3].view(-1);

    return Dz_Dy_Dx_Dt_w.to(dtype=torch.float32);
