import torch
from Integrate_net import Integrate


def Weak_Form_Loss(LHS_List, RHS_List):
    """
    LHS_List: List of left-hand side integrals
    RHS_List: List of right-hand side integrals
    """
    # Compute loss (this is (1/m)||A \xi - b ||_2^2).
    # print(len(LHS_List), len(RHS_List))
    LHS_tensor = torch.stack(LHS_List)
    # print("LHS_tensor:", LHS_tensor)
    RHS_tensor = torch.stack(RHS_List).squeeze()
    # print("RHS_tensor:", RHS_tensor)
    Residual = torch.subtract(LHS_tensor, RHS_tensor)
    return torch.mean(torch.multiply(Residual, Residual)), Residual


def l1_loss(model, lambda_l1, device):
    # device = next(model.parameters()).device
    l1_loss = torch.tensor(0.0, device=device)
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss

def huber_loss(model, lambda_l1, threshold, device):
    loss = 0.0
    for param in model.parameters():
        abs_param = torch.abs(param)
        mask = abs_param > threshold
        
        # |x| - 0.5 * threshold
        upper_loss = abs_param - 0.5 * threshold
        upper_loss = upper_loss[mask].sum()     # Sum only the values above the threshold.
        
        # 0.5 * x^2 / threshold
        lower_loss = 0.5 * (param ** 2) / threshold
        lower_loss = lower_loss[~mask].sum()     # Sum only the values at or below the threshold.
        
        loss += upper_loss + lower_loss
    
    return lambda_l1 * loss

def lp_loss(model, lambda_lp, p=1, delta=1e-6, device='cuda'):

    assert 0 < p < 2, "p must be in (0,2)"

    loss = torch.tensor(0.0, device=device)

    for param in model.parameters():
        if param.requires_grad:   # Regularize only learnable parameters.
            abs_param = torch.abs(param)
            # Compute the denominator: max(delta, |w|^(2-p)).
            denom = torch.clamp(abs_param ** (2 - p), min=delta)
            # This is equivalent to |w|^p.
            loss += torch.sum((abs_param ** 2) / denom)

    return lambda_lp * loss



