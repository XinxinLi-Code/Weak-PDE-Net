"""Gaussian feature-network utilities developed with reference to PIG.

PIG: Physics-Informed Gaussians as Adaptive Parametric Mesh Representations
Original project: https://github.com/NamGyuKang/Physics-Informed-Gaussians
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import numpy as np
import math

class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class FourierFeature(nn.Module):
    def __init__(self, in_dim, num_freqs):
        super().__init__()
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        B = torch.randn(in_dim, num_freqs) * 1   
        self.register_buffer("B", B)

    def forward(self, x):
        # x: [batch, in_dim]
        proj = (2 * math.pi * x) @ self.B      # [batch, num_freqs]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class DenseResBlock(nn.Module):
    def __init__(self, hidden_dim, activation_fn):
        super(DenseResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn
        )

    def forward(self, x):
        return x + self.block(x)

class Base(nn.Module):
    def __init__(self, full_cov,num_layers,mlp_dim,num_gaussians,sigma_init,hidden_dim,in_dim,out_dim,activation,fourier_encoding=False,num_freqs=10,res_net=False):
        super(Base, self).__init__()
        self.full_cov = full_cov
        # Model parameters.
        self.num_layers = num_layers
        self.res_net = res_net
        # Gaussian-cell configuration.
        self.mlp_dim = mlp_dim
        self.num_gaussians = num_gaussians
        self.fourier_encoding = fourier_encoding
        if self.fourier_encoding:
            self.fourier = FourierFeature(in_dim,num_freqs)
            self.gauss_scale = nn.Parameter(torch.tensor(1.0))
            self.fourier_scale = nn.Parameter(torch.tensor(1.0))
        self.sigma_init = sigma_init
        
        # Network dimensions.
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.in_dim = in_dim

        if self.in_dim == 2:
            self.mu_t = torch.nn.Parameter(torch.rand(self.mlp_dim, self.num_gaussians))
            self.mu_x = torch.nn.Parameter(torch.rand(self.mlp_dim, self.num_gaussians))
            self.mu_t.data.uniform_(-1., 1.)
            self.mu_x.data.uniform_(-1., 1.)
            self.mu_t.requires_grad = True
            self.mu_x.requires_grad = True
            self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (self.mlp_dim, self.num_gaussians)))
            self.sigma_t = torch.nn.Parameter(torch.ones((self.mlp_dim, self.num_gaussians))*self.sigma_init)
            self.sigma_x = torch.nn.Parameter(torch.ones((self.mlp_dim, self.num_gaussians))*self.sigma_init)
            self.weight.requires_grad = True
            self.sigma_t.requires_grad = True
            self.sigma_x.requires_grad = True
            

        elif self.in_dim == 3:
                self.mu_t = torch.nn.Parameter(torch.rand(self.mlp_dim, self.num_gaussians))
                self.mu_x = torch.nn.Parameter(torch.rand(self.mlp_dim, self.num_gaussians))
                self.mu_y = torch.nn.Parameter(torch.rand(self.mlp_dim, self.num_gaussians))
                self.mu_t.data.uniform_(-1, 1)
                self.mu_x.data.uniform_(-1, 1)
                self.mu_y.data.uniform_(-1, 1)
                self.mu_t.requires_grad = True
                self.mu_x.requires_grad = True
                self.mu_y.requires_grad = True
                self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (self.mlp_dim, self.num_gaussians)))
                self.sigma_t = torch.nn.Parameter(torch.ones((self.mlp_dim, self.num_gaussians))*self.sigma_init)
                self.sigma_x = torch.nn.Parameter(torch.ones((self.mlp_dim, self.num_gaussians))*self.sigma_init)
                self.sigma_y = torch.nn.Parameter(torch.ones((self.mlp_dim, self.num_gaussians))*self.sigma_init)
                self.weight.requires_grad = True
                self.sigma_t.requires_grad = True
                self.sigma_x.requires_grad = True
                self.sigma_y.requires_grad = True
        elif self.in_dim == 4:
                self.mu_t = torch.nn.Parameter(torch.rand(self.mlp_dim, self.num_gaussians))
                self.mu_x = torch.nn.Parameter(torch.rand(self.mlp_dim, self.num_gaussians))
                self.mu_y = torch.nn.Parameter(torch.rand(self.mlp_dim, self.num_gaussians))
                self.mu_z = torch.nn.Parameter(torch.rand(self.mlp_dim, self.num_gaussians))
                self.mu_t.data.uniform_(-1, 1)
                self.mu_x.data.uniform_(-1, 1)
                self.mu_y.data.uniform_(-1, 1)
                self.mu_z.data.uniform_(-1, 1)
                self.mu_t.requires_grad = True
                self.mu_x.requires_grad = True
                self.mu_y.requires_grad = True
                self.mu_z.requires_grad = True
                self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (self.mlp_dim, self.num_gaussians)))
                self.sigma_t = torch.nn.Parameter(torch.ones((self.mlp_dim, self.num_gaussians))*self.sigma_init)
                self.sigma_x = torch.nn.Parameter(torch.ones((self.mlp_dim, self.num_gaussians))*self.sigma_init)
                self.sigma_y = torch.nn.Parameter(torch.ones((self.mlp_dim, self.num_gaussians))*self.sigma_init)
                self.sigma_z = torch.nn.Parameter(torch.ones((self.mlp_dim, self.num_gaussians))*self.sigma_init)
                self.weight.requires_grad = True
                self.sigma_t.requires_grad = True
                self.sigma_x.requires_grad = True
                self.sigma_y.requires_grad = True
                self.sigma_z.requires_grad = True
        elif self.num_gaussians != 0:
                raise NotImplementedError("Gaussian embedding supports input dimensions 2, 3, and 4.")
            
        if activation=='relu':
            self.activation_fn = nn.ReLU()
        elif activation=='leaky_relu':
            self.activation_fn = nn.LeakyReLU()
        elif activation=='sigmoid':
            self.activation_fn = nn.Sigmoid()
        elif activation=='softplus':
            self.activation_fn = nn.Softplus()
        elif activation=='tanh':
            self.activation_fn = nn.Tanh()
        elif activation=='gelu':
            self.activation_fn = nn.GELU()
        elif activation =='logsigmoid':
            self.activation_fn = nn.LogSigmoid()
        elif activation =='hardsigmoid':
            self.activation_fn = nn.Hardsigmoid()
        elif activation =='elu':
            self.activation_fn = nn.ELU()
        elif activation =='celu':
            self.activation_fn = nn.CELU()            
        elif activation =='selu':
            self.activation_fn = nn.SELU() 
        elif activation =='silu':
            self.activation_fn = nn.SiLU()  
        elif activation == 'sin':
            self.activation_fn = SinActivation()
        else:
            raise NotImplementedError
      
        if self.num_layers==0:
            return
        
        ''' see the Section "Neural network and Grid Representations" in the paper.
                    we built the Neural network. '''
        self.net = []
        if self.num_gaussians != 0 and not self.fourier_encoding:
            input_dim = self.mlp_dim
        elif self.num_gaussians != 0 and self.fourier_encoding:
            input_dim = self.mlp_dim + 2*num_freqs
        else:
            input_dim = self.in_dim
        if self.num_layers < 2:
            self.net.append(self.activation_fn)
            self.net.append(torch.nn.Linear(input_dim, self.out_dim))

        else:
            if not self.res_net:
                self.net.append(torch.nn.Linear(input_dim, self.hidden_dim))
                self.net.append(self.activation_fn)
                for i in range(self.num_layers-2): 
                    self.net.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
                    self.net.append(self.activation_fn)
                self.net.append(torch.nn.Linear(self.hidden_dim, self.out_dim))
            else:
                if not self.res_net:
                    self.net.append(torch.nn.Linear(input_dim, self.hidden_dim))
                    self.net.append(self.activation_fn)
                    for i in range(self.num_layers-2): 
                        self.net.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
                        self.net.append(self.activation_fn)
                    self.net.append(torch.nn.Linear(self.hidden_dim, self.out_dim))
                else:
                    # 1. Input projection.
                    self.net.append(torch.nn.Linear(input_dim, self.hidden_dim))
                    self.net.append(self.activation_fn)
                    
                    # 2. Residual blocks.
                    for i in range(self.num_layers): 
                        self.net.append(DenseResBlock(self.hidden_dim, self.activation_fn))
                    
                    # 3. Output projection.
                    self.net.append(torch.nn.Linear(self.hidden_dim, self.out_dim))
        # Assemble the layers.
        self.net = nn.Sequential(*self.net)
        if activation == 'sin':
            for i, module in enumerate(self.net):
                if isinstance(module, nn.Linear):
                    self.init_siren(module, is_first=(i==0))
    def init_siren(self, layer, is_first, w0=30):
        n_in = layer.weight.size(-1)
        if is_first:
            bound = 1 / n_in
        else:
            bound = np.sqrt(6 / n_in) / w0
        layer.weight.data.uniform_(-bound, bound)

    def additional_params(self):
        self.r1 = torch.nn.Parameter(torch.ones((self.mlp_dim, self.num_gaussians)))
        self.r2 = torch.nn.Parameter(torch.ones((self.mlp_dim, self.num_gaussians)) * 0.0)

    def forward(self, x, norm_flag=False):
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = x.device
        x = x.to(device)
        # print("x shape:", x.shape)
        if self.fourier_encoding:
            # print("x shape:", x.shape)
            fourier_feats = self.fourier(2 * (x - x.min(dim=0, keepdim=True)[0]) / (x.max(dim=0, keepdim=True)[0] - x.min(dim=0, keepdim=True)[0] + 1e-8) - 1)
            print("fourier_feats shape:", fourier_feats.shape)
        if self.num_gaussians != 0:
            if self.in_dim==2:
                if norm_flag:
                    t_raw = x[..., 0:1]
                    x_raw = x[..., 1:2]

                    t_min, t_max = t_raw.min(), t_raw.max()
                    t_norm = (t_raw - t_min) / (t_max - t_min + 1e-8)

                    x_min, x_max = x_raw.min(), x_raw.max()
                    x_norm = 2.0 * (x_raw - x_min) / (x_max - x_min + 1e-8) - 1.0
                    x = torch.cat([t_norm, x_norm], dim=-1)

                means = torch.stack([self.mu_t, self.mu_x], -1)
                sigmas = torch.stack([self.sigma_t, self.sigma_x], -1)
    
                if self.full_cov:
                    L = self.build_scaling_rotation_2x2(sigmas, self.r1, self.r2)
                    cov = (L @ L.transpose(2, 3)).unsqueeze(1)
                    d = x.unsqueeze(0).unsqueeze(2) - means.unsqueeze(1)
                    out = cov @ d[..., None]
                    out = d[..., None, :] @ out
                    feats = (torch.exp(-0.5*out.squeeze()) * self.weight.unsqueeze(1)).sum(-1).t()

                else:
                    x = x.unsqueeze(0).repeat(self.mlp_dim, 1, 1)
                    feats = self.gaussian_sample(x, means, sigmas, self.weight)
                
            elif self.in_dim ==3:
                if norm_flag:
                    t_raw = x[..., 0:1]          # [batch, 1]
                    xy_raw = x[..., 1:3]         # [batch, 2]

                    t_min, t_max = t_raw.min(), t_raw.max()
                    t_norm = (t_raw - t_min) / (t_max - t_min + 1e-8)   # (0, 1)

                    xy_min = xy_raw.min(dim=0, keepdim=True)[0]         # [1, 2]
                    xy_max = xy_raw.max(dim=0, keepdim=True)[0]         # [1, 2]
                    xy_norm = 2.0 * (xy_raw - xy_min) / (xy_max - xy_min + 1e-8) - 1.0  # (-1, 1)

                    # Concatenate the normalized components.
                    x = torch.cat([t_norm, xy_norm], dim=-1)        # [batch, 3]

                means = torch.stack([self.mu_t, self.mu_x, self.mu_y], -1)
                sigmas = torch.stack([self.sigma_t, self.sigma_x, self.sigma_y], -1)
                x = x.unsqueeze(0).repeat(self.mlp_dim, 1, 1)
                feats = self.gaussian_sample(x, means, sigmas, self.weight)
            elif self.in_dim == 4:
                if norm_flag:
                    t_raw = x[..., 0:1]
                    xyz_raw = x[..., 1:4]

                    t_min, t_max = t_raw.min(), t_raw.max()
                    t_norm = (t_raw - t_min) / (t_max - t_min + 1e-8)

                    xyz_min = xyz_raw.min(dim=0, keepdim=True)[0]
                    xyz_max = xyz_raw.max(dim=0, keepdim=True)[0]
                    xyz_norm = 2.0 * (xyz_raw - xyz_min) / (xyz_max - xyz_min + 1e-8) - 1.0
                    x = torch.cat([t_norm, xyz_norm], dim=-1)

                means = torch.stack([self.mu_t, self.mu_x, self.mu_y, self.mu_z], -1)
                sigmas = torch.stack([self.sigma_t, self.sigma_x, self.sigma_y, self.sigma_z], -1)
                x = x.unsqueeze(0).repeat(self.mlp_dim, 1, 1)
                feats = self.gaussian_sample(x, means, sigmas, self.weight)
            else:
                raise NotImplementedError("Gaussian embedding supports input dimensions 2, 3, and 4.")
        else:
            feats = x   
        if self.fourier_encoding:
            feats = (feats - feats.mean(dim=0, keepdim=True)) / \
            (feats.std(dim=0, keepdim=True) + 1e-8)
            fourier_feats = (fourier_feats - fourier_feats.mean(dim=0, keepdim=True)) / \
                            (fourier_feats.std(dim=0, keepdim=True) + 1e-8)
            print("feats shape:", feats.shape)
            feats = torch.cat([feats*self.gauss_scale, fourier_feats*self.fourier_scale], dim=-1)
        # print(x.shape)    
        if self.num_layers > 0:
            out = self.net(feats)        
        else:
            out = feats.mean(0).squeeze().view(-1, 1)
            
        return out
    
    def gaussian_sample(self, X, means, sigmas, weight):
        means = means.unsqueeze(1)   # (k, 1, g, d)
        sigmas = sigmas.unsqueeze(1) # (k, 1, g, d)
        weight = weight.squeeze().unsqueeze(1) # (k, 1, g)
        X = X.unsqueeze(2)           # (1, p, 1, d)
        # print("X",X.device)
        # print(means.device)
        exponent = (((X - means)/sigmas)**2).sum(-1)
        gaussians = torch.exp(-0.5*exponent) * weight
        output = gaussians.sum(-1).t()

        return output

    def build_rotation_2x2(self, r):
        norm = torch.sqrt(r[..., 0]**2 + r[..., 1]**2)
        q = r / norm[..., None]

        # Define the 2 x 2 matrix elements from the normalized vector.
        r, x = q[..., 0], q[..., 1]

        # Create a 2 x 2 matrix.
        R = torch.zeros((*q.shape[:-1], 2, 2), dtype=r.dtype, device=r.device)
        R[..., 0, 0] = r
        R[..., 0, 1] = -x
        R[..., 1, 0] = x
        R[..., 1, 1] = r

        return R

    def build_scaling_rotation_2x2(self, s, r1, r2):
        r = torch.stack([r1, r2], -1)
        R = self.build_rotation_2x2(r)
        # Create a 2 x 2 scaling matrix.
        L = torch.zeros((s.shape[0], s.shape[1], 2, 2), dtype=s.dtype, device=s.device)
        L[..., 0, 0] = 1.0 / s[..., 0]
        L[..., 1, 1] = 1.0 / s[..., 1]
        
        # Apply the 2 x 2 rotation to the scaling matrix.
        L = R @ L
        
        return L
