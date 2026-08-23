import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import Regression
from matplotlib import rc
from tqdm import tqdm
from plot_pde import *
from pig_network import *

rc('text', usetex=False)


class Tune_Net(nn.Module):
    def __init__(
        self,
        problem_dim,
        Fu_list,
        derivative_list,
        var_names,
        init_weight,
        device,
        joint_optim=False,
        int_xi=None,
        reg_lambda=1E-5,
        bounds_flag=False,
        lhs_type=1,
    ):
        super().__init__()
        self.problem_dim = problem_dim
        self.Fu_list = Fu_list
        self.derivative_list = derivative_list
        self.var_names = var_names
        self.number_items = len(Fu_list)
        self.device = device
        self.init_weight = init_weight
        self.joint_optim = joint_optim
        self.int_xi = int_xi
        self.bounds_flag = bounds_flag
        self.lhs_type = lhs_type
        self.cached_LHS = None
        self.cached_RHS = None
        print("self.joint_optim:",self.joint_optim)
        print("self.bounds_flag:",self.bounds_flag)
        if self.joint_optim:
            self.reg_lambda = reg_lambda
            self.Xi = None  # Weight matrix.
            if self.int_xi is not None:  # Use the provided initial weight.
                initial_weight = self.int_xi
                if isinstance(initial_weight, np.ndarray):
                    # Convert the NumPy array to a PyTorch tensor.
                    self.initial_weight = torch.from_numpy(initial_weight).float()
                elif isinstance(initial_weight, list):
                    # Convert the Python list to a PyTorch tensor.
                    self.initial_weight = torch.tensor(initial_weight, dtype=torch.float32)
                else:
                    # Clone the PyTorch tensor.
                    self.initial_weight = initial_weight.clone().detach()
                # self.initial_weight = torch.sign(self.initial_weight)
                self.Xi = nn.Parameter(self.initial_weight)

            else:
                # Initialize the weights from a uniform distribution.
                self.Xi = nn.Parameter(torch.empty(len(self.derivative_list), 1))
                nn.init.uniform_(self.Xi, a=-self.init_uniform, b=self.init_uniform)
        self.to(self.device)
    
    def build_rhs_lhs(self, U_data, input_test, index, Random_Weight_Functions_Lists):
        u_data = U_data[:, index]
        input_data = torch.cat([input_test, U_data], dim=1)

        # Assemble weak-form matrices without gradients for faster tuning; the weak loss does not update PigNet.
        with torch.no_grad():
            self.cached_LHS = Regression.compute_lhs(
                Random_Weight_Functions_Lists[index],
                self.problem_dim,
                u_data,
                lhs_type=self.lhs_type,
            ).to(self.device)

            self.cached_RHS = Regression.compute_rhs(
                Random_Weight_Functions_Lists[index],
                self.Fu_list,
                self.derivative_list,
                input_data,
                self.var_names
            ).to(self.device)

    def forward(self):
        LHS = self.cached_LHS
        RHS = self.cached_RHS
        # print(self.joint_optim)
        if self.joint_optim:
            Xi_eff = torch.tanh(self.Xi) if self.bounds_flag else self.Xi
            Xi_vec = Xi_eff.reshape(-1)

            Residual = LHS.reshape(-1) - RHS @ Xi_vec
            mse_loss = torch.mean(Residual ** 2)
            reg_loss = torch.sum(torch.abs(Xi_vec))

            weak_loss = mse_loss + self.reg_lambda * reg_loss

            expr = Regression.print_result(
                Xi_vec.detach().cpu().numpy(),
                self.Fu_list,
                self.derivative_list
            )

        else:
            
            result, coff_list, expr = Regression.regression(RHS, LHS, self.init_weight, self.Fu_list, self.derivative_list,print=True,bounds=self.bounds_flag)
            coeffs = torch.as_tensor(coff_list, dtype=RHS.dtype, device=RHS.device)
            weak_loss = torch.mean((LHS.reshape(-1) - RHS @ coeffs.reshape(-1)) ** 2)
           
        return expr, weak_loss




def evaluate_test_mse(Pig_Net, u_test, problem_dim, Device, norm_flag, batch_size=200000):
    was_training = Pig_Net.training
    Pig_Net.eval()
    total_sq_error = 0.0
    total_count = 0
    with torch.no_grad():
        for start in range(0, u_test.shape[0], batch_size):
            batch = u_test[start:start + batch_size].to(Device)
            input_batch = batch[:, 0:problem_dim + 1]
            target_batch = batch[:, problem_dim + 1:]
            pred_batch = Pig_Net(input_batch, norm_flag)
            diff = pred_batch - target_batch
            total_sq_error += torch.sum(diff * diff).item()
            total_count += diff.numel()
    if was_training:
        Pig_Net.train()
    return total_sq_error / max(total_count, 1)


def train_tune_net(Tune_net_list, Num_datasets, input_train, u_train, input_test, Random_Weight_Functions_Lists,
                   f_scale, problem_dim, Device, output_path,
                   Pig_Net, other_data=None, norm_flag=False, u_test=None, grid_plot_size=None,grid_size=None,optimizer_type='adam', lr=1e-3, max_iter=500,
                   print_every=200,step=10):
    # Store the best state for each dataset.
    params = list(Pig_Net.parameters())
    PDE_list = []
    for i in range(Num_datasets):
        Tune_net = Tune_net_list[i]
        tune_params = list(Tune_net.parameters())
        if Tune_net.joint_optim:
            params = tune_params
            Pig_Net.eval()
            for param in Pig_Net.parameters():
                param.requires_grad_(False)
        else:
            params = list(Pig_Net.parameters()) + tune_params
            Pig_Net.train()
        # Initialize the best-state records.
        best_loss = float('inf')
        if optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(params, lr=lr)
        elif optimizer_type.lower() == 'lbfgs':
            optimizer = optim.LBFGS(params, lr=lr, max_iter=20, history_size=50, line_search_fn="strong_wolfe")
        else:
            raise ValueError("Unsupported optimizer type. Use 'adam' or 'lbfgs'.")
        if Tune_net.joint_optim:
            with torch.no_grad():
                U_pred = Pig_Net(input_test, norm_flag)
                if other_data is not None:
                    U_data = torch.cat([U_pred, other_data], dim=1)
                else:
                    U_data = U_pred

                Tune_net.build_rhs_lhs(
                    U_data,
                    input_test,
                    i,
                    Random_Weight_Functions_Lists
                )
        # Define the loss calculation shared by Adam and LBFGS.
        def compute_losses():
            if not Tune_net.joint_optim:
                U_pred = Pig_Net(input_test, norm_flag)
                if other_data is not None:
                    U_data = torch.cat([U_pred, other_data], dim=1)
                else:
                    U_data = U_pred

                Tune_net.build_rhs_lhs(
                    U_data,
                    input_test,
                    i,
                    Random_Weight_Functions_Lists
                )
            optimizer.zero_grad()
            if Tune_net.joint_optim:
                with torch.no_grad():
                    pred_u = Pig_Net(input_train,norm_flag)
            else:
                pred_u = Pig_Net(input_train,norm_flag)
            # print(input_train.device)
            # print(pred_u.device)
            Data_Loss = torch.mean((u_train - pred_u) ** 2)
            current_pde, Weak_Loss = Tune_net()
            Total_Loss = Data_Loss + f_scale * Weak_Loss
            return Total_Loss, Data_Loss, Weak_Loss, current_pde
        if max_iter == 0:
            Total_Loss, Data_Loss, Weak_Loss, best_pde= compute_losses()
        # Training loop.
        if optimizer_type.lower() == 'adam':
            for epoch in range(max_iter):
                # Run the forward and backward passes.
                Total_Loss, Data_Loss, Weak_Loss, current_pde= compute_losses()
                Total_Loss.backward()
                optimizer.step()
                with torch.no_grad():
                    # Update the best state.
                    if Total_Loss.item() < best_loss:
                        best_loss = Total_Loss.item()
                        best_pde =  current_pde
                    # Print progress periodically.
                    if epoch % print_every == 0 or epoch == max_iter - 1:
                        print(f"\n{'=' * 50}")
                        print(f"Model #{i + 1} Summary:")
                        print(f"  PDE Expression: {current_pde}")
                        print(f"  Total Loss: {Total_Loss:.6e}")
                        print(f"  Data Loss: {Data_Loss.item():.6e}")
                        print(f"  Weak Loss: {Weak_Loss:.6e}")
                        print('=' * 50)
                        if problem_dim == 1 and u_test is not None:
                            y_plot_test = u_test[:, 0].to(Device)
                            x_plot_test = u_test[:, 1:problem_dim + 1].to(Device)
                            input_plot_test = u_test[:, 0:problem_dim + 1].to(Device)
                            u_plot_test = u_test[:, problem_dim + 1:].to(Device)
                            u_pred_plot = Pig_Net(input_plot_test, norm_flag)
                            u_pred_test = Pig_Net(input_test, norm_flag)
                            # print("u_plot_test",u_plot_test.device)
                            # print("u_pred_plot",u_pred_plot.device)
                            l2_loss = torch.mean((u_plot_test - u_pred_plot) ** 2)
                            print('[Test Iter:%d, Loss: %.5e]' % (epoch, l2_loss))
                            for k in range(Num_datasets):
                                u_test_i = u_plot_test[:, k]
                                u_pred_plot_i = u_pred_plot.detach()[:, k]
                                plot_2d(epoch, y_plot_test, x_plot_test, u_pred_plot_i, u_test_i, grid_plot_size,
                                        output_path, 'Pde' + '_' + str(k) + '_' + optimizer_type)
                                if Num_datasets!=1:
                                    y_plot_test_new = input_test[:,0].to(Device)
                                    x_plot_test_new = input_test[:,1].to(Device)
                                    u_pred_test_new = u_pred_test.detach()[:,k]
                                    plot_predicted_u(epoch, y_plot_test_new, x_plot_test_new, u_pred_test_new, grid_size, output_path, 'Pde'+'_'+str(k)+'_'+optimizer_type)
                                else:
                                    y_plot_test = input_test[:,0].to(Device)
                                    x_plot_test = input_test[:,1].to(Device)
                                    u_pred_test = u_pred_test.detach()[:,k]
                                    plot_predicted_u(epoch, y_plot_test, x_plot_test, u_pred_test, grid_size, output_path, 'Pde'+'_'+str(k)+'_'+optimizer_type)
                        elif problem_dim == 2 and u_test is not None:
                            t_plot_test = u_test[:,0].to(Device)
                            x_plot_test = u_test[:,1].to(Device)
                            y_plot_test = u_test[:,2].to(Device)
                            input_plot_test = u_test[:,0:problem_dim+1].to(Device)
                            
                            # print(u_test.shape)
                            u_plot_test = u_test[:,problem_dim+1:].to(Device)
                            u_pred_plot = Pig_Net(input_plot_test, norm_flag)
                            u_pred_test = Pig_Net(input_test, norm_flag)
                            # print("u_plot_test",u_plot_test.device)
                            # print("u_pred_test",u_pred_test.device)
                            # print(u_plot_test.shape)
                            # print(u_pred_test.shape)
                            l2_loss = torch.mean((u_plot_test - u_pred_plot) ** 2)
                            print('[Test Iter:%d, Loss: %.5e]'%(epoch, l2_loss))
                            for k in range(Num_datasets):
                                u_test_i = u_plot_test[:,k]
                                u_pred_plot_i = u_pred_plot.detach()[:,k]
                                plot_3d(epoch=epoch, x=x_plot_test, y=y_plot_test, u_gt_all=u_test_i, u_pred_all=u_pred_plot_i, t=t_plot_test, grid_size=grid_plot_size, output_path=output_path, tag='Pde'+'_'+str(k)+'_'+optimizer_type, step=step)
                                t_plot_test = input_test[:,0].to(Device)
                                x_plot_test = input_test[:,1].to(Device)
                                y_plot_test = input_test[:,2].to(Device)
                                u_pred_test = u_pred_test.detach()[:,k]
                                plot_3d_predicted_u(epoch=epoch, x=x_plot_test,y=y_plot_test,u_pred_all=u_pred_test,t=t_plot_test,grid_size=grid_size,output_path=output_path, tag='Pde'+'_'+str(k)+'_'+optimizer_type, step=step)
                        elif problem_dim == 3 and u_test is not None:
                            l2_loss = evaluate_test_mse(Pig_Net, u_test, problem_dim, Device, norm_flag)
                            print('[Test Iter:%d, Loss: %.5e]'%(epoch, l2_loss))
             
        elif optimizer_type.lower() == 'lbfgs':
            for epoch in range(max_iter):
                def closure_lbfgs():
                    optimizer.zero_grad()
                    Total_Loss, Data_Loss, Weak_Loss, current_pde = compute_losses()
                    Total_Loss.backward()

                    # Update the best state.
                    nonlocal best_loss, best_pde
                    if Total_Loss.item() < best_loss:
                        best_loss = Total_Loss.item()
                        best_pde = current_pde
                    return Total_Loss

                optimizer.step(closure_lbfgs)

                # Print progress periodically.
                if epoch % print_every == 0 or epoch == max_iter - 1:
                    print(f"\n[L-BFGS] Epoch {epoch} - Model #{i + 1} Summary:")
                    print(f"  PDE Expression: {best_pde}")
                    print(f"  Total Loss: {best_loss:.6e}")
                    print('=' * 50)

                    # Evaluate the model and plot the results.
                    with torch.no_grad():
                        if problem_dim == 1 and u_test is not None:
                            y_plot_test = u_test[:, 0].to(Device)
                            x_plot_test = u_test[:, 1:problem_dim + 1].to(Device)
                            input_plot_test = u_test[:, 0:problem_dim + 1].to(Device)
                            u_plot_test = u_test[:, problem_dim + 1:].to(Device)
                            u_pred_plot = Pig_Net(input_plot_test, norm_flag)
                            u_pred_test = Pig_Net(input_test, norm_flag)
                            l2_loss = torch.mean((u_plot_test - u_pred_plot) ** 2)
                            print('[Test Iter:%d, Loss: %.5e]' % (epoch, l2_loss))
                            for k in range(Num_datasets):
                                u_test_i = u_plot_test[:, k]
                                u_pred_plot_i = u_pred_plot.detach()[:, k]
                                plot_2d(epoch, y_plot_test, x_plot_test, u_pred_plot_i, u_test_i,
                                        grid_plot_size, output_path, f'Pde_{k}_{optimizer_type}')
                                y_plot_test = input_test[:, 0]
                                x_plot_test = input_test[:, 1]
                                u_pred_test_i = u_pred_test.detach()[:, k]
                                plot_predicted_u(epoch, y_plot_test, x_plot_test, u_pred_test_i,
                                                grid_size, output_path, f'Pde_{k}_{optimizer_type}')
                        elif problem_dim == 2 and u_test is not None:
                            t_plot_test = u_test[:, 0].to(Device)
                            x_plot_test = u_test[:, 1].to(Device)
                            y_plot_test = u_test[:, 2].to(Device)
                            input_plot_test = u_test[:, 0:problem_dim + 1].to(Device)
                            u_plot_test = u_test[:, problem_dim + 1:].to(Device)
                            u_pred_plot = Pig_Net(input_plot_test, norm_flag)
                            u_pred_test = Pig_Net(input_test, norm_flag)
                            l2_loss = torch.mean((u_plot_test - u_pred_plot) ** 2)
                            print('[Test Iter:%d, Loss: %.5e]' % (epoch, l2_loss))
                            for k in range(Num_datasets):
                                u_test_i = u_plot_test[:, k]
                                u_pred_plot_i = u_pred_plot.detach()[:, k]
                                plot_3d(epoch=epoch, x=x_plot_test, y=y_plot_test,
                                        u_gt_all=u_test_i, u_pred_all=u_pred_plot_i,
                                        t=t_plot_test, grid_size=grid_plot_size,
                                        output_path=output_path, tag=f'Pde_{k}_{optimizer_type}', step=step)
                                t_plot_test = input_test[:, 0].to(Device)
                                x_plot_test = input_test[:, 1].to(Device)
                                y_plot_test = input_test[:, 2].to(Device)
                                u_pred_test_i = u_pred_test.detach()[:, k]
                                plot_3d_predicted_u(epoch=epoch, x=x_plot_test, y=y_plot_test,
                                                    u_pred_all=u_pred_test_i, t=t_plot_test,
                                                    grid_size=grid_size, output_path=output_path,
                                                    tag=f'Pde_{k}_{optimizer_type}', step=step)
                        elif problem_dim == 3 and u_test is not None:
                            l2_loss = evaluate_test_mse(Pig_Net, u_test, problem_dim, Device, norm_flag)
                            print('[Test Iter:%d, Loss: %.5e]' % (epoch, l2_loss))
        PDE_list.append(best_pde)
    return PDE_list
