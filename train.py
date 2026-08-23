import torch
import torch.nn as nn
import numpy as np
import Derivative
from symbolic_network_nas import SymbolicNet
from Integrate_net import IntegrateNet, Integrate
from loss import Weak_Form_Loss, l1_loss
from plot_pde import *


class Model(nn.Module):
    def __init__(self, problem_dim, x_dim, depth, repeats_list, funcs, max_order, device, grad_all=False, 
                 initial_weights_sym=None, initial_weights_int=None, init_uniform=1,
                 add_bias=False, depth_candidates=[], repeats_candidates=[], use_search=False,kill_small_items=False,threshold=0,
                 enforce_galilean_invariance=False,
                *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_bias = add_bias
        self.Sym_Net = SymbolicNet(x_dim=x_dim, problem_dim=problem_dim, depth=depth, repeats_list=repeats_list,
                                   funcs=funcs,device = device,
                                   initial_weights=initial_weights_sym,
                                   init_uniform=init_uniform, depth_candidates=depth_candidates,
                                   repeats_candidates=repeats_candidates,
                                   use_search=use_search,kill_small_items=kill_small_items,threshold=threshold,
                                   enforce_galilean_invariance=enforce_galilean_invariance)
        input_dim = self.Sym_Net.hidden_layers[-1].n_funcs
        print("input_dim", input_dim)
        self.Int_Net = IntegrateNet(problem_dim=problem_dim, max_order=max_order, input_dim=input_dim,
                                    init_uniform=init_uniform, grad_all=grad_all, initial_weights=initial_weights_int, add_bias=add_bias)

    def forward(self, x, weight_functions, grid_size, int_type):
        """
        x: Input data arranged as [x, t, u, v, w].
        weight_functions: A list containing K weight functions.
        """

        output_0 = self.Sym_Net(x)
        int_k_list = []
        for w in weight_functions:
            int_k = self.Int_Net(output_0, w, grid_size, int_type)
            int_k_list.append(int_k)
        return int_k_list

    def update_params(self, repeats_list, depth, initial_weights_symnet, initial_weights_int, grad_all=False,
                      kill_small_items=False, threshold=0, enforce_galilean_invariance=False,
                      init_to_ones=False):
        # Retrieve parameters from the current symbolic-network instance.
        existing_sym_net = self.Sym_Net
        x_dim = existing_sym_net.x_dim
        funcs = existing_sym_net.funcs
        init_uniform = existing_sym_net.init_uniform
        problem_dim = existing_sym_net.problem_dim
        device = existing_sym_net.device
        # Create a new symbolic-network instance using the new and retained parameters.
        self.Sym_Net = SymbolicNet(x_dim=x_dim, problem_dim=problem_dim,
                                   depth=depth, repeats_list=repeats_list,
                                   funcs=funcs, device=device,
                                   initial_weights=initial_weights_symnet,
                                   init_uniform=init_uniform, depth_candidates=[],
                                   repeats_candidates=[],
                                   use_search=False,kill_small_items=kill_small_items,threshold=threshold,
                                   enforce_galilean_invariance=enforce_galilean_invariance,
                                   init_to_ones=init_to_ones,
                                   add_bias=self.add_bias)
        # print("1:",self.Sym_Net.kill_small_items)
        input_dim = sum(repeats_list[-1])
        existing_int_net = self.Int_Net
        max_order = existing_int_net.max_order
        self.Int_Net = IntegrateNet(problem_dim=problem_dim,
                                    max_order=max_order, input_dim=input_dim,
                                    init_uniform=init_uniform,
                                    grad_all=grad_all,
                                    initial_weights=initial_weights_int,
                                    init_to_ones=init_to_ones,
                                    add_bias=self.add_bias)


def Training(Model_list, problem_dim, Inputs_train,
             Weight_Functions_List, reg_weight, Optimizer, Device,
             input_test, lhs_type, max_iter, grid_plot_size=[], grid_size=[], u_test=None, f_scale=1, Pig_Net = None, norm_flag=False, data_spase = False, 
             Writer=None, Epoch=None,output_path=None,train_pig=False, step=10, clip_grad=False, max_norm=5.0, int_type='Riemann',other_data=None):
    """
    problem_dim: The dimension of the PDE.
    Model_list: The list of models.
    Inputs_train: Training data arranged as [t, x, u, v, w]; the number of
        dependent-variable columns is determined by the number of equations.
    Weight_Functions_List: A list of weight-function groups.
    Optimizer: The optimizer.
    reg_weight: List of regularization weights.
    Device: CPU or GPU device.
    Inputs_test: Test inputs arranged as [t, x].
    data_spase: Whether PIG-network interpolation is required.
    """
    if isinstance(Optimizer, torch.optim.Adam):
        optim_type = 'adam'
    elif isinstance(Optimizer, torch.optim.LBFGS):
        optim_type = 'lbfgs'
    Inputs_train = Inputs_train.to(Device)
    for model in Model_list:
        model.to(Device)
    print(len(Weight_Functions_List))
    print( len(Model_list))
    assert len(Weight_Functions_List) == len(Model_list)
    arr = np.zeros(problem_dim + 1)
    arr[0] = lhs_type
    derivative = Derivative.Derivative(arr)
    Num_DataSets = len(Weight_Functions_List)
    # Put each U in training mode.
    for i in range(Num_DataSets):
        Model_list[i].train()
    # Initialize variables to track the residuals and losses. We need to do this
    # because these variables are used in the Closure function, which has its own
    # scope. Thus, any variables created in Closure are inaccessible from
    # outside Closure.
    Residual_List = []
    Weak_Loss_List = [0] * Num_DataSets
    Total_Loss_List = [0] * Num_DataSets
    L1_Loss_List = [0] * Num_DataSets

    for i in range(Num_DataSets):
        Residual_List.append(torch.empty(len(Weight_Functions_List[i]), dtype=torch.float32))
    Data_Loss_Container = [torch.tensor(0.0, dtype=torch.float32, device=Device)]
    # Define the closure function required by LBFGS.
    def Closure():
        Total_Loss_Value = torch.tensor(0.0, dtype=torch.float32, device=Device)
        if data_spase:
            if train_pig:
                Pig_Net.to(Device)
                Pig_Net.train()
                input_train = Inputs_train[:,0:problem_dim+1]
                u_train = Inputs_train[:,problem_dim+1: ]
                u_pred_train = Pig_Net(input_train, norm_flag)
                # print("u_train:",u_train.device)
                # print("u_pred_train:",u_pred_train.device)
                Data_Loss_Container[0] = torch.mean((u_train - u_pred_train) ** 2)
                Data_Loss_Value = Data_Loss_Container[0]
                u_pred_test = Pig_Net(input_test, norm_flag)
                if other_data is not None:
                    Inputs_combined = torch.cat([input_test,u_pred_test,other_data],dim=1)
                else:
                    Inputs_combined = torch.cat([input_test,u_pred_test],dim=1)
                # try:
                #     with torch.no_grad():
                #         if (Epoch%50 == 0 or Epoch == max_iter - 1) and problem_dim == 1 and u_test is not None:
                #             y_plot_test = u_test[:,0].to(Device)
                #             print("y_plot_test:",y_plot_test.shape)
                #             x_plot_test = u_test[:,1:problem_dim+1].to(Device)
                #             input_plot_test = u_test[:,0:problem_dim+1].to(Device)
                #             # print(u_test.shape)
                #             u_plot_test = u_test[:,problem_dim+1:].to(Device)
                #             u_pred_plot = Pig_Net(input_plot_test, norm_flag)
                #             # print("u_plot_test",u_plot_test.device)
                #             # print("u_pred_test",u_pred_test.device)
                #             # print(u_plot_test.shape)
                #             # print(u_pred_test.shape)
                #             l2_loss = torch.mean((u_plot_test - u_pred_plot) ** 2)
                #             print('[Test Iter:%d, Loss: %.5e]'%(Epoch, l2_loss))
                #             for i in range(Num_DataSets):
                #                 u_test_i = u_plot_test[:,i]
                #                 u_pred_plot_i = u_pred_plot.detach()[:,i]
                #                 plot_2d(Epoch, y_plot_test, x_plot_test, u_pred_plot_i, u_test_i, grid_plot_size, output_path, 'Pde'+'_'+str(i)+'_'+optim_type)
                #                 if Num_DataSets!=1:
                #                     y_plot_test_new = input_test[:,0].to(Device)
                #                     x_plot_test_new = input_test[:,1].to(Device)
                #                     u_pred_test_new = u_pred_test.detach()[:,i]
                #                     plot_predicted_u(Epoch, y_plot_test_new, x_plot_test_new, u_pred_test_new, grid_size, output_path, 'Pde'+'_'+str(i)+'_'+optim_type)
                #                 else:
                #                     y_plot_test = input_test[:,0].to(Device)
                #                     x_plot_test = input_test[:,1].to(Device)
                #                     u_pred_test = u_pred_test.detach()[:,i]
                #                     plot_predicted_u(Epoch, y_plot_test, x_plot_test, u_pred_test, grid_size, output_path, 'Pde'+'_'+str(i)+'_'+optim_type)
                #         elif (Epoch%50 == 0 or Epoch == max_iter - 1) and problem_dim == 2 and u_test is not None:
                #             t_plot_test = u_test[:,0].to(Device)
                #             x_plot_test = u_test[:,1].to(Device)
                #             y_plot_test = u_test[:,2].to(Device)
                #             input_plot_test = u_test[:,0:problem_dim+1].to(Device)
                #             # print(u_test.shape)
                #             u_plot_test = u_test[:,problem_dim+1:].to(Device)
                #             u_pred_plot = Pig_Net(input_plot_test, norm_flag)
                #             # print("u_plot_test",u_plot_test.device)
                #             # print("u_pred_test",u_pred_test.device)
                #             # print(u_plot_test.shape)
                #             # print(u_pred_test.shape)
                #             l2_loss = torch.mean((u_plot_test - u_pred_plot) ** 2)
                #             print('[Test Iter:%d, Loss: %.5e]'%(Epoch, l2_loss))
                #             for i in range(Num_DataSets):
                #                 u_test_i = u_plot_test[:,i]
                #                 u_pred_plot_i = u_pred_plot.detach()[:,i]
                #                 plot_3d(epoch=Epoch, x=x_plot_test, y=y_plot_test, u_gt_all=u_test_i, u_pred_all=u_pred_plot_i, t=t_plot_test, grid_size=grid_plot_size, output_path=output_path, tag='Pde'+'_'+str(i)+'_'+optim_type, step=step)
                #                 t_plot_test = input_test[:,0].to(Device)
                #                 x_plot_test = input_test[:,1].to(Device)
                #                 y_plot_test = input_test[:,2].to(Device)
                #                 u_pred_test = u_pred_test.detach()[:,i]
                #                 plot_3d_predicted_u(epoch=Epoch, x=x_plot_test,y=y_plot_test,u_pred_all=u_pred_test,t=t_plot_test,grid_size=grid_size,output_path=output_path, tag='Pde'+'_'+str(i)+'_'+optim_type, step=step)
                #     # del u_pred_plot, u_plot_test, u_test_i, u_pred_plot_i
                #     torch.cuda.empty_cache()
                # except RuntimeError as e:
                #     if 'out of memory' in str(e):
                #         torch.cuda.empty_cache()  
                #         # if (Epoch%50 == 0 or Epoch == max_iter - 1) and problem_dim == 2 and u_test is not None:
                #         #     input_plot_test = u_test[:,0:problem_dim+1].to(Device)
                #         #     l2_loss = torch.mean((u_test[:,problem_dim+1:].to(Device) - Pig_Net(input_plot_test, norm_flag)) ** 2)
                #         #     print('[Test Iter:%d, Loss: %.5e]'%(Epoch, l2_loss))
                #     else:
                #         raise e 
            
            else:
                Pig_Net.to(Device)
                Pig_Net.eval()
                with torch.no_grad():
                    Data_Loss_Container[0] = torch.tensor(0.0, dtype=torch.float32, device=Device)
                    Data_Loss_Value = Data_Loss_Container[0]
                    u_pred_test = Pig_Net(input_test, norm_flag)
                    if other_data is not None:
                        Inputs_combined = torch.cat([input_test,u_pred_test,other_data],dim=1)
                    else:
                        Inputs_combined = torch.cat([input_test,u_pred_test],dim=1)
                
        else:
            Data_Loss_Container[0] = torch.tensor(0.0, dtype=torch.float32, device=Device)
            Data_Loss_Value = Data_Loss_Container[0]
            if other_data is not None:
                Inputs_combined = torch.cat([Inputs_train,other_data],dim=1)
            else:
                Inputs_combined = Inputs_train
        grid_data = Inputs_combined [:, :problem_dim + 1].to(Device)
        # Zero the gradients if they are enabled.
        if (torch.is_grad_enabled()):
            Optimizer.zero_grad()
        
        # Calculate the losses for each dataset.
        for j in range(Num_DataSets):
            # Get the data, weak-form, and L2 losses for the i-th dataset.
            RHS_list = Model_list[j](Inputs_combined , Weight_Functions_List[j], grid_size, int_type)
            U_i = Inputs_combined[:, problem_dim + 1 + j]
            LHS_list = []
            for w in Weight_Functions_List[j]:
                int_k = Integrate(w, U_i,grid_size, grid_data, derivative, int_type)
                LHS_list.append(int_k)
            ith_Weak_Form_Loss_Value, ith_Residual = Weak_Form_Loss(LHS_list, RHS_list)
            ith_L1_Loss_Value = l1_loss(Model_list[j], reg_weight[j], Device)
            # ith_L1_Loss_Value = huber_loss(Model_list[j], reg_weight[j], 0.005, Device)
            # ith_L1_Loss_Value = lp_loss(Model_list[j], reg_weight[j])
            ith_Total_Loss_Value = f_scale*(ith_Weak_Form_Loss_Value + ith_L1_Loss_Value) + Data_Loss_Value

            # Store these losses in the buffers used by the returned dictionary.
            Residual_List[j][:] = ith_Residual.detach()
            Weak_Loss_List[j] = ith_Weak_Form_Loss_Value.detach().item()
            L1_Loss_List[j] = ith_L1_Loss_Value.detach().item()
            Total_Loss_List[j] = ith_Total_Loss_Value.detach().item()

            # Finally, accumulate the total loss.
            Total_Loss_Value += ith_Total_Loss_Value

        # Backpropagate to compute gradients of Total_Loss with respect to
        # network parameters, but only if the loss requires gradients.
        if Total_Loss_Value.requires_grad == True:
            Total_Loss_Value.backward()
        if clip_grad:
                all_params = []
                for m in Model_list:
                    all_params.extend(list(m.parameters()))
                torch.nn.utils.clip_grad_norm_(all_params, max_norm)
        if Writer is not None and Epoch is not None:
            for j, model in enumerate(Model_list):
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        Writer.add_histogram(f'Dataset_{j}/Gradients/{name}', param.grad, Epoch)
        return Total_Loss_Value

    # Update the network parameters.
    Optimizer.step(Closure)
    # Log losses to TensorBoard.
    if Writer is not None and Epoch is not None:
        Writer.add_scalar(f'Data_Loss', Data_Loss_Container[0].detach().item(), Epoch)
        for i in range(Num_DataSets):
            Writer.add_scalar(f'Dataset_{i}/Weak_Form_Loss', Weak_Loss_List[i], Epoch)
            Writer.add_scalar(f'Dataset_{i}/L1_Loss', L1_Loss_List[i], Epoch)
            Writer.add_scalar(f'Dataset_{i}/Total_Loss', Total_Loss_List[i], Epoch)
            Writer.add_histogram(f'Dataset_{i}/Residuals', Residual_List[i], Epoch)

    # Return the residual tensor.
    # return Data_Loss_Container[0], Residual_List, Weak_Loss_List, L1_Loss_List, Total_Loss_List
    return Data_Loss_Container[0].detach().cpu(), Residual_List, Weak_Loss_List, L1_Loss_List, Total_Loss_List

