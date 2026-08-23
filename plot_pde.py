import numpy as np
import torch
import sys
import os


import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable



rc('text', usetex=False)


def plot_2d(it, y, x, u, u_gt, grid_size, output_path, tag):
    # Move the tensors back to the CPU.
    y = y.detach().cpu().numpy().reshape(grid_size[0], grid_size[1])
    x = x.detach().cpu().numpy().reshape(grid_size[0], grid_size[1])
    u = u.detach().cpu().numpy().reshape(grid_size[0], grid_size[1])
    u_gt = u_gt.cpu().numpy().reshape(grid_size[0], grid_size[1])

    # Plot the results.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].set_aspect('auto')
    col0 = axes[0].pcolormesh(x, y, u_gt, cmap='rainbow', shading='auto')
    axes[0].set_xlabel('x', fontsize=12, labelpad=12)
    axes[0].set_ylabel('t', fontsize=12, labelpad=12)
    axes[0].set_title('Exact U', fontsize=18, pad=18)
    div0 = make_axes_locatable(axes[0])
    cax0 = div0.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(col0, cax=cax0)

    axes[1].set_aspect('auto')
    col1 = axes[1].pcolormesh(x, y, u, cmap='rainbow', shading='auto')
    axes[1].set_xlabel('x', fontsize=12, labelpad=12)
    axes[1].set_ylabel('t', fontsize=12, labelpad=12)
    axes[1].set_title('Predicted U', fontsize=18, pad=18)
    div1 = make_axes_locatable(axes[1])
    cax1 = div1.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(col1, cax=cax1)

    # Absolute error.
    axes[2].set_aspect('auto')
    col2 = axes[2].pcolormesh(x, y, np.abs(u-u_gt), cmap='rainbow', shading='auto')
    axes[2].set_xlabel('x', fontsize=12, labelpad=12)
    axes[2].set_ylabel('t', fontsize=12, labelpad=12)
    axes[2].set_title('Absolute error', fontsize=18, pad=18)
    div2 = make_axes_locatable(axes[2])
    cax2 = div2.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(col2, cax=cax2)
    cbar.mappable.set_clim(0, 1)
    plt.tight_layout()
    if it % 25 ==0:
        fig.savefig(output_path + "/{}_{}.png".format(tag, it))
    plt.clf()
    plt.close(fig)

def plot_sub_3d(it, y, x, u, u_gt, grid_size, output_path, tag):
    # Move the tensors back to the CPU.
    y = y.reshape(grid_size[0], grid_size[1])
    x = x.reshape(grid_size[0], grid_size[1])
    u = u.reshape(grid_size[0], grid_size[1])
    u_gt = u_gt.reshape(grid_size[0], grid_size[1])

    # Plot the results.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].set_aspect('auto')
    col0 = axes[0].pcolormesh(x, y, u_gt, cmap='rainbow', shading='auto')
    axes[0].set_xlabel('x', fontsize=12, labelpad=12)
    axes[0].set_ylabel('y', fontsize=12, labelpad=12)
    axes[0].set_title('Exact U', fontsize=18, pad=18)
    div0 = make_axes_locatable(axes[0])
    cax0 = div0.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(col0, cax=cax0)

    axes[1].set_aspect('auto')
    col1 = axes[1].pcolormesh(x, y, u, cmap='rainbow', shading='auto')
    axes[1].set_xlabel('x', fontsize=12, labelpad=12)
    axes[1].set_ylabel('y', fontsize=12, labelpad=12)
    axes[1].set_title('Predicted U', fontsize=18, pad=18)
    div1 = make_axes_locatable(axes[1])
    cax1 = div1.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(col1, cax=cax1)

    # Absolute error.
    axes[2].set_aspect('auto')
    col2 = axes[2].pcolormesh(x, y, np.abs(u-u_gt), cmap='rainbow', shading='auto')
    axes[2].set_xlabel('x', fontsize=12, labelpad=12)
    axes[2].set_ylabel('y', fontsize=12, labelpad=12)
    axes[2].set_title('Absolute error', fontsize=18, pad=18)
    div2 = make_axes_locatable(axes[2])
    cax2 = div2.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(col2, cax=cax2)
    cbar.mappable.set_clim(0, 1)
    
    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(output_path + "/{}_{}.png".format(tag, it))
    plt.clf()
    plt.close(fig)

def plot_3d(epoch, x, y, u_gt_all, u_pred_all, t, grid_size, output_path, tag, step=20):
    u_gt_all = u_gt_all.detach().cpu().numpy().reshape(grid_size[0], grid_size[1], grid_size[2])
    u_pred_all = u_pred_all.detach().cpu().numpy().reshape(grid_size[0], grid_size[1], grid_size[2])
    x = x.detach().cpu().numpy().reshape(grid_size[0], grid_size[1], grid_size[2])
    y = y.detach().cpu().numpy().reshape(grid_size[0], grid_size[1], grid_size[2])
    t = t.detach().cpu().numpy().reshape(grid_size[0], grid_size[1], grid_size[2])

    sub_outpath = output_path + "/{}_{}".format(tag, epoch)
    os.makedirs(sub_outpath, exist_ok=True)
    for i in range(0, len(t), step):
        
        plot_sub_3d(
            it=i,
            y=y[i, :, :],
            x=x[i, :, :],
            u=u_pred_all[i, :, :],
            u_gt=u_gt_all[i, :, :],
            grid_size=[grid_size[1],grid_size[2]],
            output_path=sub_outpath,
            tag=f"{tag}_t{i}"
        )


def plot_predicted_u(it, y, x, u, grid_size, output_path, tag):
    y = y.detach().cpu().numpy().reshape(grid_size[0], grid_size[1])
    x = x.detach().cpu().numpy().reshape(grid_size[0], grid_size[1])
    u = u.detach().cpu().numpy().reshape(grid_size[0], grid_size[1])

    fig, ax = plt.subplots(figsize=(8, 8)) 

    ax.set_aspect('equal') 
    col = ax.pcolormesh(x, y, u, cmap='rainbow', shading='auto')

    ax.set_xlabel('x', fontsize=12, labelpad=12)
    ax.set_ylabel('t', fontsize=12, labelpad=12)
    ax.set_title('Predicted U', fontsize=18, pad=18)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(col, cax=cax)

    plt.tight_layout()
    fig.savefig(f"{output_path}/{tag}_predictedU_{it}.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close(fig)


def plot_sub_predicted_u(it, y, x, u, grid_size, output_path, tag):
    y = y.reshape(grid_size[0], grid_size[1])
    x = x.reshape(grid_size[0], grid_size[1])
    u = u.reshape(grid_size[0], grid_size[1])

    fig, ax = plt.subplots(figsize=(8, 8))  # Square figure.

    ax.set_aspect('equal')  # Keep equal horizontal and vertical scales.
    col = ax.pcolormesh(x, y, u, cmap='rainbow', shading='auto')

    ax.set_xlabel('x', fontsize=12, labelpad=12)
    ax.set_ylabel('y', fontsize=12, labelpad=12)
    ax.set_title('Predicted U', fontsize=18, pad=18)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(col, cax=cax)

    plt.tight_layout()
    fig.savefig(f"{output_path}/{tag}_predictedU_{it}.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close(fig)

def plot_3d_predicted_u(epoch, x, y, u_pred_all, t, grid_size, output_path, tag, step=20):
    u_pred_all = u_pred_all.detach().cpu().numpy().reshape(grid_size[0], grid_size[1], grid_size[2])
    x = x.detach().cpu().numpy().reshape(grid_size[0], grid_size[1], grid_size[2])
    y = y.detach().cpu().numpy().reshape(grid_size[0], grid_size[1], grid_size[2])
    t = t.detach().cpu().numpy().reshape(grid_size[0], grid_size[1], grid_size[2])

    sub_outpath = output_path + '/only_u' + "/{}_{}".format(tag, epoch)
    os.makedirs(sub_outpath, exist_ok=True)
    for i in range(0, len(t), step):
        
        plot_sub_predicted_u(
            it=i,
            y=y[i, :, :],
            x=x[i, :, :],
            u=u_pred_all[i, :, :],
            grid_size=[grid_size[1],grid_size[2]],
            output_path=sub_outpath,
            tag=f"{tag}_t{i}"
        )

def save_loss_list(problem, loss_list, it, output_path, save_it = 50):
    if problem =='inverse':
        save_it = 5
    if it % save_it ==0:
        np.save(output_path + "/loss_{}.png".format(it), loss_list)


def test(y_test, x_test, u_test, net_u, it, loss_list, output_path, tag, num_test):
    u_pred = net_u(y_test, x_test)
    u_pred_arr = u_pred.detach().cpu().numpy()
    u_test_arr = u_test.detach().cpu().numpy()
    
    l2_loss = np.linalg.norm(u_pred_arr - u_test_arr) / np.linalg.norm(u_test_arr)
    
    loss_list.append(l2_loss)
    if it % 100 ==0 :
        print('[Test Iter:%d, Loss: %.5e]'%(it, l2_loss))
    
    sys.stdout.flush()
    save_loss_list('forward', loss_list, it, output_path)

    if it % 1 == 0 :
        plot_2d(it, y_test, x_test, u_pred.detach(), u_test, num_test, output_path, tag)

def test_3d(y_test, x_test, u_test, net_u, it, loss_list, output_path, tag, num_test):
    u_pred = net_u(y_test, x_test)
    u_pred_arr = u_pred.detach().cpu().numpy()
    u_test_arr = u_test.detach().cpu().numpy()
    
    l2_loss = np.linalg.norm(u_pred_arr - u_test_arr) / np.linalg.norm(u_test_arr)
    
    loss_list.append(l2_loss)
    if it % 100 ==0 :
        print('[Test Iter:%d, Loss: %.5e]'%(it, l2_loss))
    
    sys.stdout.flush()
    save_loss_list('forward', loss_list, it, output_path)

    if it % 1 == 0 :
        plot_3d(it, x_test[:,0], x_test[:,1], u_test, u_pred, y_test, num_test, output_path, tag, step=20)
        
