import time
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import torch
import os

from .mrc_utils import save_mrc
from .geom_utils import get_rotation_accuracy

def normalize_proj(proj):
    """
    Normalizes an image.

    Parameters
    ----------
    proj: torch.Tensor (N, C, H, W)

    Returns
    -------
    proj_norm: torch.Tensor (N, C, H, W)
    """
    num_projs = proj.shape[0]

    vmin, _ = torch.min(proj.reshape(num_projs, -1), dim=-1)
    vmin = vmin[:, None, None, None]
    vmax, _ = torch.max(proj.reshape(num_projs, -1), dim=-1)
    vmax = vmax[:, None, None, None]
    proj_norm = (proj - vmin) / (vmax - vmin)

    return proj_norm


def make_heavy_summary(writer, model, model_input, model_output, rots_pred, rots_gt, total_steps, write_mrc,
                       root_dir_path):
    """
    Heavy summary.

    Parameters
    ----------
    writer: writer
    summary_dict: Dictionary
    """
    proj_gt = model_input['proj']
    proj_pred = model_output['proj']
    fproj_gt = model_input['fproj']
    fproj_pred = model_output['fproj']
    fproj_pred_pre_ctf = model_output['fproj_prectf']

    # Visualize a single image
    proj_pred = normalize_proj(proj_pred)
    proj_gt = normalize_proj(proj_gt)
    idx = 0
    writer.add_image(f"GT (single)", proj_gt[idx, ...], global_step=total_steps)
    writer.add_image(f"Pred (single)", proj_pred[idx, ...], global_step=total_steps)

    # Visualize Fourier transforms
    idx = 0
    fproj_gt_np = fproj_gt[idx, ...].squeeze().detach().cpu().numpy()
    fproj_pred_np = fproj_pred[idx, ...].squeeze().detach().cpu().numpy()
    fproj_pred_pre_ctf_np = fproj_pred_pre_ctf[idx, ...].squeeze().detach().cpu().numpy()

    fig = plt.figure(dpi=96)
    im = plt.imshow(np.abs(fproj_gt_np), norm=colors.LogNorm(), cmap='RdPu')
    plt.colorbar(im)
    writer.add_figure(f"GT Fourier (single)", fig, global_step=total_steps)
    fig = plt.figure(dpi=96)
    im = plt.imshow(np.abs(fproj_pred_np), norm=colors.LogNorm(), cmap='RdPu')
    plt.colorbar(im)
    writer.add_figure(f"Pred Fourier (single)", fig, global_step=total_steps)
    fig = plt.figure(dpi=96)
    im = plt.imshow(np.abs(fproj_pred_pre_ctf_np), norm=colors.LogNorm(), cmap='RdPu')
    plt.colorbar(im)
    writer.add_figure(f"Pred Fourier No CTF (single)", fig, global_step=total_steps)

    # Rotation accuracy
    MSE, MedSE = get_rotation_accuracy(rots_gt, rots_pred)
    writer.add_scalar('MSE Rots', MSE, total_steps)
    writer.add_scalar('MedSE Rots', MedSE, total_steps)

    # Save mrc file
    if write_mrc:
        volume = model.pred_map.make_volume()
        volume = torch.tensor(volume).detach().cpu()
        filename = os.path.join(root_dir_path, 'reconstruction.mrc')
        save_mrc(filename, volume, voxel_size=model.ctf.resolution, header_origin=None)



def make_time_summary(time_dict, chunked_model_output):
    """
    Summarizes computation times.

    Parameters
    ----------
    time_dict: Dictionary
    chunked_model_output: Dictionary
    """
    end_time = time_dict['end_time']
    start_time = time_dict['start_time']
    to_cuda_start_time = time_dict['to_cuda_start_time']
    to_cuda_end_time = time_dict['to_cuda_end_time']
    forward_start_time = time_dict['forward_start_time']
    forward_end_time = time_dict['forward_end_time']
    loss_start_time = time_dict['loss_start_time']
    loss_end_time = time_dict['loss_end_time']
    backward_start_time = time_dict['backward_start_time']
    backward_end_time = time_dict['backward_end_time']
    optim_start_time = time_dict['optim_start_time']
    optim_end_time = time_dict['optim_end_time']
    total_time = time_dict['total_time']
    print("-----")
    print("Data loading time: " + str(start_time - end_time))
    print("To CUDA time: " + str(to_cuda_end_time - to_cuda_start_time))
    print("Forward time: " + str(forward_end_time - forward_start_time))
    print("Loss time: " + str(loss_end_time - loss_start_time))
    print("Backward time: " + str(backward_end_time - backward_start_time))
    print("Optimization step time: " + str(optim_end_time - optim_start_time))
    print("Total iteration time: " + str(time.time() - start_time))
    print("Total time: " + str(time.time() - total_time))
    print("-----")
    print("Encoding time: " + str(chunked_model_output['times']['encoder'] -
                                  chunked_model_output['times']['start']))
    print("CTF shift prep time: " + str(chunked_model_output['times']['ctf_shift_prep'] -
                                        chunked_model_output['times']['encoder']))
    print("Projection time: " + str(chunked_model_output['times']['decoder'] -
                                    chunked_model_output['times']['ctf_shift_prep']))
    print("CTF shift time: " + str(chunked_model_output['times']['end'] -
                                   chunked_model_output['times']['decoder']))
    print("-----")
