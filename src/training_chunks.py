import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
from pytorch3d.transforms import matrix_to_euler_angles
import pandas as pd
import starfile

from .utils import cond_mkdir
from .summary_utils import make_heavy_summary, make_time_summary
from .geom_utils import correct_flips_rotations

def dict2cuda(a_dict):
    """
    Loads a dictionary on GPU.

    Parameters
    ----------
    a_dict: Dictionary

    Returns
    -------
    tmp: Dictionary
    """
    tmp = {}
    for key, value in a_dict.items():
       if isinstance(value,torch.Tensor):
           tmp.update({key: value.cuda()})
       else:
           tmp.update({key: value})
    return tmp


def dict2cpu(a_dict):
    """
    Loads a dictionary on CPU.

    Parameters
    ----------
    a_dict: Dictionary

    Returns
    -------
    tmp: Dictionary
    """
    tmp = {}
    for key, value in a_dict.items():
        if isinstance(value,torch.Tensor):
            tmp.update({key: value.cpu()})
        elif isinstance(value, dict):
            tmp.update({key: dict2cpu(value)})
        else:
            tmp.update({key: value})
    return tmp


def split_dict_with_tensors(model_input, max_chunk_sz):
    """
    Splits input dictionary into chunks of sizes max_chunk_sz

    Parameters
    ----------
    model_input: Dictionary
    max_chunk_sz: int

    Returns
    -------
    list_chunked_model_input: list
    """
    model_input_chunked = []
    for key in model_input:
        chunks = torch.split(model_input[key], max_chunk_sz, dim=0)  # split along the batch dimension
        model_input_chunked.append(chunks)

    list_chunked_model_input = [{k:v for k,v in zip(model_input.keys(), curr_chunks)} \
                                    for curr_chunks in zip(*model_input_chunked)]

    return list_chunked_model_input


def initialize_optim(model, optimizer, lr_default, lr_encoder_scaling):
    """
    Initializes optimizer.

    Parameters
    ----------
    model: CryoAI instance
    optimizer: str
    lr_default: float
    lr_encoder_scaling: float

    Returns
    -------
    optim: optimizer
    """
    if optimizer == 'Adam':
        if hasattr(model, 'orientation_encoder'):
            optim = torch.optim.Adam([
                    {'params': model.pred_map.parameters()},
                    {'params': model.cnn_encoder.parameters(), 'lr': lr_default * lr_encoder_scaling},
                    {'params': model.orientation_encoder.parameters(), 'lr': lr_default * lr_encoder_scaling},
                    {'params': model.orientation_regressor.parameters(), 'lr': lr_default * lr_encoder_scaling},
                    ], lr=lr_default, amsgrad=True)
        else:
            optim = torch.optim.Adam([
                {'params': model.pred_map.parameters()},
                {'params': model.cnn_encoder.parameters(), 'lr': lr_default * lr_encoder_scaling},
                ], lr=lr_default, amsgrad=True)
    elif optimizer == 'SGD':
        optim = torch.optim.SGD([
                {'params': model.pred_map.parameters()},
                {'params': model.cnn_encoder.parameters(), 'lr': lr_default * lr_encoder_scaling},
                {'params': model.orientation_encoder.parameters(), 'lr': lr_default * lr_encoder_scaling},
                {'params': model.orientation_regressor.parameters(), 'lr': lr_default * lr_encoder_scaling},
                ], lr=lr_default)
    else:
        raise NotImplementedError
    return optim


def train(model,
          train_dataloader,
          num_particles,
          epochs,
          optimizer,
          lr,
          lr_encoder_scaling,
          steps_til_light_summary,
          epochs_til_heavy_summary,
          root_dir,
          model_dir,
          loss_dict,
          clip_grad=True,
          loss_schedules=None,
          max_chunk_sz=32,
          fast_mode=False,
          write_mrc=False,
          flip_images=False,
          print_times=False
          ):
    """
    Trains a model.

    Parameters
    ----------
    model: nn.Module
    train_dataloader: Dataloader
    num_particles: int
    epochs: int
    optimizer: str
    lr: float
    lr_encoder_scaling: float
    steps_til_light_summary: int
    epochs_til_heavy_summary: int
    root_dir: str
    model_dir: str
    loss_dict: Dictionary
    clip_grad: bool
    loss_schedules: Dictionary
    max_chunk_sz: int
    fast_mode: bool
    write_mrc: bool
    flip_images: bool
    print_times: bool
    """
    optim = initialize_optim(model, optimizer, lr, lr_encoder_scaling)

    if os.path.exists(model_dir):
        pass
    else:
        os.makedirs(model_dir)

    model_dir_postfixed = os.path.join(model_dir, '')

    summaries_dir = os.path.join(model_dir_postfixed, 'summaries')
    cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir_postfixed, 'checkpoints')
    cond_mkdir(checkpoints_dir)

    # Tensorboard writer
    writer = SummaryWriter(summaries_dir)

    # Schedulers
    for name, scheduler in model.schedulers.items():
        scheduler.set_writer(writer)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        total_time = time.time()
        rots_pred = np.empty((num_particles, 3, 3))
        rots_gt = np.empty((num_particles, 3, 3))
        print("### Training Starts Now ###")
        for epoch in range(epochs):
            end_time = time.time()
            for step, model_input in enumerate(train_dataloader):
                start_time = time.time()

                # Restart optimizer
                optim.zero_grad()

                # Chunking
                list_chunked_model_input = split_dict_with_tensors(model_input, max_chunk_sz)

                # Accumulate gradient over all the chunks
                num_chunks = len(list_chunked_model_input)
                batch_avgd_losses = {}
                batch_avgd_tot_loss = 0.
                for chunk_idx, chunked_model_input in enumerate(list_chunked_model_input):
                    to_cuda_start_time = time.time()
                    chunked_model_input = dict2cuda(chunked_model_input)
                    to_cuda_end_time = time.time()

                    forward_start_time = time.time()
                    chunked_model_output = model(chunked_model_input)
                    forward_end_time = time.time()

                    loss_start_time = time.time()
                    train_loss = 0.
                    for loss_name, loss in loss_dict.items():
                        single_loss = loss(chunked_model_output)

                        if loss_schedules is not None and loss_name in loss_schedules:
                            single_loss *= loss_schedules[loss_name](epoch, total_steps)

                        train_loss += single_loss / num_chunks
                        batch_avgd_tot_loss += float(single_loss / num_chunks)
                        if loss_name in batch_avgd_losses:
                            batch_avgd_losses[loss_name] += single_loss / num_chunks
                        else:
                            batch_avgd_losses.update({loss_name: single_loss / num_chunks})
                    loss_end_time = time.time()

                    backward_start_time = time.time()
                    train_loss.backward()
                    backward_end_time = time.time()

                    ind = chunked_model_input['idx'].detach().cpu().numpy()
                    rots_gt[ind] = chunked_model_input['rotmat'].detach().cpu().numpy()
                    if flip_images:
                        rotmat_pred = correct_flips_rotations(chunked_model_output)
                    else:
                        rotmat_pred = chunked_model_output['rotmat']
                    rots_pred[ind] = rotmat_pred.detach().cpu().numpy()

                # Write losses in tensorboard
                for loss_name, loss in batch_avgd_losses.items():
                    writer.add_scalar(loss_name, loss, total_steps)
                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](epoch, total_steps),
                                          total_steps)
                train_losses.append(batch_avgd_tot_loss)
                writer.add_scalar("total_train_loss", batch_avgd_tot_loss, total_steps)

                # Clip gradients
                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                # Optimize
                optim_start_time = time.time()
                optim.step()
                optim_end_time = time.time()

                pbar.update(1)

                # Fast summary
                if not fast_mode and not total_steps % steps_til_light_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, Step %d" % (epoch, float(batch_avgd_tot_loss), total_steps))
                    if print_times:
                        time_dict = {}
                        time_dict['end_time'] = end_time
                        time_dict['start_time'] = start_time
                        time_dict['to_cuda_start_time'] = to_cuda_start_time
                        time_dict['to_cuda_end_time'] = to_cuda_end_time
                        time_dict['forward_start_time'] = forward_start_time
                        time_dict['forward_end_time'] = forward_end_time
                        time_dict['loss_start_time'] = loss_start_time
                        time_dict['loss_end_time'] = loss_end_time
                        time_dict['backward_start_time'] = backward_start_time
                        time_dict['backward_end_time'] = backward_end_time
                        time_dict['optim_start_time'] = optim_start_time
                        time_dict['optim_end_time'] = optim_end_time
                        time_dict['total_time'] = total_time
                        make_time_summary(time_dict, chunked_model_output)

                total_steps += 1
                end_time = time.time()

            if not epoch % epochs_til_heavy_summary and not fast_mode:
                make_heavy_summary(
                    writer,
                    model,
                    model_input,
                    chunked_model_output,
                    rots_pred,
                    rots_gt,
                    total_steps,
                    write_mrc,
                    root_dir
                )
                model.save(checkpoints_dir, epoch + 1)
                print("Model saved.")


    torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'model_final.pth'))
    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'), np.array(train_losses))


def eval_to_starfile(model, dataloader, epochs, root_dir, name, config, gpu_only=False):
    """
    Evaluates an OrientationPredictor and creates a starfile.
    """

    # Optics
    rlnVoltage = config.kV
    rlnSphericalAberration = config.spherical_aberration
    rlnAmplitudeContrast = config.amplitude_contrast
    rlnOpticsGroup = 1
    rlnImageSize = config.map_shape[0]
    rlnImagePixelSize = config.resolution
    optics = {'rlnVoltage': [rlnVoltage],
              'rlnSphericalAberration': [rlnSphericalAberration],
              'rlnAmplitudeContrast': [rlnAmplitudeContrast],
              'rlnOpticsGroup': [rlnOpticsGroup],
              'rlnImageSize': [rlnImageSize],
              'rlnImagePixelSize': [rlnImagePixelSize]}

    # Particles
    rlnImageName = []
    rlnAngleRot = []
    rlnAngleTilt = []
    rlnAnglePsi = []
    rlnOriginXAngst = []
    rlnOriginYAngst = []
    rlnDefocusU = []
    rlnDefocusV = []
    rlnDefocusAngle = []
    rlnPhaseShift = []
    rlnCtfMaxResolution = []
    rlnCtfFigureOfMerit = []
    rlnRandomSubset = []
    rlnClassNumber = []
    rlnOpticsGroup = []

    mrcs_path_suffix = '.mrcs'  # 000000.mrcs

    particle_count = 0
    with tqdm(total=len(dataloader) * epochs) as pbar:
        print("### Evaluation Starts Now ###")
        total_time = time.time()
        total_transfer_time = 0
        for epoch in range(epochs):
            for step, model_input in enumerate(dataloader):
                print("Done: " + str(particle_count) + '/' + str(config.num_particles))
                print("Total time: " + str(time.time() - total_time))

                to_gpu_time = time.time()
                model_input = dict2cuda(model_input)
                print("To GPU time: " + str(time.time() - to_gpu_time))
                total_transfer_time += time.time() - to_gpu_time

                forward_time = time.time()
                model_output = model(model_input)
                print("Forward time: " + str(time.time() - forward_time))

                conversion_time = time.time()
                rotmats = model_output['rotmat'].reshape(-1, 3, 3)  # 2B, 3, 3
                euler_angles_deg = torch.rad2deg(matrix_to_euler_angles(rotmats, 'ZYZ')).reshape(2, -1, 3)[0]
                B = euler_angles_deg.shape[0]
                print("Conversion time: " + str(time.time() - conversion_time))
                indices = model_input['idx']
                defocusU = model_input['defocusU'].reshape(-1)
                defocusV = model_input['defocusV'].reshape(-1)
                angleAstigmatism = model_input['angleAstigmatism'].reshape(-1)
                shiftX = model_output['pred_shift_params']['shiftX'].reshape(-1)
                shiftY = model_output['pred_shift_params']['shiftY'].reshape(-1)
                if not gpu_only:
                    to_cpu_time = time.time()
                    euler_angles_deg = euler_angles_deg.detach()
                    euler_angles_deg = euler_angles_deg.cpu()
                    euler_angles_deg = euler_angles_deg.float().numpy()
                    indices = indices.detach().cpu().numpy()
                    defocusU = defocusU.detach().cpu().numpy()
                    defocusV = defocusV.detach().cpu().numpy()
                    angleAstigmatism = angleAstigmatism.detach().cpu().numpy()
                    shiftX = shiftX.detach().cpu().numpy()
                    shiftY = shiftY.detach().cpu().numpy()
                    print("To CPU time: " + str(time.time() - to_cpu_time))
                    total_transfer_time += time.time() - to_cpu_time
                    write_time = time.time()
                    for i in range(B):
                        rlnImageName.append(indices[i])  # change this, replace by the true path to the mrcs
                        rlnDefocusU.append(defocusU[i])
                        rlnDefocusV.append(defocusV[i])
                        rlnDefocusAngle.append(angleAstigmatism[i])
                        rlnOriginXAngst.append(shiftX[i])
                        rlnOriginYAngst.append(shiftY[i])
                        rlnAngleRot.append(-euler_angles_deg[i, 2])
                        rlnAngleTilt.append(euler_angles_deg[i, 1])
                        rlnAnglePsi.append(-euler_angles_deg[i, 0])

                        # Fixed values
                        rlnPhaseShift.append(0.)
                        rlnCtfMaxResolution.append(0.)
                        rlnCtfFigureOfMerit.append(0.)
                        rlnRandomSubset.append(1)
                        rlnClassNumber.append(1)
                        rlnOpticsGroup.append(1)
                    print("Write time: " + str(time.time() - write_time))

                pbar.update(B)
                particle_count += B

    if gpu_only:
        return 0

    particles = {'rlnImageName': rlnImageName,
                 'rlnAngleRot': rlnAngleRot,
                 'rlnAngleTilt': rlnAngleTilt,
                 'rlnAnglePsi': rlnAnglePsi,
                 'rlnOriginXAngst': rlnOriginXAngst,
                 'rlnOriginYAngst': rlnOriginYAngst,
                 'rlnDefocusU': rlnDefocusU,
                 'rlnDefocusV': rlnDefocusV,
                 'rlnDefocusAngle': rlnDefocusAngle,
                 'rlnPhaseShift': rlnPhaseShift,
                 'rlnCtfMaxResolution': rlnCtfMaxResolution,
                 'rlnCtfFigureOfMerit': rlnCtfFigureOfMerit,
                 'rlnRandomSubset': rlnRandomSubset,
                 'rlnClassNumber': rlnClassNumber,
                 'rlnOpticsGroup': rlnOpticsGroup}

    df = {}

    df['optics'] = pd.DataFrame(optics)
    df['particles'] = pd.DataFrame(particles)

    starfile_path = os.path.join(root_dir, name + '.star')

    print("Total transfer time: " + str(total_transfer_time))

    print("Writing starfile at " + str(starfile_path))
    starfile.write(df, starfile_path, overwrite=True)
    print("Success! Starfile written!")
