import starfile
import numpy as np
import mrcfile
import os
from pytorch3d.transforms import matrix_to_euler_angles
import pandas as pd
from tqdm.autonotebook import tqdm


def create_starfile(dataloader, config, root_path, relative_mrcs_path_prefix, star_file):
    '''
    Creates a starfile from a dataloader.
    '''

    ''' Optics '''
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

    ''' Particles '''
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
    print("### Startfile Creation Starts Now ###")
    with tqdm(total=config.simul_num_projs) as pbar:
        for step, model_input in enumerate(dataloader):
            print("Done: " + str(particle_count) + '/' + str(config.simul_num_projs))
            tqdm.write("Done: " + str(particle_count) + '/' + str(config.simul_num_projs))
            projs = model_input['proj'].float().numpy()  # B, 1, S, S
            B = projs.shape[0]
            S = projs.shape[-1]
            rotmats = model_input['rotmat']  # B, 3, 3
            euler_angles_deg = np.degrees(matrix_to_euler_angles(rotmats, 'ZYZ').float().numpy())  # B, 3
            defocusU = model_input['defocusU'].float().numpy()  # B, 1, 1
            defocusV = model_input['defocusV'].float().numpy()  # B, 1, 1
            angleAstigmatism = model_input['angleAstigmatism'].float().numpy()  # B, 1, 1
            shiftX = model_input['shiftX'].reshape(B, 1, 1).float().numpy()  # B, 1, 1
            shiftY = model_input['shiftY'].reshape(B, 1, 1).float().numpy()  # B, 1, 1

            filename = get_filename(step, n_char=6)

            mrc_relative_path = relative_mrcs_path_prefix + filename + mrcs_path_suffix
            mrc_path = os.path.join(root_path, mrc_relative_path)
            mrc = mrcfile.new_mmap(mrc_path, shape=(B, S, S), mrc_mode=2, overwrite=True)

            print("Writing mrcs file")
            for i in range(B):
                mrc.data[i] = projs[i].reshape(S, S)
                image_name = get_filename(i + 1, n_char=6) + '@' + mrc_relative_path
                rlnImageName.append(image_name)
                rlnDefocusU.append(defocusU[i, 0, 0] * 1e4)
                rlnDefocusV.append(defocusV[i, 0, 0] * 1e4)
                rlnDefocusAngle.append(np.degrees(angleAstigmatism[i, 0, 0]))
                rlnOriginXAngst.append(shiftX[i, 0, 0])
                rlnOriginYAngst.append(shiftY[i, 0, 0])
                rlnAngleRot.append(-euler_angles_deg[i, 2])  # to be consistent with RELION dataio (cf dataio.py)
                rlnAngleTilt.append(euler_angles_deg[i, 1])  # to be consistent with RELION dataio (cf dataio.py)
                rlnAnglePsi.append(-euler_angles_deg[i, 0])  # to be consistent with RELION dataio (cf dataio.py)

                # Fixed values
                rlnPhaseShift.append(0.)
                rlnCtfMaxResolution.append(0.)
                rlnCtfFigureOfMerit.append(0.)
                rlnRandomSubset.append(1)
                rlnClassNumber.append(1)
                rlnOpticsGroup.append(1)

            pbar.update(B)
            particle_count += B

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

    starfile_path = os.path.join(root_path, star_file + '.star')

    print("Writing starfile at " + str(starfile_path))
    starfile.write(df, starfile_path, overwrite=True)
    print("Success! Starfile written!")


def get_filename(step, n_char=6):
    if step == 0:
        return '0' * n_char
    else:
        n_dec = int(np.log10(step))
        return '0' * (n_char - n_dec) + str(step)