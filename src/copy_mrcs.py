import starfile
import mrcfile
import os
import pandas as pd

root = '/sdf/group/ml/CryoNet/axlevy/cryonettorch/simulated_starfiles_v7'
optics_created = False

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

N = 100

for i in range(N):
    id = i + 1
    print("Processing " + str(id) + " / "+str(N))
    folder = '80S_simul_128_100k_' + str(id)
    df = starfile.open(os.path.join(os.path.join(root, folder), folder + '.star'))

    if not optics_created:
        optics = df['optics']
        optics_created = True

    num_particles = len(df['particles'])

    for j in range(num_particles):
        if not j % 1000:
            print("File "+str(id)+" / "+str(N)+" ; Particle "+str(j+1)+" / "+str(num_particles))
        particle = df['particles'].iloc[j]
        imgnamedf = particle['rlnImageName'].split('@')
        n_part_in_mrc = imgnamedf[0]
        relative_path = imgnamedf[1]
        image_name = n_part_in_mrc + '@' + folder + '/' + relative_path
        rlnImageName.append(image_name)
        rlnDefocusU.append((df['particles'].iloc[j])['rlnDefocusU'])
        rlnDefocusV.append((df['particles'].iloc[j])['rlnDefocusV'])
        rlnDefocusAngle.append((df['particles'].iloc[j])['rlnDefocusAngle'])
        rlnPhaseShift.append((df['particles'].iloc[j])['rlnPhaseShift'])
        rlnCtfMaxResolution.append((df['particles'].iloc[j])['rlnCtfMaxResolution'])
        rlnCtfFigureOfMerit.append((df['particles'].iloc[j])['rlnCtfFigureOfMerit'])
        rlnRandomSubset.append((df['particles'].iloc[j])['rlnRandomSubset'])
        rlnClassNumber.append((df['particles'].iloc[j])['rlnClassNumber'])
        rlnOpticsGroup.append((df['particles'].iloc[j])['rlnOpticsGroup'])
        rlnAngleRot.append((df['particles'].iloc[j])['rlnAngleRot'])
        rlnAngleTilt.append((df['particles'].iloc[j])['rlnAngleTilt'])
        rlnAnglePsi.append((df['particles'].iloc[j])['rlnAnglePsi'])
        rlnOriginXAngst.append((df['particles'].iloc[j])['rlnOriginXAngst'])
        rlnOriginYAngst.append((df['particles'].iloc[j])['rlnOriginYAngst'])

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
df['optics'] = optics
df['particles'] = pd.DataFrame(particles)

starfile_path = os.path.join(root, '80S_simul_128_10M.star')

print("Writing starfile at " + str(starfile_path))
starfile.write(df, starfile_path, overwrite=True)
print("Success! Starfile written!")

# singularity exec -B /sdf --nv /sdf/group/ml/CryoNet/singularity_images/cryonettorch_e2cnn.sif python