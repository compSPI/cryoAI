import mrcfile, torch

import sys, os
module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)
from .utils import align_volumes

rec = mrcfile.open("/Users/axlevy/Desktop/CompSPI/mrc_files/80S/gt_poses/half_1/reconstruction.mrc").data.copy()
ref = mrcfile.open("/Users/axlevy/Desktop/CompSPI/mrc_files/80S/refined/half_A_aligned.mrc").data.copy()

ref = torch.Tensor(ref)
rec = torch.Tensor(rec)

opt_q = align_volumes(rec, ref, zoom=0.5, sigma=0.75, nscs=2, voxel_size=3.77,
                      output="/Users/axlevy/Desktop/CompSPI/mrc_files/80S/gt_poses/half_1/reconstruction_aligned.mrc")