import mrcfile
import numpy as np

def save_mrc(output, data, voxel_size=None, header_origin=None):
    """
    Save numpy array as an MRC file.
    
    Parameters
    ----------
    output : string, default None
        if supplied, save the aligned volume to this path in MRC format
    data : torch.Tensor
        image or volume to save
    voxel_size : float, default None
        if supplied, use as value of voxel size in Angstrom in the header
    header_origin : numpy.recarray
        if supplied, use the origin from this header object
    """
    data = data.detach().cpu().numpy()
    mrc = mrcfile.new(output, overwrite=True)
    mrc.header.map = mrcfile.constants.MAP_ID
    mrc.set_data(data.astype(np.float32))
    if voxel_size is not None:
        mrc.voxel_size = voxel_size
    if header_origin is not None:
        mrc.header['origin']['x'] = float(header_origin['origin']['x'])
        mrc.header['origin']['y'] = float(header_origin['origin']['y'])
        mrc.header['origin']['z'] = float(header_origin['origin']['z'])
        mrc.update_header_from_data()
        mrc.update_header_stats()
    mrc.close()
    return
