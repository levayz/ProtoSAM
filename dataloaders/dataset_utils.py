"""
Utils for datasets
"""
import functools
import numpy as np

import os
import sys
import nibabel as nib
import numpy as np
import pdb
import SimpleITK as sitk

DATASET_INFO = {
    "CHAOST2": {
            'PSEU_LABEL_NAME': ["BGD", "SUPFG"],
            'REAL_LABEL_NAME': ["BG", "LIVER", "RK", "LK", "SPLEEN"],
            '_SEP': [0, 4, 8, 12, 16, 20],
            'MODALITY': 'MR',
            'LABEL_GROUP': {
                'pa_all': set(range(1, 5)),
                0: set([1, 4]), # upper_abdomen, leaving kidneies as testing classes
                1: set([2, 3]), # lower_abdomen
                },
            },

    "SABS": {
            'PSEU_LABEL_NAME': ["BGD", "SUPFG"],

            'REAL_LABEL_NAME': ["BGD", "SPLEEN", "KID_R", "KID_l", "GALLBLADDER", "ESOPHAGUS", "LIVER", "STOMACH", "AORTA", "IVC",\
              "PS_VEIN", "PANCREAS", "AG_R", "AG_L"],
            '_SEP': [0, 6, 12, 18, 24, 30],
            'MODALITY': 'CT',
            'LABEL_GROUP':{
                'pa_all': set( [1,2,3,6]  ),
                0: set([1,6]  ), # upper_abdomen: spleen + liver as training, kidneis are testing
                1: set( [2,3] ), # lower_abdomen
                    }
            },
    "LITS17": {
            'PSEU_LABEL_NAME': ["BGD", "SUPFG"],

            'REAL_LABEL_NAME': ["BGD", "LIVER", "TUMOR"],
            '_SEP': [0, 26, 52, 78, 104],
            'MODALITY': 'CT',
            'LABEL_GROUP':{
                'pa_all': set( [1 , 2]  ),
                0: set([1 ]  ), # liver
                1: set( [ 2] ), # tumor
                2: set([1,2]) # liver + tumor
                }
        
    }

}

def read_nii_bysitk(input_fid, peel_info = False):
    """ read nii to numpy through simpleitk

        peelinfo: taking direction, origin, spacing and metadata out
    """
    img_obj = sitk.ReadImage(input_fid)
    img_np = sitk.GetArrayFromImage(img_obj)
    if peel_info:
        info_obj = {
                "spacing": img_obj.GetSpacing(),
                "origin": img_obj.GetOrigin(),
                "direction": img_obj.GetDirection(),
                "array_size": img_np.shape
                }
        return img_np, info_obj
    else:
        return img_np

        
def get_CT_statistics(scan_fids):
    """
    As CT are quantitative, get mean and std for CT images for image normalizing
    As in reality we might not be able to load all images at a time, we would better detach statistics calculation with actual data loading
    """
    total_val = 0
    n_pix = 0
    for fid in scan_fids:
        in_img = read_nii_bysitk(fid)
        total_val += in_img.sum()
        n_pix += np.prod(in_img.shape)
        del in_img
    meanval = total_val / n_pix

    total_var = 0
    for fid in scan_fids:
        in_img = read_nii_bysitk(fid)
        total_var += np.sum((in_img - meanval) ** 2 )
        del in_img
    var_all = total_var / n_pix

    global_std = var_all ** 0.5

    return meanval, global_std

def MR_normalize(x_in):
    return (x_in - x_in.mean()) / x_in.std()

def CT_normalize(x_in, ct_mean, ct_std):
    """
    Normalizing CT images, based on global statistics
    """
    return (x_in - ct_mean) / ct_std

def get_normalize_op(modality, fids, ct_mean=None, ct_std=None):
    """
    As title
    Args:
        modality:   CT or MR
        fids:       fids for the fold
    """
    if modality == 'MR':
        return MR_normalize

    elif modality == 'CT':
        if ct_mean is None or ct_std is None:
            ct_mean, ct_std = get_CT_statistics(fids)
        # debug
        print(f'###### DEBUG_DATASET CT_STATS NORMALIZED MEAN {ct_mean} STD {ct_std} ######')

        return functools.partial(CT_normalize, ct_mean=ct_mean, ct_std=ct_std)


