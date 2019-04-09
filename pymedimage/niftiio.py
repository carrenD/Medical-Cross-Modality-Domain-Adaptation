"""
nifti_io.py
Used for NIfTI-1 file IO
"""
import os
import sys
import nibabel as nib
import numpy as np
import pdb

# TODO: look into the documentation and find a way to preserve meta data

def write_nii(array_data, filename, path = "", affine = None):
    """write np array into nii file"""
    if affine is None:
        print("No information about the global coordinate system")
        affine = np.diag([1,1,1,1])
    #pdb.set_trace()
    #TODO: to check if it works
   # array_data = np.int16(array_data)
    array_img = nib.Nifti1Image(array_data, affine)
    save_fid = os.path.join(path,filename)
    try:
        array_img.to_filename(save_fid)
        print("Nii object %s has been saved!"%save_fid)
    except:
        raise Exception("file %s cannot be saved!"%save_fid)
    return save_fid

def read_nii_image(input_fid):
    """read the nii image data into numpy array"""
    img = nib.load(input_fid)
    return img.get_data()

def read_nii_object(input_fid):
    """ directly read the nii object """
    #pdb.set_trace()
    return nib.load(input_fid)

# write resampled nii file. Keep the corresponding metadata unchanged, for 3dslicer visualization
def write_resampled_nii(original_obj, data_vol, filename, new_res = None, debug = False, output_dir = None, output_prefix = False):
    """ write resampleed nii file """
    affine = original_obj.get_affine()

    if debug == True:
        old_affine = affine
    # replace the affine scale parameter with new resolution
    if new_res is None:
        print("New resolution for resampling is not provided, use old resolution")
    else:
        for i in range(3):
            #pdb.set_trace()
            affine[i,i] = new_res[i]
        affine[0,0] *= -1

    if debug == True:
        print("Old affine matrix: ")
        print(old_affine)
        print("New affine matrix: ")
        print(affine)

    output_obj = nib.Nifti1Image(data_vol, affine)
    if (output_dir is None) or (not (os.path.isdir(output_dir))):
        output_dir = os.path.dirname(filename)
        print("output directory for resample output not specified or incorrect, save to the output to the same folder as input")
    if output_dir == "":
        output_dor = "./"
    if output_prefix is False:
        filename = os.path.join(output_dir, os.path.basename(filename))
    else:
        filename = os.path.join(output_dir, "resampled_" + os.path.basename(filename))
    output_obj.to_filename(filename)
    return filename
