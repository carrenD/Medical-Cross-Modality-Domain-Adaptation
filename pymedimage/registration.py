"""registration.py

functions and classes for working with image registation
"""
import logging
from .rttypes import MaskableVolume
import SimpleITK as sitk

# initialize module logger
logger = logging.getLogger(__name__)

def register_MultiModality(ref, volume_list):
    """performs registration between ref MaskableVolume and each MaskableVolume in volume_list

    Args:
        ref          -- MaskableVolume to which others will be registered
        volume_list  -- list of MaskableVolumes that will be registered to ref
        ?HYPERPARAMS?
    """
    ### check for valid inputs
    if (not isinstance(ref, MaskableVolume)):
        logger.exception('ref of type: {:s} is an invalid type. Must be a MaskableVolume or a subtype'.format(type(ref)))
        raise TypeError
    if (isinstance(volume_list, list)):
        # check each element in volume_list
        for i, check_volume in enumerate(volume_list):
            if (not isinstance(check_volume, MaskableVolume)):
                logger.exception('volume[{idx:d}] of type: {type:s} is an invalid type. Must be a MaskableVolume or a ' \
                        'subtype'.format(idx=i, type=type(ref)))
                raise TypeError
    else:
        # volume_list was specified as a single volume, not a list holding a single volume
        if (not isinstance(volume_list, MaskableVolume)):
            logger.exception('Volume of type: {:s} is an invalid type. Must be a MaskableVolume or a subtype'.format(type(volume_list)))
            raise TypeError

    ### extract image volumes and convert to an ITK friendly type
    ref_array = sitk.GetImageFromArray(ref.vectorize(asmatrix=True))
    logger.info(ref_array.GetSize())
    ### upsample to same resolution/shape?

    ### perform affine registration using Mutual Information similarity metric

    ### inject registration parameters into each MaskableVolume, Identity parameters for ref
