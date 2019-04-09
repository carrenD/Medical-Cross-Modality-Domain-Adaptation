import os
import logging
from sklearn.decomposition import PCA as sklearnPCA
import numpy as np
from .misc import indent, g_indents, generate_heatmap_label
from .rttypes import MaskableVolume

# initialize module logger
logger = logging.getLogger(__name__)

def create_pruned_vector(volume, roi):
    """takes a BaseVolume and creates a vector of intensities where all masked voxels are excluded

    Args:
        volume -- BaseVolume containing feature voxel intensities and valid frameofreference
        roi    -- roi to use when pruning voxels from the resulting feature vector

    Returns:
        numpy 1Darray (vector) where voxel intensities are only included where roi mask is not 0
    """
    logger.debug('pre-pruning voxel count: {!s}'.format(np.prod(volume.frameofreference.size)))
    # no roi defined
    if (not roi):
        pruned_vector = volume.vectorize()

    else:
        # roi defined - actually need to prune based on roi densemask
        (size_x, size_y, size_z) = volume.frameofreference.size
        densemaskvolume = roi.makeDenseMask(volume.frameofreference)
        pruned_vector = []
        for d in range(size_z):
            for r in range(size_y):
                for c in range(size_x):
                    if (densemaskvolume.array[d, r, c] == 1):
                        pruned_vector.append(volume.array[d, r, c])

        # convert to np.array/vector
        pruned_vector = np.array(pruned_vector)

    # convert array to a vector with dims (N, 1) so concatenation can be performed
    logger.debug('post-pruning voxel count: {!s}'.format(pruned_vector.shape[0]))
    return pruned_vector.reshape((-1, 1))

def expand_pruned_vector(pruned_vector, roi, frameofreference, fill_value=-1):
    """takes a sparse vector of voxel intensities and expands into a dense MaskableVolume, filling masked
    voxel locations with fill_value
    performs opposite function of create_pruned_vector

    Args:
        pruned_vector    -- numpy 1Darray containing voxel intensities only for those positions where roi mask
                            evaluates to 1
        roi              -- roi to use when adding voxels into the MaskableVolume
        frameofreference -- informs the roi about how to create dense binary mask necessary for expansion

    Returns:
        MaskableVolume of the same size as frameofreference specified
    """
    logger.debug('pre-expansion voxel count: {!s}'.format(pruned_vector.shape[0]))
    # no roi defined
    if (not roi):
        return MaskableVolume().fromArray(pruned_vector, frameofreference)

    # roi defined - actually need to expand based on roi densemask
    (size_x, size_y, size_z) = frameofreference.size
    densemaskvolume = roi.makeDenseMask(frameofreference)
    expanded_vector = []
    pruned_idx = 0
    for d in range(size_z):
        for r in range(size_y):
            for c in range(size_x):
                if (densemaskvolume.array[d, r, c] == 1):
                    expanded_vector.append(pruned_vector[pruned_idx])
                    pruned_idx += 1
                else:
                    expanded_vector.append(fill_value)

    # convert to np.array/vector
    expanded_vector = np.array(expanded_vector)
    logger.debug('post-expansion voxel count: {!s}'.format(expanded_vector.shape[0]))

    # convert to MaskableVolume
    return MaskableVolume().fromArray(expanded_vector, frameofreference)


def create_feature_matrix(feature_volumes, roi=None, PCA=False, PCA_components=0.95):
    """takes a list of feature BaseVolumes and combines them into a numpy ndarray of N rows and D features

    where N is the number of samples in each feature vector (voxels in the image) and D is the number of
    feature vectors stored in the "feature_volumes" list.

    Args:
        feature_volumes    --  python list of BaseVolumes that are aligned
    Returns:
        NDArray     --  numpy ndarray with N rows and D columns where N is the number of voxels in the
                        aligned images (in depth-row major order) and D is the number of feature vectors
                        in the list (len(feature_volumes))
    """
    if len(feature_volumes) <= 0:
        logger.warning(indent('no features supplied. skipping', g_indents[1]))
        return None
    else:
        if (roi):
            # use roi.frameofreference as common shape
            frameofreference = roi.frameofreference
        else:
            # find highest resolution volume and use as FrameOfReference
            highest_res_volume = feature_volumes[0]
            highest_res = np.product(highest_res_volume.frameofreference.spacing)
            for volume in feature_volumes[1:]:
                res = np.product(volume.frameofreference.spacing)
                if (res < highest_res):
                    highest_res_volume = volume
                    highest_res = res
            # assign highest res FOR as common shape
            frameofreference = highest_res_volume.frameofreference

        # take the selected FORs shape to be the reference
        ref_shape = frameofreference.size[::-1]  # reverses tuple from (x,y,z) to (z,y,x)
        logger.debug('Common Spacing (z,y,x): ({:f}, {:f}, {:f})'.format(*frameofreference.spacing))
        logger.debug('Common Shape (z,y,x): ({:d}, {:d}, {:d})'.format(*ref_shape))

        # create list of commonly shaped feature vectors
        conformed_feature_list = []
        dense_feature_list = []
        feature_column_labels = []
        for i, vol in enumerate(feature_volumes):
            # check for invalid feature
            if (vol is None):
                logger.debug('empty (None) feature provided at index {:d}, removing and continuing'.format(i))
                continue

            # conform feature volumes and add to list
            conformed_volume = vol.conformTo(frameofreference)

            if (conformed_volume.array.shape != ref_shape):
                logger.warning(indent('shape mismatch. ref={ref:s} != feature[{num:d}]={shape:s}.'
                                      ' removing and continuing'.format(ref=str(ref_shape),
                                                                        num=i,
                                                                        shape=str(conformed_volume.array.shape)),
                                      g_indents[1]))
                continue
            else:
                # concatenate, need to make feat.array a 2d vector
                pruned_feature_vector = create_pruned_vector(conformed_volume, roi)
                conformed_feature_list.append(pruned_feature_vector)
                dense_feature_list.append(conformed_volume.vectorize(roi).reshape((-1, 1)))
                # label generator
                label = generate_heatmap_label(conformed_volume)
                feature_column_labels.append(label)

        # combine accepted features into array of shape (nSamples, nFeatures)
        pruned_feature_array = np.nan_to_num(np.concatenate(conformed_feature_list, axis=1))
        # create expanded/dense version for pickling and using in hierarchical clustering
        dense_feature_array = np.nan_to_num(np.concatenate(dense_feature_list, axis=1))

        # dimensionality reduction
        if PCA:
            pca = sklearnPCA(whiten=False, n_components=PCA_components)  # standard PCA
            # nfeats = pruned_feature_array.shape[1]
            pruned_feature_array = pca.fit_transform(pruned_feature_array)
            # logger.debug('pca: keeping {:d} of {:d} components'.format(pca.n_components, nfeats))

        logger.debug(indent('combined {n:d} features into pruned array of shape: {shape:s}'.format(
            n=pruned_feature_array.shape[1],
            shape=str(pruned_feature_array.shape))
            , g_indents[1]))
        return (pruned_feature_array, frameofreference, dense_feature_array, feature_column_labels)
