"""
features.py

Utility functions for calculating common image features
"""
import logging
import math
import time
import numpy as np
import scipy.ndimage
import scipy.stats
import pywt
from .rttypes import MaskableVolume, FrameOfReference
from .misc import g_indents, indent, timer

# initialize module logger
logger = logging.getLogger(__name__)

# indent shortnames
l3 = g_indents[3]
l4 = g_indents[4]

def image_iterator(processing_function, image_volume, radius=2, roi=None):
    """compute the pixel-wise feature of an image over a region defined by neighborhood

    Args:
        processing_function -- function that should be applied to at each voxel location with neighborhood
                                context. Function signature should match:
                                    fxn()
                            -- list<callables> that will all be evaluated at each patch location, results will
                                be stored to separate result MaskableVolume objects
        image -- a flattened array of pixel intensities of type imslice or a matrix shaped numpy ndarray
        radius -- describes neighborood size in each dimension. radius of 4 would be a 9x9x9
    Returns:
        feature_volume as MaskableVolume with shape=image.shape
    """
    # This is an ugly way of type-checking but cant get isinstance to see both as the same
    if (MaskableVolume.__name__ in str(type(image_volume))):
        (c, r, d) = image_volume.frameofreference.size
        def get_val(image_volume, z, y, x):
            # image boundary handling is built into BaseVolume.get_val
            return image_volume.get_val(z, y, x)
        def set_val(feature_volume, z, y, x, val):
            feature_volume.set_val(z, y, x, val)

        #instantiate a blank BaseVolume of the proper size
        feature_volume = MaskableVolume().fromArray(np.zeros((d, r, c)), image_volume.frameofreference)
        # feature_volume.modality = image_volume.modality
        # feature_volume.feature_label = 'feature'
    elif isinstance(image_volume, np.ndarray):
        if image_volume.ndim == 3:
            d, r, c = image_volume.shape
        elif image_volume.ndim == 2:
            d, r, c = (1, *image_volume.shape)
            image_volume = image_volume.reshape((d, r, c))

        # instantiate a blank np.ndarray of the proper size
        feature_volume = np.zeros((d, r, c))

        def get_val(image, z, y, x):
            if (z<0 or y<0 or x<0) or (z>=d or y>=r or x>=c):
                return 0
            else:
                return image[z, y, x]
        def set_val(image, z, y, x, val):
            image[z, y, x] = val
    else:
        logger.info('invalid image type supplied ({:s}). Please specify an image of type BaseVolume \
            or type np.ndarray'.format(str(type(image_volume))))
        return None

    # z_radius_range controls 2d neighborhood vs 3d neighborhood for 2d vs 3d images
    if d == 1:  # 2D image
        logger.debug(indent('Computing 2D feature with radius: {:d}'.format(radius), l3))
        z_radius_range = [0]
    elif d > 1:  # 3D image
        logger.debug(indent('Computing 3D feature with radius: {:d}'.format(radius), l3))
        z_radius_range = range(-radius, radius+1)

    # in plane patch range
    radius_range = range(-radius, radius+1)

    # timing
    start_feature_calc = time.time()


    # absolute max indices for imagevolume - for handling request of voxel out of bounds
    cbound = c
    rbound = r
    dbound = d

    # set calculation index bounds -- will be redefined if roi is specified
    cstart, cstop = 0, cbound
    rstart, rstop = 0, rbound
    dstart, dstop = 0, dbound

    # defines dimensionality
    d_subset = dstop - dstart
    r_subset = rstop - rstart
    c_subset = cstop - cstart

    # restrict calculation bounds to roi
    if (roi is not None):
        # get max extents of the mask/ROI to speed up calculation only within ROI cubic volume
        extents = roi.getROIExtents()
        cstart, rstart, dstart = image_volume.frameofreference.getIndices(extents.start)
        cstop, rstop, dstop = np.subtract(image_volume.frameofreference.getIndices(extents.end()), 1)
        logger.info(indent('calculation subset volume x=({xstart:d}->{xstop:d}), '
                                               'y=({ystart:d}->{ystop:d}), '
                                               'z=({zstart:d}->{zstop:d})'.format(zstart=dstart,
                                                                                  zstop=dstop,
                                                                                  ystart=rstart,
                                                                                  ystop=rstop,
                                                                                  xstart=cstart,
                                                                                  xstop=cstop ), l4))
        # redefine feature_volume
        d_subset = dstop - dstart
        r_subset = rstop - rstart
        c_subset = cstop - cstart
        feature_frameofreference = FrameOfReference((extents.start),
                                                    (image_volume.frameofreference.spacing),
                                                    (c_subset, r_subset, d_subset))
        feature_volume = feature_volume.fromArray(np.zeros((d_subset, r_subset, c_subset)), feature_frameofreference)

    # # setup an output volume for each feature in processing_function list
    # if (not isinstance(processing_function, list)):
    #     tmp = []
    #     tmp.append(processing_function)
    #     processing_function = tmp
    # feature_volumes = [feature_volume]
    # for funct in processing_function[1:]:
    #     feature_volumes.append(np.zeros_like(feature_volume))

    # nested loop approach -> slowest, try GPU next
    total_voxels = d * r * c
    subset_total_voxels = d_subset * r_subset * c_subset
    #onepercent = int(subset_total_voxels / 100)
    fivepercent = int(subset_total_voxels / 100 * 5)

    idx = -1
    subset_idx = -1
    z_idx = -1
    for z in range(dstart, dstop):
        z_idx += 1
        y_idx = -1
        x_idx = -1
        for y in range(rstart, rstop):
            y_idx += 1
            x_idx = -1
            for x in range(cstart, cstop):
                x_idx += 1
                idx += 1
                if (z<dstart or z>dstop or y<rstart or y>rstop or x<cstart or x>cstop):
                    # we shouldnt ever be here
                    logger.info('why are we here?!')
                    #fill 0 instead
                    set_val(feature_volume, z_idx, y_idx, x_idx, 0)
                else:
                    subset_idx += 1
                    patch_vals = np.zeros((len(z_radius_range), len(radius_range), len(radius_range)))
                    for p_z, k_z in enumerate(z_radius_range):
                        for p_x, k_x in enumerate(radius_range):
                            for p_y, k_y in enumerate(radius_range):
                                #logger.info('k_z:{z:d}, k_y:{y:d}, k_x:{x:d}'.format(z=k_z,y=k_y,x=k_x))
                                # handle out of bounds requests - replace with 0
                                request_z = z+k_z
                                request_y = y+k_y
                                request_x = x+k_x
                                if (request_z < 0 or request_z >= dbound or
                                    request_y < 0 or request_y >= rbound or
                                    request_x < 0 or request_x >= cbound):
                                    val = 0
                                else:
                                    val = get_val(image_volume, request_z, request_y, request_x)
                                # store to local image patch
                                patch_vals[p_z, p_y, p_x] = val

                    # for i, funct in enumerate(processing_function):
                    proc_value = processing_function(patch_vals)
                    set_val(feature_volume, z_idx, y_idx, x_idx, proc_value)

                    if (False and (subset_idx % fivepercent == 0 or subset_idx == subset_total_voxels-1)):
                        logger.debug('feature value at ({x:d}, {y:d}, {z:d})= {e:f}'.format(
                            x=z*y*x + y*x + x,
                            y=z*y*x + y,
                            z=z*y*x,
                            e=proc_value))

                    if ((subset_idx % fivepercent == 0 or subset_idx == subset_total_voxels-1)):
                        logger.debug(indent('{p:0.2%} - voxel: {i:d} of {tot:d} (of total: {abstot:d})'.format(
                            p=subset_idx/subset_total_voxels,
                            i=subset_idx,
                            tot=subset_total_voxels,
                            abstot=total_voxels), l4))

    if isinstance(image_volume, np.ndarray) and d == 1:
        # need to reshape ndarray if input was 2d
        feature_volume = feature_volume.reshape((r_subset, c_subset))
        # for i, feature_volume in enumerate(feature_volumes):
        #     feature_volumes[i] = feature_volume.reshape((r_subset, c_subset))


    end_feature_calc = time.time()
    logger.debug(timer('feature calculation time:', end_feature_calc-start_feature_calc, l3))
    # if len(features_volumes > 1):
    #     return feature_volumes
    # else:
    return feature_volume

def energy_plugin(patch_vals):
    val_counts = {}

    # get occurence counts
    for val in patch_vals.flatten().tolist():
        if val in val_counts:
            val_counts[val] += 1
        else:
            val_counts[val] = 1

    # create new dict to store class probabilities
    val_probs = np.zeros(((len(val_counts))))
    total_counts = sum(val_counts.values())
    for i, val in enumerate(val_counts.keys()):
        val_probs[i] = val_counts[val]/total_counts
    # calculate energy
    return math.sqrt(np.dot(val_probs, val_probs))

def entropy_plugin(patch_vals):
    val_counts = {}

    # get occurence counts
    for val in patch_vals.flatten().tolist():
        if val in val_counts:
            val_counts[val] += 1
        else:
            val_counts[val] = 1

    #create new dict to store class probabilities
    val_probs = np.zeros(((len(val_counts))))
    total_counts = sum(val_counts.values())
    for i, val in enumerate(val_counts.keys()):
        val_probs[i] = val_counts[val]/total_counts
    # calculate local entropy
    h = -np.sum(val_probs*np.log(val_probs))

    return h


def image_entropy(image_volume, radius=2, roi=None):
    return image_iterator(entropy_plugin, image_volume, radius, roi)

def image_energy(image_volume, radius=2, roi=None):
    return image_iterator(energy_plugin, image_volume, radius, roi)


def wavelet_decomp_3d(image_volume, wavelet_str='db1', mode_str='smooth'):
    """perform full 3d wavelet decomp and return coefficients"""
    coeffs = pywt.wavedecn(image_volume.array, wavelet_str, mode_str)
    return coeffs


def wavelet_entropy(image_volume, radius=2, roi=None, wavelet_str='db1', mode_str='smooth'):
    # compute wavelet coefficients
    logger.info(indent('performing 3d wavelet decomp using wavelet: {!s}'.format(wavelet_str), g_indents[3]))
    roi_volume = image_volume.conformTo(roi.frameofreference)
    wavelet_coeffs = wavelet_decomp_3d(roi_volume, wavelet_str, mode_str)
    nlevels = len(wavelet_coeffs) - 1
    # level_results = []
    accumulator = np.zeros(roi_volume.frameofreference.size[::-1])
    # sum voxel-wise energy across all levels
    for level in range(nlevels-1, 0, -1):
        wavelet_coeffs_diag = wavelet_coeffs[level+1]['ddd']
        logger.info(indent('computing entropy for level {:d} of shape:{!s}'.format(level, wavelet_coeffs_diag.shape), g_indents[3]))
        result = image_iterator(entropy_plugin, wavelet_coeffs_diag, radius)

        zoomfactors = tuple(np.true_divide(roi_volume.frameofreference.size[::-1], result.shape))
        # scale low-res coefficients to image res
        result = scipy.ndimage.interpolation.zoom(result, zoomfactors, order=3)
        result = MaskableVolume().fromArray(result, roi_volume.frameofreference)
        # level_results.append(result)
        accumulator = np.add(accumulator, result.array)
    return MaskableVolume().fromArray(accumulator, roi_volume.frameofreference)

def wavelet_energy(image_volume, radius=2, roi=None, wavelet_str='db1', mode_str='smooth'):
    # compute wavelet coefficients
    logger.info(indent('performing 3d wavelet decomp using wavelet: {!s}'.format(wavelet_str), g_indents[3]))
    roi_volume = image_volume.conformTo(roi.frameofreference)
    wavelet_coeffs = wavelet_decomp_3d(roi_volume, wavelet_str, mode_str)
    nlevels = len(wavelet_coeffs) - 1
    # level_results = []
    accumulator = np.zeros(roi_volume.frameofreference.size[::-1])
    # sum voxel-wise energy across all levels
    for level in range(nlevels-1, 0, -1):
        wavelet_coeffs_diag = wavelet_coeffs[level+1]['ddd']
        logger.info(indent('computing energy for level {:d} of shape:{!s}'.format(level, wavelet_coeffs_diag.shape), g_indents[3]))
        result = image_iterator(energy_plugin, wavelet_coeffs_diag, radius)

        zoomfactors = tuple(np.true_divide(roi_volume.frameofreference.size[::-1], result.shape))
        # scale low-res coefficients to image res
        result = scipy.ndimage.interpolation.zoom(result, zoomfactors, order=3)
        result = MaskableVolume().fromArray(result, roi_volume.frameofreference)
        # level_results.append(result)
        accumulator = np.add(accumulator, result.array)
    return MaskableVolume().fromArray(accumulator, roi_volume.frameofreference)

def wavelet_raw(image_volume, radius=2, roi=None, wavelet_str='db1', mode_str='smooth', level=0):
    # compute wavelet coefficients
    logger.info(indent('performing 3d wavelet decomp using wavelet: {!s}'.format(wavelet_str), g_indents[3]))
    roi_volume = image_volume.conformTo(roi.frameofreference)
    wavelet_coeffs = wavelet_decomp_3d(roi_volume, wavelet_str, mode_str)
    nlevels = len(wavelet_coeffs) - 1
    wavelet_coeffs_diag = wavelet_coeffs[nlevels-level]['ddd']
    zoomfactors = tuple(np.true_divide(roi_volume.frameofreference.size[::-1], wavelet_coeffs_diag.shape))
    # scale low-res coefficients at highest level to image res
    result = scipy.ndimage.interpolation.zoom(wavelet_coeffs_diag, zoomfactors, order=3)
    result = MaskableVolume().fromArray(result, roi_volume.frameofreference)
    return result

def glcm_polar(patch_vals, d, theta):
    """convenience function for converting spherical coords to euclidean coords"""
    #TODO
    #return glcm(patch_vals, dx, dy, dz)
    pass

def quantize(image_patch, gray_levels=12, n_stddev=2):
    """quantize the patch into specified number of bins, where first and last bins hold outliers and the rest\
            are allocated for storing values centered around mean within +-(n_stddev * stddev)
    Args:
        image_patch -- numpy ndarray
    Optional Args:
        n_stddev    -- quantize within +-(n_stddev * stddev), placing outliers in first and last bins
        gray_levels -- total bins (including outlier bins)
    """
    # compute gray level gaussian stats
    mean = np.mean(image_patch)
    stddev = np.std(image_patch)
    # logger.debug('mean:    {!s}\nstd dev: {!s}'.format(mean, stddev))
    bin_width = 2*n_stddev*stddev / (gray_levels-2)
    # logger.debug('bin_width: {!s}'.format(bin_width))

    # rebin values into new quanization, first and last bins hold outliers
    quantized_image_patch = np.zeros_like(image_patch, dtype=np.int8)
    it = np.nditer(image_patch, op_flags=['readwrite'], flags=['multi_index'])
    while not it.finished:
        val = image_patch[it.multi_index]
        quantized_image_patch[it.multi_index] = min(gray_levels-1, max(0, math.floor(((val - mean + n_stddev*stddev)/(bin_width+1e-9))+1)))
        it.iternext()

    # import matplotlib.pyplot as plt
    # xy_shape = quantized_image_patch.shape[1:]
    # for z in range(quantized_image_patch.shape[0]):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1,2,1)
    #     ax.imshow(image_patch[z,:,:].reshape(xy_shape), cmap='gray')
    #     ax = fig.add_subplot(1,2,2)
    #     ax.imshow(quantized_image_patch[z,:,:].reshape(xy_shape), cmap='gray', vmin=0, vmax=gray_levels-1)
    #     plt.show()
    return quantized_image_patch

def glcmMatrix(image_patch, dx, dy, dz, symmetric=False, normalized=False):
    # logger.debug('offsets-> dx:{:d}, dy:{:d}, dz:{:d}'.format(dx, dy, dz))
    levels = np.max(image_patch)
    glcm_matrix = np.zeros((levels+1, levels+1))
    # loop through and accumulate counts
    it = np.nditer(image_patch, op_flags=['readwrite'], flags=['multi_index'])
    while (not it.finished):
        y_idx = image_patch[it.multi_index]
        # logger.debug('parent loc: ({!s}) -> y_idx: {!s}'.format(it.multi_index, y_idx))
        x_idx_query = tuple(np.add(it.multi_index, (dz, dy, dx)))
        # boundary handling
        for i in range(3):
            s = image_patch.shape[i]
            q = x_idx_query[i]
            if (x_idx_query[i] >= image_patch.shape[i]):
                x_idx_query = list(x_idx_query)
                x_idx_query[i] = s - abs(q-s) - 1
                x_idx_query = tuple(x_idx_query)
            if (x_idx_query[i] < 0):
                x_idx_query = list(x_idx_query)
                x_idx_query[i] = s - abs(q-s) - 1
                x_idx_query = tuple(x_idx_query)
        x_idx = image_patch[x_idx_query]
        # logger.debug('  child loc: ({!s}) -> x_idx: {!s}'.format(x_idx_query, x_idx))
        glcm_matrix[y_idx, x_idx] += 1
        if (symmetric and x_idx != y_idx):
            # fill lower_diagonal values
            glcm_matrix[x_idx, y_idx] += 1
        it.iternext()

    # normalize counts
    if (normalized):
        total_counts = np.sum(glcm_matrix)
        for val in np.nditer(glcm_matrix, op_flags=["readwrite"]):
            val = val / total_counts

    return glcm_matrix

def glcm_stat_mean(glcm_matrix):
    """glcm statistic evaluation method"""
    return np.mean(glcm_matrix)

def glcm_stat_contrast(glcm_matrix):
    """glcm statistic evaluation method"""
    it = np.nditer(glcm_matrix, flags=['multi_index'])
    accum = 0
    while (not it.finished):
        accum += glcm_matrix[it.multi_index] * pow(np.diff(it.multi_index), 2)
        it.iternext()
    return accum

def glcm_stat_energy(glcm_matrix):
    """glcm statistic evaluation method"""
    return np.sum(np.square(glcm_matrix))

def glcm_stat_dissimilarity(glcm_matrix):
    """glcm statistic evaluation method"""
    it = np.nditer(glcm_matrix, flags=['multi_index'])
    accum = 0
    while (not it.finished):
        accum += glcm_matrix[it.multi_index] * abs(np.diff(it.multi_index))
        it.iternext()
    return accum

def glcm_stat_homogeneity(glcm_matrix):
    """glcm statistic evaluation method"""
    it = np.nditer(glcm_matrix, flags=['multi_index'])
    accum = 0
    while (not it.finished):
        accum += glcm_matrix[it.multi_index] / (1 + pow(np.diff(it.multi_index), 2))
        it.iternext()
    return accum


def glcm(image_volume, glcm_stat_function, radius=2, roi=None, gray_levels=12, n_stddev=2, dx=0, dy=0, dz=0):
    """feature calculation entry function"""
    # lexical scoping allows glcm to be adapted as a higher-order function then fed to image_iterator()
    def glcm_eval(patch_vals):
        """takes an ndarray of values in an image/volume patch and computes a single grey-level co-occurence matrix \
        based on the distance and angle in radians

        Args:
            gray_levels -- gray_levels are binned according to uniform thresholding within +- 2std-dev,
                           lowest and highest bins hold outliers and total bin count will be equal to requested
        """
        # if glcm_stat_function contains a collection of processing functions, return a result list
        # ALL ARE PATCH OPERATIONS
        # quantize image patch
        # logger.debug('quantizing image patch of shape: ({!s})'.format(patch_vals.shape))
        quantized_image_patch = quantize(patch_vals, gray_levels, n_stddev)
        # generate glcm using mirrored boundaries
        glcm_matrix = glcmMatrix(quantized_image_patch, dx, dy, dz)
        # calculate statistic on glcm_matrix
        result = glcm_stat_function(glcm_matrix)

        return result

    # build patch-eval function
    return image_iterator(glcm_eval, image_volume, radius, roi)
