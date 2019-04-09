import os
from string import Template
import logging
import math
import numpy as np
from .rttypes import MaskableVolume
from .quantization import QMODE_STAT, QMODE_FIXEDHU
import gc
import pycuda
import pycuda.tools
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
pycuda.compiler.DEFAULT_NVCC_FLAGS = ['--std=c++11']

# import matplotlib.pyplot as plt

# set default nvidia device number
NVDEVICE = 1

# initialize module logger
logger = logging.getLogger(__name__)

####################################################################################################
# LOCAL FEATURE COMPOSITIONS
####################################################################################################
def elementwiseMean_gpu(feature_volume_list):
    """computes the elementwise mean of the like-shaped volumes in feature_volume_list"""
    # initialize cuda context
    cuda.init()
    cudacontext = cuda.Device(NVDEVICE).make_context()

    parent_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(parent_dir, 'feature_compositions.cuh'), mode='r') as f:
        mod = SourceModule(f.read(),
                           options=['-I {!s}'.format(parent_dir),
                                    # '-g', '-G', '-lineinfo'
                                    ])
    func = mod.get_function('elementwiseMean')

    # combine volumes into linearized array
    FOR = feature_volume_list[0].frameofreference
    vols = []
    for vol in feature_volume_list:
        vols.append(vol.vectorize())
    array_length = np.product(FOR.size).item()
    while len(vols) > 1:
        num_arrays = 2
        cat = np.concatenate([vols.pop() for x in range(num_arrays)], axis=0)

        # allocate image on device in global memory
        cat = cat.astype(np.float32)
        cat_gpu = cuda.mem_alloc(cat.nbytes)
        result = np.zeros((array_length)).astype(np.float32)
        result_gpu = cuda.mem_alloc(result.nbytes)
        # transfer cat to device
        cuda.memcpy_htod(cat_gpu, cat)
        cuda.memcpy_htod(result_gpu, result)
        # call device kernel
        blocksize = 512
        gridsize = math.ceil(array_length/blocksize)
        func(cat_gpu, result_gpu, np.int32(array_length), np.int32(num_arrays), block=(blocksize, 1,1), grid=(gridsize, 1,1))
        # get result from device
        cuda.memcpy_dtoh(result, result_gpu)
        vols.append(result.reshape((-1, 1)))
    result = vols[0]

    # detach from cuda context
    # cudacontext.synchronize()
    # cudacontext.detach()
    cudacontext.pop()
    # required to successfully free device memory for created context
    del cudacontext
    gc.collect()
    pycuda.tools.clear_context_caches()

    x = MaskableVolume().fromArray(result, FOR)
    x.modality = feature_volume_list[0].modality
    return x
####################################################################################################


####################################################################################################
# LOCAL FEATURE ACCESSOR
####################################################################################################
def image_iterator_gpu(image_volume, roi=None, radius=2, gray_levels=None, binwidth=None, dx=1, dy=0, dz=0, ndev=2,
                       cadd=(0,0,0), sadd=3, csub=(0,0,0), ssub=3, i=0,
             feature_kernel='glcm_plugin_gpu', stat_name='glcm_stat_contrast_gpu'):
    """Uses PyCuda to parallelize the computation of the voxel-wise image entropy using a variable \
            neighborhood radius

    Args:
	radius -- neighborhood radius; where neighborhood size is isotropic and calculated as 2*radius+1
    """
    # initialize cuda context
    cuda.init()
    cudacontext = cuda.Device(NVDEVICE).make_context()

    parent_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(parent_dir, 'local_features.cuh'), mode='r') as f:
        cuda_template = Template(f.read())

    roimask = None
    if isinstance(image_volume, np.ndarray):
        toBaseVolume = False
        logger.debug('recognized as an np.ndarray')
        if image_volume.ndim == 3:
            d, r, c = image_volume.shape
        elif image_volume.ndim == 2:
            d, r, c = (1, *image_volume.shape)
        image = image_volume.flatten()
        # # use stat based GLCM quantization
        # quantize_mode=QMODE_STAT
    else:
        toBaseVolume = True
        logger.debug('recognized as a BaseVolume')
        image = image_volume
        if roi:
            image = image.conformTo(roi.frameofreference)
        d, r, c = image.frameofreference.size[::-1]
        image = image.vectorize()
        # if not image_volume.modality.lower() == 'ct':
        #     # use stat based GLCM quantization
        #     quantize_mode=QMODE_STAT

        # mask to roi
        if (roi):
            roimask = roi.makeDenseMask().vectorize()

    logger.debug('d:{:d}, r:{:d}, c:{:d}'.format(d, r, c))
    if d == 1:
        z_radius = 0
    elif d > 1:
        z_radius = radius

    # enforce quantization mode selection
    # fixed_start, fixed_end = -150, 350
    fixed_start, fixed_end = -175, 75
    if gray_levels and binwidth:
        logger.exception('must exclusively specify "binwidth" or "gray_levels" to select glcm quantization mode')
    elif binwidth:
        quantize_mode = QMODE_FIXEDHU
        nbins = int(math.floor((fixed_end-fixed_start)/binwidth)) + 2
        gray_levels = -1
    elif gray_levels:
        quantize_mode = QMODE_STAT
        nbins = gray_levels
        binwidth = -1
    else:
        # kernel doesn't use glcm
        quantize_mode = -1
        nbins = 1
        gray_levels = -1
        binwidth = -1

    maxrunlength = math.ceil(math.sqrt(2*(radius*2+1)*(radius*2+1) + (z_radius*2+1)))

    cuda_source = cuda_template.substitute({'RADIUS': radius,
                                            'Z_RADIUS': z_radius,
                                            'IMAGE_DEPTH': d,
                                            'IMAGE_HEIGHT': r,
                                            'IMAGE_WIDTH': c,
                                            'QUANTIZE_MODE': quantize_mode,
                                            'GRAY_LEVELS': gray_levels,
                                            'FIXED_BINWIDTH': binwidth,
                                            'FIXED_START': fixed_start,
                                            'NBINS': nbins,
                                            'MAXRUNLENGTH': maxrunlength,
                                            'DX': dx,
                                            'DY': dy,
                                            'DZ': dz,
                                            'NDEV': ndev,
                                            'CADD_X': cadd[0],
                                            'CADD_Y': cadd[1],
                                            'CADD_Z': cadd[2],
                                            'SADD': sadd,
                                            'CSUB_X': csub[0],
                                            'CSUB_Y': csub[1],
                                            'CSUB_Z': csub[2],
                                            'SSUB': ssub,
                                            'KERNEL': feature_kernel,
                                            'STAT': stat_name})
    mod2 = SourceModule(cuda_source,
                        options=['-I {!s}'.format(parent_dir),
                                '-g', '-G', '-lineinfo'
                                ])
    func = mod2.get_function('image_iterator_gpu')

    # allocate image on device in global memory
    image = image.astype(np.float32)
    image_gpu = cuda.mem_alloc(image.nbytes)
    result = np.zeros_like(image)
    result_gpu = cuda.mem_alloc(result.nbytes)
    # transfer image to device
    cuda.memcpy_htod(image_gpu, image)
    cuda.memcpy_htod(result_gpu, result)
    # call device kernel
    blocksize = 256
    gridsize = math.ceil(r*c*d/blocksize)
    func(image_gpu, result_gpu, block=(blocksize, 1, 1), grid=(gridsize, 1, 1))
    # get result from device
    cuda.memcpy_dtoh(result, result_gpu)

    # detach from cuda context
    # cudacontext.synchronize()
    # cudacontext.detach()
    cudacontext.pop()
    # required to successfully free device memory for created context
    del cudacontext
    gc.collect()
    pycuda.tools.clear_context_caches()

    logger.debug('feature result shape: {!s}'.format(result.shape))
    logger.debug('GPU done')

    # clean invalid values from result
    result = np.nan_to_num(result)

    if (roimask is not None):
        result = np.multiply(result, roimask)

    if d == 1:
        result = result.reshape(r, c)
    elif d>1:
        result = result.reshape(d, r, c)

    if toBaseVolume:
        if roi:
            FOR = roi.frameofreference
        else:
            FOR = image_volume.frameofreference
        outvolume = MaskableVolume().fromArray(result, FOR)
        outvolume.modality = image_volume.modality
        return outvolume
    else:
        return result
