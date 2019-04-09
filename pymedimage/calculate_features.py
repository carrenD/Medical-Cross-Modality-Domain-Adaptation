"""Standardized methods for mutltiprocess capable feature calculation and storage
"""
import os
import time
import logging
from multiprocessing import Pool
from .rttypes import MaskableVolume
from .notifications import pushNotification
from .multiprocess_manager import MultiprocessManager
from . import quantization
from .quantization import QMODE_STAT, QMODE_FIXEDHU

# initialize module logger
logger = logging.getLogger(__name__)

def checkCalculated(doi, local_feature_def):
    # check if already calculated
    p_doi_features = doi.getFeaturesPath()
    if os.path.exists(p_doi_features):
        matches = local_feature_def.findFiles(p_doi_features)
        if matches: return True
    return False

def loadPrecalculated(doi, local_feature_def):
    # check if already calculated
    p_doi_features = doi.getFeaturesPath()
    if os.path.exists(p_doi_features):
        matches = local_feature_def.findFiles(p_doi_features)
        if matches and len(matches)>0:
            # print(', '.join(matches))
            # return None ##QUICKFIX - corrupt pickle file errors on nextline with HYPOFRAC dataset
            return doi.loadFeatureVolume(matches[0])
    return None

def saveFeature(doi, local_feature_def, result_array):
    p_doi_features = doi.getFeaturesPath()
    p_feat_file = os.path.join(p_doi_features, local_feature_def.generateFilename())
    os.makedirs(p_doi_features, exist_ok=True)
    doi.saveFeatureVolume(result_array, p_feat_file)
    logger.debug('Feature: "{:s}" was stored to: {!s}'.format(local_feature_def.label, p_feat_file))

def calculateFeature(doi, local_feature_def, loadprecalculated=False):
    """single doi, single feature calculation sub-unit that can be multithreaded and called by a pool
    of workers

    Args:
        doi (str): string identifier unique to each patient/doi
        local_feature_def (LocalFeatureDefinition): information for feature calculation

        pickle (bool): if true, store result to pickle file, if false simply return 2-tuple of (result_code, result_array)
    Returns:
        int: status code
    """
    # load dicom data
    vol = doi.getImageVolume()
    if vol is None:
        return 1, None

    # force stat based GLCM quantization if not CT image
    local_feature_def = quantization.enforceGLCMQuantizationMode(local_feature_def, vol.modality)

    recalculated = False
    if checkCalculated(doi, local_feature_def):
        if (local_feature_def.recalculate):
            recalculated = True
        else:
            logger.debug('Feature already calculated. skipping')
            if loadprecalculated:
                loaded_feature_vol = loadPrecalculated(doi, local_feature_def)
            else: loaded_feature_vol = None
            return (10, loaded_feature_vol)

    try:
        roi = doi.getROI()
    except:
        roi = None
    if (not vol):
        logger.debug('missing image data. skipping.')
        return (1, None)

    # compute feature
    logger.debug('calculating "{!s}" for doi: {!s}'.format(local_feature_def.label, doi))
    feature_vol = local_feature_def.calculation_function(vol, roi, **local_feature_def.args)
    feature_vol.feature_label = local_feature_def.generateFeatureLabel()

    # return status
    if (recalculated):
        return (11, feature_vol)
    else:
        return (0, feature_vol)

def calculateCompositeFeature(doi, composite_feature_def, saveintermediate=False, loadprecalculated=False):
    # try to load image volume
    vol = doi.getImageVolume()
    if vol is None:
        return 1, None

    # force stat based GLCM quantization if not CT image
    composite_feature_def = quantization.enforceGLCMQuantizationMode(composite_feature_def, vol.modality)

    recalculated = False
    if checkCalculated(doi, composite_feature_def):
        if (composite_feature_def.recalculate):
            recalculated = True
        else:
            if loadprecalculated:
                loaded_composite_vol = loadPrecalculated(doi, composite_feature_def)
            else: loaded_composite_vol = None
            return (10, loaded_composite_vol)

    vol_list = []
    for lfeatdef in composite_feature_def.featdefs:
        # lfeatdef.recalculate = True  # force recalculation
        result_code, feature_vol = calculateFeature(doi, lfeatdef, loadprecalculated=True)
        if result_code not in [0, 11, 10]:
            return 2, None
        if feature_vol:
            vol_list.append(feature_vol)
            if saveintermediate: saveFeature(doi, lfeatdef, feature_vol)

    if len(vol_list) <= 0:
        return 3, None

    composite_vol = composite_feature_def.composition_function(vol_list)
    composite_vol.feature_label = composite_feature_def.generateFeatureLabel()

    # return status
    if (recalculated):
        return (11, composite_vol)
    else:
        return (0, composite_vol)

def worker_calculateFeature(args_tuple):
    (doi, local_feature_def) = args_tuple
    time_start = time.time()
    try:
        cls = local_feature_def.__class__.__name__
        if ('LocalFeatureDefinition' in cls):
            result_code, feature_vol = calculateFeature(doi, local_feature_def)
            if feature_vol:
                saveFeature(doi, local_feature_def, feature_vol)
        elif ('LocalFeatureCompositionDefinition' in cls):
            result_code, composite_result = calculateCompositeFeature(doi, local_feature_def, saveintermediate=False)
            if composite_result:
                saveFeature(doi, local_feature_def, composite_result)

        if (result_code == 0):
            result_string = 'success'
        elif (result_code == 1):
            result_string = 'missing data'
        elif (result_code == 3):
            result_string = 'composite error'
        elif (result_code == 10):
            result_string = 'skipped'
        elif (result_code == 11):
            result_string = 'recalc'
        else:
            # unknown result
            result_code = -1
            result_string = 'unknown'

    except Exception as e:
        print(e)
        result_code = 2
        result_string = 'exception'
        logger.error('{!s}'.format(e))
        raise

    time_end = time.time()
    job_time_string = time.strftime('%H:%M:%S', time.gmtime(time_end-time_start))

    return (result_code, result_string, job_time_string, doi, local_feature_def)

def logstringgenerator_calculateFeature(worker_results):
    (result_code, result_string, job_time_string, doi, local_feature_def) = worker_results
    log_string = '[{string:12s}:{code:2d}]: {doi!s:9s}  {label!s:30s}  {args!s:45s}  {time!s}'.format(
        string  = result_string,
        code    = result_code,
        doi     = doi,
        label   = local_feature_def.label,
        args    = local_feature_def.getArgsString(),
        time    = job_time_string
    )
    return log_string

def multiprocessCalculateFeatures(doi_list, feature_def_list, processes=16, notify=True):
    """multithreaded manager for calculating local image features for a large number of dois"""
    # build argmap for worker pool
    argmap = []
    for doi in doi_list:
        for local_feature_def in feature_def_list:
            argmap.append((doi, local_feature_def))

    manager = MultiprocessManager('Feature Calculation')
    manager.registerWorkerFunction(worker_calculateFeature)
    manager.registerLogStringGenerator(logstringgenerator_calculateFeature)
    manager.execute(argmap, processes=processes, notify=notify)
