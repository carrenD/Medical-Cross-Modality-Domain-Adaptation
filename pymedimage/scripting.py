"""scripting.py

A collection of functions/methods that carry us from one step to another in the study
"""

import os
import logging
import pickle
from collections import OrderedDict
from .rttypes import MaskableVolume, ROI
from .misc import indent, g_indents, findFiles
from . import dcmio, cluster, rttypes

# initialize module logger
logger = logging.getLogger(__name__)

l1_indent = g_indents[1]
l2_indent = g_indents[2]

def loadImages(images_path, modalities):
    """takes a list of modality strings and loads dicoms as a MaskableVolume instance from images_path

    Args:
        images_path --  Full path to patient specific directory containing various modality dicom images
            each modality imageset is contained in a directory within images_path where the modality string
            in modalities must match the directory name. This subdir is recursively searched for all dicoms
        modalities  --  list of modality strings that are used to identify subdirectories from which dicoms
            are loaded
    Returns:
        dictionary of {modality: imvolume} that contains loaded image data for each modality supported
    """
    # check if path specified exists
    if (not os.path.exists(images_path)):
        logger.info('Couldn\'t find specified path, nothing was loaded.')
        return None
    else:
        # load imvector and store to dictionary for each modality
        # if modality is missing, dont add to dictionary
        if (modalities is None or len(modalities)==0):
            logger.info('No modalities supplied. skipping')
            return None
        else:
            volumes = OrderedDict()
            for mod in modalities:
                logger.info(indent('Importing {mod:s} images'.format(mod=mod.upper()), l1_indent))
                dicom_path = os.path.join(images_path, '{mod:s}'.format(mod=mod))

                if (os.path.exists(dicom_path)):
                    # recursively walk modality path for dicom images, and build a dataset from it
                    try:
                        volumes[mod] = MaskableVolume().fromDir(dicom_path, recursive=True)
                    except:
                        logger.info('failed to create Volume for modality: {:s}'.format(mod))
                    else:
                        size = volumes[mod].frameofreference.size
                        logger.info(indent('stacked {len:d} datasets of shape: ({d:d}, {r:d}, {c:d})'.format(
                                      len=size[2],
                                      d=1,
                                      r=size[1],
                                      c=size[0]
                                    ), l2_indent))
                else:
                    logger.info(indent('path to {mod:s} dicoms doesn\'t exist. skipping\n'
                                 '(path: {path:s}'.format(mod=mod, path=dicom_path), l2_indent))
                logger.info('')
            return volumes

def loadROIs(rtstruct_path):
    """loads an rtstruct specified by path and returns a dict of ROI objects

    DEPRECATED IN FAVOR OF rttypes.ROI classmethod collectionFromFile(rtstruct_path)
    Args:
        rtstruct_path    -- path to rtstruct.dcm file

    Returns:
        dict<key='contour name', val=ROI>
    """
    logger.warning('scripting.loadROIs() DEPRECATED IN FAVOR OF rttypes.ROI classmethod collectionFromFile(rtstruct_path)')
    if (not os.path.exists(rtstruct_path)):
        logger.info(indent('invalid path provided: "{:s}"'.format(rtstruct_path), l2_indent))
        raise ValueError

    logger.info(indent('Importing ROIs', l1_indent))

    # check if path is file or dir
    if (os.path.isdir(rtstruct_path)):
        # search recursively for a valid rtstruct file
        ds_list = dcmio.read_dicom_dir(rtstruct_path, recursive=True)
        if (ds_list is None or len(ds_list) == 0):
            logger.info('no rtstruct datasets found at "{:s}"'.format(rtstruct_path))
            raise Exception
        ds = ds_list[0]
    elif (os.path.isfile(rtstruct_path)):
        ds = dcmio.read_dicom(rtstruct_path)

    # parse rtstruct file and instantiate maskvolume for each contour located
    # add each maskvolume to dict with key set to contour name and number?
    if (ds is not None):
        # get structuresetROI sequence
        StructureSetROI_list = ds.StructureSetROISequence
        nContours = len(StructureSetROI_list)
        if (nContours <= 0):
            logger.debug(indent('no contours were found', l2_indent))
            return None

        # Add structuresetROI to dict
        StructureSetROI_dict = {StructureSetROI.ROINumber: StructureSetROI
                                for StructureSetROI
                                in StructureSetROI_list }

        # get dict containing a contour dataset for each StructureSetROI with a paired key=ROINumber
        ROIContour_dict = {ROIContour.ReferencedROINumber: ROIContour
                           for ROIContour
                           in ds.ROIContourSequence }

        # construct a dict of ROI objects where contour name is key
        roi_dict = {}
        for ROINumber, structuresetroi in StructureSetROI_dict.items():
            roi_dict[structuresetroi.ROIName] = (ROI(frameofreference=None,
                                                     roicontour=ROIContour_dict[ROINumber],
                                                     structuresetroi=structuresetroi))
        # prune empty ROIs from dict
        for roiname, roi in dict(roi_dict).items():
            if (roi.coordslices is None or len(roi.coordslices) <= 0):
                logger.debug(indent('pruning empty ROI: {:s} from loaded ROIs'.format(roiname), l2_indent))
                del roi_dict[roiname]

        logger.info(indent('loaded {:d} ROIs succesfully'.format(len(roi_dict)), l2_indent))
        return roi_dict
    else:
        logger.info(indent('no dataset was found', l2_indent))
        return None

def getFeatureKeywords(feature_name, args):
    """generates keywords list for finding a pickled feature file using findFiles()
    DEPRECATED IN FAVOR OF CLASS METHOD: WritableFeatureDefinition.getKeywords()

    Args:
        feature_name -- string defining feature
        args         -- dict of argname: arg pairs which selectively apply to feature calculation functions

    Returns:
        list of keyword strings which must be in filename for match
    """
    logger.warning('DEPRECATED IN FAVOR OF CLASS METHOD: WritableFeatureDefinition.getKeywords()')
    keywords = ['feature={!s}'.format(feature_name)] + \
               ['{argname!s}={argval!s}'.format(argname=n, argval=v)
                for (n, v) in args.items()]
    for item in list(keywords):
        if ('function' in item.lower() or 'kernel' in item.lower()):
            keywords.remove(item)
    return keywords

def getArgsString(args, ignore_list=[]):
    """create standardized arg string based on feature args
    DEPRECATED IN FAVOR OF CLASS METHOD: WritableFeatureDefinition.getArgsString()

    Args:
        args -- ordered dict of argname: argvalue pairs
    Returns:
        string
    """
    logger.warning('DEPRECATED IN FAVOR OF CLASS METHOD: WritableFeatureDefinition.getArgsString()')
    args_string_list = []
    for k, v in args.items():
        ignore = False
        for ign in ignore_list:
            if (ign.lower() in k.lower()):
                ignore = True
                break
        if (ignore):
            continue

        if (callable(v) or 'function' in k.lower()):
            args_string_list.append('function={!s}'.format(k))
        elif ('kernel' in k.lower()):
            continue
        elif isinstance(v, int):
            args_string_list.append('{!s}={:d}'.format(k, v))
        elif isinstance(v, float):
            args_string_list.append('{!s}={:0.2f}'.format(k, v))
        else:
            args_string_list.append('{!s}={!s}'.format(k, v))
    return ','.join(args_string_list)

def checkPickle(root, label, args, mod=None, roi=None, all=False):
    """finds matching pickle files at root and returns first or list of all matches"""
    keywords = getFeatureKeywords(label, args)
    if (roi):
        keywords.append(roi.roiname)
    if mod:
        keywords.append(mod)
    matches = findFiles(root, ext='.pickle', keywordlist=keywords)

    if (matches is not None):
        return matches[0]
    else:
        return None

def loadPickle(path, mod=None, feature_label=None, nindent=l2_indent):
    logger.warning('DEPRECATED IN FAVOR OF: calculate_features.loadPrecalculated()')
    logger.info(indent('Pickled feature vector found ({!s}). Loading.'.format(mod), nindent))
    try:
        vol = MaskableVolume().fromPickle(path)
    except rttypes.PickleOutdatedError:
        # old pickle definition doesnt contain mod and feature_label add and repickle
        if mod:
            vol.mod = mod
        if feature_label:
            vol.feature_label = feature_label
        logger.info(indent('outdated pickle found, updating in filesystem', nindent))
        vol.toPickle(path)
    except:
        vol = None
    finally:
        if (vol):
            logger.info(indent('Pickled feature vector loaded successfully.', nindent))
            return vol
        else:
            logger.info(indent('there was a problem loading the file: {!s}'.format(path), nindent))
            return None

def savePickle(path, vol, mod, label, args, roi=None, nindent=l2_indent):
    logger.warning('savePickle DEPRECATED IN FAVOR OF: calculate_features.pickleFeature()')
    args_string = getArgsString(args, ignore_list=['glcm_stat_function'])
    if (roi is not None):
        roi_string = 'roi={!s}_'.format(roi.roiname)
    else:
        roi_string = ''

    # append ROIName to pickle path
    pickle_dump_path = os.path.join(path,
            'feature={featname!s}_{mod:s}_{roistring:s}args({args!s}).pickle'.format(
            featname=label,
            mod=mod,
            roistring=roi_string,
            args=args_string))
    try:
        vol.toPickle(pickle_dump_path)
    except:
        logger.info(indent('error pickling: {:s}'.format(pickle_dump_path), nindent))
    else:
        logger.info(indent('feature pickled successfully to: {:s}'.format(pickle_dump_path), nindent))

def loadFeatures(pickle_path, image_volumes, feature_defs, roi=None, savepickle=True):
    """Checks if feature vector has already been pickled at path specified and
    loads the files if so, or computes feature for each modality and pickles for later access.

    Args:
        pickle_path   -- should be the full path to the patient specific "precomputed" dir.
                         pickle file names are searched for occurence of pet, ct, and feature and will be loaded if a
                         modality string and "feature" are both present.
        image_volumes -- dictionary of {modality, BaseVolume} that contains loaded image data for
                         each modality supported
                         feature_defs  -- dict<key='feature_name', value=dict<k=argname, v-argval>>
    Returns:
        dict<key='feature_name', value=dict<key=mod, value=MaskableVolume>>
    """
    # check if path specified exists
    if (not os.path.exists(pickle_path)):
        logger.info('Couldn\'t find specified path, nothing was loaded.')
        return None
    else:
        # extract modalities from image_volumes
        if (image_volumes is None or len(image_volumes)==0):
            logger.info('No image data was provided. Skipping')
            return None
        modalities = list(image_volumes.keys())

        # load first file that matches the search and move to next modality
        feature_volumes = OrderedDict()
        for feature_def in feature_defs:
            feature_label = feature_def.label
            feature_args = feature_def.args
            calculate_feature = feature_def.calculation_function
            recalculate = feature_def.recalculate

            these_feature_volumes = OrderedDict()  # k: mod, v: BaseVolume
            logger.info('Loading Feature ({!s}):'.format(feature_label))
            for mod in modalities:
                logger.info(indent('Loading {!s} feature:'.format(mod.upper()), l1_indent))
                # initialize to None
                these_feature_volumes[mod] = None

                # get files that match settings
                match = checkPickle(pickle_path, feature_label, feature_args, mod, roi)

                if (not recalculate and match is not None):
                    # found pickled feature vector, load it and add to dict - no need to calculate feature
                    these_feature_volumes[mod] = loadPickle(os.path.join(pickle_path, match), mod, feature_label)
                else:  # Calculate feature this time
                    if (match):
                        logger.info(indent('Recalculating feature as requested', l2_indent))
                    else:
                        logger.info(indent('No pickled feature vector found ({!s})'.format(mod), l2_indent))

                    logger.info(indent('Computing feature now...'.format(mod=mod), l2_indent))
                    vol = calculate_feature(image_volumes[mod], roi=roi, **feature_args)
                    # inject metadata
                    vol.modality = mod
                    vol.feature_label = feature_label

                    these_feature_volumes[mod] = vol

                    # Check status of calculation
                    if these_feature_volumes[mod] is None:
                        logger.info(indent('Failed to compute feature for {!s} images.'.format(mod.upper()), l2_indent))
                    else:
                        logger.info(indent('feature computed successfully', l2_indent))
                        # pickle for later recall
                        savePickle(pickle_path, these_feature_volumes[mod], mod, feature_label, feature_args, roi)

                logger.info('')
                # END mod
            feature_volumes[feature_label] = these_feature_volumes
            logger.info('')
            # END feature_def

        # return dict of modality specific feature imvectors with keys defined by keys for image_volumes arg.
        return feature_volumes

def loadClusters(clusters_pickle_path, feature_volumes_list, nclusters, radius, roi=None, savepickle=True,
        recalculate=False):
    """Creates feature matrix and calculates clusters then stores the result in new pickle at path specified
    for later recall during hierarchical clustering.

    Args:
        clusters_pickle_path    -- path to search for pickle file and to store new result to
        feature_volumes_list    -- list of BaseVolumes to be used as feature vectors in clustering
        nclusters               -- desired number of clusters computed by kmeans
    Optional Args:
        savepickle              -- should we save the result to pickle?
        recalculate             -- should we recalculate anyway?
    """
    # check if path specified exists
    if (not os.path.exists(clusters_pickle_path)):
        logger.info('Couldn\'t find specified path, nothing was loaded.')
        return None
    else:
        # extract modalities
        if (feature_volumes_list is None or len(feature_volumes_list)==0):
            logger.info('No image data was provided. Skipping')
            return None
        modalities = set([vol.modality.lower()
                          for vol in feature_volumes_list
                          if (vol.modality is not None)])
        # replace pt with pet
        if ('pt' in modalities):
            modalities.remove('pt')
            modalities.add('pet')
        # reorder modalities for predictable naming
        orderedmodalities = []
        order_pref = ['ct', 'pet']
        # add any extra modalities to the end
        for mod in modalities:
            if (mod not in order_pref):
                order_pref.append(mod)
        # get modalities in preferred order
        for mod in order_pref:
            if (mod in modalities):
                orderedmodalities.append(mod)
        # turn modalities into string
        mod_string = '_'.join(orderedmodalities)

        # get files that match settings
        keywords = ['clusters',
                    'rad{:d}'.format(radius),
                    'ncl{:d}'.format(nclusters)]
        if (roi is not None):
            keywords.append(roi.roiname)
        keywords = keywords + list(modalities)
        matches = findFiles(clusters_pickle_path, ext='.pickle', keywordlist=keywords)
        if (matches is not None):
            match = matches[0]
        else:
            match = None


        # proceed with loading or recalculating clusters
        if (not recalculate and match is not None):
            # found pickled clusters, load it and add to dict - no need to calculate
            logger.info('Pickled clusters volume found. Loading from path: {:s}'.format(
                               os.path.join(clusters_pickle_path, match)))
            try:
                path = os.path.join(clusters_pickle_path, match)
                clusters = MaskableVolume().fromPickle(path)
            except:
                logger.info('there was a problem loading the file: {path:s}'.format(path=path))
                clusters = None
            else:
                logger.info('Pickled clusters volume loaded successfully.')
        else:
            # Calculate this time
            if (match is not None):
                # force calculation
                logger.info('Recalculating clusters as requested')
            else:
                # if no file is matched, calculate instead
                logger.info('No pickled clusters volume found')

            # get pruned feature matrix
            pruned_feature_matrix, clusters_frameofreference, \
                feature_matrix, feat_column_labels = cluster.create_feature_matrix(feature_volumes_list,
                                                                                   roi=roi)
            # calculate:
            clustering_result = cluster.cluster_kmeans(pruned_feature_matrix, nclusters)

            if clustering_result is None:
                logger.info('Failed to compute clusters.')
            else:
                logger.info('Clusters computed successfully')

                # expand sparse cluster assignment vector to dense MaskableVolume
                clusters = cluster.expand_pruned_vector(clustering_result, roi, clusters_frameofreference,
                                                        fill_value=-1)

                # pickle for later recall
                if (roi is not None):
                    # append ROIName to pickle path
                    pickle_dump_path = os.path.join(clusters_pickle_path,
                        'clusters_{mods:s}_roi={roiname:s}_rad{rad:d}_ncl{ncl:d}.pickle'.format(
                            mods=mod_string,
                            roiname=roi.roiname,
                            rad=radius,
                            ncl=nclusters))
                else:
                    # dont append roiname to pickle path
                    pickle_dump_path = os.path.join(clusters_pickle_path,
                            'clusters_{mods:s}_rad{rad:d}_ncl{ncl:d}.pickle'.format(
                                mods=mod_string, rad=radius, ncl=nclusters))
                try:
                    clusters.toPickle(pickle_dump_path)
                except:
                    logger.info('error pickling: {:s}'.format(pickle_dump_path))
                else:
                    logger.info('clusters pickled successfully to: {:s}'.format(pickle_dump_path))


                # store feature matrix in pickle as numpy ndarray
                featpickle_dump_path = os.path.join(clusters_pickle_path,
                        'features_{mod:s}_roi={roiname:s}_rad{rad:d}.pickle'.format(
                            mod=mod_string,
                            roiname=roi.roiname,
                            rad=radius))
                try:
                    # package feature_matrix into dict with labeling of the featues associate with each column
                    featpickle_dict = {'feature_matrix': feature_matrix,
                                       'labels':         feat_column_labels}
                    with open(featpickle_dump_path, mode='wb') as f:
                        pickle.dump(featpickle_dict, f)  # dense form
                except:
                    logger.info('error pickling: {:s}'.format(featpickle_dump_path))
                else:
                    logger.info('features successfully pickled to: {:s}'.format(featpickle_dump_path))

                logger.info('')

        # return MaskableVolume containing cluster assignments
        return clusters
