"""
dcmio.py

handles all dicom input/output tasks including reading dicom files and building
workable datasets for later use in projects
"""

import os
import sys
import logging
import dicom
from string import Template
from .misc import indent, g_indents
import copy
import numpy as np
import pdb

# initialize module logger
logger = logging.getLogger(__name__)

def write_dicom(path, dataset):
    """write a pydicom dataset to dicom file"""
    if not os.path.splitext(path)[1] == '.dcm':
        path += '.dcm'
    dicom.write_file(path, dataset)

def read_dicom(path):
    """read a dicom slice using pydicom and return the dataset object"""
    if (not os.path.exists(path)):
        raise FileNotFoundError('file at {!s} does not exist'.format(path))
    ds = dicom.read_file(path, force = True)
    return ds

def read_dicom_dir(path, recursive=False, verbosity=0):
    """read all dicom files in directory and return a list of the dataset objects.

    Keyword arguments:
    recursive -- should we walk into subdirs?
    verbosity -- 1: low, 2: high, 3: full
    """
    ds_list = []
    dicom_paths = []
    if (not os.path.exists(path)):
        logger.info('Execution Terminated. Supplied path did not exist: {:s}'.format(path))
        sys.exit(1)
    else:
        l1_indent = g_indents[2]
        l2_indent = g_indents[3]
        printstring = Template('Reading dicoms in specified path${extra}:\n"{:s}"')
        extra = ''
        if recursive:
            extra = ' and subdirs'
        printstring = printstring.substitute(extra=extra).format(path)
        logger.debug(indent(printstring, l1_indent))
        for root, dirs, files in os.walk(path, topdown=True):
            # build the list of valid dicom file paths then load them after walk
            for file in files:
                _, file_extension = os.path.splitext(file)
                if file_extension in ['.dcm', '.dicom']:
                    dicom_paths.append(root + '/' + file)
            if (not recursive):
                # clear dirs so that walk stops after this level
                del dirs[:]

        # Now read the dicom files that were located within path
        if verbosity == 0:
            #low vserbosity
            logger.debug(indent(dicom_paths[:5],l2_indent))
        elif verbosity == 1:
            #high verbosity
            logger.debug(indent(dicom_paths[:20],l2_indent))
        elif verbosity > 2:
            #full verbosity
            logger.debug(indent(dicom_paths,l2_indent))

        if (len(dicom_paths)>0):
            for file in dicom_paths:
                file_dataset = read_dicom(file)
                if file_dataset is not None:
                    ds_list.append(file_dataset)
            return ds_list
        else:
            return None

def read_dicom_3d(reference_fid, path = None, verbosity = 0):
    """ read all the slices of a same volume in a same folder given one slice as a reference, and return the whole 3d volume in together with the dicom of the beginning slice

    Args:
    path -- Path of the folder. Default value is the reference file directory
    reference_fid -- filename of one slice of the target volume

    Return:
    The dicom object of the first slice of

    """
    print("Warning: This function is still underconstruction")
    dcm_ref = read_dicom(reference_fid)

    ref_uid = dcm_ref.StudyInstanceUID
    # if size is not specified it will automatically count the number
    # of dicom slices in the folder, except for the rtstruct

    if path is None:
        path = os.path.dirname(reference_fid)

    if path == '':
        path = "./"

    _dcms = os.listdir(path)
    _min_z = 9999
    num_slices = 0
    _dcm_buffer = []
    for _slice in _dcms:
        if ("dcm" in _slice) and ("rtstruct" not in _slice):
            _tmp_dcm = read_dicom(_slice)
            if _tmp_dcm.StudyInstanceUID != ref_uid:
                # not a same scan
                continue
            num_slices += 1
            _z = _tmp_dcm.ImagePositionPatient[2]
            _dcm_buffer.append((_tmp_dcm, float(_z)))
            if _z < _min_z:
                _min_z = _z
        else:
            continue

    obj_dcm = copy.deepcopy(dcm_ref)

    _offset = [ float(dcm_ref.ImagePositionPatient[0]), float(dcm_ref.ImagePositionPatient[1]), float(_min_z) ]
    obj_dcm.ImagePositionPatient = _offset

    _size = [int(dcm_ref.Rows), int(dcm_ref.Columns), num_slices]
    obj_volume = np.zeros(list(reversed(_size)), dtype = np.float32)

    _dcm_buffer = sorted(_dcm_buffer, key = lambda x : float(x[1]))
    for idx,_slice in enumerate(_dcm_buffer):
        obj_volume[idx,:,:] = _slice[0].pixel_array


    if verbosity != 0:
        print("The detailed dicom information are listed as follows: \n ------------------------")
        print(obj_dcm)
    return obj_volume, _dcm_buffer[0]

def probeDicomProperties(root, prop_label_list, recursive=True, silent=False):
    """probe all dicoms in root for unique values of the properties defined in prop_label_list

    Returns:
        dict<k: prop_label, v: set()>: a set for each property is accumulated showing the unique values
            encountered across the entire dataset within root
    """
    sets = {}
    for l in prop_label_list:
        sets[l] = set()

    dcm_counter = 0
    for r, dirs, files in os.walk(root, topdown=True):
        # build the list of valid dicom file paths then load them after walk
        for file in files:
            _, file_extension = os.path.splitext(file)
            if file_extension in ['.dcm', '.dicom']:
                try:
                    ds = read_dicom(os.path.join(r, file))
                    dcm_counter += 1

                    for l, s in sets.items():
                        #print(l)
                        val = ds.get(l)
                        if isinstance(val, dicom.multival.MultiValue):
                            val = tuple(val)
                        #print(type(val))
                        s.add(val)
                except:
                    continue

        if (not recursive):
            # clear dirs so that walk stops after this level
            del dirs[:]

    if not silent:
        print('Finished probing {:d} dicom files.'.format(dcm_counter))
        print('')
        print('Probe Results:')
        print('--------------')
        for l, s in sets.items():
            print('| SET: {!s}'.format(l))
            for idx, item in enumerate(s):
                print('|   {!s}.  {!s}'.format(idx+1, item))
        print('--------------')

    return sets
