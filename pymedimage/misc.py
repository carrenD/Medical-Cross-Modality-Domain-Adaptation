"""misc.py

collection of miscellanious convenience functions
"""

import os
import time
import logging

# initialize module logger
logger = logging.getLogger(__name__)

def xstr(s):
    """replace None with '' (empty string)"""
    return '' if s is None else str(s)

def numpy_safe_string_from_array(array):
    if array.ndim < 1 or array.shape[0] == 0:
        return None
    else:
        return xstr(array.item())


from itertools import zip_longest
def grouper(n, iterable, fillvalue=None):
    """Unpacks iterables using groupings of n elements

    Ex:  grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    """
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def frange(x, y, jump):
    """range generator for floats between x and y with spacing jump"""
    while x < y:
        yield x
        x += jump



# global indent settings
g_indents = {1: 0,
             2: 2,
             3: 4,
             4: 6 }

def __get_indent_string(indent):
    """Contructs proper number of indent spaces"""
    return ''.join([' ' for i in range(indent)])

def timer(message, time_secs, indent=0):
    if (indent>0):
        message = ''.join([__get_indent_string(indent), message])
    return '{message:s} {time:s}'.format(message=message,
                                         time=time.strftime('%H:%M:%S', time.gmtime(time_secs)) )

def header(title, sep='*'):
   nseps = 3
   sep_string =  ''.join([sep for i in range(nseps)])
   return '{sep_string:s} {title:s} {sep_string:s}'.format(title=title, sep_string=sep_string)


def headerBlock(title, sep='-'):
   nseps = len(title)
   sep_string =  ''.join([sep for i in range(nseps)])
   return '\n{sep_string:s}\n{title:s}\n{sep_string:s}'.format(title=title, sep_string=sep_string)

def indent(message, indent=0):
    indent_string = __get_indent_string(indent)
    message = str(message)
    message = message.replace('\n', '\n' + indent_string)
    return '{indent_string:s}{message:s}'.format(indent_string=indent_string, message=message)

def findFiles(root, ext=None, keywordlist=None, casesensitive=False, recursive=False):
    """returns a list of full file paths beneath root if each path contains all of the strings in keywordlist
    and is of the type (ext) specified
    FUNCTIONALITY DUPLICATED IN FEATURES.WRITABLEFEATUREDEFINITION MEMBER METHOD - POTENTIAL DEPRECATION

    Args:
        root          -- path within which to check files

    Optional Args:
        ext          -- file extension to verify (with or without dot is okay)
        keywordlist   -- list of words which must all be present as substrings in filename
        casesensitive -- check character case?
        recursive     -- walk into subdirectories
    """
    # get list of files in root (match extension if specified)
    files = [
        f
        for f in os.listdir(root)
        if os.path.isfile(os.path.join(root, f))
        and (ext.replace('.', '') == os.path.splitext(f)[1].replace('.', '') if ext is not None else True)
    ]

    matches = []
    if (files is not None and len(files) > 0):
        # find files that contain all specified keywords
        for f in files:
            valid = True
            for key in keywordlist:
                if (casesensitive):
                    if (key not in f):
                        valid = False
                        break
                else:
                    if (key.lower() not in f.lower()):
                        valid = False
                        break
            if (valid):
                matches.append(os.path.join(root, f))

        # print results to debug
        if (len(matches) == 1):
            logger.debug('match found at path: {:s}'.format(os.path.join(root, matches[0])))
        elif (len(matches) > 1):
            logger.debug('matches found at paths:')
            for m in matches:
                logger.debug('  {:s}'.format(m))
        else:
            logger.debug('no matches found')
            return None

        return matches
    else:
        logger.debug('no files found')
        return None

def generate_heatmap_label(volume):
    """takes a BaseVolume or derivative and produces a heatmap feature label
    """
    mod = volume.modality
    feature_label = volume.feature_label
    if (feature_label):
        label = '{feat!s}({mod!s})'.format(feat = feature_label.title(),
                                           mod  = mod)
    elif mod:
        if ('ct' in mod.lower()):
            label = 'CT #'
        elif ('pet' in mod.lower() or 'pt' in mod.lower()):
            label = 'PET SUV'
        else:
            label = mod.upper()
    else:
        label = 'Unknown'

    return label

def getPatientPaths(root, includeignored=False, child_dirname='precomputed'):
    """find all patient dirs beneath root recursively and return list of full paths to these dirs

    Args:
        root -- path to search under recursively

    Optional Args:
        includeignored -- include patient_dirs with prefix '_'?
        precomputed_dirname -- set custom child dir for recognizing a patient dir
    """
    # walk filesystem to find patient_dirs
    patient_dirs = []
    # check if root dir is itself a patient
    root = root.rstrip('/')
    if (os.path.isdir(os.path.join(root, child_dirname))
        and (includeignored or os.path.basename(root)[0] != '_')):
        patient_dirs.append(root)
        return patient_dirs

    for walkroot, dirs, files in os.walk(root, followlinks=True):
        # identify patient dirs by presence of 'precomputed' directory one level deep
        if (os.path.isdir(os.path.join(walkroot, child_dirname))
            and (includeignored or os.path.basename(walkroot)[0] != '_')):
            patient_dirs.append(walkroot)
            # we've reached patient dir, dont go any deeper in this root
            del dirs[:]
            continue

    return patient_dirs
