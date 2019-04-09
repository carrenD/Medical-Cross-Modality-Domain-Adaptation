"""rttypes.py

Datatypes for general dicom processing including masking, rescaling, and fusion
"""

import sys
import os
import logging
import math
import numpy as np
import dicom as dcm# pydicom
import pickle
import scipy.io  # savemat -> save to .mat
import struct
import copy
import warnings
import pdb
import scipy.ndimage.filters as sfilter
from PIL import Image, ImageDraw
from scipy.ndimage import interpolation
from . import dcmio, misc, niftiio

# initialize module logger
logger = logging.getLogger(__name__)

rtstruct_identify = "Strctr"


class RescaleParams:
    """Defines the scale and offset necessary for linear rescaling operation of dicom data
    """
    def __init__(self, scale=1, offset=0):
        """initialize with params"""
        self.scale = scale
        self.offset = offset

    def __repr__(self):
        return 'scale:  {:g}\n'.format(self.scale) + \
               'offset: {:g}'.format(self.offset)


class FrameOfReference:
    """Defines a dicom frame of reference to which BaseVolumes can be conformed for fusion of pre-registered
    image data
    """
    def __init__(self, start, spacing, size, UID=None):
        """Define a dicom frame of reference

        Args:
            start    -- (x,y,z) describing the start of the FOR (mm)
            spacing  -- (x,y,z) describing the spacing of voxels in each direction (mm)
            size     -- (x,y,z) describing the number of voxels in each direction (integer)
            UID      -- dicom FrameOfReferenceUID can be supplied to support caching in BaseVolume

        Standard Anatomical Directions Apply:
            x -> increasing from patient right to left
            y -> increasing from patient anterior to posterior
            z -> increasing from patient inferior to superior
        """
        self.start = start
        self.spacing = spacing
        self.size = size
        self.UID = UID

    # TODO: Use the read_dicom_3d version in dcmio.py for volume parsing
    @classmethod
    def _from_dcm_image(cls, dcm_object, UID = None, direc = None, struct_identifier = rtstruct_identify):
        """ initialization with one dicom object
        Args:
            direc: directory of dicom object
            struct_identifier: partial or complete rtstruct name identifying rtstruct
        """
        #pdb.set_trace()
        _offset = ( float(dcm_object.ImagePositionPatient[0]), float(dcm_object.ImagePositionPatient[1]), 0)
        _size = (int(dcm_object.Rows), int(dcm_object.Columns), 0)
        _spacing = (float(dcm_object.PixelSpacing[0]), float(dcm_object.PixelSpacing[1]), float(dcm_object.SliceThickness) )
        ref_uid = dcm_object.StudyInstanceUID
        # if size is not specified it will automatically count the number
        # of dicom slices in the folder, except for the rtstruct
        #if _size is None:
        if (direc == None) or (direc == ''):
            direc = "./"
            print("Warning: Dicom directory not fed. Searching through current directory.")
        _dcms = os.listdir(direc)
        _min_z = 9999
        num_slices = 0
        for _slice in _dcms:
            if ("dcm" in _slice.lower()) and (struct_identifier not in _slice):
                _tmp_dcm = dcmio.read_dicom( os.path.join(direc, _slice) )
                if _tmp_dcm.StudyInstanceUID != ref_uid:
                    # not a same scan
                    continue
                try:
                    _z = _tmp_dcm.ImagePositionPatient[2]
                    num_slices += 1
                except:
                    logger.debug("this is an rtstruct file which does not contain position attribute: %s"%_slice)
                    continue
                ################
                if _z < _min_z:
                ########
                    _min_z = _z
            else:
                continue

        if num_slices == 0:
            raise FileNotFoundError("No valid dicom slice in this volume!")
        # get size and offset
        _size = list(_size)
        _offset = list(_offset)
        _size[2] = int(num_slices)
        _offset[2] = int(_min_z)
        _size = tuple(_size)
        _offset = tuple(_offset)
        #debug
        print("Sizem offset and spacing:")
        print(_size)
        print(_offset)
        print(_spacing)
        return cls(_offset, _spacing, _size)

    @classmethod
    def from_dcm_fid(cls, dcm_fid, UID = None, struct_identifier = "rtstruct"):
        """ initialization with a valid dicom slice fid """
        try:
            os.path.isfile(dcm_fid)
        except:
            raise FileNotFoundError("Reference dicom slice %s not found"%dcm_fid)
        dcm_dir = os.path.dirname(dcm_fid)
        dcm_object = dcm.read_file(dcm_fid, force = True)
        return cls._from_dcm_image(dcm_object, direc = dcm_dir, struct_identifier = struct_identifier)

    def __repr__(self):
        return 'FrameOfReference:\n' + \
               'start   <mm> (x,y,z): ({:0.3f}, {:0.3f}, {:0.3f})\n'.format(*self.start) + \
               'spacing <mm> (x,y,z): ({:0.3f}, {:0.3f}, {:0.3f})\n'.format(*self.spacing) + \
               'size    <mm> (x,y,z): ({:d}, {:d}, {:d})'.format(*self.size)

    def __eq__(self, compare):
        if (self.start   == compare.start and
            self.spacing == compare.spacing and
            self.size    == compare.size):
            return True
        else: return False

    def changeSpacing(self, new_spacing):
        """change frameofreference resolution while maintaining same bounding box
        Changes occur in place, self is returned
            Args:
                new_spacing (3-tuple<float>): spacing expressed as (X, Y, Z)
        """
        old_spacing = self.spacing
        old_size = self.size
        self.spacing = new_spacing
        self.size = tuple((np.array(old_size) * np.array(old_spacing) / np.array(self.spacing)).astype(int).tolist())
        return self

    def end(self):
        """Calculates the (x,y,z) coordinates of the end of the frame of reference (mm)
        """
        # compute ends
        end = []
        for i in range(3):
            end.insert(i, self.spacing[i] * self.size[i] + self.start[i])

        return tuple(end)

    def volume(self):
        """Calculates the volume of the frame of reference (mm^3)
        """
        length = []
        end = self.end()
        vol = 1
        for i in range(3):
            length.insert(i, end[i] - self.start[i])
            vol *= length[i]

        return vol

    def getIndices(self, position):
        """Takes a position (x, y, z) and returns the indices at that location for this FrameOfReference

        Args:
            position  -- 3-tuple of position coordinates (mm) in the format: (x, y, z)
        """
        indices = []
        for i in range(3):
            indices.insert(i, math.floor(int(round((position[i] - self.start[i]) / self.spacing[i] ))))

        return tuple(indices)


class ROI:
    """Defines a labeled RTStruct ROI for use in masking and visualization of Radiotherapy contours
    """
    def __init__(self, roicontour, structuresetroi):
        """takes FrameOfReference object and roicontour/structuresetroi dicom dataset objects and stores
        sorted contour data

        Args:
            frameofreference   -- FrameOfReference object providing details necessary for dense mask creation
            roicontour         -- dicom dataset containing contour point coords for all slices
            structuresetroi    -- dicom dataset containing additional information about contour
        """
        self.roinumber = structuresetroi.ROINumber
        self.refforuid = structuresetroi.ReferencedFrameOfReferenceUID
        self.frameofreference = None
        self.roiname = structuresetroi.ROIName
        self.coordslices = None
        # Cached variables
        self.__cache_densemask = None   # storage for BaseVolume when consecutive calls to
                                        # makeDenseMask are made
                                        # with the same frameofreference object

        # Populate list of coordslices, each containing a list of ordered coordinate points
        contoursequence = roicontour.ContourSequence
        if (len(contoursequence) <= 0):
            logger.debug('no coordinates found in roi: {:s}'.format(self.roiname))
            self.coordslices = None
        else:
            self.coordslices = []
            logger.debug('loading roi: {:s} with {:d} slices'.format(self.roiname, len(roicontour.ContourSequence)))
            for coordslice in roicontour.ContourSequence:
                points_list = []
                for x, y, z in misc.grouper(3, coordslice.ContourData):
                    points_list.append( (x, y, z) )
                self.coordslices.append(points_list)

            # sort by slice position in ascending order (inferior -> superior)
            self.coordslices.sort(key=lambda coordslice: coordslice[0][2], reverse=False)

            # create frameofreference based on the extents of the roi and apparent spacing
            self.frameofreference = self.getROIExtents()

    def __repr__(self):
        return '{!s}\n'.format(type(self)) + \
               'roiname: {!s}\n'.format(self.roiname) + \
               '[FrameOfReference]:\n{!s}'.format(self.frameofreference)

    @staticmethod
    def _loadRtstructDicom(rtstruct_path):
        """load rtstruct dicom data from a direct path or containing directory"""
        if (not os.path.exists(rtstruct_path)):
            logger.debug('invalid path provided: "{:s}"'.format(rtstruct_path))
            raise FileNotFoundError

        # check if path is file or dir
        if (os.path.isdir(rtstruct_path)):
            # search recursively for a valid rtstruct file
            ds_list = dcmio.read_dicom_dir(rtstruct_path, recursive=True)
            if (ds_list is None or len(ds_list) == 0):
                logger.debug('no rtstruct datasets found at "{:s}"'.format(rtstruct_path))
                raise Exception
            ds = ds_list[0]
        elif (os.path.isfile(rtstruct_path)):
            ds = dcmio.read_dicom(rtstruct_path)
        return ds

    @classmethod
    def collectionFromFile(cls, rtstruct_path):
        """loads an rtstruct specified by path and returns a dict of ROI objects

        Args:
            rtstruct_path    -- path to rtstruct.dcm file

        Returns:
            dict<key='contour name', val=ROI>
        """
        ds = cls._loadRtstructDicom(rtstruct_path)

        # parse rtstruct file and instantiate maskvolume for each contour located
        # add each maskvolume to dict with key set to contour name and number?
        if (ds is not None):
            # get structuresetROI sequence
            StructureSetROI_list = ds.StructureSetROISequence
            nContours = len(StructureSetROI_list)
            if (nContours <= 0):
                logger.exception('no contours were found')

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
                roi_dict[structuresetroi.ROIName] = (cls(roicontour=ROIContour_dict[ROINumber],
                                                         structuresetroi=structuresetroi))
            # prune empty ROIs from dict
            for roiname, roi in dict(roi_dict).items():
                if (roi.coordslices is None or len(roi.coordslices) <= 0):
                    logger.debug('pruning empty ROI: {:s} from loaded ROIs'.format(roiname))
                    del roi_dict[roiname]

            logger.debug('loaded {:d} ROIs succesfully'.format(len(roi_dict)))
            return roi_dict
        else:
            logger.exception('no dataset was found')

    @staticmethod
    def getROINames(rtstruct_path):
        ds = ROI._loadRtstructDicom(rtstruct_path)

        if (ds is not None):
            # get structuresetROI sequence
            StructureSetROI_list = ds.StructureSetROISequence
            nContours = len(StructureSetROI_list)
            if (nContours <= 0):
                logger.exception('no contours were found')

            roi_names = []
            for structuresetroi in StructureSetROI_list:
                roi_names.append(structuresetroi.ROIName)

            return roi_names
        else:
            logger.exception('no dataset was found')

    def makeDenseMaskSlice(self, position, frameofreference=None):
        """Takes a FrameOfReference and constructs a dense binary mask for the ROI (1 inside ROI, 0 outside)
        as a numpy 2dArray

        Args:
            position           -- position of the desired slice (mm) within the frameofreference along z-axis
            frameofreference   -- FrameOfReference that defines the position of ROI and size of dense volume

        Returns:
            numpy 2dArray
        """
        # get FrameOfReference params
        if (frameofreference is None):
            if (self.frameofreference is not None):
                frameofreference = self.frameofreference
            else:
                logger.exception('no frame of reference provided')
                raise Exception
        xstart, ystart, zstart = frameofreference.start
        xspace, yspace, zspace = frameofreference.spacing
        cols, rows, depth = frameofreference.size

        # get nearest coordslice
        minerror = 5000
        coordslice = None
        ### REVISIT THE CORRECT SETTING OF TOLERANCE TODO
        tolerance = self.frameofreference.spacing[2]*4 - 1e-9  # if upsampling too much then throw error
        for slice in self.coordslices:
            # for each list of coordinate tuples - check the slice for distance from position
            error = abs(position - slice[0][2])
            if error <= minerror:
                # if minerror != 5000:
                #     logger.debug('position:{:0.3f} | slicepos:{:0.3f}'.format(position, slice[0][2]))
                #     logger.debug('improved with error {:f}'.format(error))
                minerror = error
                coordslice = slice
                # logger.debug('updating slice')
            else:
                # we've already passed the nearest slice, break
                break

        # check if our result is actually valid or we just hit the end of the array
        if coordslice and minerror >= tolerance:
            logger.debug('No slice found within {:f} mm of position {:f}'.format(tolerance, position))
            # print(minerror, tolerance)
            # print(position)
            # print(zstart, zspace*depth)
            # for slice in self.coordslices:
            #     if abs(slice[0][2]-position) < 100:
            #         print(slice[0][2])
            return np.zeros((rows, cols))
            # raise Exception('Attempt to upsample ROI to densearray beyond 5x')
        logger.debug('slice found at {:f} for position query at {:f}'.format(coordslice[0][2], position))

        # get coordinate values
        index_coords = []
        for x, y, z in coordslice:
            # shift x and y and scale appropriately
            x_idx = int(round((x-xstart)/xspace))
            y_idx = int(round((y-ystart)/yspace))
            index_coords.append( (x_idx, y_idx) )

        # use PIL to draw the polygon as a dense image (PIL uses shape: (width, height))
        im = Image.new('1', (cols, rows), color=0)
        imdraw = ImageDraw.Draw(im)
        imdraw.polygon(index_coords, fill=1, outline=None)
        del imdraw

        # convert from PIL image to np.ndarray and threshold to binary
        return np.array(im.getdata()).reshape((rows, cols))

    def makeDenseMask(self, frameofreference=None):
        """Takes a FrameOfReference and constructs a dense binary mask for the ROI (1 inside ROI, 0 outside)
        as a BaseVolume

        Args:
            frameofreference   -- FrameOfReference that defines the position of ROI and size of dense volume

        Returns:
            BaseVolume
        """
        # get FrameOfReference params
        if (frameofreference is None):
            if (self.frameofreference is not None):
                frameofreference = self.frameofreference
            else:
                logger.exception('no frame of reference provided')
                raise Exception

        # check cache for similarity between previously and currently supplied frameofreference objects
        if (self.__cache_densemask is not None
                and frameofreference == self.__cache_densemask.frameofreference):
            # cached mask frameofreference is similar to current, return cached densemask volume
            # logger.debug('using cached dense mask volume')
            return self.__cache_densemask
        else:
            xstart, ystart, zstart = frameofreference.start
            xspace, yspace, zspace = frameofreference.spacing
            cols, rows, depth = frameofreference.size

            # generate binary mask for each slice in frameofreference
            maskslicearray_list = []
            # logger.debug('making dense mask volume from z coordinates: {:f} to {:f}'.format(
            #              zstart, (zspace * (depth+1) + zstart)))
            for i in range(depth):
                position = zstart + i * zspace
                # get a slice at every position within the current frameofreference
                densemaskslice = self.makeDenseMaskSlice(position, frameofreference)
                maskslicearray_list.append(densemaskslice.reshape((1, *densemaskslice.shape)))

            # construct BaseVolume from dense slice arrays
            densemask = BaseVolume.fromArray(np.concatenate(maskslicearray_list, axis=0), frameofreference)
            self.__cache_densemask = densemask
            return densemask

    def getROIExtents(self):
        """Creates a tightly bound frame of reference around the ROI which allows visualization in a cropped
        frame
        """
        # guess at spacing and assign arbitrarily where necessary
        # get list of points first
        point_list = []
        for slice in self.coordslices:
            for point3d in slice:
                point_list.append(point3d)

        # set actually z spacing estimated from separation of coordslice point lists
        min_z_space = 9999
        prev_z = point_list[0][2]
        for point3d in point_list[1:]:
            z = point3d[2]
            this_z_space = abs(z-prev_z)
            if (this_z_space > 0 and this_z_space < min_z_space):
                min_z_space = this_z_space
            prev_z = z

        if (min_z_space <= 0 or min_z_space > 10):
            # unreasonable result found, arbitrarily set
            new_z_space = 1
            logger.debug('unreasonable z_spacing found: {:0.3f}, setting to {:0.3f}'.format(
                min_z_space, new_z_space))
            min_z_space = new_z_space
        else:
            logger.debug('estimated z_spacing: {:0.3f}'.format(min_z_space))

        # arbitrarily set spacing
        spacing = (1, 1, min_z_space)

        # get start and end of roi volume extents
        global_limits = {'xmax': -5000,
                         'ymax': -5000,
                         'zmax': -5000,
                         'xmin': 5000,
                         'ymin': 5000,
                         'zmin': 5000 }
        for slice in self.coordslices:
            # convert coords list to ndarray
            coords = np.array(slice)
            (xmin, ymin, zmin) = tuple(coords.min(axis=0, keepdims=False))
            (xmax, ymax, zmax) = tuple(coords.max(axis=0, keepdims=False))

            # update limits
            if xmin < global_limits['xmin']:
                global_limits['xmin'] = xmin
            if ymin < global_limits['ymin']:
                global_limits['ymin'] = ymin
            if zmin < global_limits['zmin']:
                global_limits['zmin'] = zmin
            if xmax > global_limits['xmax']:
                global_limits['xmax'] = xmax
            if ymax > global_limits['ymax']:
                global_limits['ymax'] = ymax
            if zmax > global_limits['zmax']:
                global_limits['zmax'] = zmax

        # build FrameOfReference
        start = (global_limits['xmin'],
                 global_limits['ymin'],
                 global_limits['zmin'] )
        size = (int((global_limits['xmax'] - global_limits['xmin']) / spacing[0]),
                int((global_limits['ymax'] - global_limits['ymin']) / spacing[1]),
                int((global_limits['zmax'] - global_limits['zmin']) / spacing[2]) )

        logger.debug('ROIExtents:\n'
                     '    start:   {:s}\n'
                     '    spacing: {:s}\n'
                     '    size:    {:s}'.format(str(start), str(spacing), str(size)))
        frameofreference = FrameOfReference(start, spacing, size, UID=None)
        return frameofreference

    def toPickle(self, pickle_path):
        """convenience function for storing ROI to pickle file"""
        _dirname = os.path.dirname(pickle_path)
        if (_dirname and _dirname is not ''):
            os.makedirs(_dirname, exist_ok=True)
        with open(pickle_path, 'wb') as p:
            pickle.dump(self, p)

    @staticmethod
    def fromPickle(pickle_path):
        """convenience function for restoring ROI from pickle file"""
        with open(pickle_path, 'rb') as p:
            return pickle.load(p)

class BaseVolume:
    """Defines basic storage for volumetric voxel intensities within a dicom FrameOfReference
    """
    def __init__(self):
        """Entrypoint to class, initializes members
        """
        self.array = None
        self.frameofreference = None
        self.rescaleparams = None
        self.modality = None
        self.feature_label = None

    def __repr__(self):
        return '{!s}\n'.format(type(self)) + \
               'modality: {!s}\n'.format(self.modality) + \
               'feature_label: {!s}\n'.format(self.feature_label) + \
               '[FrameOfReference]:\n{!s}\n'.format(self.frameofreference) + \
               '[RescaleParams]:\n{!s}'.format(self.rescaleparams)

    # CONSTRUCTOR METHODS
    @classmethod
    def fromArray(cls, array, frameofreference):
        """Constructor: from a numpy array and FrameOfReference object

        Args:
            array             -- numpy array
            frameofreference  -- FrameOfReference object
        """
        # ensure array matches size in frameofreference
        self = cls()
        self.array = array.reshape(frameofreference.size[::-1])
        self.frameofreference = frameofreference

        return self

    @classmethod
    def fromDir(cls, path, recursive=False):
        """constructor: takes path to directory containing dicom files and builds a sorted array

        Args:
            recursive -- find dicom files in all subdirectories?
        """
        self = cls()

        # get the datasets from files
        dataset_list = dcmio.read_dicom_dir(path, recursive=recursive)

        # pass dataset list to constructor
        self.fromDatasetList(dataset_list)

        return self

    @classmethod
    def fromBinary(cls, path, frameofreference):
        """constructor: takes path to binary file (neylon .raw)
        data is organized as binary float array in row-major order

        Args:
            path (str): path to .raw file in binary format
            frameofreference (FOR): most importantly defines mapping from 1d to 3d array
        """
        if not os.path.isfile(path) or os.path.splitext(path)[1].lower() not in ['.raw', '.bin']:
            raise Exception('data is not formatted properly. must be one of [.raw, .bin]')

        if not isinstance(frameofreference, FrameOfReference):
            if not isinstance(frameofreference, tuple):
                raise TypeError('frameofreference must be a valid FrameOfReference or tuple of dimensions')
            frameofreference = FrameOfReference(start=(0,0,0), spacing=(1,1,1), size=frameofreference)

        with open(path, mode='rb') as f:
            flat = f.read()
        _shape = frameofreference.size[::-1]
        _expected_n = np.product(_shape)
        _n = int(os.path.getsize(path)/struct.calcsize('f'))
        if _n != _expected_n:
            raise Exception('filesize ({:f}) doesn\'t match expected ({:f}) size'.format(
                os.path.getsize((path)), struct.calcsize('f')*_expected_n
            ))
        s = struct.unpack('f'*_n, flat)
        vol = np.array(s).reshape(_shape)
        #  vol[vol>1e10] = 0
        #  vol[vol<-1e10] = 0
        return cls.fromArray(vol, frameofreference)

    def fromDatasetList(self, dataset_list):
        """constructor: takes a list of dicom slice datasets and builds a BaseVolume array
        Args:
            slices
        """
        if (dataset_list is None):
            raise ValueError('no valid dataset_list provided')

        # check that all elements are valid slices, if not remove and continue
        nRemoved = 0
        for i, slice in enumerate(dataset_list):
            if (not isinstance(slice, dicom.dataset.Dataset)):
                logger.debug('invalid type ({t:s}) at idx {i:d}. removing.'.format(
                    t=str(type(slice)),
                    i=i ) )
                dataset_list.remove(slice)
                nRemoved += 1
            elif (len(slice.dir('ImagePositionPatient')) == 0):
                logger.debug('invalid .dcm image at idx {:d}. removing.'.format(i))
                dataset_list.remove(slice)
                nRemoved += 1
        if (nRemoved > 0):
            logger.info('# slices removed with invalid types: {:d}'.format(nRemoved))

        # sort datasets by increasing slicePosition (inferior -> superior)
        dataset_list.sort(key=lambda dataset: dataset.ImagePositionPatient[2], reverse=False)

        # build object properties
        start = dataset_list[0].ImagePositionPatient
        spacing = (*dataset_list[0].PixelSpacing, dataset_list[0].SliceThickness)
        try:
            # some modalities don't provide NumberOfSlices attribute
            size = (dataset_list[0].Columns, dataset_list[0].Rows, dataset_list[0].NumberOfSlices)
        except:
            # use length of list instead
            size = (dataset_list[0].Columns, dataset_list[0].Rows, len(dataset_list))

        UID = dataset_list[0].FrameOfReferenceUID
        try:
            self.rescaleparams = RescaleParams(scale=dataset_list[0].RescaleSlope,
                                               offset=dataset_list[0].RescaleIntercept)
        except:
            self.rescaleparams = RescaleParams(scale=1, offset=0)

        self.frameofreference = FrameOfReference(start, spacing, size, UID)

        # standardize modality labels
        mod = dataset_list[0].Modality
        if (mod == 'PT'):
            mod = 'PET'
        self.modality = mod

        # construct 3dArray
        array_list = []
        for dataset in dataset_list:
            array = dataset.pixel_array
            array = array.reshape((1, array.shape[0], array.shape[1]))
            array_list.append(array)

        # stack arrays and perform scaling/offset
        self.array = np.concatenate(array_list, axis=0) * self.rescaleparams.scale + self.rescaleparams.offset
        self.array = self.array.astype(int)

        return self

    @classmethod
    def fromPickle(cls, pickle_path):
        """initialize BaseVolume from unchanging format so features can be stored and recalled long term
        """
        if (not os.path.exists(pickle_path)):
            logger.info('file at path: {:s} doesn\'t exists'.format(pickle_path))
        with open(pickle_path, 'rb') as p:
            # added to fix broken module refs in old pickles
            sys.modules['utils.rttypes'] = sys.modules[__name__]
            basevolumepickle = pickle.load(p)
            del sys.modules['utils.rttypes']

        # import data to this object
        try:
            self = cls()
            self.array = basevolumepickle.dataarray
            self.frameofreference = FrameOfReference(basevolumepickle.startposition,
                                                     basevolumepickle.spacing,
                                                     basevolumepickle.size)
            self.modality = basevolumepickle.modality
            self.feature_label = basevolumepickle.feature_label
        except:
            raise PickleOutdatedError()
        return self

    def toPickle(self, pickle_path):
        """store critical data to unchanging format that can be pickled long term
        """
        basevolumepickle = BaseVolumePickle()
        basevolumepickle.startposition = self.frameofreference.start
        basevolumepickle.spacing = self.frameofreference.spacing
        basevolumepickle.size = self.frameofreference.size
        basevolumepickle.dataarray = self.array
        basevolumepickle.modality = self.modality
        basevolumepickle.feature_label = self.feature_label

        _dirname = os.path.dirname(pickle_path)
        if (_dirname and _dirname is not ''):
            os.makedirs(_dirname, exist_ok=True)
        with open(pickle_path, 'wb') as p:
            pickle.dump(basevolumepickle, p)

    def toNifti(self, nifti_path):
        """store volume data to nifti file for convenience
            Args:
               target_vol: the specific volume, e.g. contour mask to be saved. Default is BaseVolume.array
        """
        print("Warning: The affine transform parameters are not tested. But the volume should be working")
        if self.frameofreference is None:
            raise AttributeError("frame of reference is not provided")
        _affine = np.diag([1,1,1,1])
        for ii,scale in enumerate(self.frameofreference.spacing):
            _affine[ii] = scale
        _affine[0,0] *= -1
        for ii,trans in enumerate(self.frameofreference.start):
            _affine[ii,-1] = trans
        niftiio.write_nii(self.array, os.path.basename(nifti_path), os.path.dirname(nifti_path), affine = _affine )

    @classmethod
    def fromMatlab(cls, path):
        """restore BaseVolume from .mat file that was created using BaseVolume.toMatlab()
        """
        extract_str = misc.numpy_safe_string_from_array
        data = scipy.io.loadmat(path, appendmat=True)
        for key, obj in data.items():
            print('{!s}({!s}: {!s}'.format(key, type(obj), obj))
        converted_data = {
            'arraydata': data['arraydata'],
            'size': tuple(data['size'][0,:])[::-1],
            'start': tuple(data['start'][0,:])[::-1],
            'spacing': tuple(data['spacing'][0,:])[::-1],
            'for_uid': extract_str(data['for_uid']),
            'modality': extract_str(data['modality']),
            'feature_label': extract_str(data['feature_label']),
            'scale': data['scale'].item(),
            'offset': data['offset'].item(),
            'order': extract_str(data['order'])
        }
        #  print('arraydata.shape: {!s}'.format(data['arraydata'].shape))
        #  print('')
        #  print('size: {!s}'.format(tuple(data['size'][0,:])[::-1]))
        #  print('start: {!s}'.format(tuple(data['start'][0,:])[::-1]))
        #  print('spacing: {!s}'.format(tuple(data['spacing'][0,:])[::-1]))
        #  print('feature_label: {!s}'.format(extract_str(data['feature_label'])))
        #  print('modality: {!s}'.format(extract_str(data['modality'])))
        #  print('for_uid: {!s}'.format(extract_str(data['for_uid'])))
        #  print('order: {!s}'.format(extract_str(data['order'])))
        #  print('scale: {:f}'.format(data['scale'].item()))
        #  print('offset: {:f}'.format(data['offset'].item()))

        # construct new volume
        self = cls()
        self.array = converted_data['arraydata']
        self.frameofreference = FrameOfReference(converted_data['start'],
                                                 converted_data['spacing'],
                                                 converted_data['size'],
                                                 converted_data['for_uid'])
        self.rescaleparams = RescaleParams(converted_data['scale'],
                                           converted_data['offset'])
        self.modality = converted_data['modality']
        self.feature_label = converted_data['feature_label']

        return self


    def toMatlab(self, path, compress=False):
        """store critical data to .mat file compatible with matlab loading
        This is essentially .toPickle() with compat. for matlab reading

        Optional Args:
            compress (bool): compress dataarray at the cost of write speed
        """
        xstr = misc.xstr  # shorter call-name for use in function
        # first represent as dictionary for savemat()
        if (not self.rescaleparams):
            rescaleparams = RescaleParams(1, 0)
        else:
            rescaleparams = self.rescaleparams

        data = {'arraydata':     self.array,
                'size':          self.frameofreference.size[::-1],
                'start':         self.frameofreference.start[::-1],
                'spacing':       self.frameofreference.spacing[::-1],
                'for_uid':       xstr(self.frameofreference.UID),
                'modality':      xstr(self.modality),
                'feature_label': xstr(self.feature_label),
                'scale':         rescaleparams.scale,
                'offset':        rescaleparams.offset,
                'order':         'ZYX'
                }

        # strip .mat extension which will be added automatically
        #  path = path.rstrip('.mat')

        # write to .mat
        scipy.io.savemat(path, data, appendmat=True, format='5', long_field_names=False,
                         do_compression=compress, oned_as='row')

    # PUBLIC METHODS
    def conformTo(self, frameofreference):
        """Resamples the current BaseVolume to the supplied FrameOfReference

        Args:
            frameofreference   -- FrameOfReference object to resample the Basevolume to

        Returns:
            BaseVolume
        """
        # conform volume to alternate FrameOfReference
        if (frameofreference is None):
            logger.exception('no FrameOfReference provided')
            raise ValueError
        elif (ROI.__name__ in (str(type(frameofreference)))):
            frameofreference = frameofreference.frameofreference
        elif (FrameOfReference.__name__ not in str(type(frameofreference))):  # This is an ugly way of type-checking but cant get isinstance to see both as the same
            logger.exception(('supplied frameofreference of type: "{:s}" must be of the type: "FrameOfReference"'.format(
                str(type(frameofreference)))))
            raise TypeError

        if self.frameofreference == frameofreference:
            return self

        # first match self resolution to requested resolution
        zoomarray, zoomFOR = self._resample(frameofreference.spacing)

        # crop to active volume of requested FrameOfReference in frameofreference
        xstart_idx, ystart_idx, zstart_idx = zoomFOR.getIndices(frameofreference.start)
        # xend_idx, yend_idx, zend_idx = zoomFOR.getIndices(frameofreference.end())
        # force new size to match requested FOR size
        xend_idx, yend_idx, zend_idx = tuple((np.array((xstart_idx, ystart_idx, zstart_idx)) + np.array(frameofreference.size)).tolist())
        try:
            cropped = zoomarray[zstart_idx:zend_idx, ystart_idx:yend_idx, xstart_idx:xend_idx]
            zoomFOR.start = frameofreference.start
            zoomFOR.size = cropped.shape[::-1]
        except:
            logger.exception('request to conform to frame outside of volume\'s frame of reference failed')
            raise Exception()

        # reconstruct volume from resampled array
        resampled_volume = MaskableVolume.fromArray(cropped, zoomFOR)
        resampled_volume.modality = self.modality
        resampled_volume.feature_label = self.feature_label
        resampled_volume.rescaleparams = self.rescaleparams
        return resampled_volume

    def _resample(self, new_voxelsize, order=3):
        if new_voxelsize == self.frameofreference.spacing:
            # no need to resample
            return (self.array, self.frameofreference)

        # voxelsize spec is in order (X,Y,Z) but array is kept in order (Z, Y, X)
        zoom_factors = np.true_divide(self.frameofreference.spacing, new_voxelsize)[::-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zoomarray = interpolation.zoom(self.array, zoom_factors, order=order, mode='nearest')
        zoomFOR = FrameOfReference(self.frameofreference.start, new_voxelsize, zoomarray.shape[::-1])
        return (zoomarray, zoomFOR)

    def resample(self, new_voxelsize, order=3):
        """resamples volume to new voxelsize

        Args:
            new_voxelsize: 3 tuple of voxel size in mm in the order (X, Y, Z)

        """
        zoomarray, zoomFOR = self._resample(new_voxelsize, order)
        new_vol = MaskableVolume.fromArray(zoomarray, zoomFOR)
        new_vol.modality = self.modality
        new_vol.feature_label = self.feature_label
        new_vol.rescaleparams = self.rescaleparams
        return new_vol

    def getSlice(self, idx, axis=0, rescale=False, flatten=False):
        """Extracts 2dArray of idx along the axis.
        Args:
            idx       -- idx identifying the slice along axis

        Optional Args:
            axis      -- specifies axis along which to extract
                            Uses depth-row major ordering:
                            axis=0 -> depth: axial slices inf->sup
                            axis=1 -> rows: coronal slices anterior->posterior
                            axis=2 -> cols: sagittal slices: pt.right->pt.left
            rescale   -- return the element-wise rescaling of pixel_array using the formula:
                            pixel[i] = pixel[i] * rescaleSlope() + rescaleIntercept()
                            If True, use self.rescaleparams, otherwise provide a RescaleParams object
            flatten   -- return a 1darray in depth-stacked row-major order
        """
        cols, rows, depth = self.frameofreference.size

        # perform index bounding
        if (axis==0):
            if (idx < 0 or idx >= depth):
                logger.exception('index out of bounds. must be between 0 -> {:d}'.format(depth-1))
                raise IndexError
            thisslice = self.array[idx, :, :]
        elif (axis==1):
            if (idx < 0 or idx >= rows):
                logger.exception('index out of bounds. must be between 0 -> {:d}'.format(rows-1))
                raise IndexError
            thisslice = self.array[:, idx, :]
        elif (axis==2):
            if (idx < 0 or idx >= cols):
                logger.exception('index out of bounds. must be between 0 -> {:d}'.format(cols-1))
                raise IndexError
            thisslice = self.array[:, :, idx]
        else:
            logger.exception('invalid axis supplied. must be between 0 -> 2')
            raise ValueError

        # RESCALE
        if (rescale is True):
            if (self.rescaleparams is not None):
                thisslice = self.rescaledArray(thisslice, rescale)
            else:
                logger.exception('No RescaleParams assigned to self.rescaleparams')
                raise Exception

        # RESHAPE
        if (flatten):
            thisslice = thisslice.flatten(order='C').reshape((-1, 1))

        return thisslice

    def rescaledArray(self, array, rescaleparams):
        factor = float(rescaleparams.factor)
        offset = float(rescaleparams.offset)
        return np.add(np.multiply(array, factor), offset)

    def vectorize(self):
        """flatten self.array in stacked-depth row-major order
        """
        return self.array.flatten(order='C').reshape((-1, 1))

    def get_val(self, z, y, x):
        """take xyz indices and return the value in array at that location
        """
        frameofreference = self.frameofreference
        # get volume size
        (cols, rows, depth) = frameofreference.size

        # perform index bounding
        if (x < 0 or x >= cols):
            logger.exception('x index ({:d}) out of bounds. must be between 0 -> {:d}'.format(x, cols-1))
            raise IndexError
        if (y < 0 or y >= rows):
            logger.exception('y index ({:d}) out of bounds. must be between 0 -> {:d}'.format(y, rows-1))
            raise IndexError
        if (z < 0 or z >= depth):
            logger.exception('z index ({:d}) out of bounds. must be between 0 -> {:d}'.format(z, depth-1))
            raise IndexError

        return self.array[z, y, x]

    def set_val(self, z, y, x, value):
        """take xyz indices and value and reassing the value in array at that location
        """
        frameofreference = self.frameofreference
        # get volume size
        (cols, rows, depth) = frameofreference.size

        # perform index bounding
        if (x < 0 or x >= cols):
            logger.exception('x index ({:d}) out of bounds. must be between 0 -> {:d}'.format(x, cols-1))
            raise IndexError
        if (y < 0 or y >= rows):
            logger.exception('y index ({:d}) out of bounds. must be between 0 -> {:d}'.format(y, rows-1))
            raise IndexError
        if (z < 0 or z >= depth):
            logger.exception('z index ({:d}) out of bounds. must be between 0 -> {:d}'.format(z, depth-1))
            raise IndexError

        # reassign value
        self.array[z, y, x] = value


class MaskableVolume(BaseVolume):
    """Subclass of BaseVolume that adds support for ROI masking of the data array
    """
    def __init__(self):
        """Entry point to class"""
        # call to base class initializer
        super().__init__()

    def conformTo(self, frameofreference):
        """Resamples the current MaskableVolume to the supplied FrameOfReference and returns a new Volume

        Args:
            frameofreference   -- FrameOfReference object to resample the MaskableVolume to

        Returns:
            MaskableVolume
        """
        base = super().conformTo(frameofreference)
        maskable = MaskableVolume().fromBaseVolume(base)
        return maskable

    # CONSTRUCTOR METHODS
    def deepCopy(self):
        """makes deep copy of self and returns the copy"""
        copy_vol = MaskableVolume()
        copy_vol.array = copy.deepcopy(self.array)
        copy_vol.frameofreference = copy.deepcopy(self.frameofreference)
        copy_vol.rescaleparams = copy.deepcopy(self.rescaleparams)
        copy_vol.modality = self.modality
        copy_vol.feature_label = self.feature_label
        return copy_vol

    def fromBaseVolume(self, base):
        """promotion constructor that converts baseVolume to MaskableVolume, retaining member variables

        Args:
            base -- BaseVolume object

        Returns:
            MaskableVolume
        """
        # copy attributes
        self.array = base.array
        self.frameofreference = copy.deepcopy(base.frameofreference)
        self.rescaleparams = copy.deepcopy(base.rescaleparams)
        self.modality = base.modality
        self.feature_label = base.feature_label
        return self

    # PUBLIC METHODS
    def getSlice(self, idx, axis=0, rescale=False, flatten=False, roi=None):
        """Extracts 2dArray of idx along the axis.
        Args:
            idx     -- idx identifying the slice along axis

        Optional Args:
            axis    --  specifies axis along which to extract
                            Uses depth-row major ordering:
                            axis=0 -> depth: axial slices inf->sup
                            axis=1 -> rows: coronal slices anterior->posterior
                            axis=2 -> cols: sagittal slices: pt.right->pt.left
            rescale -- return the element-wise rescaling of pixel_array using the formula:
                            pixel[i] = pixel[i] * rescaleSlope() + rescaleIntercept()
                        If True, use self.rescaleparams, otherwise provide a RescaleParams object
            flatten -- return a 1darray in depth-stacked row-major order
            roi     -- ROI object that can be supplied to mask the output of getSlice
         """
        # call to base class
        slicearray = super().getSlice(idx, axis, rescale, flatten)

        # get equivalent slice from densemaskarray
        if (roi is not None):
            maskslicearray = roi.makeDenseMask(self.frameofreference).getSlice(idx, axis, rescale, flatten)
            # apply mask
            slicearray = np.multiply(slicearray, maskslicearray)

        return slicearray

    def vectorize(self, roi=None):
        """flatten self.array in stacked-depth row-major order

        Args:
            roi  -- ROI object that can be supplied to mask the output of getSlice
        """
        array = self.array.flatten(order='C').reshape((-1, 1))

        # get equivalent array from densemaskarray
        if (roi is not None):
            maskarray = roi.makeDenseMask(self.frameofreference).vectorize()
            # apply mask
            array = np.multiply(array, maskarray)

        return array

    def applyMask(self, roi):
        """Applies roi mask to entire array and returns masked copy of class

        Args:
            roi -- ROI object that supplies the mask definition
        """
        volume_copy = self.deepCopy()
        masked_array = self.vectorize(roi).reshape(self.frameofreference.size[::-1])
        volume_copy.array = masked_array
        return volume_copy


class PickleOutdatedError(Exception):
    def __init__(self):
        super().__init__('a missing value was requested from a BaseVolumePickle object')


class BaseVolumePickle:
    """Defines common object that can store feature data for long term I/O
    """
    def __init__(self):
        self.dataarray     = None  # numpy ndarray
        self.startposition = None  # (x, y, z)<float>
        self.spacing       = None  # (x, y, z)<float>
        self.size          = None  # (x, y, z)<integer>
        self.modality      = None  # string
        self.feature_label = None  # string
###UNDER CONSTRUCTION###
"""
RTContourLabel type added, allowing generating dense label from rtstructs.
"""
class RTContourLabel(ROI):
    """ Generating training labels from rtstruct. Supporting pixel-wise dense label map generation and soft probability
    map generation given the location of the contour center.

    extenstion of pymedimage.rttypes.ROI class
    Member objects:
        self.reference_slice_fid: fid of the dicom image slice which it attaches to
        self.contour_loc_idx: index of the center of contour center in self.array
        self.contour_loc_coord: coordinate of the center of contour center
        self.reference_slice_FOR: rttypes.FrameOfReference of the target volume where the rtstruct is resampled to
        self.array
        self.normal : normalization upbound for labels
    Methods:
        self._find_center()
        self._loc_to_dense_array() : generate a location point array with the same size of reference slice
        self._coord2idx() : convert dicom coordinate to numpy index given FrameOfReference
        self._idx2coord() : convert numpy index to dicom coordinate given FrameOfReference
        self.smooth_prob_map() : return the smoothed probability map with the same size of reference dicom
        self.contour_map(): return pixelwise tumer mask with the same size of reference dicom

    """

    def __init__(self, base_ROI, reference_slice_fid = None, normal = 1.0):
        """ initialize from a pymedimage.rttypes.BaseVolume object """
        self.__dict__ = copy.deepcopy(base_ROI.__dict__)
        self.array = super(RTContourLabel, self).makeDenseMask().array
        self.reference_slice_FOR = None
        self.contour_loc_idx = self._find_center()
        self.contour_loc_coord = self._idx2coord(self.contour_loc_idx, self.frameofreference)
        if reference_slice_fid is not None:
            self._get_FOR(reference_slice_fid)
        self.normal = float(normal)

    def _find_center(self):
        """ find the center of this contour
            Assuming that backgound is 0 while foreground is not
        """
        print("Warning: The ROI center calculation code cannot deal with the case where multiple connected component are present")
        _foreground_idx = np.where(self.array > 0)
        return [ (np.mean(_foreground_idx[0])), (np.mean(_foreground_idx[1])), (np.mean(_foreground_idx[2])) ]

    def _center_array(self):
        idx = tuple([int(axis) for axis in self.contour_loc])
        _ctr_array = np.zeros(self.array.shape)
        _ctr_array[idx] = 1
        return _ctr_array
    # TODO: Move the following convenience functions to general utility module
    def _coord2idx(self, coord, reference):
        """ conversion between dicom coordinate and numpy indices
            Args:
                reference: FrameOfReference object
        """
        np_dims = []
        for idx in range(len(reference.spacing)):
            np_dims.append( int(( coord[idx] - reference.start[idx] ) / reference.spacing[idx]) )
        return tuple(reversed(np_dims))

    def _idx2coord(self, idx, reference):
        """ reverse of _coord2idx"""
        dcm_coords = []
        _ndims = len(reference.spacing)
        for axis in range(_ndims):
            dcm_coords.append( float(idx[_ndims - axis -1] * reference.spacing[axis] + reference.start[axis]) )
        return dcm_coords

    def _loc_to_dense_array(self, ref_fid = None):
        """ given the reference slice, generate a mask with the same size """
        if ref_fid is not None:
            _ref_filename = ref_fid
            self._get_FOR(ref_fid)
        elif self.reference_slice_FOR is None:
            raise AttributeError("reference slice fid is not specified.")

        # assume that all the other slices of this volume are saved in a same directory
        # Note: This is quite beautiful since similar implementation has been also used in the parent class
        # Note: The order of numpy array axis and dicom are opposite
        _npdim2, _npdim1, _npdim0 = self.reference_slice_FOR.size
        # convert the center coordinate into index in the dense volume
        dense_location_map = np.zeros([_npdim0, _npdim1, _npdim2])
        dense_contour_center_idx = self._coord2idx(self.contour_loc_coord, self.reference_slice_FOR)
        dense_location_map[dense_contour_center_idx] = 1
        return dense_location_map

    def _get_FOR(self, ref_fid, struct_identifier = "rtstruct"):
        """ Get the reference frame fid """
        if ref_fid == None:
            raise FileNotFoundError("No fid for the reference slice is specified!")
        self.reference_slice_FOR = FrameOfReference.from_dcm_fid(ref_fid, struct_identifier = struct_identifier)

    # TODO: Add support of other kernels
    def smooth_prob_map(self, sigma = None, ref_fid = None, kernel = 'Gaussian'):
        """ Generate a soft probability map of contour occurance
            Args:
                Sigma: the gaussian smoothing radias in pixels
                ref_fid: dicom file fid of image volume where the contour will be conformed to
                kernel: certain kernel to be used other than gaussian
        """
        dense_location_map = self._loc_to_dense_array(ref_fid)
        self.dense_location_map = dense_location_map
        if kernel == 'Gaussian':
            if sigma == None:
                sigma = 5
            label = sfilter.gaussian_filter(dense_location_map, sigma, mode = 'nearest')
            return self.normal * label * 1.0 / np.max(label)
        else:
            raise Exception(" This function is still under construction! ")

    def contour_mask(self, ref_fid = None):
        #pdb.set_trace()
        if ref_fid is not None:
            _ref_filename = ref_fid
            self._get_FOR(ref_fid)
        elif self.reference_slice_FOR is None:
            #self.reference_slice_FOR = FrameOfReference.from_dcm_fid(_ref_filename)
            raise AttributeError("reference slice frame of refernce is not initialized")
        ct = super(RTContourLabel, self).makeDenseMask(self.reference_slice_FOR).array
        return self.normal * ct * 1.0 / np.max(ct)

    def saveNifti(self, nifti_path, volume, FOR):
        """ Save file to mifti
        """
        if self.frameofreference is None:
            raise AttributeError("frame of reference is not provided")
        ###############debug###############
#        _affine = np.diag([1.0, 1.0, 1.0, 1.0])
#        for ii,scale in enumerate(list(FOR.spacing) ):
#            _affine[ii,ii] = scale
#
#        _affine[0,0] *= -1
#
#        for ii,trans in enumerate(list(FOR.start) ):
#            _affine[ii, -1] = trans
#        _affine[0, -1] *= -1
#        _affine[1,1] *= -1
#        _affine[1,-1] *= -1
#
        ############ end of debug
        _tvol = volume.T
        _affine = self._dcm2nii_affine(FOR, _tvol.shape)
        #pdb.set_trace()
        out_dir = os.path.basename(nifti_path)
        if out_dir == '':
            out_dir = "./"
       # volume = volume.T
#
        _tvol = np.flip(_tvol, axis = 1)
        niftiio.write_nii(_tvol, filename = os.path.basename(nifti_path),path = os.path.dirname(nifti_path), affine = _affine )
###END OF CONSTRUCTION###

    def _dcm2nii_affine(self,FOR, vol_shape):
        """ generate nifti affine matrix from dicom/frameOfReference coordinate system """
        if FOR is None:
            raise AttributeError("frame of reference is not provided")
        _affine = np.diag([1.0, 1.0, 1.0, 1.0])
        for ii,scale in enumerate(list(FOR.spacing) ):
            _affine[ii,ii] = scale

        _affine[0,0] *= -1

        for ii,trans in enumerate(list(FOR.start) ):
            _affine[ii, -1] = trans
        _affine[0, -1] *= -1
        # tricky part for nifti
        _dim1 = vol_shape[1]
        _dcm_trans = _affine[1,-1]
        _nii_trans = _affine[0,0] * _dim1 - _dcm_trans
        _affine[1 ,-1] = _nii_trans
        return _affine
        # end of debug

