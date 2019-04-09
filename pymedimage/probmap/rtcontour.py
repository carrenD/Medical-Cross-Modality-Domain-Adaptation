"""
Generate a binary mask given the RTStruct file and perform gaussian filtering for brain_met
probability map generation

This should be done in after dicom files are 
"""
# TODO: organize previously written nifti operation code (brainMets) and this one
import pymedimage.dcmio as dio
import numpy as np
import scipy.ndimage.filters as sfilter
import nibabel as nib
import pymedimage.visualize as viz
import os
import dicom as dcm 
import pymedimage.rttypes as rts
import copy
import pdb
# TODO: unify the usage of BaseVolume and dense np array
# Use the same base data structure BaseVolume to store masked 
# TODO: Consider one thing: the dataset is stored in nii files, while the 
# rtstruct is in dicom
class RTContourLabel(rts.ROI):
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
        #pdb.set_trace()
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
        #pdb.set_trace()
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
        #pdb.set_trace()
        dense_contour_center_idx = self._coord2idx(self.contour_loc_coord, self.reference_slice_FOR) 
        dense_location_map[dense_contour_center_idx] = 1
        return dense_location_map

    def _get_FOR(self, ref_fid):
        """ Get the reference frame fid """
        if ref_fid == None:
            raise FileNotFoundError("No fid for the reference slice is specified!")
        #pdb.set_trace()
        self.reference_slice_FOR = rts.FrameOfReference.from_dcm_fid(ref_fid)

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
        if ref_fid is not None:
            _ref_filename = ref_fid
            self._get_FOR(ref_fid)
        elif self.reference_slice_FOR is None:
            #self.reference_slice_FOR = rts.FrameOfReference.from_dcm_fid(_ref_filename)
            raise AttributeError("reference slice frame of refernce is not initialized")
        ct = super(RTContourLabel, self).makeDenseMask(self.reference_slice_FOR).array
        return self.normal * ct * 1.0 / np.max(ct)
    
    
