"""visual.py

Plotting library for rttypes
"""

import os
import logging
import math
import numpy as np

# initialize module logger
logger = logging.getLogger(__name__)

def writeFigureToFile(fig, path, removeaxes=False, overwrite=False):
    """Standardized method to write figure to file with existence checking and overwrite switch

    Args:
        fig         -- matplotlib figure instance
        path        -- string containing full output path and extension

    Optional Args:
        removeaxes  -- boolean: remove axes labels and ticks?
        overwrite   -- boolean: overwrite existing file at path?
    """
    if (removeaxes):
        for ax in fig.axes:
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticklabels([])
            ax.set_yticks([])

    # save to file
    try:
        saved = False
        exists = os.path.exists(path)
        if (overwrite or not exists):
            if (not os.path.isdir(os.path.dirname(path))):
                # path dir doesnt exist, create it
                os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path, bbox_inches='tight')
            saved = True

    except Exception as details:
        logger.exception('there was an error in saving the figure to: {:s}'.format(path))
        logger.exception(details)
    else:
        if (saved):
            if (overwrite and exists):
                logger.info('image overwritten at: {:s}'.format(path))
            else:
                logger.info('image saved to: {:s}'.format(path))

def tile(array_list, perrow, square=False, pad_width=5, pad_intensity=1000):
    """Takes a list of arrays and number of images per row and constructs a tiled array for margin-less
    visualization

    Args:
        array_list    -- list of np.ndarrays to be tiled in row-major order
        perrow        -- integer specifying number of images per row

    Optional Args:
        square        -- Try to make length and width equal by tiling vertical columns side-by-side
        pad_width     -- # columns between vertical tiling columns
        pad_intensity -- # intensity value of padding cells

    Returns:
        numpy matrix/2dArray
    """
    # setup
    if (not isinstance(array_list, list)):
        logger.debug('converting array_list to list')
        array_list_old = array_list
        ndims = len(array_list_old.shape)
        if (ndims == 3):
            array_list = []
            array_list_old_2dshape = (array_list_old.shape[1], array_list_old.shape[2])
            for i in range(array_list_old.shape[0]):
                array_list.append(array_list_old[i, :, :].reshape(array_list_old_2dshape))
        elif (ndims == 2):
            array_list = [array_list_old]
    nimages = len(array_list)
    expect_row_shape = (array_list[0].shape[0], perrow * array_list[0].shape[1])

    # make concatenated rows
    rows_list = []
    this_row_array = None
    for i in range(nimages):
        if (i % perrow == 0 or i == nimages - 1):
            # add previous row to list
            if (i > 0):
                rows_list.append(this_row_array)
                this_row_array = None
            # start new row
            this_row_array = array_list[i]
        else:
            # add to row
            this_row_array = np.concatenate((this_row_array, array_list[i]), axis=1)

    # extend short rows with zeros
    for i, row in enumerate(rows_list):
        if (row.shape != expect_row_shape):
            extra = np.zeros((expect_row_shape[0], expect_row_shape[1] - row.shape[1]))
            row = np.concatenate((row, extra), axis=1)
            del rows_list[i]
            rows_list.insert(i, row)

    # concatenate rows into matrix
    if (square):
        # try to make length and width equal by tiling vertically, leaving a space and continuing in
        # another column to the right
        if (pad_width >= 0):
            pad = pad_width
        else:
            pad = 0
        if (pad_intensity <= 0):
            pad_intensity = 0

        rows = len(rows_list) * expect_row_shape[0]
        cols = expect_row_shape[1]
        # get area, then find side length that will work best
        area = rows * cols
        pref_rows = int((math.sqrt(area) / expect_row_shape[0]))
        # pref_cols = int(area / (pref_rows * expect_row_shape[0]) / expect_row_shape[1]) + 1

        # construct matrix
        cols_list = []
        this_col_array = []
        for i, row in enumerate(rows_list):
            if (i % pref_rows == 0 or i == len(rows_list)-1):
                if (i > 0):
                    # add previous column to list
                    cols_list.append(this_col_array)
                    this_col_array = None
                    # add padding column
                    if (pad > 0 and i < len(rows_list)-1):
                        cols_list.append(pad_intensity * np.ones((pref_rows * expect_row_shape[0], pad)))

                # start new column
                this_col_array = row
            else:
                # add to column
                this_col_array = np.concatenate((this_col_array, row), axis=0)

        # extend short cols with zeros
        for i, col in enumerate(cols_list):
            if (col.shape[0] != pref_rows * expect_row_shape[0]):
                extra = np.zeros((expect_row_shape[0] * pref_rows - col.shape[0], expect_row_shape[1]))
                row = np.concatenate((col, extra), axis=0)
                del cols_list[i]
                cols_list.insert(i, row)

        tiled_array = np.concatenate(cols_list, axis=1)

    else:
        tiled_array = np.concatenate(rows_list, axis=0)
    return tiled_array
