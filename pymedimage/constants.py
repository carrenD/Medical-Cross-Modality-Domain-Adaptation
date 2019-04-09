"""constants.py

constants for patient axis specifications
    Uses depth-row major ordering:
        axis=0 -> depth: axial slices inf->sup
        axis=1 -> rows: coronal slices anterior->posterior
        axis=2 -> cols: sagittal slices: pt.right->pt.left
"""
class Axes:
    AXIAL = 0
    CORONAL = 1
    SAGITTAL = 2
