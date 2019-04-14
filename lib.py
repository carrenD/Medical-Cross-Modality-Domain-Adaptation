import os
import numpy as np
import tensorflow as tf
import nibabel as nib

#### Helpers for file IOs
def _read_lists(fid):
    """
    Read all kinds of lists from text file to python lists
    """
    if not os.path.isfile(fid):
        return None
    with open(fid,'r') as fd:
        _list = fd.readlines()

    my_list = []
    for _item in _list:
        if len(_item) < 3:
            _list.remove(_item)
        my_list.append(_item.split('\n')[0])
    return my_list

def _save(sess, model_path, global_step):
    """
    Saves the current session to a checkpoint
    """
    saver = tf.train.Saver()
    save_path = saver.save(sess, model_path, global_step = global_step)
    return save_path

def _save_nii_prediction(gth, comp_pred, ref_fid, out_folder, out_bname, debug = False):
    """
    save prediction, sample and gth to nii file given a reference
    """
    # first write prediction
    ref_obj = read_nii_object(ref_fid)
    ref_affine = ref_obj.get_affine()
    out_bname = out_bname.split(".")[0] + ".nii.gz"
    write_nii(comp_pred, out_bname, out_folder, affine = ref_affine)

    # then write sample
    _local_gth = gth.copy()
    _local_gth[_local_gth > self.num_cls - 1] = 0
    out_label_bname = "gth_" + out_bname
    write_nii(_local_gth, out_label_bname, out_folder, affine = ref_affine)

def write_nii(array_data, filename, path = "", affine = None):
    """write np array into nii file"""
    if affine is None:
        print("No information about the global coordinate system")
        affine = np.diag([1,1,1,1])
    #pdb.set_trace()
    #TODO: to check if it works
   # array_data = np.int16(array_data)
    array_img = nib.Nifti1Image(array_data, affine)
    save_fid = os.path.join(path,filename)
    try:
        array_img.to_filename(save_fid)
        print("Nii object %s has been saved!"%save_fid)
    except:
        raise Exception("file %s cannot be saved!"%save_fid)
    return save_fid

def read_nii_image(input_fid):
    """read the nii image data into numpy array"""
    img = nib.load(input_fid)
    return img.get_data()

def read_nii_object(input_fid):
    """ directly read the nii object """
    #pdb.set_trace()
    return nib.load(input_fid)

#### Helpers for evaluations
def _label_decomp(num_cls, label_vol):
    """
    decompose label for softmax classifier
    original labels are batchsize * W * H * 1, with label values 0,1,2,3...
    this function decompse it to one hot, e.g.: 0,0,0,1,0,0 in channel dimension
    numpy version of tf.one_hot
    """
    _batch_shape = list(label_vol.shape)
    _vol = np.zeros(_batch_shape)
    _vol[label_vol == 0] = 1
    _vol = _vol[..., np.newaxis]
    for i in range(num_cls):
        if i == 0:
            continue
        _n_slice = np.zeros(label_vol.shape)
        _n_slice[label_vol == i] = 1
        _vol = np.concatenate( (_vol, _n_slice[..., np.newaxis]), axis = 3 )
    return np.float32(_vol)



def _dice_eval(compact_pred, labels, n_class):
    """
    calculate standard dice for evaluation, here uses the class prediction, not the probability
    """
    dice_arr = []
    dice = 0
    eps = 1e-7
    pred = tf.one_hot(compact_pred, depth = n_class, axis = -1)
    for i in range(n_class):
        inse = tf.reduce_sum(pred[:, :, :, i] * labels[:, :, :, i])
        union = tf.reduce_sum(pred[:, :, :, i]) + tf.reduce_sum(labels[:, :, :, i])
        dice = dice + 2.0 * inse / (union + eps)
        dice_arr.append(2.0 * inse / (union + eps))

    return 1.0 * dice  / n_class, dice_arr


def _inverse_lookup(my_dict, _value):

    for key, dic_value in list(my_dict.items()):
        if dic_value == _value:
            return key
    return None


def _jaccard(conf_matrix):
    """
    calculate jaccard similarity from confusion_matrix
    """
    num_cls = conf_matrix.shape[0]
    jac = np.zeros(num_cls)
    for ii in range(num_cls):
        pp = np.sum(conf_matrix[:,ii])
        gp = np.sum(conf_matrix[ii,:])
        hit = conf_matrix[ii,ii]
        if (pp + gp -hit) == 0:
            jac[ii] = 0
        else:
            jac[ii] = hit * 1.0 / (pp + gp - hit )
    return jac


def _dice(conf_matrix):
    """
    calculate dice coefficient from confusion_matrix
    """
    num_cls = conf_matrix.shape[0]
    dic = np.zeros(num_cls)
    for ii in range(num_cls):
        pp = np.sum(conf_matrix[:,ii])
        gp = np.sum(conf_matrix[ii,:])
        hit = conf_matrix[ii,ii]
        if (pp + gp) == 0:
            dic[ii] = 0
        else:
            dic[ii] = 2.0 * hit / (pp + gp)
    return dic


def _indicator_eval(cm):
    """
    Decompose confusion matrix and get statistics
    """
    contour_map = { # a map used for mapping label value to its name, used for output
    "bg": 0,
    "la_myo": 1,
    "la_blood": 2,
    "lv_blood": 3,
    "aa": 4
    }

    dice = _dice(cm)
    jaccard = _jaccard(cm)
    print(cm)
    for organ, ind in list(contour_map.items()):
        print(( "organ: %s"%organ  ))
        print(( "dice: %s"%(dice[int(ind)] ) ))
        print(( "jaccard: %s"%(jaccard[int(ind)] ) ))

    return dice, jaccard

