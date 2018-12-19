"""
This is the basic training script for the baseline MRI or CT Model
It is used to train the source segmenter
"""
import os
import sys
import logging
import datetime
import argparse

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import drn_dilate_base_leaky as drn
import numpy as np


logging.basicConfig(filename = "general_log", level = logging.DEBUG)

test_flag = False # set True if testing the segmenter
restore = True # set True if resume training from stored model
verbose = True # set True if want to see detailed training logs

# manual_bn_train = False
currtime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

train_fid = "./mr_train_list"
val_fid = "./mr_val_list"
test_label_fid = "unfinished"
test_nii_fid = "unfinished"
output_path = "./tmp_exps/mr_baseline"

restored_path = output_path
lr_update_flag = False # Set True if want to use a new learning rate for fine-tuning

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

num_cls = 5
batch_size = 10
if test_flag is True:
    batch_size = 128
training_iters = 10
epochs = 50000
checkpoint_space = 1500
image_summeris = True

optimizer = 'adam'

cost_kwargs = {
    "cross_flag": True, # use cross entropy loss
    "miu_cross": 1.0,
    "dice_flag": True, # use dice loss
    "miu_dice": 1.0,
    "regularizer": 1e-4,
    "miu_dis": 1.0, # miu_dis and miu_gen are weights for GAN, so no use when training the source model
    "miu_gen": 1.0}

opt_kwargs = {
    "learning_rate": 1e-3
}


def _read_lists(fid):
    """ read train list and test list """
    if not os.path.isfile(fid):
        return None
    with open(fid,'r') as fd:
        _list = fd.readlines()

    my_list = []
    for _item in _list:
        if len(_item) < 5:
            _list.remove(_item)
        my_list.append(_item.split('\n')[0])
    return my_list

def main():

    train_list = _read_lists(train_fid)
    val_list = _read_lists(val_fid)
    test_label_list = _read_lists(test_label_fid)
    test_nii_list = _read_lists(test_nii_fid)

    try:
        os.makedirs(output_path)
    except:
        print("folder exist!")

    if verbose:
        print("Start building the data generator...")

    my_net = drn.Full_DRN(channels = 3, batch_size = batch_size,  n_class = num_cls, image_summeris = image_summeris, test_flag = test_flag, cost_kwargs = cost_kwargs)
    if verbose:
        print("Network has been built!")
    my_trainer = drn.Trainer(my_net, train_list = train_list, val_list = val_list, test_label_list = test_label_list, test_nii_list = test_nii_list, num_cls = num_cls, \
                             batch_size = batch_size, opt_kwargs = opt_kwargs, checkpoint_space = checkpoint_space,\
                             optimizer = optimizer, lr_update_flag = lr_update_flag)

    # start tensorboard before getting started
    command1 = "tensorboard --logdir=" + output_path + " --port=6999 " +  " &"
    os.system(command1)

    if test_flag is True:
        my_trainer.test(output_path = output_path, restored_path = restored_path)
        return 0

    print("Now start training...")
    if restore is True:
        my_trainer.train(output_path = output_path, training_iters = training_iters, epochs = epochs, restore = True, restored_path = restored_path)
    else:
        my_trainer.train(output_path = output_path, training_iters = training_iters, epochs = epochs)

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--lr_update_flag", action = "store_true", default = False)
    # args = parser.parse_args()
    # lr_update_flag = args.lr_update_flag

    main()