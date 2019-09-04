"""
Here is the implementation for MRI to CT unsupervised domain adaptation with adversarial loss for segmentation network
"""
import os
import sys
import logging
import datetime
import argparse

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import random

import adversarial as drn
from lib import _read_lists

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

random.seed(456)
logging.basicConfig(filename = "general_log", level = logging.DEBUG)
currtime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

rate = 0.3
date = "1221"

cost_kwargs = {
    "regularizer": 1e-4, # L2 norm regularizer segmentation model
    "gan_regularizer": 1e-4, # L2 norm regularizer for WGAN variables
    "miu_gen": 0.002, # weighing of generator loss
    "miu_dis": 0.002, # weighing of discriminator loss
    "lambda_mask_loss": None, # the trade-off parameter for mask discriminator, set it as 0.3
}

opt_kwargs = {
    "learning_rate": 3e-4,
}

network_config = {
    "mr_front_trainable": False,  # whether mri segmenter early layers are trainable, set it as False
    "joint_trainable": False, # whether common higher layers shared by MRI and CT trainable or not, set it as False
    "ct_front_trainable": None,  # whether CT adaptation (DAM) variables are trainable
    "cls_trainable": True,  # whether domain discriminator for CNN features are trainable, set it as True
    "m_cls_trainable": True,  # whether domain discriminator for segmentation mask are trainable, set it as True
    "restore_skip_kwd": ["Adam", "RMS", "cls"],  # when manually RESTORE a checkpoint, what should be ignored, for implementation purpose
}

train_config = {
    "restore_from_baseline": None, # restore from the source segmenter and manually initialize DAM layers with learned early layers
    "copy_main": None,  # only for rerun the zip experiment with cls6 pretrained classifier
    "clear_rms": None,  # restore from the baseline module and manually copy parameters to the ct branch
    "lr_update": None, # if true, when the model is first run, the learning rate specified above will be used to update learning rate in the checkpoint
    "dis_interval": 1,  # frequency of updating discriminator, normally, just set it to 1
    "gen_interval": 1,  # frequency of updating generator (CT adaptation layers), normally, just set it to 1
    "dis_sub_iter": 20, # number of sub iteration in one update, set as 1 for pre-train, other wise 20
    "gen_sub_iter": 1,
    "tag": "gan-"+str(rate)+"_"+date,  # name postfix of tensorboard log file for identifying this run
    "iter_upd_interval": 300,  # interval for increasing number of *_sub_iter
    "dis_sub_iter_inc": 1,  # number of iteraion increase when updating
    "gen_sub_iter_inc": 0,
    "lr_decay_factor": 0.98,
    "checkpoint_space": 100, # intervals between model save and learning rate decay
    "training_iters": 200,
    "epochs": 600
}

def main(phase):

    mr_train_list = _read_lists("./lists/mr_train_list") # load a list of tfrecord samples for CT training samples
    mr_val_list = _read_lists("./lists/mr_val_list") # load  list of tfrecord samples for CT validation
    ct_train_list = _read_lists("./lists/ct_train_list") # load a list of tfrecord samples for MRI training
    ct_val_list = _read_lists("./lists/ct_val_list") # load a list of tfrecord samples for MRI validation

    adapt_var_list = _read_lists("./lists/half_zip_ct_vars") # load a list of all variables for opened CT layers
    mr_var_list = _read_lists("./lists/half_zip_mri_vars") # load a list of all MRI variables corresponding to adapt_variable_list, These variables are used for initializing CT adaptation variables
    old_bn_list = _read_lists("./lists/old_bn_list") # load a list of batch normalization internal variables for source segmenter model
    new_bn_list = _read_lists("./lists/pred_bn_list")  # load a list of batch normalization internal variables for current adaptation model.

    num_cls = 5 # number of classes, 0: background, 1: la_myo, 2: la_blood, 3: lv_blood, 4: aa
    batch_size = 6

    output_path = "./tmp_exps/mr2ct"+date+str(rate)[0]+str(rate)[2]
    restored_path = output_path

    if phase == 'pre-train':  # pre-train the discriminator for CNN feature, before update the DAM and segmentation mask discriminator together

        network_config["ct_front_trainable"] = False

        train_config["restore_from_baseline"] = True
        train_config["copy_main"] = True
        train_config["clear_rms"] = True
        train_config["lr_update"] = True
        train_config["gen_interval"] = 0
        train_config["dis_sub_iter"] = 1
        train_config["dis_sub_iter_inc"] = 0
        train_config["checkpoint_space"] = 2000  # intervals between model save and learning rate decayU
        train_config["training_iters"] = 201
        train_config["epochs"] = 100

        cost_kwargs["lambda_mask_loss"] = 0 # do not take into account for the mask discriminator in pre-training, as ct prediction masks are initially unmeaningful

    elif phase == 'train-gan':  # After warming-up, train the DAM and DCM together

        network_config["ct_front_trainable"] = True

        train_config["restore_from_baseline"] = False
        train_config["copy_main"] = False
        train_config["clear_rms"] = False
        train_config["lr_update"] = True
        train_config["tag"] = train_config["tag"] + "-gan"

        cost_kwargs["lambda_mask_loss"] = rate

    elif phase == 'fine-tune':  # continue to train the GAN from a breakpoint

        network_config["ct_front_trainable"] = True

        train_config["restore_from_baseline"] = False
        train_config["copy_main"] = False
        train_config["clear_rms"] = False
        training_config["lr_update"] = False
        train_config["gen_interval"] = 1
        train_config["dis_sub_iter"] = 30
        train_config["tag"] = train_config["tag"] + "-fine_tune"

        cost_kwargs["lambda_mask_loss"] = rate

    else:
        raise Exception("Please set a training phase!")

    net = drn.Full_DRN(channels = 3, batch_size = batch_size,  n_class = num_cls, cost_kwargs = cost_kwargs, network_config = network_config)
    print("Network has been built ...")

    trainer = drn.Trainer(net, mr_train_list, mr_val_list, ct_train_list, ct_val_list, \
                             adapt_var_list = adapt_var_list,\
                             mr_var_list = mr_var_list,\
                             old_bn_list = old_bn_list,\
                             new_bn_list = new_bn_list,\
                             num_cls = num_cls, \
                             batch_size = batch_size,\
                             opt_kwargs = opt_kwargs,\
                             train_config = train_config)

    print("Now start training...")
    trainer.train(output_path = output_path,\
                  restored_path = restored_path,\
                  training_iters = train_config["training_iters"],\
                  epochs = train_config["epochs"])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type = str, default = None)
    args = parser.parse_args()
    phase = args.phase

    main(phase = phase)
