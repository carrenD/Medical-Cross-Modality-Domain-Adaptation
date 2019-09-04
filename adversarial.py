import os
import glob
import time
import shutil
import numpy as np
from collections import OrderedDict
import __future__
import logging
import matplotlib
import tensorflow as tf
import csv

from tensorflow.python import debug as tf_debug
from layers import *
from ops import *
from lib import _dice_eval, _save, _save_nii_prediction, _jaccard, _dice, _label_decomp, _indicator_eval, read_nii_image

np.random.seed(0)
contour_map = { # a map used for mapping label value to its name, used for output
    "bg": 0,
    "la_myo": 1,
    "la_blood": 2,
    "lv_blood": 3,
    "aa": 4
}

verbose = True
logging.basicConfig(filename = "curr_log", level=logging.DEBUG, format='%(asctime)s %(message)s')
if verbose == True:
    logging.getLogger().addHandler(logging.StreamHandler())
raw_size = [256, 256, 3] # original raw input size
volume_size = [256, 256, 3] # volume size after processing, for the tfrecord file
label_size = [256, 256, 1] # size of label
decomp_feature = { # configuration for decoding tf_record file
            'dsize_dim0': tf.FixedLenFeature([], tf.int64),
            'dsize_dim1': tf.FixedLenFeature([], tf.int64),
            'dsize_dim2': tf.FixedLenFeature([], tf.int64),
            'lsize_dim0': tf.FixedLenFeature([], tf.int64),
            'lsize_dim1': tf.FixedLenFeature([], tf.int64),
            'lsize_dim2': tf.FixedLenFeature([], tf.int64),
            'data_vol': tf.FixedLenFeature([], tf.string),
            'label_vol': tf.FixedLenFeature([], tf.string)}

class Full_DRN(object):

    def __init__(self, channels, n_class, batch_size, cost_kwargs={}, network_config = {}):

        ##### Done this function

        tf.reset_default_graph()

        self.n_class = n_class # please note background is another class
        self.batch_size = batch_size

        self.mr_front_weights = [] # conv weights of MR path
        self.ct_front_weights = [] # conv weights of CT path
        self.cls_weights = []   # weights of feature discriminator
        self.m_cls_weights = [] # weights for segmentation mask discriminator
        self.joint_weights = [] # weights of joint part between CT and MRI. The final segmentor in our case

        self.mr = tf.placeholder("float", shape=[None, volume_size[0], volume_size[1], channels], name = "mr_ph")
        self.ct = tf.placeholder("float", shape=[None, volume_size[0], volume_size[1], channels])
        self.ct_y = tf.placeholder("float", shape=[None, label_size[0], label_size[1], self.n_class])
        self.mr_y = tf.placeholder("float", shape=[None, label_size[0], label_size[1], self.n_class])

        self.mr_front_bn = tf.placeholder_with_default(False, shape = None, name = "main_batchnorm_training_switch")
        self.joint_bn = tf.placeholder_with_default(False, shape = None, name = "joint_batchnorm_training_switch")
        self.ct_front_bn = tf.placeholder_with_default(True, shape = None, name = "adapt_batchnorm_training_switch")

        # these two are useless. They are not passed into the program
        self.cls_bn = tf.placeholder_with_default(True, shape = None, name = "cls_batchnorm_training_switch")
        self.m_cls_bn = tf.placeholder_with_default(True, shape = None, name = "mask_cls_batchnorm_training_switch")

        self.network_config = network_config
        self.mr_front_trainable = self.network_config["mr_front_trainable"]
        self.ct_front_trainable = self.network_config["ct_front_trainable"]
        self.joint_trainable = self.network_config["joint_trainable"]
        self.cls_trainable = self.network_config["cls_trainable"]
        self.m_cls_trainable = self.network_config["m_cls_trainable"]

        self.keep_prob = tf.placeholder(tf.float32) # dropout keep probability

        # Get features from MRI and CT path, for early layers
        _mr_c4_2, _ct_c4_2, _mr_c6_2, _ct_c6_2 = self.create_zip_network(input_channel = channels,\
                                        feature_base = 16, num_cls = n_class, keep_prob = self.keep_prob,\
                                        main_bn = self.mr_front_bn, main_trainable = self.mr_front_trainable,\
                                        adapt_bn = self.ct_front_bn, adapt_trainable = self.ct_front_trainable)

        # Get features from MRI and CT, fromt the shared higher layers
        with tf.variable_scope("", reuse = tf.AUTO_REUSE) as scope:
            _ct_c9_2, _ct_b8, _ct_b7, _ct_logits = self.create_second_half( _ct_c6_2, feature_base = 16, input_channel = 3, num_cls = n_class, keep_prob = self.keep_prob, joint_bn = self.joint_bn, joint_trainable = self.joint_trainable)
            _mr_c9_2, _mr_b8, _mr_b7, _mr_logits = self.create_second_half( _mr_c6_2, feature_base = 16, input_channel = 3, num_cls = n_class, keep_prob = self.keep_prob, joint_bn = self.joint_bn, joint_trainable = self.joint_trainable)

        self.ct_conv9_2 = _ct_c9_2
        self.mr_conv9_2 = _mr_c9_2

        with tf.variable_scope("cls_scope", reuse = tf.AUTO_REUSE) as scope:
            self._ct_class_logits = self.create_classifier(_ct_c4_2, _ct_c6_2, _ct_b7, _ct_c9_2, _ct_logits)
            self._mr_class_logits = self.create_classifier(_mr_c4_2, _mr_c6_2, _mr_b7, _mr_c9_2, _mr_logits)

        self.predictor = pixel_wise_softmax_2(_ct_logits) # segmentation logits of CT
        self.compact_pred = tf.argmax(self.predicter, 3) # predictions

        self.compact_y = tf.argmax(self.ct_y, 3) # ground truth
        self.ct_dice_eval, self.ct_dice_eval_arr = _dice_eval(self.compact_pred, self.ct_y, self.n_class) # used for monitoring training process
        self.ct_dice_eval_c1 = self.ct_dice_eval_arr[1]
        self.ct_dice_eval_c2 = self.ct_dice_eval_arr[2]
        self.ct_dice_eval_c3 = self.ct_dice_eval_arr[3]
        self.ct_dice_eval_c4 = self.ct_dice_eval_arr[4]

        self.mr_seg_valid = pixel_wise_softmax_2(_mr_logits) # segmentation logits of MRI
        self.compact_mr_valid = tf.argmax(self.mr_seg_valid, 3)

        self.compact_mr_y = tf.argmax(self.mr_y, 3)
        self.mr_dice_eval, self.mr_dice_eval_arr = _dice_eval(self.compact_mr_valid, self.mr_y, self.n_class)

        with tf.variable_scope("mask_cls_scope", reuse = tf.AUTO_REUSE) as scope:
            self._ct_mask_logits = self.create_mask_critic(_ct_logits, num_cls = n_class)  # auxilary D loss for masks
            self._mr_mask_logits = self.create_mask_critic(_mr_logits, num_cls = n_class)

        self.cost_kwargs = cost_kwargs
        self.dis_loss, self.ct_gen_loss, self.fixed_coeff_reg, self.dis_reg, self.gen_reg = self._get_cost(_ct_logits, _mr_logits,  self._ct_class_logits, self._mr_class_logits,\
                                                self._ct_mask_logits, self._mr_mask_logits, self.cost_kwargs) # get cost

        self.confusion_matrix = tf.confusion_matrix( tf.reshape(self.compact_y,[-1]), tf.reshape(self.compact_pred, [-1]), num_classes = self.n_class )

    def create_zip_network(self, main_bn, main_trainable, adapt_bn, adapt_trainable, num_cls, feature_base = 16,  input_channel = 3, keep_prob = 0.75):

        # MR path starts from here
        with tf.variable_scope('group_1') as scope:
            w1_1 = weight_variable(shape = [3, 3, input_channel, feature_base], trainable = main_trainable)
            conv1_1 = conv2d(self.mr, w1_1, keep_prob )
            wr1_1 = weight_variable(shape = [ 3, 3, feature_base,feature_base], trainable = main_trainable)
            wr1_2 = weight_variable(shape = [3, 3, feature_base, feature_base], trainable = main_trainable)
            block1_1 = residual_block(conv1_1, wr1_1, wr1_2, keep_prob , is_train = main_bn, leak = True, bn_trainable = main_trainable , scope = 'pred_1_1'   ) # here the scope is for bn
            out1 = max_pool2d(block1_1, n = 2)
            self.mr_front_weights.append(w1_1)
            self.mr_front_weights.append(wr1_1)
            self.mr_front_weights.append(wr1_2)

        with tf.variable_scope('group_2') as scope:
            wr2_1 = weight_variable(shape = [3, 3, feature_base, feature_base * 2], trainable = main_trainable)
            wr2_2 = weight_variable(shape = [3, 3, feature_base * 2, feature_base * 2], trainable = main_trainable)
            block2_1 = residual_block(out1, wr2_1, wr2_2, inc_dim = True,keep_prob = keep_prob, leak = True, is_train = main_bn, bn_trainable = main_trainable, scope = 'pred_2_1'  )
            out2 = max_pool2d(block2_1, n = 2)
            self.mr_front_weights.append(wr2_1)
            self.mr_front_weights.append(wr2_2)

        with tf.variable_scope('group_3') as scope:
            wr3_1 = weight_variable( shape = [3, 3, feature_base * 2, feature_base * 4], trainable = main_trainable  )
            wr3_2 = weight_variable( shape = [3, 3, feature_base * 4, feature_base * 4], trainable = main_trainable  )
            block3_1 = residual_block( out2, wr3_1, wr3_2, keep_prob, inc_dim = True, is_train = main_bn, leak = True, bn_trainable = main_trainable , scope = 'pred_3_1'       )
            wr3_3 = weight_variable( shape = [3, 3, feature_base * 4, feature_base * 4], trainable = main_trainable  )
            wr3_4 = weight_variable( shape = [3, 3, feature_base * 4, feature_base * 4], trainable = main_trainable  )
            block3_2 = residual_block( block3_1, wr3_3, wr3_4,keep_prob = keep_prob, is_train = main_bn, leak = True, bn_trainable = main_trainable , scope = 'pred_3_2'      )
            out3 = max_pool2d(block3_2, n = 2)
            self.mr_front_weights.append(wr3_1)
            self.mr_front_weights.append(wr3_2)
            self.mr_front_weights.append(wr3_3)
            self.mr_front_weights.append(wr3_4)

        with tf.variable_scope('group_4') as scope:
            wr4_1 = weight_variable( shape = [3, 3, feature_base * 4, feature_base * 8], trainable = main_trainable  )
            wr4_2 = weight_variable( shape = [3, 3, feature_base * 8, feature_base * 8], trainable = main_trainable   )
            block4_1 = residual_block( out3, wr4_1, wr4_2, keep_prob,  inc_dim = True, is_train = main_bn, leak = True, bn_trainable = main_trainable , scope = 'pred_4_1'   )
            wr4_3 = weight_variable( shape = [3, 3, feature_base * 8, feature_base * 8], trainable = main_trainable  )
            wr4_4 = weight_variable( shape = [3, 3, feature_base * 8, feature_base * 8], trainable = main_trainable  )
            block4_2 = residual_block( block4_1, wr4_3, wr4_4, keep_prob, is_train = main_bn, leak = True, bn_trainable = main_trainable , scope = 'pred_4_2'    )
            self.mr_front_weights.append(wr4_1)
            self.mr_front_weights.append(wr4_2)
            self.mr_front_weights.append(wr4_3)
            self.mr_front_weights.append(wr4_4)

        with tf.variable_scope('group_5') as scope:
            wr5_1 = sharable_weight_variable( shape = [3, 3, feature_base * 8, feature_base * 16], trainable = main_trainable, name = "Variable"  )
            wr5_2 = sharable_weight_variable( shape = [3, 3, feature_base * 16, feature_base * 16], trainable = main_trainable , name = "Variable_1"  )
            block5_1 = residual_block( block4_2, wr5_1, wr5_2, keep_prob = keep_prob, inc_dim = True, leak = True,  is_train = main_bn, bn_trainable = main_trainable, scope = 'pred_5_1'     )
            wr5_3 = sharable_weight_variable( shape = [3, 3, feature_base * 16, feature_base * 16], trainable = main_trainable , name = "Variable_2"  )
            wr5_4 = sharable_weight_variable( shape = [3, 3, feature_base * 16, feature_base * 16], trainable = main_trainable , name = "Variable_3"  )
            block5_2 = residual_block( block5_1, wr5_3, wr5_4, keep_prob = keep_prob, is_train = main_bn, leak = True, bn_trainable = main_trainable , scope = 'pred_5_2'  )
            self.mr_front_weights.append( wr5_1  )
            self.mr_front_weights.append( wr5_2  )
            self.mr_front_weights.append( wr5_3  )
            self.mr_front_weights.append( wr5_4  )

        with tf.variable_scope('group_6') as scope:
            wr6_1 = sharable_weight_variable( shape = [3, 3, feature_base * 16, feature_base * 16], trainable = main_trainable , name = "Variable"  )
            wr6_2 = sharable_weight_variable( shape = [3, 3, feature_base * 16, feature_base * 16], trainable = main_trainable , name = "Variable_1"  )
            block6_1 = residual_block( block5_2, wr6_1, wr6_2, keep_prob = keep_prob,  is_train = main_bn, leak = True, bn_trainable = main_trainable , scope = 'pred_6_1'       )
            wr6_3 = sharable_weight_variable( shape = [3, 3, feature_base * 16, feature_base * 16], trainable = main_trainable , name = "Variable_2"  )
            wr6_4 = sharable_weight_variable( shape = [3, 3, feature_base * 16, feature_base * 16], trainable = main_trainable,  name = "Variable_3"   )
            block6_2 = residual_block( block6_1, wr6_3, wr6_4, keep_prob = keep_prob, is_train = main_bn, leak = True, bn_trainable = main_trainable , scope = 'pred_6_2'       )
            self.mr_front_weights.append( wr6_1  )
            self.mr_front_weights.append( wr6_2  )
            self.mr_front_weights.append( wr6_3  )
            self.mr_front_weights.append( wr6_4  )

        # DAM for CT path starts from here
        with tf.variable_scope('adapt_1') as scope:
            w1_1a = sharable_weight_variable(shape = [3, 3, input_channel, feature_base ], trainable = adapt_trainable, name = "Variable")
            conv1_1a = conv2d(self.ct, w1_1a, keep_prob )
            wr1_1a = sharable_weight_variable(shape = [ 3, 3, feature_base ,feature_base ], trainable = adapt_trainable, name = "Variable_1")
            wr1_2a = sharable_weight_variable(shape = [3, 3, feature_base , feature_base ], trainable = adapt_trainable, name = "Variable_2")
            block1_1a = residual_block(conv1_1a, wr1_1a, wr1_2a, keep_prob , is_train = adapt_bn, leak = True, bn_trainable = adapt_trainable, scope = 'adapt_1'   )
            out1a = max_pool2d(block1_1a, n = 2)
            self.ct_front_weights.append(w1_1a)
            self.ct_front_weights.append(wr1_1a)
            self.ct_front_weights.append(wr1_2a)

        with tf.variable_scope('adapt_2') as scope:
            wr2_1a = sharable_weight_variable(shape = [3, 3, feature_base , feature_base * 2], trainable = adapt_trainable, name = "Variable")
            wr2_2a = sharable_weight_variable(shape = [3, 3, feature_base * 2, feature_base * 2], trainable = adapt_trainable, name = "Variable_1")
            block2_1a = residual_block(out1a, wr2_1a, wr2_2a, inc_dim = True,keep_prob = keep_prob, leak = True, is_train = adapt_bn, bn_trainable = adapt_trainable, scope = 'adapt_2'   )
            out2a = max_pool2d(block2_1a, n = 2)
            self.ct_front_weights.append(wr2_1a)
            self.ct_front_weights.append(wr2_2a)

        with tf.variable_scope('adapt_3') as scope:
            wr3_1a = sharable_weight_variable( shape = [3, 3, feature_base * 2, feature_base * 4], trainable = adapt_trainable, name = "Variable"  )
            wr3_2a = sharable_weight_variable( shape = [3, 3, feature_base * 4, feature_base * 4], trainable = adapt_trainable, name = "Variable_1"  )
            block3_1a = residual_block( out2a, wr3_1a, wr3_2a, keep_prob, inc_dim = True, leak = True, is_train = adapt_bn, bn_trainable = adapt_trainable , scope = 'adapt_3_1'   )
            wr3_3a = sharable_weight_variable( shape = [3, 3, feature_base * 4, feature_base * 4], trainable = adapt_trainable, name = "Variable_2"  )
            wr3_4a = sharable_weight_variable( shape = [3, 3, feature_base * 4, feature_base * 4], trainable = adapt_trainable , name = "Variable_3"  )
            block3_2a = residual_block( block3_1a, wr3_3a, wr3_4a,keep_prob = keep_prob, leak = True, is_train = adapt_bn, bn_trainable = adapt_trainable, scope = 'adapt_3_2'    )

            out3a = max_pool2d(block3_2a, n = 2)
            self.ct_front_weights.append(wr3_1a)
            self.ct_front_weights.append(wr3_2a)
            self.ct_front_weights.append(wr3_3a)
            self.ct_front_weights.append(wr3_4a)

        with tf.variable_scope('adapt_4') as scope:
            wr4_1a = sharable_weight_variable( shape = [3, 3, feature_base * 4, feature_base * 8], trainable = adapt_trainable, name  = "Variable"  )
            wr4_2a = sharable_weight_variable( shape = [3, 3, feature_base * 8, feature_base * 8], trainable = adapt_trainable , name  = "Variable_1"   )
            block4_1a = residual_block( out3a, wr4_1a, wr4_2a, keep_prob, inc_dim = True, leak = True, is_train = adapt_bn, bn_trainable = adapt_trainable, scope = 'adapt_4_1'     )

            wr4_3a = sharable_weight_variable( shape = [3, 3, feature_base * 8, feature_base * 8], trainable = adapt_trainable , name  = "Variable_2"  )
            wr4_4a = sharable_weight_variable( shape = [3, 3, feature_base * 8, feature_base * 8], trainable = adapt_trainable  , name  = "Variable_3" )
            block4_2a = residual_block( block4_1a, wr4_3a, wr4_4a, keep_prob, is_train = adapt_bn, leak = True, bn_trainable = adapt_trainable, scope = 'adapt_4_2'      )
            self.ct_front_weights.append(wr4_1a)
            self.ct_front_weights.append(wr4_2a)
            self.ct_front_weights.append(wr4_3a)
            self.ct_front_weights.append(wr4_4a)

        with tf.variable_scope('adapt_5') as scope:
            wr5_1a = sharable_weight_variable( shape = [3, 3, feature_base * 8, feature_base * 16], trainable = adapt_trainable, name = "Variable"  )
            wr5_2a = sharable_weight_variable( shape = [3, 3, feature_base * 16, feature_base * 16], trainable = adapt_trainable , name = "Variable_1"  )
            block5_1a = residual_block( block4_2a, wr5_1a, wr5_2a, keep_prob = keep_prob, leak = True, inc_dim = True,  is_train = adapt_bn, bn_trainable = adapt_trainable, scope = 'adapt_5_1'     )

            wr5_3a = sharable_weight_variable( shape = [3, 3, feature_base * 16, feature_base * 16], trainable = adapt_trainable , name = "Variable_2"  )
            wr5_4a = sharable_weight_variable( shape = [3, 3, feature_base * 16, feature_base * 16], trainable = adapt_trainable , name = "Variable_3"  )
            block5_2a = residual_block( block5_1a, wr5_3a, wr5_4a, keep_prob = keep_prob, leak = True, is_train = adapt_bn, bn_trainable = adapt_trainable , scope = 'adapt_5_2'  )
            self.ct_front_weights.append( wr5_1a  )
            self.ct_front_weights.append( wr5_2a  )
            self.ct_front_weights.append( wr5_3a  )
            self.ct_front_weights.append( wr5_4a  )

        with tf.variable_scope('adapt_6') as scope:
            wr6_1a = sharable_weight_variable( shape = [3, 3, feature_base * 16, feature_base * 16], trainable = adapt_trainable , name = "Variable"  )
            wr6_2a = sharable_weight_variable( shape = [3, 3, feature_base * 16, feature_base * 16], trainable = adapt_trainable , name = "Variable_1"  )
            block6_1a = residual_block( block5_2a, wr6_1a, wr6_2a, keep_prob = keep_prob, leak = True,  is_train = adapt_bn, bn_trainable = adapt_trainable , scope = 'adapt_6_1'       )

            wr6_3a = sharable_weight_variable( shape = [3, 3, feature_base * 16, feature_base * 16], trainable = adapt_trainable , name = "Variable_2"  )
            wr6_4a = sharable_weight_variable( shape = [3, 3, feature_base * 16, feature_base * 16], trainable = adapt_trainable,  name = "Variable_3"   )
            block6_2a = residual_block( block6_1a, wr6_3a, wr6_4a, keep_prob = keep_prob, leak = True, is_train = adapt_bn, bn_trainable = adapt_trainable , scope = 'adapt_6_2'       )
            self.ct_front_weights.append( wr6_1a  )
            self.ct_front_weights.append( wr6_2a  )
            self.ct_front_weights.append( wr6_3a  )
            self.ct_front_weights.append( wr6_4a  )

        return block4_2, block4_2a, block6_2, block6_2a

    def create_second_half(self, input_feature, joint_bn, joint_trainable, num_cls, feature_base = 16,  input_channel = 3, keep_prob = 0.75):

        with tf.variable_scope('group_7', reuse = tf.AUTO_REUSE) as scope:
            wr7_1 = sharable_weight_variable( shape = [3, 3, feature_base * 16, feature_base * 32], trainable = joint_trainable  , name = "Variable" )
            wr7_2 = sharable_weight_variable( shape = [3, 3, feature_base * 32, feature_base * 32], trainable = joint_trainable  , name = "Variable_1" )
            block7_1 = residual_block( input_feature, wr7_1, wr7_2, keep_prob = keep_prob, leak = True, inc_dim = True,  is_train = joint_bn, bn_trainable = joint_trainable , scope = 'pred_7_1'     )
            wr7_3 = sharable_weight_variable( shape = [3, 3, feature_base * 32, feature_base * 32], trainable = joint_trainable , name = "Variable_2"  )
            wr7_4 = sharable_weight_variable( shape = [3, 3, feature_base * 32, feature_base * 32], trainable = joint_trainable  , name = "Variable_3" )
            block7_2 = residual_block( block7_1, wr7_3, wr7_4, keep_prob = keep_prob, leak = True,  is_train = joint_bn, bn_trainable = joint_trainable , scope = 'pred_7_2'      )
            self.mr_front_weights.append( wr7_1  )
            self.mr_front_weights.append( wr7_2  )
            self.mr_front_weights.append( wr7_3  )
            self.mr_front_weights.append( wr7_4  )

        with tf.variable_scope('group_8', reuse = tf.AUTO_REUSE) as scope:
            wr8_1 = sharable_weight_variable( shape = [3, 3, feature_base * 32, feature_base * 32], trainable = joint_trainable , name = "Variable"  )
            wr8_2 = sharable_weight_variable( shape = [3, 3, feature_base * 32, feature_base * 32], trainable = joint_trainable  , name = "Variable_1" )
            block8_1 = DR_block( block7_2, wr8_1, wr8_2, keep_prob = keep_prob, leak = True, is_train = joint_bn, rate = 2, bn_trainable = joint_trainable , scope = 'pred_8_1'       )
            wr8_3 = sharable_weight_variable( shape = [3, 3, feature_base * 32, feature_base * 32], trainable = joint_trainable  , name = "Variable_2" )
            wr8_4 = sharable_weight_variable( shape = [3, 3, feature_base * 32, feature_base * 32], trainable = joint_trainable  , name = "Variable_3" )
            block8_2 = DR_block( block8_1, wr8_3, wr8_4, keep_prob = keep_prob, leak = True,  is_train = joint_bn, rate = 2, bn_trainable = joint_trainable , scope = 'pred_8_2'   )
            self.mr_front_weights.append( wr8_1  )
            self.mr_front_weights.append( wr8_2  )
            self.mr_front_weights.append( wr8_3  )
            self.mr_front_weights.append( wr8_4  )

        with tf.variable_scope('group_9', reuse = tf.AUTO_REUSE) as scope:
            w9_1 = sharable_weight_variable( shape = [3, 3, feature_base * 32, feature_base * 32], trainable = joint_trainable  , name = "Variable" )
            conv9_1 = conv_bn_relu2d( block8_2, w9_1, keep_prob, leak = True, is_train = joint_bn, bn_trainable = joint_trainable , scope = 'pred_9_1'   )
            w9_2 = sharable_weight_variable( shape = [3, 3, feature_base * 32, feature_base * 32], trainable = joint_trainable  , name = "Variable_1" )
            conv9_2 = conv_bn_relu2d( conv9_1, w9_2, keep_prob, leak = True, is_train = joint_bn, bn_trainable = joint_trainable , scope = 'pred_9_2'    )
            self.mr_front_weights.append( w9_1  )
            self.mr_front_weights.append( w9_2  )

        with tf.variable_scope('group_10', reuse = tf.AUTO_REUSE) as scope:
            local_size = 8 * 8
            w10_1 = sharable_weight_variable( shape = [3, 3, feature_base * 32, local_size * num_cls * 8], trainable = joint_trainable , name = "Variable" )
            conv10_1 = conv2d( conv9_2, w10_1, keep_prob_ = keep_prob, padding = 'SYMMETRIC')
            self.mr_front_weights.append(w10_1)
            flat_conv10_1 = PS(conv10_1, r = 8, n_channel = num_cls * 8, batch_size = self.batch_size) # phase shift

        with tf.variable_scope('output', reuse = tf.AUTO_REUSE) as scope:
            w11_1 = sharable_weight_variable( shape = [5, 5, num_cls * 8, num_cls], trainable = joint_trainable  , name = "Variable" )
            logits = conv2d( flat_conv10_1, w11_1, keep_prob_ = 1., padding = 'SYMMETRIC'  )

        return conv9_2, block8_2, block7_2, logits

    def create_classifier(self, input_conv4, input_conv6, input_b7, input_conv9, seg_logits, feature_base = 16, keep_prob = 0.75, cls_bn = True, cls_trainable = True):
        """
        domain discriminator for MRI features and CT features
        """
        with tf.variable_scope('cls_0') as scope:
            flat_input_conv4 = PS(input_conv4, r=8, n_channel=2, batch_size=self.batch_size)  # 2
            flat_input_conv4 = tf.tile(flat_input_conv4, [1, 1, 1, 3]) # 6 in total
            flat_input_conv6 = PS(input_conv6, r=8, n_channel=4, batch_size=self.batch_size)  # 10 in total
            flat_input_b7 = PS(input_b7, r=8, n_channel=8, batch_size=self.batch_size)  # 18 in total
            flat_input_conv9 = PS(input_conv9, r = 8, n_channel = 8, batch_size = self.batch_size) # 26 in total

            input_comp = simple_concat2d(flat_input_conv4, flat_input_conv6) # 10
            input_comp = simple_concat2d(input_comp, flat_input_b7) # 18
            input_comp = simple_concat2d(input_comp, flat_input_conv9) # 26
            input_comp = simple_concat2d(input_comp, seg_logits) # 31 in total
            input_comp = simple_concat2d(input_comp, tf.expand_dims(tf.cast(tf.argmax(seg_logits, 3), tf.float32), 3))  # 1

        with tf.variable_scope('cls_1') as scope:
            wr1_1c = sharable_weight_variable( shape = [3, 3, feature_base * 2, feature_base * 4], trainable = cls_trainable  , name = "Variable" )
            wr1_2c = sharable_weight_variable( shape = [3, 3, feature_base * 4, feature_base * 4], trainable = cls_trainable  , name = "Variable_1" )
            block1_1c = residual_block( input_comp, wr1_1c, wr1_2c, keep_prob = keep_prob, inc_dim = True, is_train = cls_bn, bn_trainable = cls_trainable, scope = 'cls_1'   , leak = True   )
            wr1_3d = sharable_weight_variable( shape = [3,3, feature_base * 4, feature_base * 4], trainable = cls_trainable, name = "Variable_2"  )
            out1c = conv_bn_relu2d( block1_1c, wr1_3d, keep_prob, strides = [1,2,2,1], is_train = cls_bn, bn_trainable = cls_trainable, scope = 'cls_1_3', leak = True  )
            self.cls_weights.append( wr1_1c  )
            self.cls_weights.append( wr1_2c  )
            self.cls_weights.append( wr1_3d  )

        with tf.variable_scope('cls_2') as scope:
            wr2_1c = sharable_weight_variable( shape = [3, 3, feature_base * 4, feature_base *8], trainable = cls_trainable  , name = "Variable" )
            wr2_2c = sharable_weight_variable( shape = [3, 3, feature_base * 8, feature_base *8], trainable = cls_trainable  , name = "Variable_1" )
            block2_1c = residual_block( out1c, wr2_1c, wr2_2c, keep_prob = keep_prob, inc_dim = True, is_train = cls_bn, bn_trainable = cls_trainable, scope = 'cls_2'   , leak = True   )
            wr2_3d = sharable_weight_variable( shape = [5,5, feature_base * 8, feature_base * 8], trainable = cls_trainable, name = "Variable_2"  )
            out2c = conv_bn_relu2d( block2_1c, wr2_3d, keep_prob, strides = [1,2,2,1], is_train = cls_bn, bn_trainable = cls_trainable, scope = 'cls_2_3', leak = True  )
            self.cls_weights.append( wr2_1c  )
            self.cls_weights.append( wr2_2c  )
            self.cls_weights.append( wr2_3d  )
            self.debug_out2c = out2c
            self.debug_wr2_2c = wr2_2c

        with tf.variable_scope('cls_3') as scope:
            wr3_1c = sharable_weight_variable( shape = [3, 3, feature_base * 8, feature_base *16], trainable = cls_trainable  , name = "Variable" )
            wr3_2c = sharable_weight_variable( shape = [3, 3, feature_base * 16, feature_base *16], trainable = cls_trainable  , name = "Variable_1" )
            block3_1c = residual_block( out2c, wr3_1c, wr3_2c,  keep_prob = keep_prob, inc_dim = True, is_train = cls_bn, bn_trainable = cls_trainable, scope = 'cls_3'   , leak = True   )
            wr3_3d = sharable_weight_variable( shape = [3,3, feature_base * 16, feature_base * 16], trainable = cls_trainable, name = "Variable_2"  )
            out3c = conv_bn_relu2d( block3_1c, wr3_3d, keep_prob, strides = [1,2,2,1], is_train = cls_bn, bn_trainable = cls_trainable, scope = 'cls_3_3', leak = True  )
            self.cls_weights.append( wr3_1c  )
            self.cls_weights.append( wr3_2c  )
            self.cls_weights.append( wr3_3d  )

        with tf.variable_scope('cls_4') as scope:
            wr4_1c = sharable_weight_variable( shape = [3, 3, feature_base * 16, feature_base *32], trainable = cls_trainable  , name = "Variable" )
            wr4_2c = sharable_weight_variable( shape = [3, 3, feature_base * 32, feature_base *32], trainable = cls_trainable  , name = "Variable_1" )
            block4_1c = residual_block( out3c, wr4_1c, wr4_2c,  keep_prob = keep_prob, inc_dim = True, is_train = cls_bn, bn_trainable = cls_trainable, scope = 'cls_4'   , leak = True   )
            wr4_3d = sharable_weight_variable( shape = [3,3, feature_base * 32, feature_base * 32], trainable = cls_trainable, name = "Variable_2"  )
            out4c = conv_bn_relu2d( block4_1c, wr4_3d, keep_prob, strides = [1,2,2,1], is_train = cls_bn, bn_trainable = cls_trainable, scope = 'cls_4_3', leak = True  )
            self.cls_weights.append( wr4_1c  )
            self.cls_weights.append( wr4_2c  )
            self.cls_weights.append( wr4_3d  )

        with tf.variable_scope('cls_5') as scope:
            wr5_1c = sharable_weight_variable( shape = [3, 3, feature_base * 32, feature_base *32], trainable = cls_trainable  , name = "Variable" )
            wr5_2c = sharable_weight_variable( shape = [3, 3, feature_base * 32, feature_base *32], trainable = cls_trainable  , name = "Variable_1" )
            block5_1c = residual_block( out4c, wr5_1c, wr5_2c,  keep_prob = keep_prob, is_train = cls_bn, bn_trainable = cls_trainable, scope = 'cls_5'   , leak = True   )
            wr5_3d = sharable_weight_variable( shape = [5,5, feature_base * 32, feature_base * 32], trainable = cls_trainable, name = "Variable_2"  )
            out5c = conv_bn_relu2d( block5_1c, wr5_3d, keep_prob, strides = [1,4,4,1], is_train = cls_bn, bn_trainable = cls_trainable, scope = 'cls_5_3', leak = True  )
            self.cls_weights.append( wr5_1c  )
            self.cls_weights.append( wr5_2c  )
            self.cls_weights.append( wr5_3d  )

        with tf.variable_scope('cls_6') as scope:
            wr6_1c = sharable_weight_variable( shape = [3, 3, feature_base * 32, feature_base *32], trainable = cls_trainable  , name = "Variable" )
            conv_6c = conv_bn_relu2d(out5c, wr6_1c, strides = [1,2,2,1], keep_prob = keep_prob, padding = "SYMMETRIC", scope = 'cls_6', is_train = cls_bn, bn_trainable = cls_trainable, leak = True)
            self.cls_weights.append( wr6_1c  )

        with tf.variable_scope('cls_out') as scope:
            wc_out = sharable_weight_variable( shape = [ feature_base* 32 * 4,1 ], trainable = cls_trainable , name = "Variable"  )
            out6c_flat = tf.reshape(conv_6c, [-1, feature_base * 32 * 4])
            cls_logits = tf.matmul(out6c_flat, wc_out)
            self.cls_weights.append(wc_out)

        return cls_logits

    def create_mask_critic(self, input_mask, feature_base = 16, keep_prob = 0.75, num_cls = 5, m_cls_bn = True, m_cls_trainable = True):
        """
        domain discriminator for MRI and CT segmentation maskS

        """
        with tf.variable_scope('mask_cls_1') as scope:
            wr1_1m = sharable_weight_variable( shape = [3, 3, num_cls, feature_base], trainable = m_cls_trainable  , name = "Variable" )
            out1m = conv_bn_relu2d( input_mask, wr1_1m, keep_prob, strides = [1,2,2,1], is_train = m_cls_bn, bn_trainable = m_cls_trainable, scope = 'mask_cls_1', leak = True  ) # use strided conv instead of maxpool to
            self.m_cls_weights.append( wr1_1m )

        with tf.variable_scope('mask_cls_2') as scope:
            wr2_1m = sharable_weight_variable( shape = [3, 3, feature_base, feature_base ], trainable = m_cls_trainable  , name = "Variable" )
            wr2_2m = sharable_weight_variable( shape = [3, 3, feature_base, feature_base ], trainable = m_cls_trainable  , name = "Variable_1" )
            block2_1m = residual_block( out1m, wr2_1m, wr2_2m, keep_prob = keep_prob, inc_dim = False, is_train = m_cls_bn, bn_trainable = m_cls_trainable, scope = 'm_cls_2'   , leak = True   )
            wr2_3d = sharable_weight_variable( shape = [5,5, feature_base, feature_base * 2], trainable = m_cls_trainable, name = "Variable_2"  )
            out2m = conv_bn_relu2d( block2_1m, wr2_3d, keep_prob, strides = [1,4,4,1], is_train = m_cls_bn, bn_trainable = m_cls_trainable, scope = 'm_cls_2_3', leak = True  )
            self.m_cls_weights.append( wr2_1m  )
            self.m_cls_weights.append( wr2_2m  )
            self.m_cls_weights.append( wr2_3d  )

        with tf.variable_scope('mask_cls_3') as scope:
            wr3_1m = sharable_weight_variable( shape = [3, 3, feature_base * 2, feature_base * 4], trainable = m_cls_trainable  , name = "Variable" )
            wr3_2m = sharable_weight_variable( shape = [3, 3, feature_base * 4, feature_base * 4 ], trainable = m_cls_trainable  , name = "Variable_1" )
            block3_1m = residual_block( out2m, wr3_1m, wr3_2m, keep_prob = keep_prob, inc_dim = True, is_train = m_cls_bn, bn_trainable = m_cls_trainable, scope = 'm_cls_3'   , leak = True   )
            wr3_3d = sharable_weight_variable( shape = [5,5, feature_base * 4, feature_base * 8], trainable = m_cls_trainable, name = "Variable_2"  )
            out3m = conv_bn_relu2d( block3_1m, wr3_3d, keep_prob, strides = [1,4,4,1], is_train = m_cls_bn, bn_trainable = m_cls_trainable, scope = 'm_cls_3_3', leak = True  )
            self.m_cls_weights.append( wr3_1m  )
            self.m_cls_weights.append( wr3_2m  )
            self.m_cls_weights.append( wr3_3d  )

        with tf.variable_scope('mask_cls_4') as scope:
            wr4_1m = sharable_weight_variable( shape = [5, 5, feature_base * 8, feature_base * 16], trainable = m_cls_trainable  , name = "Variable" )
            conv_4m = conv_bn_relu2d(out3m, wr4_1m, strides = [1,4,4,1], keep_prob = keep_prob, padding = "SYMMETRIC", scope = 'm_cls_4', is_train = m_cls_bn, bn_trainable = m_cls_trainable, leak = True)
            self.m_cls_weights.append( wr4_1m  )

        with tf.variable_scope('m_cls_out') as scope:
            wm_out = sharable_weight_variable( shape = [ feature_base* 16 * 4,1 ], trainable = m_cls_trainable , name = "Variable"  )
            out5m_flat = tf.reshape(conv_4m, [-1, feature_base * 16 * 4])
            m_cls_logits = tf.matmul(out5m_flat, wm_out)
            self.m_cls_weights.append(wm_out)

        return m_cls_logits

    def _get_cost(self, ct_logits, mr_logits,  ct_cls_logits, mr_cls_logits, ct_mask_logits, mr_mask_logits, cost_kwargs):

        miu_dis = cost_kwargs["miu_dis"] # coefficient for discriminator loss
        miu_gen = cost_kwargs["miu_gen"] # used to be 0.5 0.5 1
        lambda_mask_loss = cost_kwargs.pop("lambda_mask_loss", 1.0) # weighting of mask critic score

        self.miu_dis = tf.Variable(miu_dis, name = "miu_dis") # coefficient for discrminator
        self.miu_gen = tf.Variable(miu_gen, name = "miu_gen")

        # loss for main critic and mask critic
        dis_loss = -1 * self.miu_dis * tf.reduce_mean( mr_cls_logits - ct_cls_logits  ) # loss functions of WGAN
        gen_loss = -1 * self.miu_gen * tf.reduce_mean( ct_cls_logits  )

        m_dis_loss = -1 * self.miu_dis * tf.reduce_mean( mr_mask_logits - ct_mask_logits  )
        m_gen_loss = -1 * self.miu_gen * tf.reduce_mean( ct_mask_logits  )

        ############  L2 norm regularizer  ######################
        reg_coeff = cost_kwargs.pop("regularizer", 1.0e-4) # regularizer coefficients for non-GAN parts
        mr_front_reg = sum([tf.nn.l2_loss(variable) for variable in self.mr_front_weights]) # regulizer for MRI varibles, fixed for the unsupervised setting
        joint_reg = sum([tf.nn.l2_loss(variable) for variable in self.joint_weights]) # regularizer for joint part, fixed for the unsupervised setting
        fixed_coeff_reg = reg_coeff * (mr_front_reg + joint_reg) # for training observation to confirm the source segmenter is not updated

        gan_reg_coeff = cost_kwargs.pop("gan_regularizer", 1.0e-4)  # regularizer coefficients for GAN parts, note, seems that it works well when it is larger
        gen_reg = gan_reg_coeff * self.miu_gen * sum([tf.nn.l2_loss(variable) for variable in self.ct_front_weights]) # regulizers for WGAN
        dis_reg = gan_reg_coeff * self.miu_dis * sum([tf.nn.l2_loss(variable) for variable in self.cls_weights])
        m_dis_reg = gan_reg_coeff * self.miu_dis * sum([tf.nn.l2_loss(variable) for variable in self.m_cls_weights])

        dis_loss += lambda_mask_loss * m_dis_loss
        gen_loss += lambda_mask_loss * m_gen_loss
        dis_reg += lambda_mask_loss * m_dis_reg

        return dis_loss, gen_loss, fixed_coeff_reg, dis_reg, gen_reg

    def _get_variables_by_scope(self):
        """
        Group different variables (MR, CT, GAN, etc)to different groups
        """
        logging.info("extent of joint part and segmenter need to be manually set, including variables and bns")

        self.adapt_vars = [] # variables for adaptation (CT)
        self.cls_vars = [] # variables for domain-classifier (i.e. discriminator) for WGAN
        self.seg_vars = [] # variables for segmentation, fixed higher layers in source segmenter
        self.mri_seg_vars = [] # variables for segmentation, MRI early players, fixed as well

        var_list = tf.contrib.framework.get_variables()
        for var in var_list:
            if "cls" in var.name:
                self.cls_vars.append(var)
            elif "adapt" in var.name:
                self.adapt_vars.append(var)
            elif "output" in var.name:
                self.seg_vars.append(var)
                self.mri_seg_vars.append(var)
            elif "group" in var.name:
                _group_name = var.name.split("/")[0]
                _group_no = float(_group_name.split("_")[-1] )
                self.mri_seg_vars.append(var)

    def restore(self, sess, model_path, no_gan=False, clear_rms=False):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        :param no_gan: only restore mr variables
        :param clear_rms: does not restore RMSprop internal variables, please set is true
        """
        saver = tf.train.Saver(tf.contrib.framework.get_variables() + tf.get_collection_ref("internal_batchnorm_variables") )
        logging.info("Model restored from file: %s" % model_path)
        if no_gan is True:
            logging.info("I only load the main variables! without batchnorm!!!")
            variables = tf.global_variables()
            reader = tf.pywrap_tensorflow.NewCheckpointReader(model_path)
            var_keep_dic = reader.get_variable_to_shape_map()
            variables_to_restore = []
            for v in variables:
                if v.name.split(':')[0] in var_keep_dic:
                    if ("adapt" in v.name) or ("cls" in v.name) or("Adam" in v.name):
                        continue
                    if ("group" in v.name) or ("output" in v.name):
                        logging.info("restoring "+str(v.name))
                        variables_to_restore.append(v)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, model_path)

            logging.info("Model restored from file: %s, the pre-trained MRI model (without bn params)" % model_path)

            return 0

        if clear_rms is True:
            logging.info("Calculating RMS parameters from beginning")
            variables = tf.global_variables()
            reader = tf.pywrap_tensorflow.NewCheckpointReader(model_path)
            var_keep_dic = reader.get_variable_to_shape_map()
            variables_to_restore = []
            for v in variables:
                if v.name.split(':')[0] in var_keep_dic:
                    if ("RMS" in v.name) :
                        continue
                    else:
                        logging.info("restoring "+str(v.name))
                        variables_to_restore.append(v)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, model_path)

            logging.info("Model restored from file: %s and RMS variables are ignored" % model_path)
            return 0

        try: # else, just restore as much as possible
            saver.restore(sess, model_path)
            logging.info("Model restored from file: %s" % model_path)
        except:
            variables = tf.global_variables()
            reader = tf.pywrap_tensorflow.NewCheckpointReader(model_path)
            var_keep_dic = reader.get_variable_to_shape_map()
            variables_to_restore = []
            for v in variables:
                if v.name.split(':')[0] in var_keep_dic:
                    skip_flg = False
                    for kwd in self.network_config["restore_skip_kwd"]: # if it is manully specified to be skipped, don't restore it
                        if kwd in v.name:
                            skip_flg = True
                            break
                    if skip_flg is False:
                        variables_to_restore.append(v)
                        logging.info("cannot fully restore the model, restoring "+str(v.name))
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, model_path)

            logging.info("Model restored from file: %s with relaxation" % model_path)

class Trainer(object):
    """
    Train a unet instance
    """
    def __init__(self, net, mr_train_list, mr_val_list, ct_train_list, ct_val_list,\
                 adapt_var_list, mr_var_list, old_bn_list, new_bn_list,\
                 test_label_list = None, test_nii_list = None,\
                 num_cls=None, batch_size = 6,\
                 opt_kwargs={}, train_config = {}):

        self.net = net
        self.batch_size = batch_size
        self.num_cls = num_cls # including background
        self.opt_kwargs = opt_kwargs
        self.ct_train_list = ct_train_list # a list of training files
        self.ct_val_list = ct_val_list # a list of validation files
        self.mr_train_list = mr_train_list # a list of training files for MRI
        self.mr_val_list = mr_val_list
        self.test_label_list = test_label_list # test files (npz format)
        self.test_nii_list = test_nii_list # test files (npz format)
        self.adapt_var_list = adapt_var_list # a list of variables in CT path
        self.mr_var_list = mr_var_list # a list of variables in MRI path in correspondance with variables in adapt_var_list, this is used for manually initialize variables in CT path with those of MRI path
        self.old_bn_list = old_bn_list # a list of batch_norm internal variables in baseline model
        self.new_bn_list = new_bn_list # a list of batch_norm internal variables for the MRI path in current model
        self.ct_train_queue = tf.train.string_input_producer(ct_train_list, num_epochs = None, shuffle = True) # tensorflow input queue for CT supervision (disabled), CT and MRI
        self.ct_val_queue = tf.train.string_input_producer(ct_val_list, num_epochs = None, shuffle = True)
        self.mr_train_queue = tf.train.string_input_producer(mr_train_list, num_epochs = None, shuffle = True)
        self.mr_val_queue = tf.train.string_input_producer(mr_val_list, num_epochs = None, shuffle = True)
        self.train_config = train_config # configuations for training
        self.lr_update_flag = train_config["lr_update"]

    def next_batch(self, input_queue, capacity = 120, num_threads = 2, min_after_dequeue = 30, label_type = 'float'):

        reader = tf.TFRecordReader()
        fid, serialized_example = reader.read(input_queue)
        parser = tf.parse_single_example(serialized_example, features = decomp_feature)
        dsize_dim0 = tf.cast(parser['dsize_dim0'], tf.int32)
        dsize_dim1 = tf.cast(parser['dsize_dim1'], tf.int32)
        dsize_dim2 = tf.cast(parser['dsize_dim2'], tf.int32)
        lsize_dim0 = tf.cast(parser['lsize_dim0'], tf.int32)
        lsize_dim1 = tf.cast(parser['lsize_dim1'], tf.int32)
        lsize_dim2 = tf.cast(parser['dsize_dim2'], tf.int32)
        data_vol = tf.decode_raw(parser['data_vol'], tf.float32)
        label_vol = tf.decode_raw(parser['label_vol'], tf.float32)

        data_vol = tf.reshape(data_vol, raw_size)
        label_vol = tf.reshape(label_vol, raw_size)
        data_vol = tf.slice(data_vol, [0,0,0],volume_size)
        label_vol = tf.slice(label_vol, [0,0,1], label_size)

        data_feed, label_feed, fid_feed = tf.train.shuffle_batch([data_vol, label_vol, fid], batch_size =self.batch_size , capacity = capacity, \
                                                            num_threads = num_threads, min_after_dequeue = min_after_dequeue)

        pair_feed = tf.concat([data_feed, label_feed], axis = 3) # concatenate them

        return pair_feed, fid_feed

    def _get_optimizer(self, training_iters, global_step):
        """
        Use RMSprop instead of Adam for training WGAN
        """
        learning_rate = self.opt_kwargs.pop("learning_rate", None) # default set to 0.0002
        self.LR_refresh = learning_rate
        self.learning_rate_node = tf.Variable(learning_rate)


        # optimizer for discriminator/ domain classifier
        dis_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_node,
                                                            **self.opt_kwargs).minimize(self.net.dis_loss + 1.0 / self.train_config['dis_sub_iter'] * self.net.dis_reg,
                                                            global_step=global_step,\
                                                            var_list = self.net.cls_vars)

        # optimizer for training generator
        gen_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_node,
                                                            **self.opt_kwargs).minimize(self.net.ct_gen_loss + 1.0 / self.train_config['gen_sub_iter'] * self.net.gen_reg,
                                                            global_step=global_step,\
                                                            var_list = self.net.adapt_vars)
        # clip operation for WGAN for Lipschitz constrain
        self.clip_op = [tf.assign(var, tf.clip_by_value(var, -0.03, 0.03)) for var in self.net.cls_vars if "Variable" in var.name]

        return dis_optimizer, gen_optimizer

    def _initialize(self, training_iters, output_path):
        """
        initialization and tensorboard setting
        """
        self.global_step = tf.Variable(0)

        scalar_summaries = [] # tensorboard summaries
        scalar_summaries.append(tf.summary.scalar('fixed_coeff_reg', self.net.fixed_coeff_reg)) # regulizer of MRI segemter weights, monitor MRI weights unchanged
        scalar_summaries.append(tf.summary.scalar('discriminator_loss', self.net.dis_loss))
        scalar_summaries.append(tf.summary.scalar('generator_loss', self.net.ct_gen_loss))

        scalar_summaries.append(tf.summary.scalar('ct_dice_eval_c1_lv_myo', self.net.ct_dice_eval_c1))
        scalar_summaries.append(tf.summary.scalar('ct_dice_eval_c2_la_blood', self.net.ct_dice_eval_c2))
        scalar_summaries.append(tf.summary.scalar('ct_dice_eval_c3_lv_blood', self.net.ct_dice_eval_c3))
        scalar_summaries.append(tf.summary.scalar('ct_dice_eval_c4_aa', self.net.ct_dice_eval_c4))

        scalar_summaries.append(tf.summary.scalar('mri_dice', self.net.mr_dice_eval)) # set to show absolute value for mr segmentation

        train_images = []
        train_images.append(tf.summary.image('ct_pred', tf.expand_dims(tf.cast(self.net.compact_pred, tf.float32), 3 )) ) # ct prediction
        train_images.append(tf.summary.image('ct_image', tf.expand_dims(tf.cast(self.net.ct[:,:,:,1], tf.float32), 3 )) )
        train_images.append(tf.summary.image('ct_gt', tf.expand_dims(tf.cast(self.net.compact_y, tf.float32), 3))) # ground truth for CT segmentation
        train_images.append(tf.summary.image('mri_validation_pred', tf.expand_dims(tf.cast(self.net.compact_mr_valid, tf.float32), 3 )) ) # mri segmentation for debugging
        train_images.append(tf.summary.image('mri_image', tf.expand_dims(tf.cast(self.net.mr[:,:,:,1], tf.float32), 3 )) )
        train_images.append(tf.summary.image('mri_gt', tf.expand_dims(tf.cast(self.net.compact_mr_y, tf.float32), 3))) # ground truth for CT segmentation

        val_images = []
        val_images.append(tf.summary.image('ct_val_pred', tf.expand_dims(tf.cast(self.net.compact_pred, tf.float32), 3))) # prediction for validation
        val_images.append(tf.summary.image('ct_image', tf.expand_dims(tf.cast(self.net.ct[:,:,:,1], tf.float32), 3)))
        val_images.append(tf.summary.image('ct_val_gt', tf.expand_dims(tf.cast(self.net.compact_y, tf.float32), 3)))

        self.net._get_variables_by_scope() # get variable groups
        self.dis_optimizer, self.gen_optimizer = self._get_optimizer(training_iters, self.global_step) # get optimizers

        scalar_summaries.append(tf.summary.scalar('learning_rate', self.learning_rate_node))

        # get summary writers
        self.scalar_summary_op = tf.summary.merge(scalar_summaries)
        self.train_image_summary_op = tf.summary.merge(train_images)
        self.val_image_summary_op = tf.summary.merge(val_images)

        # variable initializers
        init_glb = tf.global_variables_initializer()
        init_loc = tf.variables_initializer(tf.local_variables())

        return init_glb, init_loc


    def _adapt_copy_weights(self, internal = False):

        if internal is False:
            if len(self.mr_var_list) != len(self.adapt_var_list):
                raise ValueError("cannot copy weight to adaptation because of incorrect varaible lists")
            with tf.variable_scope("", reuse = True):
                for idx in range(len(self.mr_var_list)):
                    logging.info("Now initializing adaptation variable %s with mainstream variable %s"%( self.adapt_var_list[idx], self.mr_var_list[idx]   ))
                    _curr_mr_var = tf.get_default_graph().get_tensor_by_name(self.mr_var_list[idx])
                    _curr_adapt_var = tf.get_default_graph().get_tensor_by_name(self.adapt_var_list[idx])
                    upd_op = tf.assign(_curr_adapt_var,_curr_mr_var)
                    upd_op.eval()

        else:
            logging.info("automatically seeks for variable correspondance")
            all_var_list = tf.contrib.framework.get_variables()
            self.mr_var_list = []
            self.adapt_var_list = []
            for v in all_var_list:
                if ("RMS" in v.name) or ("Adam" in v.name):
                    continue
                else:
                    if "group" in v.name:
                        self.mr_var_list.append(v)
                    elif "adapt" in v.name:
                        self.adapt_var_list.append(v)
                    else:
                        continue
            if len(self.mr_var_list) != len(self.adapt_var_list):
                raise ValueError("cannot copy weight to adaptation because of incorrect varaible list")
            for _curr_adapt_var, _curr_mr_var in zip(self.adapt_var_list, self.mr_var_list):
                upd_op = tf.assign(_curr_adapt_var,_curr_mr_var)
                upd_op.eval()

        logging.info("adaptation module has been initialized! Please remember that it is a one-time operation")


    def _load_batch_norm_weights(self, output_path):
        """
        convenience function for loading weights from eariler version of baseline model for the CT/MR segmentation network
        old_bn_list: a list of bn variable names in baseline model
        new_bn_list: a list of bn Variable names in current model
        """
        if len(self.old_bn_list) != len(self.new_bn_list):
            raise ValueError("two mappings mismatch")

        checkpoint = tf.train.get_checkpoint_state(output_path)
        self.copy_bn_dict = {}
        for old_var, new_var in zip(self.old_bn_list, self.new_bn_list):
            n_group = new_var.split("_")[1]
            new_var = "group_" + n_group + "/" + new_var
            self.copy_bn_dict[old_var] = new_var
            old_variable = tf.contrib.framework.load_variable( output_path, old_var  )
            new_variable = tf.get_default_graph().get_tensor_by_name(new_var)
            upd_op = tf.assign(new_variable, old_variable)
            upd_op.eval()

            logging.info("%s has send value to %s"%(old_var, new_var))

        return 0

    def train(self, output_path, restore=True, restored_path=None, training_iters=200, epochs=1000, dropout=0.75, display_step=5):

        self.output_path = output_path
        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)

        self._initialize_logs()
        save_path = os.path.join(output_path, "model.cpkt")
        if epochs == 0:
            return save_path
        init_glb, init_loc = self._initialize(training_iters, output_path)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True # False
        with tf.Session(config=config) as sess:
            sess.run([ init_glb, init_loc] )
            coord = tf.train.Coordinator()
            # For restore models, there are three situations:
            # 1. warming up discriminator, init from MRI segmenter: "restore_from_baseline=True, clear_rms=True"
            #    if restore_from_baseline set True, clear_rms whatever,
            #    this would restore the pre-trained MRI segmenter (without BN), this works together with following lines 1076-1079 to manually load BN
            # 2. after warming up discriminator, start training GAN: "restpre_from_baseline=False, clear_rms=True"
            #    this would restore the entire GAN system with warmed up discriminator (excluding RMS from optimizer)
            # 3. fine-tune GAN from a breakpoint: "restore_from_baseline=False, clear_rms=False"
            if restore:
                if restored_path is None:
                    raise Exception("No restore path is provided")
                ckpt = tf.train.get_checkpoint_state(restored_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path, no_gan = self.train_config["restore_from_baseline"], clear_rms = self.train_config["clear_rms"])

            if self.train_config["restore_from_baseline"] is True:  # here initialize the MRI and CT part with the pre-trained MRI segmenter, only call once beginning train
                self._load_batch_norm_weights(restored_path)  # load batchnorm variables of MRI-specific and joint part
                print("initializing from baseline model!")
                self._adapt_copy_weights()  # copy MRI weights to CT adaptation layers for initialization

            if self.lr_update_flag is True: # manually reset learning rate when needed
                sess.run( tf.assign(self.learning_rate_node, self.LR_refresh)  )
                logging.info("New learning rate %s has been loaded"%str(self.LR_refresh))

            train_summary_writer = tf.summary.FileWriter(output_path + "/train_log" + self.train_config['tag'], graph=sess.graph)
            val_summary_writer = tf.summary.FileWriter(output_path + "/val_log" + self.train_config['tag'], graph=sess.graph)

            ct_feed_all, ct_feed_fid = self.next_batch(self.ct_train_queue)
            ct_feed_val, ct_feed_val_fid = self.next_batch(self.ct_val_queue)

            mr_feed_all, mr_feed_fid = self.next_batch(self.mr_train_queue)
            mr_feed_val, mr_feed_val_fid = self.next_batch(self.mr_val_queue)

            threads = tf.train.start_queue_runners(sess = sess, coord = coord, start = True)

            # read iteration configurations
            dis_interval = self.train_config['dis_interval'] # frequency of discriminator updates, default 1. if set 2, update discriminator every 2 iterations
            gen_interval = self.train_config['gen_interval'] # frequency of generator updates, default 1. if set 2, update generator every 2 iterations

            dis_sub_iter = self.train_config['dis_sub_iter'] # number of sub-iteration in one updates, recommended to be larger than gen_sub_iter
            gen_sub_iter = self.train_config['gen_sub_iter']

            # set if we what to increase *_sub_iter every <sub_iter_upd_interval>.
            # for example, if this is set 1, and sub_iter_upd_interval is 100, then increase dis_sub_iter by 1 every 100 iterations
            dis_sub_iter_inc = self.train_config.pop('dis_sub_iter_inc', 0)
            gen_sub_iter_inc = self.train_config.pop('gen_sub_iter_inc', 0)

            sub_iter_upd_interval = self.train_config.pop('iter_upd_interval', 999999999999)
            for epoch in range(epochs):
                for step in range((epoch*training_iters), ((epoch+1)*training_iters)):
                    logging.info("Running step %s epoch %s ..."%(str(step), str(epoch)))
                    start = time.time()
                    # according to DCGAN paper, first update discriminator

                    if dis_interval == 0:
                        pass
                    elif (step % dis_interval == 0) and (step != 0):
                        for itr_dummy in range(dis_sub_iter):
                            # read samples from the pipeline, decomp them and feed them into the discriminator
                            ct_batch, ct_fid = sess.run([ct_feed_all, ct_feed_fid])
                            ct_raw_y = ct_batch[:,:,:,3]
                            ct_batch = ct_batch[:,:,:,0:3]
                            ct_batch_y = _label_decomp(self.num_cls, ct_raw_y)

                            mr_batch, mr_fid = sess.run([mr_feed_all, mr_feed_fid])
                            mr_raw_y = mr_batch[:,:,:,3]
                            mr_batch = mr_batch[:,:,:,0:3]
                            mr_batch_y = _label_decomp(self.num_cls, mr_raw_y)

                            _, _ = sess.run((self.dis_optimizer, self.learning_rate_node),
                                                        feed_dict={ self.net.mr: mr_batch,
                                                                    self.net.ct: ct_batch,
                                                                    self.net.mr_front_bn: False,
                                                                    self.net.joint_bn: False,
                                                                    self.net.ct_front_bn: False,
                                                                    self.net.cls_bn: True,
                                                                    self.net.keep_prob: dropout})
                            # clip operation
                            sess.run(self.clip_op)
                            logging.info("discriminator updated %s of %s"%(str(itr_dummy),str(dis_sub_iter)))


                    # update generator
                    if gen_interval == 0:
                        pass
                    elif (step % gen_interval == 0) and (step != 0):
                        for _ in range(gen_sub_iter):
                            ct_batch, ct_fid = sess.run([ct_feed_all, ct_feed_fid])
                            ct_raw_y = ct_batch[:,:,:,3]
                            ct_batch = ct_batch[:,:,:,0:3]
                            ct_batch_y = _label_decomp(self.num_cls, ct_raw_y)

                            _, _ = sess.run((self.gen_optimizer, self.learning_rate_node),
                                                            feed_dict={ self.net.ct: ct_batch,
                                                                        self.net.mr_front_bn: False,
                                                                        self.net.joint_bn: False,
                                                                        self.net.ct_front_bn: True,
                                                                        self.net.cls_bn: False,
                                                                        self.net.keep_prob: dropout})
                            logging.info("generator updated")

                    # if we need to update iteration configurations, do it here
                    if (step % sub_iter_upd_interval == 0) and (step != 0):
                        dis_sub_iter += dis_sub_iter_inc
                        gen_sub_iter += gen_sub_iter_inc
                        logging.info("sub iterations updated!")

                    logging.info("Training step %s epoch %s has been finished!"%(str(step), str(epoch)))
                    logging.info("Time elapsed %s seconds"%(str(time.time() - start)))

                    # evaluation and write them to tensorboard
                    if step % display_step == 0:

                        # training batch
                        train_ct_batch = sess.run(ct_feed_all)
                        train_ct_raw_y = train_ct_batch[:,:,:,3]
                        train_ct_batch = train_ct_batch[:,:,:,0:3]
                        train_ct_batch_y = _label_decomp(self.num_cls, train_ct_raw_y)

                        mr_batch, mr_fid = sess.run([mr_feed_all, mr_feed_fid])
                        mr_raw_y = mr_batch[:,:,:,3]
                        mr_batch = mr_batch[:,:,:,0:3]
                        mr_batch_y = _label_decomp(self.num_cls, mr_raw_y)

                        self.output_minibatch_stats(sess, train_summary_writer, step, train_ct_batch, train_ct_batch_y, mr_batch, mr_batch_y)

                    if step % (display_step * 1) == 0:

                        # validation batch
                        ct_batch = sess.run(ct_feed_val)
                        ct_raw_y = ct_batch[:,:,:,3]
                        ct_batch = ct_batch[:,:,:,0:3]
                        ct_batch_y = _label_decomp(self.num_cls, ct_raw_y)

                        mr_batch = sess.run(mr_feed_val)
                        mr_raw_y = mr_batch[:,:,:,3]
                        mr_batch = mr_batch[:,:,:,0:3]
                        mr_batch_y = _label_decomp(self.num_cls, mr_raw_y)

                        self.output_minibatch_stats(sess, val_summary_writer, step, ct_batch, ct_batch_y, mr_batch, mr_batch_y, detail = True)

                    # save and restore the model periodically
                    if step % (self.train_config["checkpoint_space"]) == 0:
                        if step == 0:
                            continue
                        else:
                            save_path = _save(sess, save_path, global_step = self.global_step.eval())
                            print('*********************** save path ******************: ', save_path)
                            logging.info("Model has been saved ...")
                            last_ckpt = tf.train.get_checkpoint_state(output_path)
                            if last_ckpt and last_ckpt.model_checkpoint_path:
                                self.net.restore(sess, last_ckpt.model_checkpoint_path)
                            logging.info("Model has been restored for re-allocation")
                            # learning rate decay
                            _pre_lr = sess.run(self.learning_rate_node)
                            sess.run( tf.assign(self.learning_rate_node, _pre_lr *\
                                        self.train_config['lr_decay_factor'])  )

                logging.info("Global step %s"%str(self.global_step.eval()))

            logging.info("Optimization Finished!")
            coord.request_stop()
            coord.join(threads)
            return save_path

    def output_minibatch_stats(self, sess, summary_writer, step, ct_batch, ct_batch_y, mr_batch, mr_batch_y, detail = False):

        """
        minibatch stats for tensorboard observation
        """
        if detail is not True:
            summary_str, summary_img = sess.run([\
                                                    self.scalar_summary_op,
                                                    self.train_image_summary_op],
                                                    feed_dict={\
                                                    self.net.ct_front_bn : False,
                                                    self.net.mr_front_bn : False,
                                                    self.net.joint_bn : False,
                                                    self.net.cls_bn : False,
                                                    self.net.mr: mr_batch,
                                                    self.net.mr_y: mr_batch_y,
                                                    self.net.ct: ct_batch,
                                                    self.net.ct_y: ct_batch_y,
                                                    self.net.keep_prob: 1.\
                                                    })

        else:
            _, curr_conf_mat, summary_str, summary_img = sess.run([\
                                                    self.net.compact_pred,
                                                    self.net.confusion_matrix,
                                                    self.scalar_summary_op,
                                                    self.train_image_summary_op],
                                                    feed_dict={\
                                                    self.net.ct_front_bn : False,
                                                    self.net.mr_front_bn : False,
                                                    self.net.joint_bn : False,
                                                    self.net.cls_bn : False,
                                                    self.net.mr: mr_batch,
                                                    self.net.mr_y: mr_batch_y,
                                                    self.net.ct: ct_batch,
                                                    self.net.ct_y: ct_batch_y,
                                                    self.net.keep_prob: 1.\
                                                    })


            _indicator_eval(curr_conf_mat)
        summary_writer.add_summary(summary_str, step)
        summary_writer.add_summary(summary_img, step)
        summary_writer.flush()

    def test_eval(self, sess, output_path, flip_correction = True):

        all_cm = np.zeros([self.num_cls, self.num_cls])

        pred_folder = os.path.join(output_path, "dense_pred")
        try:
            os.makedirs(pred_folder)
        except:
            logging.info("prediction folder exists")

        self.test_pair_list = list(zip(self.test_label_list, self.test_nii_list))

        sample_eval_list = [] # evaluation of each sample

        for idx_file, pair in enumerate(self.test_pair_list):
            sample_cm = np.zeros([self.num_cls, self.num_cls]) # confusion matrix for each sample
            label_fid = pair[0]
            nii_fid = pair[1]
            if not os.path.isfile(nii_fid):
                raise Exception("cannot find sample %s"%str(nii_fid))
            raw = read_nii_image(nii_fid)
            raw_y = read_nii_image(label_fid)

            if flip_correction is True:
                raw = np.flip(raw, axis = 0)
                raw = np.flip(raw, axis = 1)
                raw_y = np.flip(raw_y, axis = 0)
                raw_y = np.flip(raw_y, axis = 1)

            tmp_y = np.zeros(raw_y.shape)

            frame_list = [kk for kk in range(1, raw.shape[2] - 1)]
            np.random.shuffle(frame_list)
            for ii in range( int( floor( raw.shape[2] // self.net.batch_size  )  )  ):
                vol = np.zeros( [self.net.batch_size, raw_size[0], raw_size[1], raw_size[2]]  )
                slice_y = np.zeros( [self.net.batch_size, label_size[0], label_size[1]]  )
                for idx, jj in enumerate(frame_list[ ii * self.net.batch_size : (ii + 1) * self.net.batch_size  ]):
                    vol[idx, ...] = raw[ ..., jj -1: jj+2  ].copy()
                    slice_y[idx,...] = raw_y[..., jj ].copy()

                vol_y = _label_decomp(self.num_cls, slice_y)
                pred, curr_conf_mat= sess.run([self.net.compact_pred, self.net.confusion_matrix], feed_dict =\
                                              {self.net.ct: vol, self.net.ct_y: vol_y, self.net.keep_prob: 1.0, self.net.mr_front_bn : False,\
                                               self.net.ct_front_bn: False})

                for idx, jj in enumerate(frame_list[ii * self.net.batch_size: (ii + 1) * self.net.batch_size]):
                    tmp_y[..., jj] = pred[idx, ...].copy()

                sample_cm += curr_conf_mat

            all_cm += sample_cm
            sample_dice = _dice(sample_cm)
            sample_jaccard = _jaccard(sample_cm)
            sample_eval_list.append((sample_dice, sample_jaccard))

        subject_dice_list, subject_jaccard_list = self.sample_metric_stddev(sample_eval_list)

        np.savetxt(os.path.join(output_path, "cm.csv"), all_cm)

        return subject_dice_list, subject_jaccard_list

    def sample_metric_stddev(self, sample_eval_list):
        """
        calculate stddev of each organ across samples
        """
        metric_mat = np.zeros( [len(sample_eval_list), self.num_cls, 2]  )
        for organ, ind in list(contour_map.items()):
            for ii in range(len(sample_eval_list)):
                metric_mat[ii, int(ind), 0] = sample_eval_list[ii][0][int(ind)] # dice
                metric_mat[ii, int(ind), 1] = sample_eval_list[ii][1][int(ind)] # jaccard

        print("------- inside the sample_metric_stddev file ---- ")
        for organ, ind in list(contour_map.items()):
            print(( "organ: %s"%organ ))
            print(( "dice_stddev: %s"%( np.std(metric_mat[:, int(ind), 0] ) ) ))
            print(( "jaccard_stddev: %s"%( np.std(metric_mat[:, int(ind), 1] )  )  ))

        print("------- inside the sample_metric_stddev file ----  ")
        for organ, ind in list(contour_map.items()):
            print(( "organ: %s"%organ ))
            print(( "dice_mean: %s"%( np.mean(metric_mat[:, int(ind), 0] ) ) ))
            print(( "jaccard_mean %s"%( np.mean(metric_mat[:, int(ind), 1] )  )  ))

        print("-------")
        print(( "all_dice_mean: %s"%( np.mean(metric_mat[:, 1:, 0] ) ) ))
        print(("all_jaccard_mean: %s" % (np.mean(metric_mat[:, 1:, 1] ) )))

        subject_level_list = np.mean(metric_mat, axis=0)
        subject_level_list_dice = subject_level_list[:,0]
        subject_level_list_jaccard = subject_level_list[:1]

        return subject_level_list_dice, subject_level_list_jaccard

    def _initialize_logs(self):
        """
        This log is actually useless so ignore it
        """
        self.acc_dice_dict = {}
        self.acc_jaccard_dict = {}
        self.log_eval_fid = os.path.join(self.output_path, "acc_eval.csv")
        for organ, ind in list(contour_map.items()):
            self.acc_dice_dict[organ] = [organ]
            self.acc_jaccard_dict[organ] = [organ]

    def test_model(self, this_model, output_path):

        init_glb, init_loc = self._initialize(1, output_path)

        with tf.Session() as sess:
            sess.run([init_glb, init_loc])
            self.net.restore(sess, this_model)
            logging.info("model has been loaded!")
            dice, jac = self.test_eval(sess, output_path)
            logging.info("testing finished")

        return dice, jac
