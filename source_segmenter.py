'''
Here are implementations for source segmenter trained on single modality
'''
import os
import time
import shutil
import numpy as np
from collections import OrderedDict
import __future__
import logging
import matplotlib
from tensorflow.python import debug as tf_debug
from layers import *
from ops import *
from lib import _dice_eval, _save, _save_nii_prediction, _jaccard, _dice, _label_decomp, _indicator_eval, read_nii_image, read_nii_object


np.random.seed(0)

contour_map = {
    "bg": 0,
    "lv_myo": 1,
    "la_blood": 2,
    "lv_blood": 3,
    "aa": 4,
}

verbose = True
if verbose == True:
    logging.getLogger().addHandler(logging.StreamHandler())
view = True
logging.basicConfig(filename = "curr_log", level=logging.DEBUG, format='%(asctime)s %(message)s')

raw_size = [256, 256, 3] # original raw input size
volume_size = [256, 256, 3] # volume size after processing
label_size = [256, 256, 1]

decomp_feature = {
            'dsize_dim0': tf.FixedLenFeature([], tf.int64),
            'dsize_dim1': tf.FixedLenFeature([], tf.int64),
            'dsize_dim2': tf.FixedLenFeature([], tf.int64),
            'lsize_dim0': tf.FixedLenFeature([], tf.int64),
            'lsize_dim1': tf.FixedLenFeature([], tf.int64),
            'lsize_dim2': tf.FixedLenFeature([], tf.int64),
            'data_vol': tf.FixedLenFeature([], tf.string),
            'label_vol': tf.FixedLenFeature([], tf.string)}

class Full_DRN(object):

    def __init__(self, channels, n_class, batch_size, adapt_module = True, main_trainable = True, adapt_trainable = True, cost_kwargs={}, **kwargs):

        """
        Dilated Residual Network
        :param channels:    number of channels in the input image, set as 3
        :param n_class:     number of output labels, set as 5
        :param batch_size:  number of batch_size
        :param adapt_module: (optional)
        :param main_trainable: (optional)
        :param adapt_trainable: (optional)
        :param cost_kwargs: (optional) kwargs passed to the cost function
        """

        tf.reset_default_graph()

        self.n_class = n_class
        self.batch_size = batch_size
        self.summaries = kwargs.get("summaries", True)
        self.conv_weights = []
        self.x = tf.placeholder("float", shape=[None, volume_size[0], volume_size[1], channels])
        self.y = tf.placeholder("float", shape=[None, label_size[0], label_size[1], self.n_class])
        self.main_bn = tf.placeholder_with_default(True, shape = None, name = "main_batchnorm_training_switch")
        self.main_trainable = main_trainable
        self.adapt_trainable = adapt_trainable

        self.adapt_bn = tf.placeholder_with_default(True, shape = None, name = "adapt_batchnorm_training_switch")
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        logits =  self.create_network(input_size = raw_size, input_channel = channels, num_cls = self.n_class, feature_base = 16, keep_prob = self.keep_prob, adapt_module = adapt_module,\
                                      main_bn = self.main_bn, main_trainable = self.main_trainable,\
                                      adapt_bn = self.adapt_bn, adapt_trainable = self.adapt_trainable)
        self.predicter = pixel_wise_softmax_2(logits)
        self.compact_pred = tf.argmax(self.predicter, 3)
        self.compact_y = tf.argmax(self.y, 3)

        self.cost, self.regularizer_loss = self._get_cost(logits, cost_kwargs)
        self.confusion_matrix = tf.confusion_matrix(tf.reshape(self.compact_y,[-1]), tf.reshape(self.compact_pred, [-1]), num_classes = self.n_class)


    def create_network(self, input_size, input_channel, num_cls, feature_base = 16, keep_prob = 0.75, main_bn = True, main_trainable = True,\
                       adapt_module = True, adapt_bn = True, adapt_trainable = True):

        with tf.name_scope('group_1') as scope:
            w1_1 = weight_variable(shape = [3, 3, input_channel, feature_base], trainable = adapt_trainable)
            conv1_1 = conv2d(self.x, w1_1, keep_prob )
            wr1_1 = weight_variable(shape = [3, 3, feature_base, feature_base], trainable = adapt_trainable)
            wr1_2 = weight_variable(shape = [3, 3, feature_base, feature_base], trainable = adapt_trainable)
            block1_1 = residual_block(conv1_1, wr1_1, wr1_2, keep_prob , is_train = adapt_bn, leak = True, bn_trainable = adapt_trainable)
            out1 = max_pool2d(block1_1, n = 2)
            self.conv_weights.append(w1_1)
            self.conv_weights.append(wr1_1)
            self.conv_weights.append(wr1_2)

        with tf.name_scope('group_2') as scope:
            wr2_1 = weight_variable(shape = [3, 3, feature_base, feature_base * 2], trainable = adapt_trainable)
            wr2_2 = weight_variable(shape = [3, 3, feature_base * 2, feature_base * 2], trainable = adapt_trainable)
            block2_1 = residual_block(out1, wr2_1, wr2_2, inc_dim = True, leak = True, keep_prob = keep_prob, is_train = adapt_bn, bn_trainable = adapt_trainable)
            out2 = max_pool2d(block2_1, n = 2)
            self.conv_weights.append(wr2_1)
            self.conv_weights.append(wr2_2)

        with tf.name_scope('group_3') as scope:
            wr3_1 = weight_variable( shape = [3, 3, feature_base * 2, feature_base * 4], trainable = adapt_trainable  )
            wr3_2 = weight_variable( shape = [3, 3, feature_base * 4, feature_base * 4], trainable = adapt_trainable  )
            block3_1 = residual_block( out2, wr3_1, wr3_2, keep_prob, inc_dim = True, leak = True, is_train = adapt_bn, bn_trainable = adapt_trainable      )

            wr3_3 = weight_variable( shape = [3, 3, feature_base * 4, feature_base * 4], trainable = adapt_trainable  )
            wr3_4 = weight_variable( shape = [3, 3, feature_base * 4, feature_base * 4], trainable = adapt_trainable  )
            block3_2 = residual_block( block3_1, wr3_3, wr3_4,keep_prob = keep_prob, leak = True, is_train = adapt_bn, bn_trainable = adapt_trainable     )
            out3 = max_pool2d(block3_2, n = 2)
            self.conv_weights.append(wr3_1)
            self.conv_weights.append(wr3_2)
            self.conv_weights.append(wr3_3)
            self.conv_weights.append(wr3_4)

        with tf.name_scope('group_4') as scope:
            wr4_1 = weight_variable( shape = [3, 3, feature_base * 4, feature_base * 8], trainable = adapt_trainable  )
            wr4_2 = weight_variable( shape = [3, 3, feature_base * 8, feature_base * 8], trainable = adapt_trainable   )
            block4_1 = residual_block( out3, wr4_1, wr4_2, keep_prob,  inc_dim = True, leak = True, is_train = adapt_bn, bn_trainable = adapt_trainable     )

            wr4_3 = weight_variable( shape = [3, 3, feature_base * 8, feature_base * 8], trainable = adapt_trainable  )
            wr4_4 = weight_variable( shape = [3, 3, feature_base * 8, feature_base * 8], trainable = adapt_trainable  )
            block4_2 = residual_block( block4_1, wr4_3, wr4_4, keep_prob, is_train = adapt_bn, leak = True, bn_trainable = adapt_trainable     )
            self.conv_weights.append(wr4_1)
            self.conv_weights.append(wr4_2)
            self.conv_weights.append(wr4_4)
            self.conv_weights.append(wr4_4)

        with tf.name_scope('group_5') as scope:
            wr5_1 = weight_variable( shape = [3, 3, feature_base * 8, feature_base * 16], trainable = main_trainable  )
            wr5_2 = weight_variable( shape = [3, 3, feature_base * 16, feature_base * 16], trainable = main_trainable  )
            block5_1 = residual_block( block4_2, wr5_1, wr5_2, keep_prob = keep_prob, leak = True, inc_dim = True,  is_train = main_bn, bn_trainable = main_trainable    )

            wr5_3 = weight_variable( shape = [3, 3, feature_base * 16, feature_base * 16], trainable = main_trainable  )
            wr5_4 = weight_variable( shape = [3, 3, feature_base * 16, feature_base * 16], trainable = main_trainable  )
            block5_2 = residual_block( block5_1, wr5_3, wr5_4, keep_prob = keep_prob, leak = True,  is_train = main_bn, bn_trainable = main_trainable    )
            self.conv_weights.append( wr5_1  )
            self.conv_weights.append( wr5_2  )
            self.conv_weights.append( wr5_3  )
            self.conv_weights.append( wr5_4  )

        with tf.name_scope('group_6') as scope:
            wr6_1 = weight_variable( shape = [3, 3, feature_base * 16, feature_base * 16], trainable = main_trainable  )
            wr6_2 = weight_variable( shape = [3, 3, feature_base * 16, feature_base * 16], trainable = main_trainable  )
            block6_1 = residual_block( block5_2, wr6_1, wr6_2, keep_prob = keep_prob, leak = True, is_train = main_bn,  bn_trainable = main_trainable    )

            wr6_3 = weight_variable( shape = [3, 3, feature_base * 16, feature_base * 16], trainable = main_trainable  )
            wr6_4 = weight_variable( shape = [3, 3, feature_base * 16, feature_base * 16], trainable = main_trainable  )
            block6_2 = residual_block( block6_1, wr6_3, wr6_4, keep_prob = keep_prob, leak = True,  is_train = main_bn,  bn_trainable = main_trainable    )
            self.conv_weights.append( wr6_1  )
            self.conv_weights.append( wr6_2  )
            self.conv_weights.append( wr6_3  )
            self.conv_weights.append( wr6_4  )

        with tf.name_scope('group_7') as scope:
            wr7_1 = weight_variable( shape = [3, 3, feature_base * 16, feature_base * 32], trainable = main_trainable  )
            wr7_2 = weight_variable( shape = [3, 3, feature_base * 32, feature_base * 32], trainable = main_trainable  )
            block7_1 = residual_block( block6_2, wr7_1, wr7_2, keep_prob = keep_prob, leak = True,  inc_dim = True, is_train = main_bn, bn_trainable = main_trainable    )

            wr7_3 = weight_variable( shape = [3, 3, feature_base * 32, feature_base * 32], trainable = main_trainable  )
            wr7_4 = weight_variable( shape = [3, 3, feature_base * 32, feature_base * 32], trainable = main_trainable  )
            block7_2 = residual_block( block7_1, wr7_3, wr7_4, keep_prob = keep_prob, leak = True,  is_train = main_bn, bn_trainable = main_trainable    )
            self.conv_weights.append( wr7_1  )
            self.conv_weights.append( wr7_2  )
            self.conv_weights.append( wr7_3  )
            self.conv_weights.append( wr7_4  )

        with tf.name_scope('group_8') as scope:
            wr8_1 = weight_variable( shape = [3, 3, feature_base * 32, feature_base * 32], trainable = main_trainable  )
            wr8_2 = weight_variable( shape = [3, 3, feature_base * 32, feature_base * 32], trainable = main_trainable  )
            block8_1 = DR_block( block7_2, wr8_1, wr8_2, keep_prob = keep_prob, leak = True, rate = 2, is_train = main_bn, bn_trainable = main_trainable    )

            wr8_3 = weight_variable( shape = [3, 3, feature_base * 32, feature_base * 32], trainable = main_trainable  )
            wr8_4 = weight_variable( shape = [3, 3, feature_base * 32, feature_base * 32], trainable = main_trainable  )
            block8_2 = DR_block( block8_1, wr8_3, wr8_4, keep_prob = keep_prob, leak = True,  rate = 2, is_train = main_bn, bn_trainable = main_trainable    )
            self.conv_weights.append( wr8_1  )
            self.conv_weights.append( wr8_2  )
            self.conv_weights.append( wr8_3  )
            self.conv_weights.append( wr8_4  )

        with tf.name_scope('group_9') as scope:
            w9_1 = weight_variable( shape = [3, 3, feature_base * 32, feature_base * 32], trainable = main_trainable  )
            conv9_1 = conv_bn_relu2d( block8_2, w9_1, keep_prob, is_train = main_bn, bn_trainable = main_trainable, leak = True    )
            w9_2 = weight_variable( shape = [3, 3, feature_base * 32, feature_base * 32], trainable = main_trainable  )
            conv9_2 = conv_bn_relu2d( conv9_1, w9_2, keep_prob, is_train = main_bn, bn_trainable = main_trainable, leak = True    )
            self.conv_weights.append( w9_1  )
            self.conv_weights.append( w9_2  )

        with tf.name_scope('group_10') as scope:
            local_size = 8 * 8 # (r^2)
            w10_1 = weight_variable( shape = [3, 3, feature_base * 32, local_size * num_cls * 8], trainable = main_trainable  )
            conv10_1 = conv2d( conv9_2, w10_1, keep_prob_ = keep_prob, padding = 'SYMMETRIC')
            self.conv_weights.append(w10_1)
            flat_conv10_1 = PS(conv10_1, r = 8, n_channel = num_cls * 8, batch_size = self.batch_size)

        with tf.name_scope('output') as scope:
            w11_1 = weight_variable( shape = [5, 5, num_cls * 8, num_cls], trainable = main_trainable  )
            logits = conv2d( flat_conv10_1, w11_1, keep_prob_ = 1., padding = 'SYMMETRIC'  )
            self.conv_weights.append(w11_1)

        return logits

    def _get_cost(self, logits, cost_kwargs):
        """
        Compute cost for segmentation network
        Here we jointly use weighted cross-entropy (for class imbalance) and Dice loss
        """
        loss = 0
        dice_flag = cost_kwargs.pop("dice_flag", True)
        cross_flag = cost_kwargs.pop("cross_flag", False)
        miu_dice = cost_kwargs.pop("miu_dice", None)
        miu_cross = cost_kwargs.pop("miu_cross", None)
        reg_coeff = cost_kwargs.pop("regularizer", 1e-4)

        if cross_flag is True:
            self.weighted_loss = self._softmax_weighted_loss(logits)
            loss += miu_cross * self.weighted_loss

        if dice_flag is True:
            self.dice_loss = self._dice_loss_fun(logits)
            loss += miu_dice * self.dice_loss

        self.dice_eval, self.dice_eval_arr = _dice_eval(self.compact_pred, self.y, self.n_class)
        self.dice_eval_c1 = self.dice_eval_arr[1]
        self.dice_eval_c2 = self.dice_eval_arr[2]
        self.dice_eval_c3 = self.dice_eval_arr[3]
        self.dice_eval_c4 = self.dice_eval_arr[4]

        regularizers = sum([tf.nn.l2_loss(variable) for variable in self.conv_weights])

        return loss, reg_coeff * regularizers

    def _softmax_weighted_loss(self, logits):
        '''
        calculate weighted cross-entropy loss, the weight is dynamic dependent on the data
        '''
        softmaxpred = tf.nn.softmax(logits)

        for i in range(self.n_class):
            gti = self.y[:,:,:,i]
            predi = softmaxpred[:,:,:,i]
            weighted = 1-(tf.reduce_sum(gti) / tf.reduce_sum(self.y))
            if i == 0:
                raw_loss = -1.0 * weighted * gti * tf.log(tf.clip_by_value(predi, 0.005, 1))
            else:
                raw_loss += -1.0 * weighted * gti * tf.log(tf.clip_by_value(predi, 0.005, 1))

        loss = tf.reduce_mean(raw_loss)

        return loss

    def _dice_loss_fun(self, logits):
        '''
        calculate dice loss, - 2*interesction/union, with relaxed for gradients backpropagation
        '''
        dice = 0
        eps = 1e-7
        softmaxpred = tf.nn.softmax(logits)
        for i in range(self.n_class):
            inse = tf.reduce_sum(softmaxpred[:, :, :, i]*self.y[:, :, :, i])
            l = tf.reduce_sum(softmaxpred[:, :, :, i]*softmaxpred[:, :, :, i])
            r = tf.reduce_sum(self.y[:, :, :, i])
            dice += 2.0 * inse/(l+r+eps)

        return -1.0 * dice / self.n_class

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint
        :param sess: current session instance
        :param model_path: path to checkpoint file location
        """
        saver = tf.train.Saver(tf.contrib.framework.get_variables() + tf.get_collection_ref("internal_batchnorm_variables") )
        logging.info("Model restored from file: %s" % model_path)
        try:
            saver.restore(sess, model_path)
            logging.info("Model restored from file: %s" % model_path)
        except:
            variables = tf.global_variables()
            reader = tf.pywrap_tensorflow.NewCheckpointReader(model_path)
            var_keep_dic = reader.get_variable_to_shape_map()
            variables_to_restore = []
            for v in variables:
                if v.name.split(':')[0] in var_keep_dic:
                    variables_to_restore.append(v)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, model_path)

            logging.info("Model restored from file: %s with relaxation" % model_path)
            logging.info("Restored variables: ")
            for vname in list(var_keep_dic.keys()):
                logging.info(str(vname))


class Trainer(object):
    """
    Train a network instance
    :param net: the network instance to train
    :param train_list: image files for training
    :param val_list: image files for validation
    :param test_nii_list: image files used at testing mode
    """

    def __init__(self, net, train_list, val_list, num_cls, batch_size, test_nii_list = None, test_label_list = None, optimizer="momentum", \
                 opt_kwargs={}, num_epochs = 100, checkpoint_space = 500, lr_update_flag = False):
        self.net = net
        self.batch_size = batch_size
        self.num_cls = num_cls
        self.checkpoint_space = checkpoint_space
        self.opt_kwargs = opt_kwargs
        self.optimizer = optimizer
        self.train_list = train_list
        self.val_list =val_list
        self.test_label_list = test_label_list
        self.test_nii_list = test_nii_list
        self.train_queue = tf.train.string_input_producer(train_list, num_epochs = None, shuffle = True)
        self.val_queue = tf.train.string_input_producer(val_list, num_epochs = None, shuffle = True)
        self.dice = tf.Variable( -1 * np.ones( self.num_cls))
        self.jaccard = tf.Variable( -1 * np.ones( self.num_cls))
        self.loss_dict = {}
        self.lr_update_flag = lr_update_flag

    def next_batch(self, input_queue, capacity = 120, num_threads = 4, min_after_dequeue = 30, label_type = 'float'):
        """ move original input pipeline here"""
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
        data_vol = tf.slice(data_vol, [0,0,0], volume_size)
        label_vol = tf.slice(label_vol, [0,0,1], label_size)

        data_feed, label_feed, fid_feed = tf.train.shuffle_batch([data_vol, label_vol, fid], batch_size =self.batch_size , capacity = capacity, \
                                                            num_threads = num_threads, min_after_dequeue = min_after_dequeue)

        pair_feed = tf.concat([data_feed, label_feed], axis = 3)

        return pair_feed, fid_feed

    def _get_optimizer(self, training_iters, global_step):

        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)

            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                        global_step=global_step,
                                                        decay_steps=training_iters,
                                                        decay_rate=decay_rate,
                                                        staircase=True)

            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs).minimize(self.net.cost + self.net.regularizer_loss,
                                                                                global_step=global_step)
        elif self.optimizer == "adam":

            learning_rate = self.opt_kwargs.pop("learning_rate", None)
            self.learning_rate_node = tf.Variable(learning_rate)
            self._new_LR = learning_rate # this for using a new specified learning rate when RESTORING a model
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node,
                                                                **self.opt_kwargs).minimize(self.net.cost + self.net.regularizer_loss ,\
                                                                global_step=global_step)
        return optimizer

    def _initialize(self, training_iters, output_path,  restore):

        self.global_step = tf.Variable(0)
        scalar_summaries = []
        scalar_summaries.append(tf.summary.scalar('loss', self.net.cost))
        scalar_summaries.append(tf.summary.scalar('regularizer_loss', self.net.regularizer_loss))
        scalar_summaries.append(tf.summary.scalar('weighted_loss', self.net.weighted_loss))
        scalar_summaries.append(tf.summary.scalar('dice_loss', self.net.dice_loss))
        scalar_summaries.append(tf.summary.scalar('dice_eval', self.net.dice_eval))

        scalar_summaries.append(tf.summary.scalar('dice_eval_c1', self.net.dice_eval_c1))
        scalar_summaries.append(tf.summary.scalar('dice_eval_c2', self.net.dice_eval_c2))
        scalar_summaries.append(tf.summary.scalar('dice_eval_c3', self.net.dice_eval_c3))
        scalar_summaries.append(tf.summary.scalar('dice_eval_c4', self.net.dice_eval_c4))

        train_images = []
        train_images.append(tf.summary.image('train_pred', tf.expand_dims(tf.cast(self.net.compact_pred, tf.float32), 3 )) )
        train_images.append(tf.summary.image('image', tf.expand_dims(tf.cast(self.net.x[:,:,:,1], tf.float32), 3 )) )
        train_images.append(tf.summary.image('GND', tf.expand_dims(tf.cast(self.net.compact_y, tf.float32), 3)))
        val_images = []
        val_images.append(tf.summary.image('val_pred', tf.expand_dims(tf.cast(self.net.compact_pred, tf.float32), 3)))
        val_images.append(tf.summary.image('image', tf.expand_dims(tf.cast(self.net.x[:,:,:,1], tf.float32), 3)))
        val_images.append(tf.summary.image('validation_GND', tf.expand_dims(tf.cast(self.net.compact_y, tf.float32), 3)))

        self.scalar_summary_op = tf.summary.merge(scalar_summaries)
        self.train_image_summary_op = tf.summary.merge(train_images)
        self.val_image_summary_op = tf.summary.merge(val_images)

        self.optimizer = self._get_optimizer(training_iters, self.global_step)
        scalar_summaries.append(tf.summary.scalar('learning_rate', self.learning_rate_node))

        init_glb = tf.global_variables_initializer()
        init_loc = tf.variables_initializer(tf.local_variables())

        output_path = os.path.abspath(output_path)

        if not restore:
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)

        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)

        return init_glb, init_loc

    def train(self, output_path, restored_path=None, restore=False, training_iters=100, epochs=100, display_step=5, dropout=0.75):

        """
        Lauches the training process
        :param output_path: path where to store checkpoints
        :param restored_path: path where checkpoints are read from, for resume training
        :param restore: Flag if previous model should be restored
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param display_step: number of steps till outputting stats
        :param dropout: keep probability for dropout rate
        """
        save_path = os.path.join(output_path, "model.cpkt")
        if epochs == 0:
            return save_path
        init_glb, init_loc = self._initialize(training_iters, output_path, restore)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False
        with tf.Session(config=config) as sess:
            sess.run([init_glb, init_loc])
            coord = tf.train.Coordinator()

            if restore:
                if restored_path is None:
                    raise Exception("No restore path is provided")
                ckpt = tf.train.get_checkpoint_state(restored_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print("Unable to restore, start from beginning")

                if self.lr_update_flag is True:
                    sess.run(tf.assign(self.learning_rate_node, self._new_LR))
                    logging.info("New learning rate %s has been loaded"%str(self._new_LR))

            train_summary_writer = tf.summary.FileWriter(output_path + "/train_log", graph=sess.graph)
            val_summary_writer = tf.summary.FileWriter(output_path + "/val_log", graph=sess.graph)
            feed_all, feed_fid = self.next_batch(self.train_queue)
            feed_val, feed_val_fid = self.next_batch(self.val_queue)
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)

            train_vars = tf.trainable_variables()
            for _var in train_vars:
                logging.info(_var.name)

            for epoch in range(epochs):
                for step in range((epoch*training_iters), ((epoch+1)*training_iters)):
                    logging.info("Running step %s epoch %s ..."%(str(step), str(epoch)))
                    start = time.time()
                    batch, fid = sess.run([feed_all, feed_fid])
                    batch_x = batch[:,:,:,0:3]
                    raw_y = batch[:,:,:,3] # a single map with multi-classes
                    batch_y = _label_decomp(self.num_cls, raw_y) # n_class binary maps
                    fids = [ _single.decode('utf-8').split(":")[0] for _single in fid ]

                    _, loss, lr = sess.run((self.optimizer, self.net.cost, self.learning_rate_node),
                                                      feed_dict={self.net.x: batch_x,
                                                                 self.net.y: batch_y,
                                                                 self.net.main_bn: True,
                                                                 self.net.adapt_bn: True,
                                                                 self.net.keep_prob: dropout})
                    if verbose:
                        logging.info("Training at step %s epoch %s , loss is %0.4f"%(str(step), str(epoch), loss))
                        logging.info("Time elapsed %s seconds"%(str(time.time() - start)))

                    if step % display_step == 0:
                        self.output_minibatch_stats(sess, train_summary_writer, step, batch_x, batch_y, raw_y)

                    if step % (display_step * 1) == 0:
                        val_batch = feed_val.eval()
                        val_x = val_batch[:,:,:,0:3]
                        val_y = val_batch[:,:,:,3]
                        val_y = _label_decomp(self.num_cls, val_y)
                        detail_flag = False
                        if step % (1 * display_step) == 0:
                            detail_flag = True
                        self.val_stats(sess, val_summary_writer, step, val_x, val_y, detail_flag)

                    if step % (self.checkpoint_space) == 0 and step > 10000:
                        if step == 0:
                            pass
                        else:
                            save_path = _save(sess, save_path, global_step = self.global_step.eval())
                            last_ckpt = tf.train.get_checkpoint_state(output_path)
                            if last_ckpt and last_ckpt.model_checkpoint_path:
                                self.net.restore(sess, last_ckpt.model_checkpoint_path)
                            logging.info("Model has been restored for re-allocation")
                            _pre_lr = sess.run(self.learning_rate_node)
                            sess.run( tf.assign(self.learning_rate_node, _pre_lr * 0.9 )  )

                logging.info("Global step %s"%str(self.global_step.eval()))
            logging.info("Optimization Finished!")
            coord.request_stop()
            coord.join(threads)
            return save_path

    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y, compact_y = None):
        """
        minibatch stats for tensorboard observation
        """
        summary_str, summary_img, loss= sess.run([\
                                            self.scalar_summary_op,
                                            self.train_image_summary_op,
                                            self.net.cost],
                                            feed_dict={self.net.x: batch_x,
                                            self.net.y: batch_y,
                                            self.net.keep_prob: 1.})

        summary_writer.add_summary(summary_str, step)
        summary_writer.add_summary(summary_img, step)
        summary_writer.flush()

    def val_stats(self, sess, summary_writer, step, batch_x, batch_y, detail = False):

        if detail is False:
            summary_str, summary_img, loss= sess.run([\
                                                self.scalar_summary_op,
                                                self.val_image_summary_op,
                                                self.net.cost],
                                                feed_dict ={self.net.x: batch_x,
                                                            self.net.y: batch_y,
                                                            self.net.main_bn: False,
                                                            self.net.adapt_bn: False,
                                                            self.net.keep_prob: 1.})

        else:
            pred, curr_conf_mat, summary_str, summary_img, loss = sess.run([\
                                                self.net.compact_pred,
                                                self.net.confusion_matrix,\
                                                self.scalar_summary_op,
                                                self.val_image_summary_op,
                                                self.net.cost],
                                                feed_dict ={self.net.x: batch_x,
                                                            self.net.y: batch_y,
                                                            self.net.main_bn: False,
                                                            self.net.adapt_bn: False,
                                                            self.net.keep_prob: 1.0 })

            _indicator_eval(curr_conf_mat)
        summary_writer.add_summary(summary_str, step)
        summary_writer.add_summary(summary_img, step)
        summary_writer.flush()

    def test_eval(self, sess, output_path, flip_correction = True, save_result = False):
        """
        Doing inference given test cases, in the format of .nii file
        Args:
            flip correction: use this to correct orientation mismatch between tfrecords and nii file
        """
        pred_folder = os.path.join(output_path, "test_pred")
        try:
            os.makedirs(pred_folder)
        except:
            logging.info("Cannot create prediction result folder")

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

            nii_pred_bname = "dense_pred_" + os.path.basename(nii_fid)

            if flip_correction is True:
                raw = np.flip(raw, axis = 0)
                raw = np.flip(raw, axis = 1)
                raw_y = np.flip(raw_y, axis = 0)
                raw_y = np.flip(raw_y, axis = 1)

            tmp_y = np.zeros(raw_y.shape)

            for ii in range( int(floor( raw.shape[2] // self.net.batch_size  )  ) ):
                vol = np.zeros( [self.net.batch_size, raw_size[0], raw_size[1], raw_size[2]]  )
                slice_y = np.zeros( [self.net.batch_size, label_size[0], label_size[1]]  )

                for idx, jj in enumerate(range(ii * self.net.batch_size : (ii + 1) * self.net.batch_size)):
                    vol[idx,...] = raw[ ..., jj -1: jj+2  ].copy()
                    slice_y[idx,...] = raw_y[..., jj ].copy()
                vol_y = _label_decomp(self.num_cls, slice_y)
                pred, curr_conf_mat= sess.run([self.net.compact_pred, self.net.confusion_matrix], \
                                                feed_dict = {self.net.x: vol, self.net.y: vol_y, self.net.keep_prob: 1.0, \
                                                             self.net.main_bn: False, self.net.adapt_bn: False})

                for idx, jj in enumerate(range(ii * self.net.batch_size : (ii + 1) * self.net.batch_size)):
                    tmp_y[..., jj] = pred[idx, ... ].copy()
                logging.info(" part %s of %s of sample %s has been processed.."%(str(ii), str(floor(raw.shape[2] // self.net.batch_size)), str(idx_file)))
                sample_cm += curr_conf_mat

            sample_dice = _dice(sample_cm)
            sample_jaccard = _jaccard(sample_cm)
            sample_eval_list.append((sample_dice, sample_jaccard))

            if save_result is True:
                _save_nii_prediction(raw_y, tmp_y, nii_fid, pred_folder, out_bname = nii_pred_bname)

        subject_dice_list, subject_jaccard_list = self.sample_metric_stddev(sample_eval_list)
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

    def test_choose_model(self, this_model, output_path):
        init_glb, init_loc = self._initialize(1, output_path, True)

        with tf.Session() as sess:
            sess.run([init_glb, init_loc])
            self.net.restore(sess, this_model)
            logging.info("model has been loaded!")
            dice, jac = self.test_eval(sess, output_path)
            logging.info("testing finished")
        return dice, jac

#    def _indicator_eval(self, cm, verbose = True):
#        """
#        Decompose confusion matrix and get statistics, for logging training procedure
#        """
#        my_dice = _dice(cm)
#        my_jaccard = _jaccard(cm)
#        print(cm)
#        for organ, ind in list(contour_map.items()):
#            print(("organ: %s "%organ))
#            print(("dice: %s " %(my_dice[int(ind)])))
#            print(("jaccard: %s " %(my_jaccard[int(ind)])))
#        return my_dice, my_jaccard
#
#    def test(self, output_path, restored_path):
#        """
#        Launches the test process
#
#        :param output_path: path where to store checkpoints
#        :param restored_path: path where checkpoints are read from
#        """
#        init_glb, init_loc = self._initialize(1, output_path, True)
#
#        with tf.Session() as sess:
#            sess.run([ init_glb, init_loc] )
#            ckpt = tf.train.get_checkpoint_state(restored_path)
#            self.net.restore(sess, ckpt.model_checkpoint_path)
#            logging.info("model has been loaded!")
#            self.test_eval(sess, output_path)
#            logging.info("testing finished")
