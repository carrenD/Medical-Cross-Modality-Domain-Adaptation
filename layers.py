'''
Layer definations
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import pdb
import tensorflow as tf
from math import floor

def conv_bn_relu2d(x, W, keep_prob, padding = 'SAME', strides = [1,1,1,1], is_train = True, scope = None, bn_trainable = True, leak = False):
    bn = conv_bn_2d(x, W, keep_prob, padding = padding, strides = strides, is_train = is_train, scope = scope, bn_trainable = bn_trainable )
    if leak is True:
    	return tf.nn.leaky_relu(bn)
    else:
    	return tf.nn.relu(bn)

def conv_bn_2d(x, W, keep_prob, padding = 'SAME', strides = [1,1,1,1], is_train = True, scope = None, bn_trainable = True):
    if padding == 'SAME':
        conv_2d = tf.nn.conv2d(x, W, strides=strides, padding = 'SAME')
    elif padding == 'SYMMETRIC': # to deal with boundary effect!
        k_shape = W.get_shape().as_list()
        pd_offset = tf.constant( [[0, 0], [  floor(k_shape[0] / 2 ) , floor(k_shape[0] / 2 )], [  floor(k_shape[1] / 2 ) , floor(k_shape[1] / 2)], [0, 0 ]] )
        pd_offset = tf.cast(pd_offset, tf.int32)
        x = tf.pad(x, pd_offset, 'SYMMETRIC' )
        conv_2d = tf.nn.conv2d(x, W, strides=strides, padding = 'VALID')
    conv_2d = tf.nn.dropout(conv_2d, keep_prob)
    bn = batch_norm(conv_2d, is_training = is_train, scope = scope, trainable = bn_trainable)
    return bn

def dilate_conv_bn_relu2d(x, W, keep_prob, padding = 'SAME', rate = 2, is_train = True,scope = None, bn_trainable = True, leak = False):
    """
    Meow!
    """
    bn = dilate_conv_bn( x, W, keep_prob, padding = padding, rate = rate, is_train = is_train, scope = scope, bn_trainable = bn_trainable  )
    if leak is True:
    	return tf.nn.leaky_relu(bn)
    else:
    	return tf.nn.relu(bn)

def dilate_conv_bn(x, W, keep_prob, padding = 'SAME', rate = 2, is_train = True, scope = None, bn_trainable = True):
    """
    Meow!
    """
    di_conv = dilate_conv2d(x, W, keep_prob_ = keep_prob, rate = rate, padding = padding)
    bn = batch_norm(di_conv, is_training = is_train, trainable = bn_trainable, scope = scope)
    return bn

def weight_variable(shape, stddev=0.01, trainable = True):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, trainable = trainable)

def sharable_weight_variable(shape, stddev=0.1, trainable = True, name = "IhaveNoName"):
    """
    sharable through variable scope reuse mechnism
    """
    return tf.get_variable(name = name, shape = shape, initializer = tf.truncated_normal_initializer(stddev = stddev), trainable = trainable)

def weight_variable_deconv(shape, stddev=0.1):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W,keep_prob_,strides = [1,1,1,1], padding = 'SAME'):
#    pdb.set_trace()
    if padding == 'SAME':
        conv_2d = tf.nn.conv2d(x, W, strides = strides, padding = 'SAME')
    elif padding == 'SYMMETRIC':
        k_shape = W.get_shape().as_list()
        pd_offset = tf.constant( [[0, 0], [  floor(k_shape[0] / 2 ) , floor(k_shape[0] / 2 )], [  floor(k_shape[1] / 2 ) , floor(k_shape[1] / 2)], [0, 0 ]] )
        pd_offset = tf.cast(pd_offset, tf.int32)
        x = tf.pad(x, pd_offset, 'SYMMETRIC' )
        conv_2d = tf.nn.conv2d(x, W, strides= strides, padding = 'VALID')
    return tf.nn.dropout(conv_2d, keep_prob_)

# layers for gpwgan
def conv_relu2d(x, W, keep_prob, padding = 'SAME', strides = [1,1,1,1],leak = False):
    cv = conv2d(x, W, keep_prob, padding = padding, strides = strides )
    if leak is True:
    	return tf.nn.leaky_relu(cv)
    else:
    	return tf.nn.relu(cv)

def dilate_conv2d(x, W, keep_prob_, rate = 2, padding = 'SAME'):
    if padding == 'SAME':
        di_conv_2d = tf.nn.atrous_conv2d(x, W, rate = rate, padding = 'SAME')
    elif padding == 'SYMMETRIC':
        k_shape = W.get_shape().as_list()
        pd_offset = tf.constant( [[0, 0], [  floor(k_shape[0] / 2 ) , floor(k_shape[0] / 2 )], [  floor(k_shape[1] / 2 ) , floor(k_shape[1] / 2)], [0, 0 ]] )
        pd_offset = tf.cast(pd_offset, tf.int32)
        x = tf.pad(x, pd_offset, 'SYMMETRIC' )
        di_conv_2d = tf.nn.atrous_conv2d(x, W, rate = rate, padding = 'VALID')
    return tf.nn.dropout(di_conv_2d, keep_prob_)

def batch_norm(x, is_training = True, scope = None, trainable = True):
    """
    Note:
        For training and testing the discriminator the batch norm is actually using batch statistics instead of global ones, making this somehow work as instance normalization
    """
    return tf.contrib.layers.batch_norm(x, is_training = is_training, decay = 0.90, scale = True, center = True, scope = scope, variables_collections = ["internal_batchnorm_variables"], updates_collections = None, trainable = trainable)

def max_pool2d(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

def avg_pool2d(x,n):
    return tf.nn.avg_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

def crop_and_concat(x1,x2, name = "default"):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3, name = "concat_" + name)

def simple_concat2d(x1,x2):
    """ concatenation without offset check"""
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    try:
        tf.equal(x1_shape[0:-2], x2_shape[0: -2])
    except:
        print("x1_shape: %s"%str(x1.get_shape().as_list()))
        print("x2_shape: %s"%str(x2.get_shape().as_list()))
        raise ValueError("Cannot concatenate tensors with different shape, igonoring feature map depth")
    return tf.concat([x1, x2], 3)

def pixel_wise_softmax(output_map):
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map,tf.reverse(exponential_map,[False,False,False,True]))
    return tf.div(exponential_map,evidence, name="pixel_wise_softmax")

def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1,  tf.shape(output_map)[3]]))
    return tf.clip_by_value( tf.div(exponential_map,tensor_sum_exp), -1.0 * 1e15, 1.0* 1e15, name = "pixel_softmax_2d")

def cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")
#     return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output_map), reduction_indices=[1]))
# residual_block

def residual_block(x, w1, w2, keep_prob, inc_dim = False, is_train = True, scope = None, bn_trainable = True, leak = False, padding = 'SAME'):
    """Args:
        adapt_scope: a flag indicating the variable scope for batch_norm
    """
    _x_channel = x.get_shape().as_list()[-1]
    if scope is None:
        _loc_scope1 = None
        _loc_scope2 = None
    else:
        _loc_scope1 = scope + "_1"
        _loc_scope2 = scope + "_2"

    _inner_conv = conv_bn_relu2d(x, w1, keep_prob = keep_prob, is_train = is_train, scope = _loc_scope1, bn_trainable = bn_trainable, leak = leak, padding = padding)
    _inner_conv = conv_bn_2d(_inner_conv, w2, keep_prob = keep_prob, is_train = is_train, scope = _loc_scope2, bn_trainable = bn_trainable, padding = padding)
    if inc_dim is True:
        x_s = tf.pad(x, [ [0,0], [0,0], [0,0], [_x_channel // 2, _x_channel // 2]])
    else:
        x_s = x
    if leak is False:
        return tf.nn.relu(x_s + _inner_conv)
    else:
        return tf.nn.leaky_relu(x_s + _inner_conv)

def DR_block(x, w1, w2, rate, keep_prob, inc_dim = False, is_train = True, bn_trainable = True, scope = None, leak = False):
    _x_channel = x.get_shape().as_list()[-1]

    if scope is None:
        _loc_scope1 = None
        _loc_scope2 = None
    else:
        _loc_scope1 = scope + "_1"
        _loc_scope2 = scope + "_2"

    _inner_conv = dilate_conv_bn_relu2d(x, w1, keep_prob = keep_prob, is_train = is_train,scope = _loc_scope1, bn_trainable = bn_trainable, leak = leak)
    _inner_conv = dilate_conv_bn(_inner_conv, w2, keep_prob = keep_prob, is_train = is_train, scope = _loc_scope2, bn_trainable = bn_trainable)

    if inc_dim is True:
        x_s = tf.pad(x, [ [0,0], [0,0], [0,0], [_x_channel // 2, _x_channel // 2]])
    else:
        x_s = x

    if leak is True:
        return tf.nn.leaky_relu( x_s + _inner_conv )
    else:
        return tf.nn.relu(  x_s + _inner_conv )
